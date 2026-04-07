"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

import inspect
import logging
import math
import os
import time
import datetime as dt
import pytz
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from threading import Lock

import numpy as np
import pandas as pd

from ibkr_multi_asset import create_database as cd
from ibkr_multi_asset import trading_functions as tf
from ibkr_multi_asset import setup_functions as sf
from ibkr_multi_asset.report_generator import generate_live_portfolio_report
from ibkr_multi_asset.setup import trading_app
from ibkr_multi_asset.strategy_runtime import stra, load_strategy_module, get_strategy_file


now_ = dt.datetime.now()
logging.basicConfig(
    filename=f"data/log/log_file_{now_.year}_{now_.month:02d}_{now_.day:02d}_{now_.hour:02d}_{now_.minute:02d}_{now_.second:02d}.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class _SharedOrderIdAllocator:
    def __init__(self, start_id=1):
        self._lock = Lock()
        self._next_id = int(start_id)

    def reserve(self, count=1):
        count = max(1, int(count))
        with self._lock:
            start = self._next_id
            self._next_id += count
            return start


def _format_sleep_duration(total_seconds):
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _format_sleep_label(total_seconds, cadence_label=None):
    total_seconds = max(0, int(total_seconds))
    if cadence_label:
        try:
            cadence_seconds = int(tf.get_frequency_change(cadence_label).total_seconds())
            if abs(total_seconds - cadence_seconds) <= 60:
                return cadence_label
        except Exception:
            pass
    return _format_sleep_duration(total_seconds)


def _load_strategy_state(database_path):
    path = Path(database_path)
    if not path.exists():
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
    try:
        df = pd.read_excel(path, sheet_name='strategy_state')
    except Exception:
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
    if df.empty:
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
    unnamed = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if unnamed:
        df = df.drop(columns=unnamed)
    canonical_cols = ['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime']
    rename_map = {}
    for canonical in canonical_cols:
        if canonical in df.columns:
            continue
        duplicate_candidates = [col for col in df.columns if str(col).split('.', 1)[0] == canonical]
        if duplicate_candidates:
            rename_map[duplicate_candidates[0]] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    existing_cols = [col for col in canonical_cols if col in df.columns]
    if existing_cols:
        df = df.loc[:, existing_cols].copy()
    for canonical in canonical_cols:
        if canonical not in df.columns:
            df[canonical] = pd.NA
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    return df


def _portfolio_period_marker_path(database_path):
    workbook_path = Path(database_path)
    return workbook_path.with_name("last_completed_portfolio_period.txt")


def _carry_protection_marker_path(database_path):
    workbook_path = Path(database_path)
    return workbook_path.with_name("last_carry_protection_refresh.txt")


def _read_portfolio_period_marker(database_path):
    marker_path = _portfolio_period_marker_path(database_path)
    if not marker_path.exists():
        return None
    try:
        raw = marker_path.read_text(encoding='utf-8').strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return pd.Timestamp(raw).isoformat()
    except Exception:
        return raw


def _write_portfolio_period_marker(database_path, current_period):
    marker_path = _portfolio_period_marker_path(database_path)
    marker_value = pd.Timestamp(current_period).isoformat()
    tmp_path = marker_path.with_suffix(f"{marker_path.suffix}.tmp")
    tmp_path.write_text(marker_value, encoding='utf-8')
    tmp_path.replace(marker_path)


def _read_carry_protection_marker(database_path):
    marker_path = _carry_protection_marker_path(database_path)
    if not marker_path.exists():
        return None
    try:
        return marker_path.read_text(encoding='utf-8').strip() or None
    except Exception:
        return None


def _write_carry_protection_marker(database_path, marker_value):
    marker_path = _carry_protection_marker_path(database_path)
    tmp_path = marker_path.with_suffix(f"{marker_path.suffix}.tmp")
    tmp_path.write_text(str(marker_value), encoding='utf-8')
    tmp_path.replace(marker_path)


def _was_carry_protection_refreshed(database_path, reopen_datetime):
    marker_value = _read_carry_protection_marker(database_path)
    return marker_value == pd.Timestamp(reopen_datetime).isoformat()


def _in_carry_protection_window(now_datetime, reopen_datetime, day_start_datetime):
    return pd.Timestamp(reopen_datetime) <= pd.Timestamp(now_datetime) < pd.Timestamp(day_start_datetime)


def _was_portfolio_period_completed(database_path, current_period):
    marker_value = _read_portfolio_period_marker(database_path)
    current_value = pd.Timestamp(current_period).isoformat()
    if marker_value == current_value:
        return True
    state_df = _load_strategy_state(database_path)
    if state_df.empty:
        return False
    mask = (
        state_df['Symbol'].astype(str).str.upper() == 'PORTFOLIO'
    ) & (
        state_df['Scope'].astype(str).str.upper() == 'PORTFOLIO'
    ) & (
        state_df['StateKey'].astype(str).str.upper() == 'LAST_COMPLETED_PERIOD'
    )
    if not mask.any():
        return False
    latest = state_df.loc[mask].sort_values('datetime').iloc[-1]
    return str(latest.get('StateValue', '')) == current_value


def _were_all_symbols_period_traded(database_path, current_period, symbol_specs):
    path = Path(database_path)
    if not path.exists():
        return False
    try:
        periods_df = pd.read_excel(path, sheet_name='periods_traded')
    except Exception:
        return False
    if periods_df.empty or 'trade_time' not in periods_df.columns or 'trade_done' not in periods_df.columns or 'Symbol' not in periods_df.columns:
        return False

    periods_df = periods_df.copy()
    unnamed = [col for col in periods_df.columns if str(col).startswith('Unnamed:')]
    if unnamed:
        periods_df = periods_df.drop(columns=unnamed)
    periods_df['trade_time'] = pd.to_datetime(periods_df['trade_time'], errors='coerce')
    periods_df['Symbol'] = periods_df['Symbol'].astype(str).str.upper()
    periods_df['trade_done'] = pd.to_numeric(periods_df['trade_done'], errors='coerce').fillna(0)
    periods_df = periods_df.dropna(subset=['trade_time'])
    if periods_df.empty:
        return False

    current_period_ts = pd.Timestamp(current_period)
    period_rows = periods_df[periods_df['trade_time'] == current_period_ts].copy()
    if period_rows.empty:
        return False

    portfolio_rows = period_rows[period_rows['Symbol'].astype(str).str.upper() == 'PORTFOLIO']
    if portfolio_rows.empty:
        return False
    return bool((portfolio_rows['trade_done'] == 1).any())


def _apps_traded_current_period(apps, current_period):
    current_period_ts = pd.Timestamp(current_period)
    for app in apps or []:
        periods_df = getattr(app, 'periods_traded', pd.DataFrame())
        if not isinstance(periods_df, pd.DataFrame) or periods_df.empty:
            continue
        if 'trade_time' not in periods_df.columns or 'trade_done' not in periods_df.columns:
            continue
        local = periods_df.copy()
        local['trade_time'] = pd.to_datetime(local['trade_time'], errors='coerce')
        local['trade_done'] = pd.to_numeric(local['trade_done'], errors='coerce').fillna(0)
        period_rows = local[local['trade_time'] == current_period_ts]
        if not period_rows.empty and bool((period_rows['trade_done'] == 1).any()):
            return True
    return False


def _mark_portfolio_period_completed(database_path, current_period, apps=None):
    workbook_path = Path(database_path)
    completed_at = dt.datetime.now()
    state_value = pd.Timestamp(current_period).isoformat()
    _write_portfolio_period_marker(database_path, current_period)
    row = pd.DataFrame([
        {
            'Symbol': 'PORTFOLIO',
            'Scope': 'PORTFOLIO',
            'StateKey': 'LAST_COMPLETED_PERIOD',
            'StateValue': state_value,
            'datetime': completed_at,
        }
    ])
    state_df = _load_strategy_state(workbook_path)
    state_df = pd.concat([state_df, row], ignore_index=True)
    state_df['datetime'] = pd.to_datetime(state_df['datetime'], errors='coerce')
    state_df = state_df.dropna(subset=['datetime'])
    state_df = state_df.sort_values('datetime')
    state_df = state_df.drop_duplicates(subset=['Symbol', 'Scope', 'StateKey'], keep='last')
    with pd.ExcelWriter(workbook_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        state_df.to_excel(writer, sheet_name='strategy_state', index=False)

    if apps:
        for app in apps:
            try:
                app_state_df = getattr(app, 'strategy_state_df', pd.DataFrame()).copy()
                if not app_state_df.empty:
                    if 'datetime' not in app_state_df.columns:
                        app_state_df = app_state_df.reset_index()
                    unnamed = [col for col in app_state_df.columns if str(col).startswith('Unnamed:')]
                    if unnamed:
                        app_state_df = app_state_df.drop(columns=unnamed)
                else:
                    app_state_df = pd.DataFrame(columns=['datetime', 'Symbol', 'Scope', 'StateKey', 'StateValue'])
                app_state_df = pd.concat([app_state_df, row], ignore_index=True)
                app_state_df['datetime'] = pd.to_datetime(app_state_df['datetime'], errors='coerce')
                app_state_df = app_state_df.dropna(subset=['datetime'])
                app_state_df = app_state_df.sort_values('datetime')
                app_state_df = app_state_df.drop_duplicates(subset=['Symbol', 'Scope', 'StateKey'], keep='last')
                app_state_df.set_index('datetime', inplace=True)
                app_state_df.index.name = ''
                app.strategy_state_df = app_state_df
            except Exception:
                pass


def _validate_user_configuration(variables):
    required = ['host', 'account', 'timezone', 'port', 'account_currency', 'client_id']
    missing = [key for key in required if key not in variables]
    if missing:
        raise ValueError(f"Missing required variables in main.py: {', '.join(missing)}")

    required_strategy_attrs = ['prepare_base_df', 'get_signal', 'get_asset_frequency', 'get_asset_train_span']
    missing_attrs = [name for name in required_strategy_attrs if not hasattr(stra, name)]
    if missing_attrs:
        raise ValueError(f"strategy.py is missing required callables: {', '.join(missing_attrs)}")

    universe_lists = ['fx_pairs', 'futures_symbols', 'metals_symbols', 'crypto_symbols']
    for name in universe_lists:
        value = variables.get(name, [])
        if value is None:
            variables[name] = []
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple in main.py")


def _bootstrap_shared_workbook(variables):
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'historical'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'base_frames'), exist_ok=True)
    database_path = os.path.join(data_dir, 'database.xlsx')
    email_info_path = os.path.join(data_dir, 'email_info.xlsx')
    cd.ensure_trading_info_workbook(
        variables.get('smtp_username'),
        variables.get('to_email'),
        variables.get('password'),
        database_path=database_path,
        email_info_path=email_info_path,
    )
    return database_path, email_info_path


def _normalize_symbol_specs(variables):
    specs = []
    for fx in variables.get("fx_pairs", []):
        specs.append({"symbol": str(fx).upper(), "asset_class": "forex", "exchange": variables.get("forex_exchange", "IDEALPRO"), "currency": variables.get("forex_currency", "USD"), "sec_type": "CASH"})
    for fut in variables.get("futures_symbols", []):
        specs.append({"symbol": str(fut).upper(), "asset_class": "futures", "exchange": variables.get("futures_exchange", "CME"), "currency": variables.get("futures_currency", "USD"), "sec_type": "FUT", "roll_policy": variables.get("futures_roll_policy", "AUTO_FRONT_MONTH"), "contract_month": variables.get("futures_contract_month"), "multiplier": variables.get("futures_multiplier")})
    for metal in variables.get("metals_symbols", []):
        specs.append({"symbol": str(metal).upper(), "asset_class": "metals", "exchange": variables.get("metals_exchange", "IDEALPRO"), "currency": variables.get("metals_currency", "USD"), "sec_type": variables.get("metals_sec_type", "CASH"), "contract_month": variables.get("metals_contract_month"), "multiplier": variables.get("metals_multiplier")})
    for crypto in variables.get("crypto_symbols", []):
        specs.append({"symbol": str(crypto).upper(), "asset_class": "crypto", "exchange": variables.get("crypto_exchange", "PAXOS"), "currency": variables.get("crypto_currency", "USD"), "sec_type": "CRYPTO", "contract_month": None, "multiplier": None})
    if not specs:
        raise ValueError("No tradable symbols found. Define fx_pairs/futures_symbols/metals_symbols/crypto_symbols in main.py")
    seen = set()
    dedup = []
    for spec in specs:
        if spec["symbol"] in seen:
            continue
        seen.add(spec["symbol"])
        dedup.append(spec)
    return dedup


def _ensure_history_file(path, seed_dt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    idx = pd.to_datetime([seed_dt - dt.timedelta(days=3), seed_dt - dt.timedelta(days=2)])
    pd.DataFrame({"Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0], "Close": [1.0, 1.0]}, index=idx).to_csv(path)


def _build_strategy_call_args(func, variables, extra_context=None):
    context = dict(variables)
    if extra_context:
        context.update(extra_context)

    params = []
    signature = inspect.signature(func)
    for name, param in signature.parameters.items():
        if name in context:
            params.append(context[name])
        elif param.default is not inspect.Parameter.empty:
            params.append(param.default)
        else:
            raise KeyError(name)
    return params


def _ensure_strategy_optimization_ready(variables, symbol_specs, optimization_frequency=None, optimization_bucket=None, optimized_at=None):
    optimization_func = getattr(stra, 'strategy_parameter_optimization', None)
    if not callable(optimization_func):
        raise ValueError('strategy.py must define strategy_parameter_optimization() before live trading.')

    optimization_result = optimization_func(*_build_strategy_call_args(
        optimization_func,
        variables,
        {
            'symbol_specs': symbol_specs,
            'optimization_frequency': optimization_frequency,
            'optimization_bucket': optimization_bucket,
            'optimized_at': optimized_at,
        },
    ))

    validation_func = getattr(stra, 'validate_strategy_optimization', None)
    if callable(validation_func):
        validation_func(*_build_strategy_call_args(
            validation_func,
            variables,
            {
                'symbol_specs': symbol_specs,
                'optimization_result': optimization_result,
                'optimization_frequency': optimization_frequency,
                'optimization_bucket': optimization_bucket,
            },
        ))

    return optimization_result


def _optimization_schedule_path():
    return Path("data/models/strategy_optimization_schedule.json")


def _read_optimization_schedule():
    path = _optimization_schedule_path()
    if not path.exists():
        return {}
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        try:
            import json
            return json.loads(path.read_text())
        except Exception:
            return {}


def _write_optimization_schedule(payload):
    path = _optimization_schedule_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _optimization_bucket(now_dt, timezone_name, optimization_frequency):
    freq = str(optimization_frequency or "weekly").strip().lower()
    local_tz = pytz.timezone(timezone_name)
    if now_dt.tzinfo is None:
        local_now = local_tz.localize(now_dt)
    else:
        local_now = now_dt.astimezone(local_tz)
    local_now = local_now.replace(microsecond=0)
    if freq == "daily":
        return local_now.strftime("%Y-%m-%d")
    if freq == "weekly":
        iso_year, iso_week, _ = local_now.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    return "startup"


def _ensure_strategy_optimization_for_schedule(variables, symbol_specs, now_dt=None):
    timezone_name = variables["timezone"]
    optimization_frequency = str(variables.get("optimization_frequency", "weekly")).strip().lower()
    strategy_file = str(variables.get("strategy_file", "strategy.py")).strip() or "strategy.py"
    now_dt = now_dt or dt.datetime.now()
    bucket = _optimization_bucket(now_dt, timezone_name, optimization_frequency)
    schedule = _read_optimization_schedule()

    same_strategy = schedule.get("strategy_file") == strategy_file
    same_frequency = schedule.get("optimization_frequency") == optimization_frequency
    same_bucket = schedule.get("bucket") == bucket

    validation_func = getattr(stra, 'validate_strategy_optimization', None)
    optimization_needed = not (same_strategy and same_frequency and same_bucket)

    if not optimization_needed and callable(validation_func):
        try:
            validation_func(*_build_strategy_call_args(
                validation_func,
                variables,
                {
                    'symbol_specs': symbol_specs,
                    'optimization_frequency': optimization_frequency,
                    'optimization_bucket': bucket,
                },
            ))
            return schedule
        except Exception:
            optimization_needed = True

    if optimization_needed:
        result = _ensure_strategy_optimization_ready(
            variables,
            symbol_specs,
            optimization_frequency=optimization_frequency,
            optimization_bucket=bucket,
            optimized_at=pd.Timestamp(now_dt).isoformat(),
        )
        schedule_payload = {
            "strategy_file": strategy_file,
            "optimization_frequency": optimization_frequency,
            "bucket": bucket,
            "optimized_at": pd.Timestamp(now_dt).isoformat(),
            "strategy_module": getattr(getattr(stra, "__class__", None), "__name__", "StrategyProxy"),
        }
        if isinstance(result, dict) and "config_hash" in result:
            schedule_payload["config_hash"] = result["config_hash"]
        _write_optimization_schedule(schedule_payload)
        return schedule_payload

    return schedule


def _resolve_engine_cycle_frequency(symbol_specs):
    frequencies = []
    for spec in symbol_specs:
        symbol = spec['symbol']
        try:
            freq = stra.get_asset_frequency(symbol)
        except Exception as exc:
            raise ValueError(f"strategy.py could not resolve a data frequency for {symbol}: {exc}") from exc
        if not isinstance(freq, str) or not freq.strip():
            raise ValueError(f"strategy.py returned an invalid data frequency for {symbol}: {freq!r}")
        frequencies.append(freq)
    return min(frequencies, key=lambda value: tf.get_frequency_change(value))


def _get_asset_runtime_policy(symbol_spec):
    defaults = {
        "session": "weekdays",
        "flatten_at_day_end": False,
        "daily_maintenance_utc_start": "00:00",
        "daily_maintenance_minutes": 15,
    }
    try:
        if hasattr(stra, "get_asset_runtime_policy"):
            policy = stra.get_asset_runtime_policy(
                symbol_spec.get("symbol"),
                asset_class=symbol_spec.get("asset_class"),
            )
            if isinstance(policy, dict):
                return {
                    "session": str(policy.get("session", defaults["session"])),
                    "flatten_at_day_end": bool(policy.get("flatten_at_day_end", defaults["flatten_at_day_end"])),
                    "daily_maintenance_utc_start": str(policy.get("daily_maintenance_utc_start", defaults["daily_maintenance_utc_start"])),
                    "daily_maintenance_minutes": int(policy.get("daily_maintenance_minutes", defaults["daily_maintenance_minutes"]) or 0),
                }
    except Exception:
        pass
    return defaults


def _is_asset_session_open(now_dt, symbol_spec, market_open_time, market_close_time, timezone_name):
    policy = _get_asset_runtime_policy(symbol_spec)
    session = str(policy.get("session", "weekdays")).strip().lower()
    local_tz = pytz.timezone(timezone_name)
    if now_dt.tzinfo is None:
        now_local = local_tz.localize(now_dt)
    else:
        now_local = now_dt.astimezone(local_tz)
    now_utc = now_local.astimezone(pytz.UTC)

    if session in {"24_7", "24x7", "always", "always_on"}:
        is_weekly_open = True
    elif session == "weekdays":
        # UTC week gate: open from Sunday 00:15 UTC through Saturday 00:00 UTC,
        # with the daily maintenance window applied separately below.
        is_weekly_open = now_utc.weekday() != 5
    elif session == "auto":
        # 'auto' indicates we should rely on IBKR contract details LiquidHours.
        # We return True here to allow the bootstrap to proceed; sf._is_within_asset_trading_hours
        # will do the actual gatekeeping once contract details are available.
        return True
    else:
        is_weekly_open = market_open_time <= now_dt <= market_close_time
    if not is_weekly_open:
        return False

    maintenance_start = str(policy.get("daily_maintenance_utc_start", "00:00")).strip()
    maintenance_minutes = int(policy.get("daily_maintenance_minutes", 0) or 0)
    if maintenance_minutes <= 0:
        return True

    try:
        start_hour, start_minute = [int(x) for x in maintenance_start.split(":", 1)]
        maintenance_begin = now_utc.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        maintenance_end = maintenance_begin + dt.timedelta(minutes=maintenance_minutes)
        if maintenance_begin <= now_utc < maintenance_end:
            return False
    except Exception:
        pass
    return True


def _should_flatten_at_day_end(symbol_spec):
    policy = _get_asset_runtime_policy(symbol_spec)
    return bool(policy.get("flatten_at_day_end", False))


def _create_symbol_app(host, port, account, client_id, timezone, account_currency, symbol_spec, data_frequency, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, train_span, test_span, trail, leverage, historical_data_address, base_df_address, database_path, email_info_path, strict_targets_validation, allowed_symbols, client_id_offset=0):
    symbol = symbol_spec["symbol"]
    print(f"[{symbol}] Running period {current_period}")
    logging.info(f"[{symbol}] Running period {current_period}")

    app = trading_app(
        logging,
        account,
        account_currency,
        symbol,
        timezone,
        data_frequency,
        historical_data_address,
        base_df_address,
        market_open_time,
        market_close_time,
        previous_day_start_datetime,
        trading_day_end_datetime,
        day_end_datetime,
        current_period,
        previous_period,
        next_period,
        train_span,
        test_span,
        trail,
        leverage,
        asset_spec=symbol_spec,
        strict_targets_validation=strict_targets_validation,
        allowed_symbols=allowed_symbols,
        database_path=database_path,
        email_info_path=email_info_path,
    )

    app.connect(host=host, port=port, clientId=client_id + client_id_offset)
    thread = Thread(target=app.run, daemon=True)
    thread.start()
    return app


def _run_carry_protection_refresh(host, port, account, client_id, timezone, account_currency, symbol_specs, trail, strict_targets_validation, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, current_period, previous_period, next_period, data_dir, database_path, email_info_path):
    apps = []
    allowed_symbols = [s["symbol"] for s in symbol_specs]
    refreshed_any = False

    def _carry_refresh_worker(app):
        refreshed = bool(sf.restore_carry_risk_management(app))
        if refreshed:
            # Persist the newly sent carry-protection orders and state immediately,
            # otherwise the next startup can still read stale workbook data.
            time.sleep(2)
            sf.request_orders(app)
            sf.update_submitted_orders(app)
            sf.save_data(app)
        return refreshed

    try:
        print("Refreshing carry risk management after market reopen...")
        for idx, symbol_spec in enumerate(symbol_specs):
            symbol = symbol_spec["symbol"]
            asset_data_frequency = stra.get_asset_frequency(symbol)
            asset_train_span = stra.get_asset_train_span(symbol)
            asset_test_span = max(1, tf.get_periods_per_day(asset_data_frequency))
            historical_dir = os.path.join(data_dir, "historical")
            base_frames_dir = os.path.join(data_dir, "base_frames")
            os.makedirs(historical_dir, exist_ok=True)
            os.makedirs(base_frames_dir, exist_ok=True)
            historical_data_address = os.path.join(historical_dir, f"historical_{symbol}.csv")
            base_df_address = os.path.join(base_frames_dir, f"app_base_df_{symbol}.csv")
            _ensure_history_file(historical_data_address, current_period)
            app = _create_symbol_app(
                host=host,
                port=port,
                account=account,
                client_id=client_id,
                timezone=timezone,
                account_currency=account_currency,
                symbol_spec=symbol_spec,
                data_frequency=asset_data_frequency,
                current_period=current_period,
                previous_period=previous_period,
                next_period=next_period,
                market_open_time=market_open_time,
                market_close_time=market_close_time,
                previous_day_start_datetime=previous_day_start_datetime,
                trading_day_end_datetime=trading_day_end_datetime,
                day_end_datetime=day_end_datetime,
                train_span=asset_train_span,
                test_span=asset_test_span,
                trail=trail,
                leverage=0.0,
                historical_data_address=historical_data_address,
                base_df_address=base_df_address,
                database_path=database_path,
                email_info_path=email_info_path,
                strict_targets_validation=strict_targets_validation,
                allowed_symbols=allowed_symbols,
                client_id_offset=idx,
            )
            apps.append(app)

        time.sleep(5)
        allocator_seed = max(int(getattr(app, "nextValidOrderId", 1) or 1) for app in apps)
        shared_order_id_allocator = _SharedOrderIdAllocator(allocator_seed)
        for app in apps:
            app.shared_order_id_allocator = shared_order_id_allocator

        if apps:
            with ThreadPoolExecutor(len(apps)) as executor:
                futures = [executor.submit(_carry_refresh_worker, app) for app in apps]
                for future in futures:
                    refreshed_any = bool(future.result()) or refreshed_any
    finally:
        for app in apps:
            try:
                sf.stop(app)
            except Exception:
                pass
    return refreshed_any


def run_portfolio_setup_loop(host, port, account, client_id, timezone, account_currency, symbol_specs, trail, strict_targets_validation):
    variables = tf.extract_variables("main.py")
    trading_day_origin = variables.get("trading_day_origin")
    engine_cycle_frequency = _resolve_engine_cycle_frequency(symbol_specs)
    restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour = tf.get_end_hours(
        timezone,
        trading_day_origin=trading_day_origin,
    )
    market_open_time, market_close_time = tf.define_trading_week(timezone, trading_start_hour, day_end_minute)

    now_dt = dt.datetime.now()
    any_session_open_now = any(
        _is_asset_session_open(now_dt, spec, market_open_time, market_close_time, timezone)
        for spec in symbol_specs
    )
    if not any_session_open_now:
        print("Waiting for market open...")
        while not any(
            _is_asset_session_open(dt.datetime.now(), spec, market_open_time, market_close_time, timezone)
            for spec in symbol_specs
        ):
            time.sleep(1)

    allowed_symbols = [s["symbol"] for s in symbol_specs]
    while True:
        try:
            _ensure_strategy_optimization_for_schedule(variables, symbol_specs, dt.datetime.now())
        except Exception as exc:
            print(f"CRITICAL ERROR while running scheduled strategy optimization: {exc}")
            logging.critical(f"Scheduled strategy optimization failed: {exc}")
            time.sleep(5)
            continue

        market_open_time, market_close_time = tf.define_trading_week(timezone, trading_start_hour, day_end_minute)
        _, _, _, day_start_datetime, _, trading_day_end_datetime, day_end_datetime, previous_day_start_datetime = tf.get_restart_and_day_close_datetimes(
            engine_cycle_frequency,
            dt.datetime.now(),
            day_end_hour,
            day_end_minute,
            restart_hour,
            restart_minute,
            trading_start_hour,
        )
        previous_period, current_period, next_period = tf.get_the_closest_periods(
            dt.datetime.now(),
            engine_cycle_frequency,
            trading_day_end_datetime,
            previous_day_start_datetime,
            day_start_datetime,
            market_close_time,
        )

        apps = []
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        database_path = os.path.join(data_dir, "database.xlsx")
        email_info_path = os.path.join(data_dir, "email_info.xlsx")

        for _ in range(5):
            print("=" * 100)
        print(f"Portfolio trading period: {current_period}")
        print(f"Next period: {next_period}")

        now_before_cycle = dt.datetime.now()
        local_market_reopen = tf.get_local_market_reopen_datetime(now_before_cycle, timezone)
        carry_window_end = now_before_cycle.replace(
            hour=restart_hour,
            minute=restart_minute,
            second=0,
            microsecond=0,
        )
        if _in_carry_protection_window(now_before_cycle, local_market_reopen, carry_window_end):
            print(f"Within carry-protection window from market reopen until trading_day_origin ({carry_window_end}).")
            _run_carry_protection_refresh(
                host=host,
                port=port,
                account=account,
                client_id=client_id,
                timezone=timezone,
                account_currency=account_currency,
                symbol_specs=symbol_specs,
                trail=trail,
                strict_targets_validation=strict_targets_validation,
                market_open_time=market_open_time,
                market_close_time=market_close_time,
                previous_day_start_datetime=previous_day_start_datetime,
                trading_day_end_datetime=trading_day_end_datetime,
                day_end_datetime=day_end_datetime,
                current_period=current_period,
                previous_period=previous_period,
                next_period=next_period,
                data_dir=data_dir,
                database_path=database_path,
                email_info_path=email_info_path,
            )
            sleep_seconds = max(1, math.ceil((carry_window_end - dt.datetime.now()).total_seconds()))
            print(
                f"Carry protection window active. Sleeping until "
                f"{carry_window_end.strftime('%H:%M')} ({sleep_seconds}s)."
            )
            time.sleep(sleep_seconds)
            continue

        if _were_all_symbols_period_traded(database_path, current_period, symbol_specs):
            print(f"Portfolio trading period {current_period} was already completed. Skipping cycle work before app bootstrap...")
            now_after_cycle = dt.datetime.now()
            if now_after_cycle < next_period:
                sleep_target = next_period
            else:
                _, _, sleep_target = tf.get_the_closest_periods(
                    now_after_cycle,
                    engine_cycle_frequency,
                    trading_day_end_datetime,
                    previous_day_start_datetime,
                    day_start_datetime,
                    market_close_time,
                )
            sleep_seconds = max(1, math.ceil((sleep_target - dt.datetime.now()).total_seconds()))
            print(f"Cycle done. Sleeping {_format_sleep_label(sleep_seconds, engine_cycle_frequency)} until next period ({sleep_seconds}s).")
            time.sleep(sleep_seconds)
            continue

        try:
            for idx, symbol_spec in enumerate(symbol_specs):
                symbol = symbol_spec["symbol"]
                asset_data_frequency = stra.get_asset_frequency(symbol)
                asset_train_span = stra.get_asset_train_span(symbol)
                asset_test_span = max(1, tf.get_periods_per_day(asset_data_frequency))
                historical_dir = os.path.join(data_dir, "historical")
                base_frames_dir = os.path.join(data_dir, "base_frames")
                os.makedirs(historical_dir, exist_ok=True)
                os.makedirs(base_frames_dir, exist_ok=True)
                historical_data_address = os.path.join(historical_dir, f"historical_{symbol}.csv")
                base_df_address = os.path.join(base_frames_dir, f"app_base_df_{symbol}.csv")

                _ensure_history_file(historical_data_address, current_period)
                app = _create_symbol_app(
                    host=host,
                    port=port,
                    account=account,
                    client_id=client_id,
                    timezone=timezone,
                    account_currency=account_currency,
                    symbol_spec=symbol_spec,
                    data_frequency=asset_data_frequency,
                    current_period=current_period,
                    previous_period=previous_period,
                    next_period=next_period,
                    market_open_time=market_open_time,
                    market_close_time=market_close_time,
                    previous_day_start_datetime=previous_day_start_datetime,
                    trading_day_end_datetime=trading_day_end_datetime,
                    day_end_datetime=day_end_datetime,
                    train_span=asset_train_span,
                    test_span=asset_test_span,
                    trail=trail,
                    leverage=0.0,
                    historical_data_address=historical_data_address,
                    base_df_address=base_df_address,
                    database_path=database_path,
                    email_info_path=email_info_path,
                    strict_targets_validation=strict_targets_validation,
                    allowed_symbols=allowed_symbols,
                    client_id_offset=idx,
                )
                apps.append(app)

            time.sleep(5)
            allocator_seed = max(
                int(getattr(app, "nextValidOrderId", 1) or 1)
                for app in apps
            )
            shared_order_id_allocator = _SharedOrderIdAllocator(allocator_seed)
            for app in apps:
                app.shared_order_id_allocator = shared_order_id_allocator

            tradable_apps = []
            closable_apps = []
            now_dt = dt.datetime.now()
            for app in apps:
                period_rows = app.periods_traded.loc[app.periods_traded["trade_time"] == current_period]
                traded = (not period_rows.empty) and (period_rows["trade_done"].iloc[-1] == 1)
                session_open = _is_asset_session_open(
                    now_dt,
                    getattr(app, "asset_spec", {}),
                    market_open_time,
                    market_close_time,
                    timezone,
                )
                if traded or (not session_open):
                    sf.stop(app)
                    continue

                if now_dt >= trading_day_end_datetime and _should_flatten_at_day_end(getattr(app, "asset_spec", {})):
                    closable_apps.append(app)
                else:
                    tradable_apps.append(app)

            if tradable_apps:
                sf.run_portfolio_cycle_for_the_period(tradable_apps)
            if closable_apps:
                for app in closable_apps:
                    sf.update_and_close_positions(app)
            if _apps_traded_current_period(apps, current_period):
                _mark_portfolio_period_completed(database_path, current_period, apps)
            else:
                for app in apps:
                    sf.stop(app)
                now_after_skip = dt.datetime.now()
                maintenance_reopen = tf.get_local_market_reopen_datetime(now_after_skip, timezone)
                if now_after_skip < maintenance_reopen < next_period:
                    sleep_seconds = max(1, math.ceil((maintenance_reopen - dt.datetime.now()).total_seconds()))
                    print(f"No tradable assets are currently in session. Sleeping until market reopen ({sleep_seconds}s).")
                    time.sleep(sleep_seconds)
                    continue
        except Exception as exc:
            logging.exception(f"Portfolio cycle error: {exc}")
            for app in apps:
                try:
                    sf.stop(app)
                except Exception:
                    pass

        try:
            import collections
            DummyApp = collections.namedtuple("DummyApp", ["logging"])
            generate_live_portfolio_report(DummyApp(logging=logging))
        except Exception as exc:
            logging.error(f"Background report generation error: {exc}")

        now_after_cycle = dt.datetime.now()
        if now_after_cycle < next_period:
            sleep_target = next_period
        else:
            _, _, sleep_target = tf.get_the_closest_periods(
                now_after_cycle,
                engine_cycle_frequency,
                trading_day_end_datetime,
                previous_day_start_datetime,
                day_start_datetime,
                market_close_time,
            )
        sleep_seconds = max(1, math.ceil((sleep_target - dt.datetime.now()).total_seconds()))
        print(f"Cycle done. Sleeping {_format_sleep_label(sleep_seconds, engine_cycle_frequency)} until next period ({sleep_seconds}s).")
        time.sleep(sleep_seconds)


def main():
    try:
        variables = tf.extract_variables("main.py")
    except FileNotFoundError:
        print("CRITICAL ERROR: user_config/main.py not found.")
        logging.critical("user_config/main.py not found.")
        return

    try:
        _validate_user_configuration(variables)
    except Exception as exc:
        print(f"CRITICAL ERROR: {exc}")
        logging.critical(str(exc))
        return

    _bootstrap_shared_workbook(variables)

    host = variables["host"]
    account = variables["account"]
    timezone = variables["timezone"]
    port = variables["port"]
    account_currency = variables["account_currency"]
    client_id = variables["client_id"]
    trail = variables.get("trail", False)
    strict_targets_validation = bool(variables.get("strict_targets_validation", True))
    strategy_file = str(variables.get("strategy_file", "strategy.py")).strip() or "strategy.py"

    try:
        load_strategy_module(strategy_file)
    except Exception as exc:
        print(f"CRITICAL ERROR while loading strategy file '{strategy_file}': {exc}")
        logging.critical(f"Strategy load failed for {strategy_file}: {exc}")
        return

    try:
        symbol_specs = _normalize_symbol_specs(variables)
    except Exception as exc:
        print(f"CRITICAL ERROR while parsing symbol universe: {exc}")
        logging.critical(f"Symbol universe error: {exc}")
        return

    np.random.seed(int(getattr(stra, "STRATEGY_PARAMS", {}).get("seed", 20260308)))
    engine_cycle_frequency = _resolve_engine_cycle_frequency(symbol_specs)

    asset_frequencies = {spec["symbol"]: stra.get_asset_frequency(spec["symbol"]) for spec in symbol_specs}

    print("=" * 100)
    print("Running multi-asset setup")
    print(f"Universe: {[s['symbol'] for s in symbol_specs]}")
    print(f"Strategy file: {strategy_file}")
    print(f"Portfolio loop cadence: {engine_cycle_frequency}")
    print(f"Asset frequencies: {asset_frequencies}")
    print("=" * 100)

    try:
        _ensure_strategy_optimization_for_schedule(variables, symbol_specs, dt.datetime.now())
    except Exception as exc:
        print(f"CRITICAL ERROR while validating strategy optimization: {exc}")
        logging.critical(f"Strategy optimization validation failed: {exc}")
        return

    run_portfolio_setup_loop(
        host=host,
        port=port,
        account=account,
        client_id=client_id,
        timezone=timezone,
        account_currency=account_currency,
        symbol_specs=symbol_specs,
        trail=trail,
        strict_targets_validation=strict_targets_validation,
    )

if __name__ == "__main__":
    main()
