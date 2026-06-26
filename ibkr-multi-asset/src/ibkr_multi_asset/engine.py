"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

import datetime as dt
import logging
import math
import os
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import numpy as np
import pandas as pd

from ibkr_multi_asset import ib_functions as ibf
# Module-level cache for portfolio allocation targets (strategy-agnostic)
_last_allocation_targets = {}
_last_allocation_attrs = {}

from ibkr_multi_asset import setup_functions as sf
from ibkr_multi_asset import trading_functions as tf
from ibkr_multi_asset.create_database import ensure_trading_info_workbook
from ibkr_multi_asset.report_generator import generate_live_portfolio_report
from ibkr_multi_asset.setup import trading_app
from ibkr_multi_asset.strategy_runtime import stra


# Set the logging level to INFO
_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "user_config", "data", "log", f"log_file_{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
os.makedirs(os.path.dirname(_log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=_log_path,
    filemode='w'
)

CONNECT_HANDSHAKE_TIMEOUT_SECONDS = 45


def _main_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "user_config", "main.py")


# Cached path to user_config/ for functions that need to read main.py
_USER_CONFIG_DIR = os.path.dirname(_main_config_path())


def _extract_main_variable(name, default=None):
    """Read a single variable from user_config/main.py with a fallback."""
    try:
        variables = tf.extract_variables(_main_config_path())
        return variables.get(name, default)
    except Exception:
        return default


def _is_within_portfolio_trading_week(now_dt, market_open_time, market_close_time):
    return (now_dt >= market_open_time) and (now_dt <= market_close_time)


def _create_symbol_app(host, port, account, client_id, timezone, account_currency, symbol_spec, data_frequency, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, train_span, test_span, trail, leverage, historical_data_address, base_df_address, database_path, email_info_path, strict_targets_validation, allowed_symbols, client_id_offset=0, announce=True):
    symbol = symbol_spec["symbol"]
    if announce:
        print(f"[{symbol}] Running period {current_period}")
    logging.info(f"[{symbol}] Running period {current_period}")
    effective_client_id = int(client_id) + int(client_id_offset)

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

    print(f"[PORTFOLIO] Connecting to IBKR at {host}:{port} with client id {effective_client_id}...")
    app.connect(host=host, port=port, clientId=effective_client_id)
    
    # Launch the run thread immediately as EClient needs it to process the handshake
    thread = Thread(target=app.run, daemon=True)
    thread.start()

    # Wait for connection to be established and handshake (nextValidOrderId) to complete
    timeout = CONNECT_HANDSHAKE_TIMEOUT_SECONDS
    start_time = time.time()
    connected = False
    while (time.time() - start_time) < timeout:
        if app.isConnected() and getattr(app, 'nextValidOrderId', None) is not None:
            connected = True
            break
        time.sleep(0.1)

    if not connected:
        error_tail = getattr(app, "errors_dict", {})
        error_tail = dict(list(error_tail.items())[-5:]) if isinstance(error_tail, dict) else error_tail
        print(
            f"[PORTFOLIO] Failed to fully connect/handshake with IBKR within {timeout} seconds "
            f"at {host}:{port} client_id={effective_client_id}. Connected={app.isConnected()} "
            f"nextValidOrderId={getattr(app, 'nextValidOrderId', None)} recent_errors={error_tail}"
        )
        logging.error(
            "[PORTFOLIO] Failed to fully connect/handshake with IBKR within %s seconds at %s:%s client_id=%s. Connected=%s nextValidOrderId=%s recent_errors=%s",
            timeout,
            host,
            port,
            effective_client_id,
            app.isConnected(),
            getattr(app, 'nextValidOrderId', None),
            error_tail,
        )
        try:
            app.disconnect()
        except:
            pass
        return None

    return app


def _symbol_runtime_metadata(symbol_spec, data_dir, current_period):
    symbol = str(symbol_spec["symbol"]).upper()
    asset_data_frequency = stra.get_asset_frequency(symbol)
    asset_train_span = stra.get_asset_train_span(symbol)
    asset_test_span = max(1, tf.get_periods_per_day(asset_data_frequency))
    historical_data_address = os.path.join(data_dir, "historical", f"historical_{symbol}.csv")
    base_df_address = os.path.join(data_dir, "base_frames", f"app_base_df_{symbol}.csv")
    os.makedirs(os.path.dirname(base_df_address), exist_ok=True)
    _ensure_history_file(historical_data_address, current_period)
    return {
        "symbol": symbol,
        "data_frequency": asset_data_frequency,
        "train_span": asset_train_span,
        "test_span": asset_test_span,
        "historical_data_address": historical_data_address,
        "base_df_address": base_df_address,
    }


def _load_symbol_history_for_app(app, metadata):
    historical_data = pd.read_csv(metadata["historical_data_address"], index_col=0)
    historical_data.index = pd.to_datetime(historical_data.index, errors="coerce")
    historical_data = historical_data[~historical_data.index.isna()].sort_index()
    keep_rows = int(metadata["train_span"])
    if str(app.asset_spec.get("asset_class", "forex")).lower() == "crypto":
        keep_rows = max(keep_rows, int(tf.get_periods_per_day(metadata["data_frequency"]) * 40))
    return historical_data.tail(keep_rows)


def _configure_portfolio_app_for_symbol(app, symbol_spec, metadata, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, leverage, strict_targets_validation, allowed_symbols):
    symbol = metadata["symbol"]
    app.ticker = symbol
    app.asset_spec = dict(symbol_spec)
    app.data_frequency = metadata["data_frequency"]
    app.frequency_number, app.frequency_string = tf.get_data_frequency_values(app.data_frequency)
    app.historical_data_address = metadata["historical_data_address"]
    app.base_df_address = metadata["base_df_address"]
    app.current_period = current_period
    app.previous_period = previous_period
    app.next_period = next_period
    app.market_open_time = market_open_time
    app.market_close_time = market_close_time
    app.previous_day_start_datetime = previous_day_start_datetime
    app.trading_day_end_datetime = trading_day_end_datetime
    app.day_end_datetime = day_end_datetime
    app.train_span = metadata["train_span"]
    app.test_span = metadata["test_span"]
    app.leverage = leverage
    app.strict_targets_validation = strict_targets_validation
    app.allowed_symbols = [str(item).upper() for item in allowed_symbols]
    app.contract = ibf.build_contract_from_spec(app.asset_spec)
    app.resolved_contract = None
    app.historical_data = _load_symbol_history_for_app(app, metadata)
    app.base_df = pd.DataFrame()
    if not getattr(app, 'new_df', None):
        app.new_df = {}
    app.hist_data_events = getattr(app, 'hist_data_events', {}) or {}
    app.hist_request_errors = getattr(app, 'hist_request_errors', {}) or {}
    app.errors_dict = {}
    app.bid_price = np.nan
    app.ask_price = np.nan
    app.last_trade_price = np.nan
    app.last_value = np.nan
    app.last_value_count = 0
    app.active_market_data_req_id = None
    app.risk_management_price_overrides = {}
    app.risk_management_position_sign = np.nan
    app.force_new_risk_management_prices = False
    app.temp_contract_details = pd.DataFrame(columns=app.contract_details_df.columns)
    app.strategy_state = app._deserialize_strategy_state(app.strategy_state_df, symbol)
    return app


def _create_portfolio_app(host, port, account, client_id, timezone, account_currency, first_symbol_spec, metadata, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, trail, leverage, database_path, email_info_path, strict_targets_validation, allowed_symbols, client_id_offset=0):
    print(f"[PORTFOLIO] Running period {current_period} with one IB app for {len(allowed_symbols)} symbols")
    return _create_symbol_app(
        host=host,
        port=port,
        account=account,
        client_id=client_id,
        timezone=timezone,
        account_currency=account_currency,
        symbol_spec=first_symbol_spec,
        data_frequency=metadata["data_frequency"],
        current_period=current_period,
        previous_period=previous_period,
        next_period=next_period,
        market_open_time=market_open_time,
        market_close_time=market_close_time,
        previous_day_start_datetime=previous_day_start_datetime,
        trading_day_end_datetime=trading_day_end_datetime,
        day_end_datetime=day_end_datetime,
        train_span=metadata["train_span"],
        test_span=metadata["test_span"],
        trail=trail,
        leverage=leverage,
        historical_data_address=metadata["historical_data_address"],
        base_df_address=metadata["base_df_address"],
        database_path=database_path,
        email_info_path=email_info_path,
        strict_targets_validation=strict_targets_validation,
        allowed_symbols=allowed_symbols,
        client_id_offset=client_id_offset,
        announce=False,
    )


def _copy_portfolio_target_attrs(app):
    return {
        "strategy_targets": deepcopy(getattr(app, "strategy_targets", {}) or {}),
        "target_weights": deepcopy(getattr(app, "target_weights", {}) or {}),
        "applied_target_weights": deepcopy(getattr(app, "applied_target_weights", {}) or {}),
        "leverage": getattr(app, "leverage", 1.0),
        "margin_scale": getattr(app, "margin_scale", 1.0),
        "required_capital_frac": getattr(app, "required_capital_frac", 0.0),
        "used_capital_frac": getattr(app, "used_capital_frac", 0.0),
        "cash_weight": getattr(app, "cash_weight", 0.0),
    }


def _apply_portfolio_target_attrs(app, attrs):
    app.strategy_targets = deepcopy(attrs.get("strategy_targets", {}) or {})
    app.target_weights = deepcopy(attrs.get("target_weights", {}) or {})
    app.applied_target_weights = deepcopy(attrs.get("applied_target_weights", {}) or {})
    app.leverage = float(pd.to_numeric(pd.Series([attrs.get("leverage", 1.0)]), errors="coerce").iloc[0])
    app.margin_scale = attrs.get("margin_scale", 1.0)
    app.required_capital_frac = attrs.get("required_capital_frac", 0.0)
    app.used_capital_frac = attrs.get("used_capital_frac", 0.0)
    app.cash_weight = attrs.get("cash_weight", 0.0)


def _broadcast_shared_account_state(lead_app, apps):
    shared_acc_update = getattr(lead_app, "shared_acc_update", pd.DataFrame()).copy(deep=True)
    shared_cash_balance = getattr(lead_app, "shared_cash_balance", pd.DataFrame()).copy(deep=True)
    shared_portfolio_snapshots = getattr(lead_app, "shared_portfolio_snapshots_df", pd.DataFrame()).copy(deep=True)
    shared_unlevered_capital = float(getattr(lead_app, "shared_unlevered_capital", np.nan))
    live_cash_balance = getattr(lead_app, "cash_balance", pd.DataFrame()).copy(deep=True)
    live_portfolio_snapshots = getattr(lead_app, "portfolio_snapshots_df", pd.DataFrame()).copy(deep=True)
    for app in apps:
        app.shared_acc_update = shared_acc_update.copy(deep=True)
        app.shared_cash_balance = shared_cash_balance.copy(deep=True)
        app.shared_portfolio_snapshots_df = shared_portfolio_snapshots.copy(deep=True)
        app.shared_unlevered_capital = shared_unlevered_capital
        app.cash_balance = live_cash_balance.copy(deep=True)
        app.portfolio_snapshots_df = live_portfolio_snapshots.copy(deep=True)


def _broadcast_shared_broker_state(lead_app, apps):
    shared_state = deepcopy(getattr(lead_app, "shared_broker_state", {}) or {})
    pos_df = getattr(lead_app, "pos_df", pd.DataFrame()).copy(deep=True)
    open_orders = getattr(lead_app, "open_orders", pd.DataFrame()).copy(deep=True)
    orders_status = getattr(lead_app, "orders_status", pd.DataFrame()).copy(deep=True)
    exec_df = getattr(lead_app, "exec_df", pd.DataFrame()).copy(deep=True)
    comm_df = getattr(lead_app, "comm_df", pd.DataFrame()).copy(deep=True)
    current_open_orders_snapshot = getattr(lead_app, "current_open_orders_snapshot", pd.DataFrame()).copy(deep=True)
    snapshot_ready = bool(getattr(lead_app, "shared_broker_snapshot_ready", False))
    snapshot_period = getattr(lead_app, "shared_broker_snapshot_period", None)
    for app in apps:
        app.shared_broker_state = deepcopy(shared_state)
        app.shared_broker_snapshot_ready = snapshot_ready
        app.shared_broker_snapshot_period = snapshot_period
        app.pos_df = pos_df.copy(deep=True)
        app.open_orders = open_orders.copy(deep=True)
        app.orders_status = orders_status.copy(deep=True)
        app.exec_df = exec_df.copy(deep=True)
        app.comm_df = comm_df.copy(deep=True)
        app.current_open_orders_snapshot = current_open_orders_snapshot.copy(deep=True)


def _broadcast_shared_contract_details(lead_app, apps):
    contract_details = getattr(lead_app, "contract_details_df", pd.DataFrame()).copy(deep=True)
    for app in apps:
        app.contract_details_df = contract_details.copy(deep=True)


def _worker_cycle_error_row(app, exc):
    symbol = str(getattr(app, "ticker", "")).upper()
    msg = f"[{symbol}] Order preparation failed: {exc}"
    print(msg)
    logging.exception(msg)
    sf.append_runtime_audit(app, "send_orders_failed", str(exc))
    return msg


def _run_parallel_worker_send_orders(app):
    app.parallel_isolated_order_worker = True
    app.defer_posttrade_sync = True
    app.defer_synthetic_monitors = True
    app.pending_synthetic_monitors = []
    app.use_shared_pretrade_snapshot = True
    try:
        sf.send_orders(app)
    finally:
        app.parallel_isolated_order_worker = False
    return app


def _run_parallel_app_portfolio_cycle(apps, tradable_symbol_specs, metadata_by_symbol, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, leverage, strict_targets_validation, allowed_symbols):
    global _last_allocation_targets, _last_allocation_attrs
    active_apps = [app for app in apps if app is not None and app.isConnected()]
    if len(active_apps) == 0:
        return False

    lead_app = active_apps[0]

    print("Refreshing all symbols before portfolio decisioning with isolated apps...")
    with ThreadPoolExecutor(max_workers=len(active_apps)) as executor:
        futures = [executor.submit(sf.refresh_symbol_market_data, app) for app in active_apps]
        for future in futures:
            future.result()

    # Capture each app's fresh historical data for the strategy's portfolio-level
    # get_signal call (mirrors single-app path's _portfolio_symbol_histories).
    _portfolio_symbol_histories = {}
    for app in active_apps:
        sym = str(getattr(app, 'ticker', '')).upper()
        hist = getattr(app, 'historical_data', None)
        if sym and hist is not None and not hist.empty:
            _portfolio_symbol_histories[sym] = hist.copy()
    lead_app._portfolio_symbol_histories = _portfolio_symbol_histories

    print("Collecting shared account updates once for the full universe...")
    sf.collect_shared_account_snapshot([lead_app])
    _broadcast_shared_account_state(lead_app, active_apps)

    if _is_portfolio_allocation_bar(current_period):
        print("Computing portfolio targets once for the full universe...")
        targets = sf.compute_portfolio_targets_once(active_apps)
        portfolio_attrs = _copy_portfolio_target_attrs(lead_app)
        _last_allocation_targets = dict(targets) if isinstance(targets, dict) else {}
        _last_allocation_attrs = dict(portfolio_attrs) if isinstance(portfolio_attrs, dict) else {}
    else:
        if _last_allocation_targets:
            targets = _last_allocation_targets
            portfolio_attrs = _last_allocation_attrs
        else:
            print("Computing portfolio targets once for the full universe (first cycle)...")
            targets = sf.compute_portfolio_targets_once(active_apps)
            portfolio_attrs = _copy_portfolio_target_attrs(lead_app)
            _last_allocation_targets = dict(targets) if isinstance(targets, dict) else {}
            _last_allocation_attrs = dict(portfolio_attrs) if isinstance(portfolio_attrs, dict) else {}

    print("Resolving contract details once for the full universe...")
    sf.collect_shared_contract_details(lead_app, tradable_symbol_specs)
    _broadcast_shared_contract_details(lead_app, active_apps)

    print("Collecting shared positions, open orders, and executions once for the full universe...")
    sf.collect_shared_broker_snapshot(lead_app)
    _broadcast_shared_broker_state(lead_app, active_apps)

    for app in active_apps:
        _apply_portfolio_target_attrs(app, portfolio_attrs)
        symbol = str(app.ticker).upper()
        target = targets.get(symbol, {"signal": 0}) if isinstance(targets, dict) else {"signal": 0}
        app.signal = int(np.sign(float(target.get("signal", 0.0))))
        # Per-asset trading gate + refresh: only trade at this asset's bar frequency
        if not _is_asset_trading_bar(symbol, current_period):
            app.trading_status = "SKIPPED"
            continue
        # Refresh per-symbol signal from latest data
        try:
            stra.refresh_symbol_signal(app)
        except (AttributeError, Exception):
            pass
        app.trading_status = "TRADING"

    trading_apps = [app for app in active_apps if getattr(app, 'trading_status', '') == "TRADING"]
    if not trading_apps:
        print("No assets are at their trading bar this cycle. Skipping order submission.")
        print("Portfolio order summary...")
        sf.print_portfolio_order_summary(active_apps)
        for app in active_apps:
            app.strategy_end = True
        print("Saving the data and sending the email...")
        sf.save_portfolio_cycle_data(active_apps, send_email_summary=True)
        return True

    # Shared capital enforcement: respect strategy's cash_weight (strategy-agnostic)
    cash_weight = float(getattr(lead_app, 'cash_weight', 0.0))
    if cash_weight > 0 and trading_apps:
        max_deployed = 1.0 - cash_weight
        total_weight = sum(
            float(getattr(a, 'target_weights', {}).get(str(getattr(a, 'ticker', '')).upper(), 0.0))
            for a in trading_apps
        )
        if total_weight > max_deployed and max_deployed > 0:
            scale = max_deployed / total_weight
            for a in trading_apps:
                a.leverage = float(a.leverage) * scale
    print("Preparing and sending orders in parallel with isolated IB apps...")
    cycle_errors = []
    with ThreadPoolExecutor(max_workers=len(trading_apps)) as executor:
        futures = {executor.submit(_run_parallel_worker_send_orders, app): app for app in trading_apps}
        for future, app in futures.items():
            try:
                future.result()
            except Exception as exc:
                cycle_errors.append(_worker_cycle_error_row(app, exc))

    aggregated_pending = []
    for app in active_apps:
        aggregated_pending.extend(list(getattr(app, "pending_synthetic_monitors", []) or []))
    lead_app.pending_synthetic_monitors = aggregated_pending

    print("Collecting shared post-trade positions, open orders, and executions once for the full universe...")
    sf.collect_shared_broker_snapshot(lead_app)
    _broadcast_shared_broker_state(lead_app, active_apps)

    if getattr(lead_app, "pending_synthetic_monitors", []):
        _run_deferred_synthetic_monitor_phase(
            lead_app,
            tradable_symbol_specs,
            metadata_by_symbol,
            current_period,
            previous_period,
            next_period,
            market_open_time,
            market_close_time,
            previous_day_start_datetime,
            trading_day_end_datetime,
            day_end_datetime,
            leverage,
            strict_targets_validation,
            allowed_symbols,
        )
        print("Collecting final shared broker snapshot once after the deferred synthetic monitor phase...")
        sf.collect_shared_broker_snapshot(lead_app)
        _broadcast_shared_broker_state(lead_app, active_apps)

    print("Portfolio order summary...")
    sf.print_portfolio_order_summary(active_apps)

    if cycle_errors:
        print("Portfolio cycle warnings:")
        for msg in cycle_errors:
            print(msg)

    for app in active_apps:
        app.strategy_end = True

    print("Saving the data and sending the email...")
    sf.save_portfolio_cycle_data(active_apps, send_email_summary=True)
    return True


def _run_deferred_synthetic_monitor_phase(app, tradable_symbol_specs, metadata_by_symbol, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, leverage, strict_targets_validation, allowed_symbols):
    pending = list(getattr(app, "pending_synthetic_monitors", []) or [])
    if app is None or not app.isConnected() or len(pending) == 0:
        return

    print(f"Running deferred synthetic crypto monitor sweep until {next_period}...")
    symbol_specs_by_symbol = {
        str(symbol_spec["symbol"]).upper(): symbol_spec
        for symbol_spec in tradable_symbol_specs
    }

    while app.isConnected() and dt.datetime.now() < pd.Timestamp(next_period).to_pydatetime() and len(pending) > 0:
        sf.collect_shared_broker_snapshot(app)
        remaining = []

        for item in pending:
            symbol = str(item.get("symbol", "")).upper()
            symbol_spec = symbol_specs_by_symbol.get(symbol)
            metadata = metadata_by_symbol.get(symbol)
            if symbol_spec is None or metadata is None:
                continue

            _configure_portfolio_app_for_symbol(
                app,
                symbol_spec,
                metadata,
                current_period,
                previous_period,
                next_period,
                market_open_time,
                market_close_time,
                previous_day_start_datetime,
                trading_day_end_datetime,
                day_end_datetime,
                leverage,
                strict_targets_validation,
                allowed_symbols,
            )
            if item.get("resolved_contract") is not None:
                app.resolved_contract = deepcopy(item["resolved_contract"])
                app.contract = deepcopy(item["resolved_contract"])

            live_quantity = float(sf._latest_position_for_symbol(app, refresh=False, verbose=False))
            if sf._is_effectively_flat_quantity(app, live_quantity):
                continue

            stop_price = float(item.get("stop_price", np.nan))
            live_price = sf._request_live_market_price_snapshot(app, timeout_seconds=2)
            if not sf._stop_is_breached(live_quantity, live_price, stop_price):
                remaining.append(item)
                continue

            print(
                f'[{symbol}] Synthetic stop breached at {live_price:.8f} '
                f'against stop {stop_price:.8f}. Canceling take-profit and sending market exit...'
            )
            tp_order_id = item.get("take_profit_order_id")
            if isinstance(tp_order_id, int):
                try:
                    app.cancelOrder(int(tp_order_id))
                except Exception:
                    pass
                time.sleep(1)

            sf.collect_shared_broker_snapshot(app)
            live_quantity = float(sf._latest_position_for_symbol(app, refresh=False, verbose=False))
            if sf._is_effectively_flat_quantity(app, live_quantity):
                continue

            original_signal = float(getattr(app, "signal", 0.0))
            app.signal = -1.0 if live_quantity > 0 else 1.0
            try:
                exit_order_id = sf._next_order_id(app)
                market_sent = sf.send_market_order(app, exit_order_id, abs(live_quantity))
            finally:
                app.signal = original_signal

            if market_sent:
                print(
                    f'[{symbol}] Synthetic stop exit order sent: '
                    f'order_id={exit_order_id}, side={"SELL" if live_quantity > 0 else "BUY"}, '
                    f'quantity={abs(live_quantity):.8f}'
                )
            else:
                remaining.append(item)

        pending = remaining
        app.pending_synthetic_monitors = list(remaining)
        if len(pending) == 0:
            break
        time.sleep(min(5, max(1, math.ceil((pd.Timestamp(next_period).to_pydatetime() - dt.datetime.now()).total_seconds()))))


def _run_single_app_portfolio_cycle(app, tradable_symbol_specs, metadata_by_symbol, current_period, previous_period, next_period, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, leverage, strict_targets_validation, allowed_symbols):
    global _last_allocation_targets, _last_allocation_attrs
    if app is None or not app.isConnected() or len(tradable_symbol_specs) == 0:
        return False

    print("Refreshing all symbols before portfolio decisioning with one connected app...")
    # Capture each symbol's freshly-updated historical data before the loop overwrites it.
    _portfolio_symbol_histories = {}
    for symbol_spec in tradable_symbol_specs:
        metadata = metadata_by_symbol[str(symbol_spec["symbol"]).upper()]
        _configure_portfolio_app_for_symbol(
            app,
            symbol_spec,
            metadata,
            current_period,
            previous_period,
            next_period,
            market_open_time,
            market_close_time,
            previous_day_start_datetime,
            trading_day_end_datetime,
            day_end_datetime,
            leverage,
            strict_targets_validation,
            allowed_symbols,
        )
        sf.refresh_symbol_market_data(app)
        _portfolio_symbol_histories[str(symbol_spec["symbol"]).upper()] = app.historical_data.copy()
    app._portfolio_symbol_histories = _portfolio_symbol_histories

    print("Collecting shared account updates once for the full universe...")
    sf.collect_shared_account_snapshot([app])

    # Portfolio allocation: only at daily origin bar (once per trading day).
    # The strategy internally caches heavy portfolio weights; per-symbol signals are
    # set here once and orders are gated per-asset by _is_asset_trading_bar below.
    if _is_portfolio_allocation_bar(current_period):
        print("Computing portfolio targets once for the full universe...")
        first_spec = tradable_symbol_specs[0]
        _configure_portfolio_app_for_symbol(
            app, first_spec,
            metadata_by_symbol[str(first_spec["symbol"]).upper()],
            current_period, previous_period, next_period,
            market_open_time, market_close_time,
            previous_day_start_datetime, trading_day_end_datetime, day_end_datetime,
            leverage, strict_targets_validation, allowed_symbols,
        )
        targets = sf.compute_portfolio_targets_once([app])
        portfolio_attrs = _copy_portfolio_target_attrs(app)
        _last_allocation_targets = dict(targets) if isinstance(targets, dict) else {}
        _last_allocation_attrs = dict(portfolio_attrs) if isinstance(portfolio_attrs, dict) else {}
    elif _last_allocation_targets:
        app.logging.info(f"[{app.ticker}] Not a portfolio allocation bar. Reusing previous targets.")
        targets = _last_allocation_targets
        portfolio_attrs = _last_allocation_attrs
        for attr_name, attr_val in _last_allocation_attrs.items():
            if hasattr(app, attr_name) and attr_val is not None:
                try:
                    setattr(app, attr_name, attr_val)
                except Exception:
                    pass
    else:
        print("Computing portfolio targets once for the full universe (first cycle)...")
        first_spec = tradable_symbol_specs[0]
        _configure_portfolio_app_for_symbol(
            app, first_spec,
            metadata_by_symbol[str(first_spec["symbol"]).upper()],
            current_period, previous_period, next_period,
            market_open_time, market_close_time,
            previous_day_start_datetime, trading_day_end_datetime, day_end_datetime,
            leverage, strict_targets_validation, allowed_symbols,
        )
        targets = sf.compute_portfolio_targets_once([app])
        portfolio_attrs = _copy_portfolio_target_attrs(app)
        _last_allocation_targets = dict(targets) if isinstance(targets, dict) else {}
        _last_allocation_attrs = dict(portfolio_attrs) if isinstance(portfolio_attrs, dict) else {}

    print("Resolving contract details once for the full universe...")
    sf.collect_shared_contract_details(app, tradable_symbol_specs)

    print("Collecting shared positions, open orders, and executions once for the full universe...")
    sf.collect_shared_broker_snapshot(app)

    print("Preparing and sending orders sequentially through the single app...")
    app.defer_posttrade_sync = True
    app.defer_synthetic_monitors = True
    app.pending_synthetic_monitors = []
    # Shared capital enforcement: respect strategy's cash_weight (strategy-agnostic)
    _enforce_shared_capital = float(getattr(app, 'cash_weight', 0.0))
    if _enforce_shared_capital > 0:
        _max_deployed = 1.0 - _enforce_shared_capital
        _tw = sum(float(app.target_weights.get(str(s["symbol"]).upper(), 0.0)) for s in tradable_symbol_specs)
        _capital_scale = min(1.0, _max_deployed / max(_tw, 1e-8)) if _tw > _max_deployed else 1.0
    else:
        _capital_scale = 1.0
    rows = []
    cycle_errors = []
    print("=" * 50)
    for symbol_spec in tradable_symbol_specs:
        symbol = str(symbol_spec["symbol"]).upper()
        metadata = metadata_by_symbol[symbol]
        _configure_portfolio_app_for_symbol(
            app,
            symbol_spec,
            metadata,
            current_period,
            previous_period,
            next_period,
            market_open_time,
            market_close_time,
            previous_day_start_datetime,
            trading_day_end_datetime,
            day_end_datetime,
            leverage,
            strict_targets_validation,
            allowed_symbols,
        )
        _apply_portfolio_target_attrs(app, portfolio_attrs)
        target = targets.get(symbol, {"signal": 0}) if isinstance(targets, dict) else {"signal": 0}
        app.signal = int(np.sign(float(target.get("signal", 0.0))))
        app.use_shared_pretrade_snapshot = True
        
        # Per-asset trading gate: only submit orders when bar aligns with asset frequency
        if not _is_asset_trading_bar(symbol, current_period):
            print(f"[{symbol}] SKIPPED: not this asset's trading bar")
            app.trading_status = "SKIPPED"
            rows.append([
                symbol,
                "SKIPPED",
                str(int(getattr(app, "signal", 0))),
                f"{float(getattr(app, 'leverage', 0.0)):.6f}",
                "0.000000",
            ])
            continue
        
        # Refresh per-symbol signal from latest data (matches backtest per-bar logic)
        try:
            stra.refresh_symbol_signal(app)
        except (AttributeError, Exception):
            pass

        app.trading_status = "TRADING"
        print(f"[{symbol}] TRADING: sending orders for this bar")
        # Apply shared capital scale if cash_weight is enforced
        if _capital_scale < 1.0:
            app.leverage = float(app.leverage) * _capital_scale
        try:
            sf.send_orders(app)
        except Exception as exc:
            msg = f"[{symbol}] Order preparation failed: {exc}"
            print(msg)
            logging.exception(msg)
            sf.append_runtime_audit(app, "send_orders_failed", str(exc))
            cycle_errors.append(msg)
        rows.append([
            symbol,
            "TRADING",
            str(int(getattr(app, "signal", 0))),
            f"{float(getattr(app, 'leverage', 0.0)):.6f}",
            f"{float(getattr(app, 'ordered_quantity', 0.0)):.6f}",
        ])
    print("=" * 50)

    print("Collecting shared post-trade positions, open orders, and executions once for the full universe...")
    sf.collect_shared_broker_snapshot(app)

    if getattr(app, "pending_synthetic_monitors", []):
        _run_deferred_synthetic_monitor_phase(
            app,
            tradable_symbol_specs,
            metadata_by_symbol,
            current_period,
            previous_period,
            next_period,
            market_open_time,
            market_close_time,
            previous_day_start_datetime,
            trading_day_end_datetime,
            day_end_datetime,
            leverage,
            strict_targets_validation,
            allowed_symbols,
        )
        print("Collecting final shared broker snapshot once after the deferred synthetic monitor phase...")
        sf.collect_shared_broker_snapshot(app)

    print("Portfolio order summary...")
    _print_portfolio_order_summary_rows(rows)

    if cycle_errors:
        print("Portfolio cycle warnings:")
        for msg in cycle_errors:
            print(msg)

    app.strategy_end = True
    print("Saving the data and sending the email...")
    sf.save_portfolio_cycle_data([app], send_email_summary=True)
    return True


def _print_portfolio_order_summary_rows(rows):
    headers = ["Asset", "Status", "Signal", "Leverage", "OrderedQty"]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def _resolve_engine_cycle_frequency(symbol_specs):
    """Return the engine heartbeat: the finest of all asset signal frequencies
    and the user's portfolio_mark_frequency from main.py."""
    freqs = []
    for spec in symbol_specs:
        try:
            freqs.append(stra.get_asset_frequency(spec["symbol"]))
        except Exception:
            pass
    mark = _extract_main_variable("portfolio_mark_frequency", "5min")
    if mark:
        freqs.append(str(mark).strip())
    return _finest_frequency(freqs) if freqs else "5min"


def _finest_frequency(freq_strings):
    """Return the frequency with the smallest period among a list of frequency strings
    (e.g. '5min' < '20min' < '1h' < '1D').  Falls back to '5min' on parse errors."""
    best_period_min = float('inf')
    best_freq = "5min"
    for f in freq_strings:
        try:
            n, unit = tf.get_data_frequency_values(str(f).strip())
        except Exception:
            continue
        if unit == 'min':
            period = n
        elif unit == 'h':
            period = n * 60
        elif unit == 'D':
            period = n * 24 * 60
        else:
            continue
        if period < best_period_min:
            best_period_min = period
            best_freq = str(f).strip()
    return best_freq


def _is_asset_trading_bar(symbol, current_period_dt):
    """Check if current_period aligns with this asset's trading frequency.
    Sub-hour frequencies (5min, 10min, etc.) always trade.
    Hour/daily frequencies trade only at aligned bars from trading_day_origin.
    
    Returns True if orders should be submitted for this symbol at this bar."""
    freq = stra.get_asset_frequency(symbol)
    freq_val, freq_unit = tf.get_data_frequency_values(freq)
    
    # Convert to minutes for alignment check
    if freq_unit == 'D':
        freq_minutes = freq_val * 24 * 60
    elif freq_unit == 'h':
        freq_minutes = freq_val * 60
    else:  # min or sub-minute
        freq_minutes = freq_val
    
    # Resolve trading_day_origin for alignment
    try:
        origin_str = str(_extract_main_variable("trading_day_origin", "18:00")).strip()
        parts = origin_str.split(":")
        origin_hour, origin_minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        origin_hour, origin_minute = 18, 0
    
    period_minutes = current_period_dt.hour * 60 + current_period_dt.minute
    origin_minutes = origin_hour * 60 + origin_minute
    delta_minutes = (period_minutes - origin_minutes) % (24 * 60)
    
    return delta_minutes % freq_minutes == 0



def _is_portfolio_allocation_bar(current_period_dt):
    """Portfolio allocation only at bars aligned with the strategy's
    rebalance frequency (default '1D'). The gate is strategy-agnostic:
    it uses get_portfolio_rebalance_frequency() from the strategy module."""
    try:
        reb_freq = stra.get_portfolio_rebalance_frequency()
    except Exception:
        reb_freq = "1D"
    try:
        reb_n, reb_unit = tf.get_data_frequency_values(str(reb_freq).strip())
    except Exception:
        reb_n, reb_unit = 1, 'D'
    if reb_unit == 'min':
        reb_minutes = reb_n
    elif reb_unit == 'h':
        reb_minutes = reb_n * 60
    else:
        reb_minutes = reb_n * 24 * 60  # daily or coarser

    try:
        origin_str = str(_extract_main_variable("trading_day_origin", "18:00")).strip()
        parts = origin_str.split(":")
        origin_hour, origin_minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        origin_hour, origin_minute = 18, 0
    period_minutes = current_period_dt.hour * 60 + current_period_dt.minute
    origin_minutes = origin_hour * 60 + origin_minute
    delta_minutes = (period_minutes - origin_minutes) % (24 * 60)
    return delta_minutes % reb_minutes == 0

# NOTE: Duplicate _is_asset_trading_bar removed (used incorrect freq_seconds extraction).
# The correct implementation above is the active one.


def _in_carry_protection_window(now_dt, market_reopen_dt, window_end_dt):
    if now_dt < market_reopen_dt:
        return False
    return now_dt < window_end_dt


def _run_carry_protection_refresh(host, port, account, client_id, timezone, account_currency, symbol_specs, trail, strict_targets_validation, market_open_time, market_close_time, previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, current_period, previous_period, next_period, data_dir, database_path, email_info_path, client_id_offset=0):
    portfolio_app = None
    refreshed_any = False
    try:
        allowed_symbols = [s["symbol"] for s in symbol_specs]
        metadata_by_symbol = {
            str(symbol_spec["symbol"]).upper(): _symbol_runtime_metadata(symbol_spec, data_dir, current_period)
            for symbol_spec in symbol_specs
        }
        if not symbol_specs:
            return

        first_spec = symbol_specs[0]
        first_metadata = metadata_by_symbol[str(first_spec["symbol"]).upper()]
        portfolio_app = _create_portfolio_app(
            host=host,
            port=port,
            account=account,
            client_id=client_id,
            timezone=timezone,
            account_currency=account_currency,
            first_symbol_spec=first_spec,
            metadata=first_metadata,
            current_period=current_period,
            previous_period=previous_period,
            next_period=next_period,
            market_open_time=market_open_time,
            market_close_time=market_close_time,
            previous_day_start_datetime=previous_day_start_datetime,
            trading_day_end_datetime=trading_day_end_datetime,
            day_end_datetime=day_end_datetime,
            trail=trail,
            leverage=1.0,
            database_path=database_path,
            email_info_path=email_info_path,
            strict_targets_validation=strict_targets_validation,
            allowed_symbols=allowed_symbols,
            client_id_offset=client_id_offset,
        )
        if portfolio_app is None:
            print("No portfolio app was able to connect for carry protection refresh.")
            return False

        for symbol_spec in symbol_specs:
            symbol = str(symbol_spec["symbol"]).upper()
            _configure_portfolio_app_for_symbol(
                portfolio_app,
                symbol_spec,
                metadata_by_symbol[symbol],
                current_period,
                previous_period,
                next_period,
                market_open_time,
                market_close_time,
                previous_day_start_datetime,
                trading_day_end_datetime,
                day_end_datetime,
                1.0,
                strict_targets_validation,
                allowed_symbols,
            )
            try:
                sf.refresh_symbol_market_data(portfolio_app)
                refreshed_any = bool(sf.restore_carry_risk_management(portfolio_app)) or refreshed_any
            except Exception as exc:
                msg = f"[{symbol}] Carry refresh error: {exc}"
                print(msg)
                logging.exception(msg)
    finally:
        if portfolio_app is not None and portfolio_app.isConnected():
            try:
                sf.stop(portfolio_app)
            except Exception:
                pass
    return refreshed_any


def _ensure_history_file(path, current_period):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        df.index.name = 'datetime'
        df.loc[pd.Timestamp(current_period) - dt.timedelta(days=1), :] = [0,0,0,0]
        df.to_csv(path)


def _were_all_symbols_period_traded(database_path, current_period, symbol_specs):
    state_df = sf._load_strategy_state(database_path)
    if not state_df.empty:
        mask = (
            state_df['Symbol'].astype(str).str.upper() == 'PORTFOLIO'
        ) & (
            state_df['Scope'].astype(str).str.upper() == 'PORTFOLIO'
        ) & (
            state_df['StateKey'].astype(str).str.upper() == 'LAST_COMPLETED_PERIOD'
        )
        if mask.any():
            latest = state_df.loc[mask].sort_values('datetime').iloc[-1]
            import json
            try:
                val = json.loads(latest['StateValue'])
            except:
                val = str(latest['StateValue'])
            if str(val) == str(current_period):
                return True

    # Fallback to periods_traded sheet
    try:
        database = pd.ExcelFile(database_path)
        if 'periods_traded' not in database.sheet_names:
            return False
        periods_df = database.parse('periods_traded')
    except:
        return False
    if periods_df.empty:
        return False
    unnamed = [col for col in periods_df.columns if str(col).startswith('Unnamed:')]
    if unnamed:
        periods_df = periods_df.drop(columns=unnamed)
    periods_df['trade_time'] = pd.to_datetime(periods_df['trade_time'], errors='coerce')
    current_period_ts = pd.Timestamp(current_period)
    periods_df['trade_done'] = pd.to_numeric(periods_df['trade_done'], errors='coerce').fillna(0)
    period_rows = periods_df[periods_df['trade_time'] == current_period_ts]
    if not period_rows.empty and bool((period_rows['trade_done'] == 1).any()):
        return True
    return False


def _mark_portfolio_period_completed(database_path, current_period, apps):
    state_value = str(current_period)
    row = pd.DataFrame([
        {
            'Symbol': 'PORTFOLIO',
            'Scope': 'PORTFOLIO',
            'StateKey': 'LAST_COMPLETED_PERIOD',
            'StateValue': state_value,
            'datetime': dt.datetime.now().replace(microsecond=0)
        }
    ])
    state_df = sf._load_strategy_state(database_path)
    state_df = pd.concat([state_df, row], ignore_index=True)
    sf._save_strategy_state(database_path, state_df)


def _optimization_bucket_start(previous_day_start_datetime, optimization_frequency):
    bucket_start = pd.Timestamp(previous_day_start_datetime).to_pydatetime().replace(microsecond=0)
    frequency = str(optimization_frequency or "daily").strip().lower()
    if frequency == "weekly":
        bucket_start = bucket_start - dt.timedelta(days=bucket_start.weekday())
    return bucket_start


def _bulk_download_historical_data(symbol_specs, data_dir, timezone, variables):
    """Download bulk historical data from IB for symbols whose CSV files have fewer
    than 200 rows.  Resampled OHLC is saved to data/historical/historical_{symbol}.csv."""
    from ibkr_multi_asset import setup_for_download_data as sdd
    trading_day_origin = str(variables.get("trading_day_origin", "18:00")).strip()
    _, _, _, _, trading_start_hour = tf.get_end_hours(timezone, trading_day_origin=trading_day_origin)
    market_open_time, _ = tf.define_trading_week(timezone, trading_start_hour, 0)

    os.makedirs(os.path.join(data_dir, "historical"), exist_ok=True)

    def _download_one(idx_spec):
        idx, spec = idx_spec
        symbol = str(spec["symbol"]).upper()
        freq = stra.get_asset_frequency(symbol)
        train_span = stra.get_asset_train_span(symbol)
        hist_path = os.path.join(data_dir, "historical", f"historical_{symbol}.csv")
        raw_path = os.path.join(data_dir, "historical", f"raw_minute_{symbol}.csv")

        if os.path.exists(hist_path):
            try:
                existing = pd.read_csv(hist_path, index_col=0)
                if len(existing) >= 200:
                    print(f"[{symbol}] Historical data sufficient ({len(existing)} bars). Skipping.")
                    return
            except Exception:
                pass

        is_daily = str(freq).strip().upper() in ('1D', '1 D', '1DAY', 'DAILY', 'DAY')
        span = '2 Y' if is_daily else '10 D'
        print(f"[{symbol}] Bulk-downloading historical data from IB (span={span}, bars={freq})...")
        try:
            sdd.run_hist_data_download_app(
                raw_path, hist_path, symbol, timezone, freq,
                "false", span, train_span, market_open_time,
                client_id=1000 + idx * 10, silent=True,
            )
            print(f"[{symbol}] Bulk download complete.")
        except Exception as exc:
            print(f"[{symbol}] IB bulk download failed: {exc}")

    print("=" * 80)
    print("Bulk-downloading historical data from IB for all symbols...")
    print("=" * 80)
    dl_workers = min(6, len(symbol_specs))
    with ThreadPoolExecutor(max_workers=dl_workers) as executor:
        list(executor.map(_download_one, enumerate(symbol_specs)))


def _ensure_strategy_optimization_for_schedule(variables, symbol_specs, previous_day_start_datetime, now_dt):
    optimization_frequency = str(variables.get("optimization_frequency", "weekly")).strip().lower()
    bucket_start = _optimization_bucket_start(previous_day_start_datetime, optimization_frequency)
    optimization_bucket = bucket_start.isoformat(sep=" ")
    try:
        payload = stra.validate_strategy_optimization(
            symbol_specs=symbol_specs,
            optimization_frequency=optimization_frequency,
            optimization_bucket=optimization_bucket,
        )
    except Exception:
        payload = stra.strategy_parameter_optimization(
            symbol_specs=symbol_specs,
            optimization_frequency=optimization_frequency,
            optimization_bucket=optimization_bucket,
            optimized_at=now_dt.replace(microsecond=0).isoformat(sep=" "),
        )
    return payload


def _resolve_portfolio_leverage(variables, symbol_specs, optimization_frequency=None, optimization_bucket=None):
    base_leverage = pd.to_numeric(pd.Series([variables.get("portfolio_leverage", 1.0)]), errors="coerce").iloc[0]
    fallback = float(base_leverage) if np.isfinite(base_leverage) else 1.0
    try:
        payload = stra.validate_strategy_optimization(
            symbol_specs=symbol_specs,
            optimization_frequency=optimization_frequency,
            optimization_bucket=optimization_bucket,
        )
    except Exception:
        return fallback
    optimized = pd.to_numeric(pd.Series([payload.get("portfolio_leverage_multiplier", fallback)]), errors="coerce").iloc[0]
    return float(optimized) if np.isfinite(optimized) else fallback


def _is_asset_session_open(now_dt, asset_spec, market_open_time, market_close_time, timezone):
    symbol = asset_spec.get("symbol")
    asset_class = asset_spec.get("asset_class")
    policy = stra.get_asset_runtime_policy(symbol, asset_class=asset_class)
    session = str(policy.get("session", "weekdays")).strip().lower()
    
    if session == "24_7":
        return True
    
    # Check weekday session
    if not (market_open_time <= now_dt <= market_close_time):
        return False
        
    # Maintenance window check
    maintenance_start = str(policy.get("daily_maintenance_utc_start", "00:00")).strip()
    maintenance_minutes = int(policy.get("daily_maintenance_minutes", 0) or 0)
    if maintenance_minutes > 0:
        import pytz
        now_utc = dt.datetime.now(pytz.UTC)
        try:
            h, m = map(int, maintenance_start.split(":"))
            maintenance_begin = now_utc.replace(hour=h, minute=m, second=0, microsecond=0)
            maintenance_end = maintenance_begin + dt.timedelta(minutes=maintenance_minutes)
            if maintenance_begin <= now_utc < maintenance_end:
                return False
        except:
            pass
    return True


def _should_flatten_at_day_end(asset_spec):
    policy = stra.get_asset_runtime_policy(asset_spec.get("symbol"), asset_class=asset_spec.get("asset_class"))
    return bool(policy.get("flatten_at_day_end", False))


def _apps_traded_current_period(apps, current_period):
    for app in apps:
        rows = app.periods_traded[app.periods_traded["trade_time"] == current_period]
        if not rows.empty and (rows["trade_done"].iloc[-1] == 1):
            return True
    return False


def _format_sleep_label(seconds, freq):
    if seconds < 60:
        return f"{seconds}s"
    return freq


def run_portfolio_setup_loop(host, port, account, client_id, timezone, account_currency, symbol_specs, trail, strict_targets_validation, portfolio_client_id_offset=100):
    variables = tf.extract_variables(_main_config_path())
    trading_day_origin = variables.get("trading_day_origin")
    engine_cycle_frequency = _resolve_engine_cycle_frequency(symbol_specs)
    restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour = tf.get_end_hours(
        timezone,
        trading_day_origin=trading_day_origin,
    )
    
    allowed_symbols = [s["symbol"] for s in symbol_specs]

    while True:
        market_open_time, market_close_time = tf.define_trading_week(timezone, trading_start_hour, day_end_minute, close_hour=day_end_hour, close_minute=day_end_minute)
        now_dt = dt.datetime.now()
        if not _is_within_portfolio_trading_week(now_dt, market_open_time, market_close_time):
            sleep_seconds = max(1, math.ceil((market_open_time - now_dt).total_seconds()))
            print(f"Portfolio is outside the trading week. Sleeping until market reopen at {market_open_time}.")
            time.sleep(sleep_seconds)
            continue

        effective_portfolio_client_id_offset = int(variables.get("portfolio_client_id_offset", portfolio_client_id_offset))
        
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
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # Bulk-download historical data from IB before optimization so the
        # strategy has enough bars to compute meaningful signals.
        _bulk_download_historical_data(symbol_specs, data_dir, timezone, variables)

        optimization_payload = _ensure_strategy_optimization_for_schedule(
            variables,
            symbol_specs,
            previous_day_start_datetime,
            dt.datetime.now(),
        )
        optimization_frequency = str(variables.get("optimization_frequency", "daily")).strip().lower()
        optimization_bucket = str(optimization_payload.get("optimization_bucket") or _optimization_bucket_start(previous_day_start_datetime, optimization_frequency).isoformat(sep=" "))
        portfolio_leverage = _resolve_portfolio_leverage(
            variables,
            symbol_specs,
            optimization_frequency=optimization_frequency,
            optimization_bucket=optimization_bucket,
        )
        database_path = os.path.join(data_dir, "database.xlsx")
        email_info_path = os.path.join(data_dir, "email_info.xlsx")
        ensure_trading_info_workbook(
            smtp_username=variables.get("smtp_username", ""),
            to_email=variables.get("to_email", ""),
            password=variables.get("password", ""),
            database_path=database_path,
            email_info_path=email_info_path,
        )

        print("=" * 100)
        print(f"Portfolio trading period: {current_period}")
        print(f"Next period: {next_period}")

        if _were_all_symbols_period_traded(database_path, current_period, symbol_specs):
            print(f"Portfolio trading period {current_period} was already completed. Skipping...")
            sleep_seconds = max(1, math.ceil((pd.Timestamp(next_period).to_pydatetime() - dt.datetime.now()).total_seconds()))
            print(f"Sleeping until next period ({sleep_seconds}s, next: {next_period}).")
            time.sleep(sleep_seconds)
            continue

        portfolio_app = None
        portfolio_apps = []
        try:
            metadata_by_symbol = {
                str(symbol_spec["symbol"]).upper(): _symbol_runtime_metadata(symbol_spec, data_dir, current_period)
                for symbol_spec in symbol_specs
            }
            tradable_symbol_specs = [
                symbol_spec
                for symbol_spec in symbol_specs
                if _is_asset_session_open(dt.datetime.now(), symbol_spec, market_open_time, market_close_time, timezone)
            ]

            if not tradable_symbol_specs:
                print("No configured symbols are inside their trading session for this period.")
                sleep_seconds = max(1, math.ceil((pd.Timestamp(next_period).to_pydatetime() - dt.datetime.now()).total_seconds()))
                print(f"Cycle done. Sleeping until next period ({sleep_seconds}s, next: {next_period}).")
                time.sleep(sleep_seconds)
                continue

            parallel_order_submission = bool(variables.get("portfolio_parallel_order_submission", True))
            if parallel_order_submission:
                for index, symbol_spec in enumerate(tradable_symbol_specs):
                    symbol = str(symbol_spec["symbol"]).upper()
                    metadata = metadata_by_symbol[symbol]
                    worker_app = _create_symbol_app(
                        host=host,
                        port=port,
                        account=account,
                        client_id=client_id,
                        timezone=timezone,
                        account_currency=account_currency,
                        symbol_spec=symbol_spec,
                        data_frequency=metadata["data_frequency"],
                        current_period=current_period,
                        previous_period=previous_period,
                        next_period=next_period,
                        market_open_time=market_open_time,
                        market_close_time=market_close_time,
                        previous_day_start_datetime=previous_day_start_datetime,
                        trading_day_end_datetime=trading_day_end_datetime,
                        day_end_datetime=day_end_datetime,
                        train_span=metadata["train_span"],
                        test_span=metadata["test_span"],
                        trail=trail,
                        leverage=portfolio_leverage,
                        historical_data_address=metadata["historical_data_address"],
                        base_df_address=metadata["base_df_address"],
                        database_path=database_path,
                        email_info_path=email_info_path,
                        strict_targets_validation=strict_targets_validation,
                        allowed_symbols=allowed_symbols,
                        client_id_offset=effective_portfolio_client_id_offset + index,
                        announce=False,
                    )
                    if worker_app is not None:
                        worker_app.optimization_frequency = optimization_frequency
                        worker_app.optimization_bucket = optimization_bucket
                        portfolio_apps.append(worker_app)

                if len(portfolio_apps) == 0:
                    print("No portfolio worker apps were able to connect.")
                    sleep_seconds = max(1, math.ceil((pd.Timestamp(next_period).to_pydatetime() - dt.datetime.now()).total_seconds()))
                    print(f"Sleeping until next period ({sleep_seconds}s, next: {next_period}).")
                    time.sleep(sleep_seconds)
                    continue

                _run_parallel_app_portfolio_cycle(
                    portfolio_apps,
                    tradable_symbol_specs,
                    metadata_by_symbol,
                    current_period,
                    previous_period,
                    next_period,
                    market_open_time,
                    market_close_time,
                    previous_day_start_datetime,
                    trading_day_end_datetime,
                    day_end_datetime,
                    portfolio_leverage,
                    strict_targets_validation,
                    allowed_symbols,
                )
            else:
                first_spec = tradable_symbol_specs[0]
                first_metadata = metadata_by_symbol[str(first_spec["symbol"]).upper()]
                portfolio_app = _create_portfolio_app(
                    host=host,
                    port=port,
                    account=account,
                    client_id=client_id,
                    timezone=timezone,
                    account_currency=account_currency,
                    first_symbol_spec=first_spec,
                    metadata=first_metadata,
                    current_period=current_period,
                    previous_period=previous_period,
                    next_period=next_period,
                    market_open_time=market_open_time,
                    market_close_time=market_close_time,
                    previous_day_start_datetime=previous_day_start_datetime,
                    trading_day_end_datetime=trading_day_end_datetime,
                    day_end_datetime=day_end_datetime,
                    trail=trail,
                    leverage=portfolio_leverage,
                    database_path=database_path,
                    email_info_path=email_info_path,
                    strict_targets_validation=strict_targets_validation,
                    allowed_symbols=allowed_symbols,
                    client_id_offset=effective_portfolio_client_id_offset,
                )

                if portfolio_app is None:
                    print("Portfolio app failed to connect.")
                    sleep_seconds = max(1, math.ceil((pd.Timestamp(next_period).to_pydatetime() - dt.datetime.now()).total_seconds()))
                    print(f"Sleeping until next period ({sleep_seconds}s, next: {next_period}).")
                    time.sleep(sleep_seconds)
                    continue
                portfolio_app.optimization_frequency = optimization_frequency
                portfolio_app.optimization_bucket = optimization_bucket

                _run_single_app_portfolio_cycle(
                    portfolio_app,
                    tradable_symbol_specs,
                    metadata_by_symbol,
                    current_period,
                    previous_period,
                    next_period,
                    market_open_time,
                    market_close_time,
                    previous_day_start_datetime,
                    trading_day_end_datetime,
                    day_end_datetime,
                    portfolio_leverage,
                    strict_targets_validation,
                    allowed_symbols,
                )
                portfolio_apps = [portfolio_app]

            if _apps_traded_current_period(portfolio_apps, current_period):
                _mark_portfolio_period_completed(database_path, current_period, portfolio_apps)

        except Exception as exc:
            logging.exception(f"Error: {exc}")
            print(f"Portfolio cycle error: {exc}")
        finally:
            for app in portfolio_apps:
                if app is not None and app.isConnected():
                    try:
                        sf.stop(app)
                    except Exception:
                        pass
            if portfolio_app is not None and portfolio_app not in portfolio_apps and portfolio_app.isConnected():
                try:
                    sf.stop(portfolio_app)
                except Exception:
                    pass

        sleep_seconds = max(1, math.ceil((pd.Timestamp(next_period).to_pydatetime() - dt.datetime.now()).total_seconds()))
        print(f"Cycle done. Sleeping until next period ({sleep_seconds}s, next: {next_period}).")
        time.sleep(sleep_seconds)


def main():
    try:
        variables = tf.extract_variables(_main_config_path())
    except FileNotFoundError:
        return

    run_portfolio_setup_loop(
        host=variables.get("host", "127.0.0.1"),
        port=variables.get("port", 7497),
        account=variables.get("account", "account"),
        client_id=variables.get("client_id", 1),
        timezone=variables.get("timezone", "America/Lima"),
        account_currency=variables.get("account_currency", "USD"),
        symbol_specs=_normalize_symbol_specs(variables),
        trail=variables.get("trail", False),
        strict_targets_validation=variables.get("strict_targets_validation", True),
        portfolio_client_id_offset=variables.get("portfolio_client_id_offset", 100),
    )


def _normalize_symbol_specs(variables):
    specs = []
    for fx in variables.get("fx_pairs", []):
        specs.append({"symbol": str(fx).upper(), "asset_class": "forex", "exchange": "IDEALPRO", "currency": "USD"})
    for fut in variables.get("futures_symbols", []):
        specs.append({"symbol": str(fut).upper(), "asset_class": "futures", "exchange": "CME", "currency": "USD", "multiplier": "5"})
    metal_quantity_steps = variables.get("metals_quantity_steps", {}) or {}
    metal_default_quantity_step = float(variables.get("metals_quantity_step", variables.get("metals_default_quantity_step", 1.0)))
    for metal in variables.get("metals_symbols", []):
        symbol = str(metal).upper()
        specs.append({
            "symbol": symbol,
            "asset_class": "metals",
            "exchange": "SMART",
            "currency": "USD",
            "sec_type": "CMDTY",
            "quantity_step": float(metal_quantity_steps.get(symbol, metal_default_quantity_step)),
        })
    for crypto in variables.get("crypto_symbols", []):
        specs.append({"symbol": str(crypto).upper(), "asset_class": "crypto", "exchange": "PAXOS", "currency": "USD", "sec_type": "CRYPTO"})
    stock_primary_exchanges = variables.get("stock_primary_exchanges", {}) or {}
    stock_quantity_steps = variables.get("stock_quantity_steps", {}) or {}
    stock_fractional_shares = bool(variables.get("stock_fractional_shares", False))
    default_stock_step = float(variables.get("stock_default_quantity_step", 0.0001 if stock_fractional_shares else 1.0))
    for stock in variables.get("stock_symbols", []):
        symbol = str(stock).upper()
        specs.append({
            "symbol": symbol,
            "asset_class": "stock",
            "exchange": str(variables.get("stock_exchange", "SMART") or "SMART").upper(),
            "currency": str(variables.get("stock_currency", "USD") or "USD").upper(),
            "sec_type": "STK",
            "primary_exchange": stock_primary_exchanges.get(symbol, variables.get("stock_primary_exchange", "NASDAQ")),
            "quantity_step": float(stock_quantity_steps.get(symbol, default_stock_step)),
            "fractional_shares": stock_fractional_shares,
        })
    return specs


if __name__ == "__main__":
    main()
