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
from ibkr_multi_asset import setup_functions as sf
from ibkr_multi_asset import trading_functions as tf
from ibkr_multi_asset.create_database import ensure_trading_info_workbook
from ibkr_multi_asset.report_generator import generate_live_portfolio_report
from ibkr_multi_asset.setup import trading_app
from ibkr_multi_asset.strategy_runtime import stra


# Set the logging level to INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "user_config", "data", "log", f"log_file_{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"),
    filemode='w'
)

CONNECT_HANDSHAKE_TIMEOUT_SECONDS = 45


def _main_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "user_config", "main.py")


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
    app.new_df = {"0": pd.DataFrame(), "1": pd.DataFrame()}
    app.errors_dict = {}
    app.hist_request_errors = {}
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
    active_apps = [app for app in apps if app is not None and app.isConnected()]
    if len(active_apps) == 0:
        return False

    lead_app = active_apps[0]

    print("Refreshing all symbols before portfolio decisioning with isolated apps...")
    with ThreadPoolExecutor(max_workers=len(active_apps)) as executor:
        futures = [executor.submit(sf.refresh_symbol_market_data, app) for app in active_apps]
        for future in futures:
            future.result()

    print("Collecting shared account updates once for the full universe...")
    sf.collect_shared_account_snapshot([lead_app])
    _broadcast_shared_account_state(lead_app, active_apps)

    print("Computing portfolio targets once for the full universe...")
    targets = sf.compute_portfolio_targets_once(active_apps)
    portfolio_attrs = _copy_portfolio_target_attrs(lead_app)

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

    print("Preparing and sending orders in parallel with isolated IB apps...")
    cycle_errors = []
    with ThreadPoolExecutor(max_workers=len(active_apps)) as executor:
        futures = {executor.submit(_run_parallel_worker_send_orders, app): app for app in active_apps}
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
    if app is None or not app.isConnected() or len(tradable_symbol_specs) == 0:
        return False

    print("Refreshing all symbols before portfolio decisioning with one connected app...")
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

    print("Collecting shared account updates once for the full universe...")
    sf.collect_shared_account_snapshot([app])

    print("Computing portfolio targets once for the full universe...")
    first_spec = tradable_symbol_specs[0]
    _configure_portfolio_app_for_symbol(
        app,
        first_spec,
        metadata_by_symbol[str(first_spec["symbol"]).upper()],
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
    targets = sf.compute_portfolio_targets_once([app])
    portfolio_attrs = _copy_portfolio_target_attrs(app)

    print("Resolving contract details once for the full universe...")
    sf.collect_shared_contract_details(app, tradable_symbol_specs)

    print("Collecting shared positions, open orders, and executions once for the full universe...")
    sf.collect_shared_broker_snapshot(app)

    print("Preparing and sending orders sequentially through the single app...")
    app.defer_posttrade_sync = True
    app.defer_synthetic_monitors = True
    app.pending_synthetic_monitors = []
    rows = []
    cycle_errors = []
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
            str(int(getattr(app, "signal", 0))),
            f"{float(getattr(app, 'leverage', 0.0)):.6f}",
            f"{float(getattr(app, 'ordered_quantity', 0.0)):.6f}",
        ])

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
    headers = ["Asset", "Signal", "Leverage", "OrderedQty"]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def _resolve_engine_cycle_frequency(symbol_specs):
    frequencies = [stra.get_asset_frequency(s["symbol"]) for s in symbol_specs]
    if not frequencies:
        return "5min"
    # Take the minimum frequency in seconds
    freq_seconds = [tf.get_data_frequency_values(f)[0] for f in frequencies]
    min_seconds = min(freq_seconds)
    # Convert back to string representation
    for f in frequencies:
        if tf.get_data_frequency_values(f)[0] == min_seconds:
            return f
    return "5min"


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
        # Create a skeleton history file if it doesn't exist
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        df.index.name = 'datetime'
        # Set a dummy index to avoid empty-dataframe errors in some tf functions
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
        market_open_time, market_close_time = tf.define_trading_week(timezone, trading_start_hour, day_end_minute)
        now_dt = dt.datetime.now()
        if not _is_within_portfolio_trading_week(now_dt, market_open_time, market_close_time):
            sleep_seconds = max(1, math.ceil((market_open_time - now_dt).total_seconds()))
            print(f"Portfolio is outside the trading week. Sleeping until market reopen ({sleep_seconds}s).")
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

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
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
            print(f"Sleeping until next period ({sleep_seconds}s).")
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
                print(f"Cycle done. Sleeping until next period ({sleep_seconds}s).")
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
                    print(f"Sleeping until next period ({sleep_seconds}s).")
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
                    print(f"Sleeping until next period ({sleep_seconds}s).")
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
        print(f"Cycle done. Sleeping until next period ({sleep_seconds}s).")
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
