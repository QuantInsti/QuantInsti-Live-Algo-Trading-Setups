from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path
from threading import Event

import traceback
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_cancel import OrderCancel
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ibkr_multi_asset import trading_functions as tf


USER_CONFIG_DIR = Path(__file__).resolve().parent
DATA_DIR = USER_CONFIG_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "log"
HISTORICAL_DIR = DATA_DIR / "historical"
BASE_FRAMES_DIR = DATA_DIR / "base_frames"

USER_MAIN_PATH = USER_CONFIG_DIR / "main.py"
user_main = tf.extract_variables(str(USER_MAIN_PATH))

CONNECT_WAIT_SECONDS = 15
REQUEST_WAIT_SECONDS = 20
FLAT_WAIT_SECONDS = 15

GENERATED_FILES = [
    DATA_DIR / "database.xlsx",
    DATA_DIR / "email_info.xlsx",
    DATA_DIR / "portfolio_report.pdf",
    MODELS_DIR / "optimal_features_df.xlsx",
    MODELS_DIR / "strategy_optimization_manifest.json",
    MODELS_DIR / "strategy_optimization_schedule.json",
    MODELS_DIR / "strategy2_optimal_features_df.xlsx",
    MODELS_DIR / "strategy2_optimization_manifest.json",
    USER_CONFIG_DIR / "tws.png",
    DATA_DIR / "database.xlsx.bak",
]

GENERATED_LOG_GLOBS = [
    "log_file_*.log",
]

EXTRA_DIRS = [
    USER_CONFIG_DIR / "__pycache__",
    HISTORICAL_DIR,
    BASE_FRAMES_DIR,
]

MARKER_FILES = [
    DATA_DIR / "last_completed_portfolio_period.txt",
    DATA_DIR / "last_carry_protection_refresh.txt",
]


class ClosePositionsApp(EWrapper, EClient):
    def __init__(self, target_account: str):
        EClient.__init__(self, self)
        self.target_account = str(target_account)
        self.positions = []
        self.open_orders = []
        self.errors = []
        self.next_order_id = None
        self.bid_price = None
        self.ask_price = None
        self.last_price = None
        self.next_id_event = Event()
        self.open_orders_event = Event()
        self.positions_event = Event()
        self.market_data_event = Event()

    def nextValidId(self, orderId: int):
        self.next_order_id = int(orderId)
        self.next_id_event.set()

    def error(self, reqId, errorCode, errorString, *args):
        actual_code = errorCode
        actual_msg = errorString
        if actual_code == 0 and isinstance(errorString, int):
            actual_code = errorString
            actual_msg = args[0] if len(args) > 0 else ""
        if int(actual_code) == 0:
            return
        self.errors.append((int(reqId), int(actual_code), str(actual_msg)))
        print(f"IB error {actual_code} (req {reqId}): {actual_msg}")

    def position(self, account, contract, position, avgCost):
        if str(account) != self.target_account:
            return
        qty = float(position)
        if abs(qty) <= 1e-12:
            return
        self.positions.append(
            {
                "account": account,
                "contract": contract,
                "position": qty,
                "avg_cost": float(avgCost),
            }
        )

    def positionEnd(self):
        self.positions_event.set()

    def openOrder(self, orderId, contract, order, orderState):
        order_account = str(getattr(order, "account", "") or "")
        if order_account and order_account != self.target_account:
            return
        self.open_orders.append(
            {
                "order_id": int(orderId),
                "contract": contract,
                "order": order,
                "order_state": orderState,
            }
        )

    def openOrderEnd(self):
        self.open_orders_event.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        try:
            value = float(price)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        if tickType in {1, 66}:
            self.bid_price = value
        elif tickType in {2, 67}:
            self.ask_price = value
        elif tickType in {4, 68, 9, 75}:
            self.last_price = value
        self.market_data_event.set()

    def reserve_order_id(self) -> int:
        if self.next_order_id is None:
            raise RuntimeError("nextValidId was not received from IB")
        order_id = int(self.next_order_id)
        self.next_order_id += 1
        return order_id


def market_order(action: str, quantity: float) -> Order:
    order = Order()
    order.action = action
    order.orderType = "MKT"
    order.totalQuantity = quantity
    order.transmit = True
    if not float(quantity).is_integer():
        order.tif = "IOC"
    return order


def ioc_limit_order(action: str, quantity: float, price: float) -> Order:
    order = Order()
    order.action = action
    order.orderType = "LMT"
    order.tif = "IOC"
    order.totalQuantity = quantity
    order.lmtPrice = price
    order.transmit = True
    order.eTradeOnly = 0
    order.firmQuoteOnly = 0
    order.overridePercentageConstraints = True
    return order


def normalize_quantity(raw_qty: float) -> float:
    qty = abs(float(raw_qty))
    rounded = round(qty)
    if abs(qty - rounded) < 1e-9:
        return int(rounded)
    return qty


def normalized_close_contract(contract) -> Contract:
    close_contract = Contract()
    for attr in (
        "conId",
        "symbol",
        "secType",
        "lastTradeDateOrContractMonth",
        "strike",
        "right",
        "multiplier",
        "exchange",
        "primaryExchange",
        "currency",
        "localSymbol",
        "tradingClass",
    ):
        try:
            value = getattr(contract, attr)
        except Exception:
            continue
        if value not in (None, "", 0):
            setattr(close_contract, attr, value)

    sec_type = str(getattr(close_contract, "secType", "")).upper()
    exchange = str(getattr(close_contract, "exchange", "") or "").upper()
    if not exchange:
        if sec_type == "CASH":
            close_contract.exchange = "IDEALPRO"
        elif sec_type == "FUT":
            close_contract.exchange = "CME"
        elif sec_type == "CMDTY":
            close_contract.exchange = "SMART"
        elif sec_type == "CRYPTO":
            close_contract.exchange = "PAXOS"
    return close_contract


def contract_key(contract: Contract) -> tuple[str, str, str, str]:
    return (
        str(getattr(contract, "symbol", "")).upper(),
        str(getattr(contract, "secType", "")).upper(),
        str(getattr(contract, "currency", "")).upper(),
        str(getattr(contract, "localSymbol", "")).upper(),
    )


def refresh_open_orders(app: ClosePositionsApp) -> list[dict]:
    app.open_orders = []
    app.open_orders_event.clear()
    app.reqAllOpenOrders()
    if not app.open_orders_event.wait(timeout=REQUEST_WAIT_SECONDS):
        raise TimeoutError("Timed out waiting for openOrderEnd from IB")
    return list(app.open_orders)


def refresh_positions(app: ClosePositionsApp) -> list[dict]:
    app.positions = []
    app.positions_event.clear()
    app.reqPositions()
    if not app.positions_event.wait(timeout=REQUEST_WAIT_SECONDS):
        raise TimeoutError("Timed out waiting for positionEnd from IB")
    return list(app.positions)


def current_position_for_contract(app: ClosePositionsApp, contract: Contract) -> float:
    target_key = contract_key(contract)
    matching = [
        item for item in app.positions
        if contract_key(item["contract"]) == target_key and abs(float(item["position"])) > 1e-12
    ]
    if not matching:
        return 0.0
    return float(matching[-1]["position"])


def best_live_price(app: ClosePositionsApp) -> float | None:
    for value in (app.last_price, app.bid_price, app.ask_price):
        if value is not None and float(value) > 0:
            return float(value)
    return None


def marketable_exit_limit_price(app: ClosePositionsApp, action: str) -> float:
    live_price = best_live_price(app)
    if live_price is None:
        raise RuntimeError("No live market price is available for fallback IOC limit close")
    if action.upper() == "SELL":
        return round(live_price * 0.98, 8)
    return round(live_price * 1.02, 8)


def wait_until_flat(app: ClosePositionsApp, contract: Contract, timeout_seconds: int) -> float:
    deadline = time.time() + max(int(timeout_seconds), 1)
    last_qty = 0.0
    while time.time() < deadline:
        refresh_positions(app)
        last_qty = current_position_for_contract(app, contract)
        if abs(last_qty) <= 1e-12:
            return 0.0
        time.sleep(1)
    return last_qty


def flatten_contract_position(app: ClosePositionsApp, position_item: dict) -> None:
    contract = normalized_close_contract(position_item["contract"])
    position = float(position_item["position"])
    action = "SELL" if position > 0 else "BUY"
    quantity = normalize_quantity(position)
    symbol = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")

    order_id = app.reserve_order_id()
    print(f"- {symbol} {getattr(contract, 'secType', '')} {position} -> {action} {quantity} (order id {order_id})")
    app.placeOrder(order_id, contract, market_order(action, quantity))

    remaining_qty = wait_until_flat(app, contract, FLAT_WAIT_SECONDS)
    if abs(remaining_qty) <= 1e-12:
        print(f"  Flattened {symbol} with market order.")
        return

    print(f"  {symbol} was not flat after market close attempt. Trying aggressive IOC limit close...")
    market_data_req_id = app.reserve_order_id()
    app.market_data_event.clear()
    app.reqMktData(market_data_req_id, contract, "", False, False, [])
    app.market_data_event.wait(timeout=5)
    fallback_price = marketable_exit_limit_price(app, action)
    app.cancelMktData(market_data_req_id)

    fallback_order_id = app.reserve_order_id()
    print(f"  Fallback order {fallback_order_id}: {action} {abs(remaining_qty)} @ {fallback_price}")
    app.placeOrder(fallback_order_id, contract, ioc_limit_order(action, abs(remaining_qty), fallback_price))
    remaining_qty = wait_until_flat(app, contract, FLAT_WAIT_SECONDS)
    if abs(remaining_qty) > 1e-12:
        raise RuntimeError(f"{symbol} still has an open position after flatten attempt: {remaining_qty}")
    print(f"  Flattened {symbol} with fallback IOC limit order.")


def connect_app() -> ClosePositionsApp:
    account = user_main["account"]
    host = user_main["host"]
    port = int(user_main["port"])
    client_id = int(user_main["client_id"]) + 900

    app = ClosePositionsApp(target_account=account)
    print(f"Connecting to IB paper account {account} on {host}:{port} with client id {client_id}...")
    app.connect(host, port, clientId=client_id)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    if not app.next_id_event.wait(timeout=CONNECT_WAIT_SECONDS):
        app.disconnect()
        error_tail = app.errors[-3:] if app.errors else []
        raise TimeoutError(f"Timed out waiting for nextValidId from IB. Errors: {error_tail}")
    return app


def run_close_flow() -> None:
    app = connect_app()
    try:
        print("Submitting global cancel for all open orders...")
        app.reqGlobalCancel(OrderCancel())
        time.sleep(2)

        print("Requesting current open orders...")
        open_orders = refresh_open_orders(app)
        if open_orders:
            print("Canceling open orders for the configured paper account...")
            seen_order_ids = set()
            for item in open_orders:
                order_id = int(item["order_id"])
                if order_id in seen_order_ids:
                    continue
                seen_order_ids.add(order_id)
                contract = item["contract"]
                symbol = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
                print(f"- Canceling open order {order_id} for {symbol}")
                app.cancelOrder(order_id, OrderCancel())
                time.sleep(0.2)
            time.sleep(2)
        else:
            print("No open orders found in the configured account.")

        print("Requesting current positions...")
        positions = refresh_positions(app)
        if not positions:
            print("No open positions found in the configured account.")
            return

        print("Submitting orders to flatten these positions:")
        for item in positions:
            flatten_contract_position(app, item)

        print("Verifying account is flat...")
        remaining_positions = refresh_positions(app)
        if remaining_positions:
            outstanding = [
                (
                    getattr(item["contract"], "localSymbol", "") or getattr(item["contract"], "symbol", ""),
                    float(item["position"]),
                )
                for item in remaining_positions
            ]
            raise RuntimeError(f"Some positions remain open after flattening: {outstanding}")
        print("All positions were flattened successfully.")
    finally:
        app.disconnect()


def delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    else:
        path.unlink()
        print(f"Deleted file: {path}")


def cleanup_generated_outputs() -> None:
    for path in GENERATED_FILES:
        delete_path(path)

    for pattern in GENERATED_LOG_GLOBS:
        for path in sorted(LOG_DIR.glob(pattern)):
            delete_path(path)

    for path in EXTRA_DIRS:
        delete_path(path)


def drop_marker_files() -> None:
    for path in MARKER_FILES:
        delete_path(path)


def main() -> None:
    print("Step 1: Closing all open positions in the configured paper account...")
    try:
        run_close_flow()
    except Exception as exc:
        print(f"Warning: close-all-positions step failed: {exc}")
        traceback.print_exc()
        print("Continuing with local paper-trading state cleanup...")

    print("Step 2: Deleting generated paper-trading outputs under user_config...")
    cleanup_generated_outputs()
    print("Step 3: Removing period marker files (optional)...")
    drop_marker_files()

    print("Reset complete. Core setup files were preserved.")


if __name__ == "__main__":
    main()
