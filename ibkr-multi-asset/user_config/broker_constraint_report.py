from __future__ import annotations

import datetime as dt
import json
import math
import threading
import time
from pathlib import Path

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Connection
HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 901
ACCOUNT = "DU1682711"
TIMEZONE = "America/Lima"

# Universe to inspect :  all assets from best_best_strategy.py / josgt_strategy_backtesting.py
ASSETS = [
    # FX pairs (IDEALPRO)
    {"symbol": "EUR", "secType": "CASH", "exchange": "IDEALPRO", "currency": "USD", "label": "EURUSD"},
    {"symbol": "GBP", "secType": "CASH", "exchange": "IDEALPRO", "currency": "USD", "label": "GBPUSD"},
    {"symbol": "AUD", "secType": "CASH", "exchange": "IDEALPRO", "currency": "USD", "label": "AUDUSD"},
    {"symbol": "USD", "secType": "CASH", "exchange": "IDEALPRO", "currency": "CAD", "label": "USDCAD"},
    {"symbol": "USD", "secType": "CASH", "exchange": "IDEALPRO", "currency": "CHF", "label": "USDCHF"},
    {"symbol": "USD", "secType": "CASH", "exchange": "IDEALPRO", "currency": "JPY", "label": "USDJPY"},
    # Futures
    {"symbol": "MES", "secType": "FUT", "exchange": "CME", "currency": "USD", "multiplier": "5", "label": "MES"},
    # Spot metals
    {"symbol": "XAUUSD", "secType": "CMDTY", "exchange": "SMART", "currency": "USD", "label": "XAUUSD"},
    # Crypto (PAXOS)
    {"symbol": "BTC", "secType": "CRYPTO", "exchange": "PAXOS", "currency": "USD", "label": "BTCUSD"},
    {"symbol": "ETH", "secType": "CRYPTO", "exchange": "PAXOS", "currency": "USD", "label": "ETHUSD"},
    {"symbol": "SOL", "secType": "CRYPTO", "exchange": "PAXOS", "currency": "USD", "label": "SOLUSD"},
    {"symbol": "LTC", "secType": "CRYPTO", "exchange": "PAXOS", "currency": "USD", "label": "LTCUSD"},
    {"symbol": "BCH", "secType": "CRYPTO", "exchange": "PAXOS", "currency": "USD", "label": "BCHUSD"},
]

# Optional manual notes that are useful for a backtester but are not exposed cleanly by the IB API.
POLICY_HINTS = {
    "CRYPTO": [
        "IBKR crypto concentration limits are not exposed directly by the API.",
        "Validate account-specific crypto caps with paper-order probes before backtesting live sizing.",
        "Observed/common rule in some IBKR accounts: crypto exposure may be capped near 30% of account equity.",
    ],
    "XAUUSD": [
        "Spot metals quantity-step constraints can differ by venue and account configuration.",
        "Validate whether XAUUSD accepts only integer quantities in your account before backtesting.",
    ],
}

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "reports"
JSON_OUTPUT = OUTPUT_DIR / "broker_constraints_report.json"
PDF_OUTPUT = OUTPUT_DIR / "broker_constraints_report.pdf"
CONNECT_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 20
ACCOUNT_SUMMARY_TAGS = ",".join(
    [
        "AccountType",
        "NetLiquidation",
        "NetLiquidationByCurrency",
        "TotalCashValue",
        "AvailableFunds",
        "BuyingPower",
        "ExcessLiquidity",
        "GrossPositionValue",
        "Leverage",
        "Cushion",
    ]
)


def _safe_float(value):
    try:
        converted = float(value)
    except Exception:
        return None
    if math.isfinite(converted):
        return converted
    return None


def _safe_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_contract(asset: dict) -> Contract:
    contract = Contract()
    contract.symbol = str(asset["symbol"]).upper()
    contract.secType = str(asset["secType"]).upper()
    contract.exchange = str(asset["exchange"]).upper()
    contract.currency = str(asset["currency"]).upper()
    if asset.get("multiplier") is not None:
        contract.multiplier = str(asset["multiplier"])
    if asset.get("lastTradeDateOrContractMonth") is not None:
        contract.lastTradeDateOrContractMonth = str(asset["lastTradeDateOrContractMonth"])
    if asset.get("localSymbol") is not None:
        contract.localSymbol = str(asset["localSymbol"])
    if asset.get("tradingClass") is not None:
        contract.tradingClass = str(asset["tradingClass"])
    return contract


def _parse_expiry(value) -> str | None:
    text = _safe_text(value)
    if text is None:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    if len(digits) == 6:
        return digits
    return None


def _select_contract_detail(details: list, asset: dict):
    if not details:
        return None
    sec_type = str(asset.get("secType", "")).upper()
    if sec_type != "FUT":
        return details[0]

    today = dt.datetime.utcnow().strftime("%Y%m%d")
    candidates = []
    for detail in details:
        expiry = _parse_expiry(getattr(detail.contract, "lastTradeDateOrContractMonth", None))
        if expiry is None:
            continue
        if len(expiry) >= 8 and expiry < today:
            continue
        candidates.append((expiry, detail))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return details[0]


def _quantity_step(detail, asset: dict):
    size_increment = _safe_float(getattr(detail, "sizeIncrement", None))
    suggested_increment = _safe_float(getattr(detail, "suggestedSizeIncrement", None))
    min_size = _safe_float(getattr(detail, "minSize", None))
    sec_type = str(asset.get("secType", "")).upper()

    if size_increment is not None and size_increment > 0:
        return size_increment, "broker:sizeIncrement"
    if suggested_increment is not None and suggested_increment > 0:
        return suggested_increment, "broker:suggestedSizeIncrement"
    if sec_type in {"FUT", "CMDTY"}:
        return 1.0, "heuristic:whole-units"
    if sec_type == "CASH":
        return 1.0, "heuristic:base-currency-units"
    if sec_type == "CRYPTO" and min_size is not None and min_size > 0:
        return min_size, "heuristic:minSize"
    return None, "unknown"


def _fractional_allowed(step):
    if step is None:
        return None
    return bool(step < 1.0)


def _market_rule_rows(price_increments):
    rows = []
    for item in price_increments or []:
        rows.append(
            {
                "low_edge": _safe_float(getattr(item, "lowEdge", None)),
                "increment": _safe_float(getattr(item, "increment", None)),
            }
        )
    return rows


class ConstraintProbeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_order_id = None
        self.connected_event = threading.Event()
        self.account_summary_end = threading.Event()
        self.contract_events = {}
        self.market_rule_events = {}
        self.account_rows = []
        self.contract_details_map = {}
        self.market_rules = {}
        self.errors = []

    def nextValidId(self, orderId: int):
        self.next_valid_order_id = orderId
        self.connected_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="", *args, **kwargs):
        self.errors.append(
            {
                "reqId": reqId,
                "errorCode": errorCode,
                "errorString": errorString,
                "advancedOrderRejectJson": advancedOrderRejectJson if advancedOrderRejectJson else "",
                "timestamp": dt.datetime.utcnow().isoformat(),
            }
        )

    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_rows.append(
            {
                "reqId": reqId,
                "account": account,
                "tag": tag,
                "value": value,
                "currency": currency,
            }
        )

    def accountSummaryEnd(self, reqId):
        self.account_summary_end.set()

    def contractDetails(self, reqId, contractDetails):
        self.contract_details_map.setdefault(reqId, []).append(contractDetails)

    def contractDetailsEnd(self, reqId):
        self.contract_events.setdefault(reqId, threading.Event()).set()

    def marketRule(self, marketRuleId, priceIncrements):
        self.market_rules[marketRuleId] = list(priceIncrements or [])
        self.market_rule_events.setdefault(marketRuleId, threading.Event()).set()


def _connect_app() -> tuple[ConstraintProbeApp, threading.Thread]:
    app = ConstraintProbeApp()
    app.connect(HOST, PORT, CLIENT_ID)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    if not app.connected_event.wait(CONNECT_TIMEOUT_SECONDS):
        raise TimeoutError("Timed out waiting for IBKR connection handshake.")
    return app, thread


def _request_account_summary(app: ConstraintProbeApp):
    app.reqAccountSummary(7001, "All", ACCOUNT_SUMMARY_TAGS)
    if not app.account_summary_end.wait(REQUEST_TIMEOUT_SECONDS):
        raise TimeoutError("Timed out waiting for account summary.")
    return pd.DataFrame(app.account_rows)


def _request_contract_details(app: ConstraintProbeApp, asset: dict, req_id: int):
    app.contract_events[req_id] = threading.Event()
    app.reqContractDetails(req_id, _build_contract(asset))
    if not app.contract_events[req_id].wait(REQUEST_TIMEOUT_SECONDS):
        raise TimeoutError(f"Timed out waiting for contract details for {asset.get('label') or asset.get('symbol')}.")
    details = app.contract_details_map.get(req_id, [])
    if not details:
        raise RuntimeError(f"No contract details returned for {asset.get('label') or asset.get('symbol')}.")
    return details


def _request_market_rule(app: ConstraintProbeApp, market_rule_id: int):
    if market_rule_id in app.market_rules:
        return app.market_rules[market_rule_id]
    app.market_rule_events[market_rule_id] = threading.Event()
    app.reqMarketRule(market_rule_id)
    app.market_rule_events[market_rule_id].wait(REQUEST_TIMEOUT_SECONDS)
    return app.market_rules.get(market_rule_id, [])


def _account_snapshot_rows(account_df: pd.DataFrame):
    if account_df.empty:
        return []
    rows = []
    latest = account_df.copy()
    latest["tag"] = latest["tag"].astype(str)
    latest["currency"] = latest["currency"].astype(str)
    grouped = latest.groupby(["tag", "currency"], as_index=False).tail(1)
    for _, row in grouped.sort_values(["tag", "currency"]).iterrows():
        rows.append(
            {
                "tag": row["tag"],
                "currency": row["currency"],
                "value": row["value"],
            }
        )
    return rows


def _asset_report_row(asset: dict, detail, market_rules: dict):
    contract = detail.contract
    sec_type = str(asset.get("secType", "")).upper()
    expiry = _parse_expiry(getattr(contract, "lastTradeDateOrContractMonth", None))
    min_tick = _safe_float(getattr(detail, "minTick", None))
    min_size = _safe_float(getattr(detail, "minSize", None))
    quantity_step, quantity_step_source = _quantity_step(detail, asset)
    market_rule_ids = [int(item) for item in str(getattr(detail, "marketRuleIds", "") or "").split(",") if item.strip().isdigit()]
    order_types = sorted({item.strip() for item in str(getattr(detail, "orderTypes", "") or "").split(",") if item.strip()})
    valid_exchanges = sorted({item.strip() for item in str(getattr(detail, "validExchanges", "") or "").split(",") if item.strip()})
    asset_key = str(asset.get("label") or asset.get("symbol")).upper()
    policy_notes = list(POLICY_HINTS.get(asset_key, [])) + list(POLICY_HINTS.get(sec_type, []))

    verification_items = []
    if sec_type == "CRYPTO":
        verification_items.append("Validate broker crypto concentration caps with paper-order probes.")
        verification_items.append("Validate whether native stop orders are accepted on the crypto venue.")
    if sec_type == "CMDTY":
        verification_items.append("Validate whether fractional quantities are accepted for the selected commodity venue.")
    if sec_type == "CASH":
        verification_items.append("Validate practical minimum order size if you want to avoid odd-lot FX routing.")
    if sec_type == "FUT":
        verification_items.append("Model front-month roll logic explicitly in backtests.")
    if not market_rule_ids:
        verification_items.append("No market-rule id returned by IBKR; validate price rounding with paper orders.")
    if quantity_step is None:
        verification_items.append("Quantity step was not exposed by IBKR; validate with paper orders.")

    return {
        "asset": asset,
        "contract": {
            "symbol": _safe_text(getattr(contract, "symbol", None)),
            "label": asset.get("label") or asset.get("symbol"),
            "secType": sec_type,
            "exchange": _safe_text(getattr(contract, "exchange", None)),
            "currency": _safe_text(getattr(contract, "currency", None)),
            "conId": getattr(contract, "conId", None),
            "localSymbol": _safe_text(getattr(contract, "localSymbol", None)),
            "tradingClass": _safe_text(getattr(contract, "tradingClass", None)),
            "multiplier": _safe_text(getattr(contract, "multiplier", None)),
            "expiry": expiry,
        },
        "execution_constraints": {
            "minTick": min_tick,
            "priceMagnifier": getattr(detail, "priceMagnifier", None),
            "minSize": min_size,
            "sizeIncrement": _safe_float(getattr(detail, "sizeIncrement", None)),
            "suggestedSizeIncrement": _safe_float(getattr(detail, "suggestedSizeIncrement", None)),
            "quantityStep": quantity_step,
            "quantityStepSource": quantity_step_source,
            "fractionalQuantityAllowed": _fractional_allowed(quantity_step),
            "marketRuleIds": market_rule_ids,
            "marketRules": {str(rule_id): _market_rule_rows(market_rules.get(rule_id, [])) for rule_id in market_rule_ids},
            "orderTypes": order_types,
            "validExchanges": valid_exchanges,
        },
        "backtest_translation": {
            "price_tick": min_tick,
            "quantity_step": quantity_step,
            "min_quantity": min_size if min_size is not None else quantity_step,
            "whole_units_only": False if _fractional_allowed(quantity_step) else True if quantity_step is not None else None,
            "requires_front_month_roll": sec_type == "FUT",
            "long_only_recommended": sec_type == "CRYPTO",
        },
        "notes": {
            "policy_hints": policy_notes,
            "verification_needed": verification_items,
        },
    }


def _summary_table_rows(asset_reports: list[dict]):
    rows = []
    for report in asset_reports:
        contract = report["contract"]
        constraints = report["execution_constraints"]
        rows.append(
            [
                contract.get("label"),
                contract.get("secType"),
                contract.get("exchange"),
                contract.get("currency"),
                contract.get("expiry") or "",
                constraints.get("minTick"),
                constraints.get("quantityStep"),
                constraints.get("minSize"),
                constraints.get("fractionalQuantityAllowed"),
            ]
        )
    return rows


def _table_page(pdf: PdfPages, title: str, columns: list[str], rows: list[list], fontsize: int = 8):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if not rows:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _chunk_long_line(line: str, max_chars: int = 95) -> list[str]:
    """Split a long line into sub-lines that each fit on one visual row."""
    if len(line) <= max_chars:
        return [line]
    # Try to break at a comma+space boundary near max_chars
    chunks = []
    remaining = line
    while len(remaining) > max_chars:
        split_at = remaining.rfind(", ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        else:
            split_at += 1  # include the comma
        chunks.append(remaining[:split_at].rstrip())
        remaining = "  " + remaining[split_at:].lstrip()
    if remaining.strip():
        chunks.append(remaining)
    return chunks


def _text_page(pdf: PdfPages, title: str, lines: list[str]):
    """Render a text-only page. Long lines chunked; page breaks only between logical lines."""
    # Pre-flatten: each logical line → one or more display sub-lines
    flat: list[str] = []
    for line in lines:
        flat.extend(_chunk_long_line(line))

    lines_per_page = 26  # ~0.96 / 0.035 ≈ 27, leave margin
    total = len(flat)
    start = 0

    while start < total:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        # If fewer than half a page remains, try to squeeze onto current page
        remaining = total - start
        take = remaining if remaining <= int(lines_per_page * 1.15) else lines_per_page
        y = 0.96
        for i in range(start, min(start + take, total)):
            ax.text(0.03, y, flat[i], ha="left", va="top", fontsize=10, wrap=True)
            y -= 0.035
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        start += take


def _write_pdf_report(payload: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_OUTPUT) as pdf:
        overview_lines = [
            f"Generated at: {payload['generated_at']}",
            f"Host: {payload['connection']['host']}:{payload['connection']['port']}",
            f"Client ID: {payload['connection']['client_id']}",
            f"Account: {payload['connection']['account']}",
            f"Timezone reference: {payload['connection']['timezone']}",
            "",
            "Purpose:",
            "This report is a pre-backtest broker-constraint inventory.",
            "Use the JSON output as the machine-readable source for future backtesting rules.",
            "",
            "Important:",
            "Some execution constraints are not exposed directly by the IBKR API.",
            "Those items are listed under verification_needed and should be validated with paper probes.",
        ]
        _text_page(pdf, "IBKR Constraint Discovery Overview", overview_lines)

        account_rows = [[row["tag"], row["currency"], row["value"]] for row in payload["account_snapshot"]]
        _table_page(pdf, "Account Snapshot", ["Tag", "Currency", "Value"], account_rows, fontsize=9)

        _table_page(
            pdf,
            "Asset Constraint Summary",
            ["Asset", "SecType", "Exchange", "CCY", "Expiry", "MinTick", "QtyStep", "MinSize", "Fractional?"],
            _summary_table_rows(payload["asset_reports"]),
            fontsize=8,
        )

        for report in payload["asset_reports"]:
            contract = report["contract"]
            constraints = report["execution_constraints"]
            lines = [
                f"Asset: {contract.get('label')}",
                f"SecType: {contract.get('secType')}",
                f"Exchange: {contract.get('exchange')}",
                f"Currency: {contract.get('currency')}",
                f"ConId: {contract.get('conId')}",
                f"LocalSymbol: {contract.get('localSymbol')}",
                f"TradingClass: {contract.get('tradingClass')}",
                f"Multiplier: {contract.get('multiplier')}",
                f"Expiry: {contract.get('expiry')}",
                f"MinTick: {constraints.get('minTick')}",
                f"MinSize: {constraints.get('minSize')}",
                f"QuantityStep: {constraints.get('quantityStep')} ({constraints.get('quantityStepSource')})",
                f"FractionalQuantityAllowed: {constraints.get('fractionalQuantityAllowed')}",
                f"OrderTypes: {', '.join(constraints.get('orderTypes', [])) or 'n/a'}",
                f"ValidExchanges: {', '.join(constraints.get('validExchanges', [])) or 'n/a'}",
                "",
                "Policy hints:",
            ]
            lines.extend([f"- {item}" for item in report["notes"]["policy_hints"]] or ["- none"])
            lines.append("")
            lines.append("Verification needed:")
            lines.extend([f"- {item}" for item in report["notes"]["verification_needed"]] or ["- none"])
            _text_page(pdf, f"Asset Detail: {contract.get('label')}", lines)

        rule_lines = []
        for report in payload["asset_reports"]:
            contract = report["contract"]
            translation = report["backtest_translation"]
            rule_lines.append(
                f"{contract.get('label')}: price_tick={translation.get('price_tick')}, "
                f"quantity_step={translation.get('quantity_step')}, "
                f"min_quantity={translation.get('min_quantity')}, "
                f"whole_units_only={translation.get('whole_units_only')}, "
                f"front_month_roll={translation.get('requires_front_month_roll')}, "
                f"long_only_recommended={translation.get('long_only_recommended')}"
            )
        _text_page(pdf, "Backtest Translation Rules", rule_lines)

        if payload["errors"]:
            error_rows = [
                [item.get("timestamp"), item.get("reqId"), item.get("errorCode"), item.get("errorString")]
                for item in payload["errors"]
            ]
            _table_page(pdf, "IBKR Errors Captured During Discovery", ["Timestamp", "ReqId", "Code", "Message"], error_rows, fontsize=7)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app, thread = _connect_app()
    try:
        account_df = _request_account_summary(app)

        asset_reports = []
        used_market_rules = {}
        req_id = 8000
        for asset in ASSETS:
            details = _request_contract_details(app, asset, req_id)
            selected = _select_contract_detail(details, asset)
            market_rule_ids = [int(item) for item in str(getattr(selected, "marketRuleIds", "") or "").split(",") if item.strip().isdigit()]
            for market_rule_id in market_rule_ids:
                used_market_rules[market_rule_id] = _request_market_rule(app, market_rule_id)
            asset_reports.append(_asset_report_row(asset, selected, used_market_rules))
            req_id += 1

        payload = {
            "generated_at": dt.datetime.now().replace(microsecond=0).isoformat(sep=" "),
            "connection": {
                "host": HOST,
                "port": PORT,
                "client_id": CLIENT_ID,
                "account": ACCOUNT,
                "timezone": TIMEZONE,
            },
            "selected_assets": ASSETS,
            "account_snapshot": _account_snapshot_rows(account_df),
            "asset_reports": asset_reports,
            "errors": app.errors,
        }

        JSON_OUTPUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _write_pdf_report(payload)
        print(f"Constraint JSON saved to: {JSON_OUTPUT}")
        print(f"Constraint PDF saved to: {PDF_OUTPUT}")
    finally:
        try:
            app.disconnect()
        except Exception:
            pass
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
