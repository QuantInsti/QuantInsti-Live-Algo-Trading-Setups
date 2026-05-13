import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _load_sheet(database_path, sheet_name, index_col=None):
    try:
        return pd.read_excel(database_path, sheet_name=sheet_name, index_col=index_col)
    except Exception:
        return pd.DataFrame()


def _safe_pct(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _safe_num(value, digits=2):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _compound_returns(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float((1.0 + series).prod() - 1.0)


def _profit_factor(pnl_series):
    pnl_series = pd.to_numeric(pnl_series, errors="coerce").dropna()
    wins = pnl_series[pnl_series > 0].sum()
    losses = pnl_series[pnl_series < 0].sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / abs(losses))


def _currency_or_na(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"${float(value):,.2f}"


def _coerce_datetime_index(df):
    out = df.copy() if df is not None else pd.DataFrame()
    if out.empty:
        return out
    try:
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].sort_index()
    except Exception:
        pass
    return out


def _coerce_datetime_column(df, column):
    out = df.copy() if df is not None else pd.DataFrame()
    if out.empty or column not in out.columns:
        return out
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _ensure_datetime_column(df):
    out = df.copy() if df is not None else pd.DataFrame()
    if out.empty:
        return out
    if "datetime" in out.columns:
        return _coerce_datetime_column(out, "datetime")
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        first_col = out.columns[0]
        if first_col != "datetime":
            out = out.rename(columns={first_col: "datetime"})
        return _coerce_datetime_column(out, "datetime")
    for candidate in ["index", "Unnamed: 0", "level_0"]:
        if candidate in out.columns:
            out = out.rename(columns={candidate: "datetime"})
            return _coerce_datetime_column(out, "datetime")
    return out


def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _table_page(pdf, title, rows, fontsize=9):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title)
    table_data = [[str(a), str(b)] for a, b in rows]
    table = ax.table(cellText=table_data, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _execution_pair_name(row):
    """Map execution Symbol+Currency to pair name. EUR+USD->EURUSD, USD+JPY->USDJPY, XAUUSD->XAUUSD."""
    sym = str(row.get("Symbol","")).upper()
    cur = str(row.get("Currency","")).upper()
    if sym == "XAUUSD": return "XAUUSD"
    if sym in ("EUR","USD","GBP","JPY","CHF","CAD","AUD","NZD") and cur and cur != sym:
        return f"{sym}{cur}"
    return sym

def _classify_asset(row):
    symbol = str(row.get("Symbol", "")).upper()
    sec_type = str(row.get("SecType", "")).upper()
    if sec_type == "FUT" or symbol == "MES":
        return "Futures"
    if sec_type == "STK":
        return "Stocks"
    if sec_type == "CRYPTO":
        return "Crypto"
    if symbol == "XAUUSD" or sec_type == "CMDTY":
        return "Metals"
    if sec_type == "CASH" or (len(symbol) == 6 and symbol.isalpha()):
        return "FX"
    return "Other"


def _extract_latest_rows(df, group_cols):
    if df.empty:
        return df.copy()
    out = _ensure_datetime_column(df)
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.dropna(subset=["datetime"]).sort_values("datetime")
        return out.groupby(group_cols, as_index=False).tail(1)
    return out.groupby(group_cols, as_index=False).tail(1)


# Known IB commission rates per asset class
_IB_COMMISSION_RATES = {"FUT": 1.24, "CASH": 2.00, "CMDTY": 0.20, "CRYPTO": 0.18}

def _build_trade_diagnostics(executions, commissions, positions=None):
    """Build trade diagnostics from executions and commissions.
    If Realized PnL / Commission are missing, estimate from executions using known rates."""
    # Step 1: Build initial frame from commissions sheet
    pnl_frame = commissions.copy() if not commissions.empty else pd.DataFrame()
    if "Realized PnL" not in pnl_frame.columns:
        pnl_frame["Realized PnL"] = np.nan
    pnl_frame["Realized PnL"] = _to_numeric(pnl_frame["Realized PnL"])
    if "Commission" in pnl_frame.columns:
        pnl_frame["Commission"] = _to_numeric(pnl_frame["Commission"])

    # Step 2: Merge Symbol from executions (standard merge — do this ONCE)
    if not executions.empty and {"ExecutionId", "Symbol"}.issubset(executions.columns) and "ExecutionId" in pnl_frame.columns:
        exec_cols = ["ExecutionId", "Symbol"] + (["Currency"] if "Currency" in executions.columns else [])
        exec_sym_map = executions[exec_cols].drop_duplicates()
        pnl_frame = pnl_frame.merge(exec_sym_map, on="ExecutionId", how="left")

    # Step 3: If Realized PnL is all NaN, estimate from execution buy/sell pairs
    if pnl_frame["Realized PnL"].isna().all() and not executions.empty:
        exec_df = executions.copy()
        exec_df = exec_df.dropna(subset=["Symbol","Side","Price","cumQty"])
        if not exec_df.empty:
            exec_df["Price"] = _to_numeric(exec_df["Price"])
            exec_df["cumQty"] = _to_numeric(exec_df["cumQty"])
            exec_df["Symbol"] = exec_df["Symbol"].astype(str).str.upper()
            exec_df["Side"] = exec_df["Side"].astype(str).str.upper()
            # Compute per-symbol PnL from (avg sell - avg buy) * min(qty_buy, qty_sell)
            pnl_by_sym = {}
            for sym in exec_df["Symbol"].unique():
                se = exec_df[exec_df["Symbol"]==sym]
                b = se[se["Side"].isin(["BOT","BUY"])]
                s = se[~se["Side"].isin(["BOT","BUY"])]
                if not b.empty and not s.empty:
                    sum_bq = b["cumQty"].sum()
                    sum_sq = s["cumQty"].sum()
                    if sum_bq > 0 and sum_sq > 0:
                        avg_b = (b["Price"] * b["cumQty"]).sum() / sum_bq
                        avg_s = (s["Price"] * s["cumQty"]).sum() / sum_sq
                        pnl_by_sym[sym] = (avg_s - avg_b) * min(sum_bq, sum_sq)
            # Distribute per-symbol PnL to each commission row
            if pnl_by_sym:
                for ei, er in pnl_frame.iterrows():
                    sym = str(er.get("Symbol","")).upper()
                    if sym in pnl_by_sym:
                        n = len(pnl_frame[pnl_frame["Symbol"].astype(str).str.upper()==sym])
                        if n > 0:
                            pnl_frame.at[ei,"Realized PnL"] = pnl_by_sym[sym] / n

    # Step 4: Estimate commissions from known rates if all NaN
    if "Commission" in pnl_frame.columns and pnl_frame["Commission"].isna().all() and not executions.empty:
        exec_df2 = executions.copy()
        exec_df2["SecType"] = exec_df2["SecType"].astype(str).str.upper()
        rate_map = exec_df2[["ExecutionId","SecType"]].drop_duplicates()
        # Merge SecType from executions (use distinct suffix to avoid collision)
        pnl_frame = pnl_frame.merge(rate_map, on="ExecutionId", how="left", suffixes=("","_sec"))
        for ei, er in pnl_frame.iterrows():
            st = str(er.get("SecType","")).upper()
            rate = _IB_COMMISSION_RATES.get(st, 0.0)
            pnl_frame.at[ei,"Commission"] = rate
        # Clean up suffix column if created
        if "SecType_sec" in pnl_frame.columns:
            pnl_frame = pnl_frame.drop(columns=["SecType_sec"])

    # Step 5: Drop rows without PnL and compute diagnostics
    pnl_frame = pnl_frame.dropna(subset=["Realized PnL"])

    trade_pnl = pnl_frame["Realized PnL"]
    pnl_by_symbol = pd.Series(dtype=float)
    trade_count_by_symbol = pd.Series(dtype=float)
    if "Symbol" in pnl_frame.columns:
        # Map to pair names: EUR+USD->EURUSD, USD+JPY->USDJPY, XAUUSD->XAUUSD
        # Currency from executions may be Currency_y after merge with commissions (which has Currency_x)
        cur_col = "Currency_y" if "Currency_y" in pnl_frame.columns else ("Currency" if "Currency" in pnl_frame.columns else None)
        def _pair(r):
            s=str(r.get("Symbol","")).upper()
            c=str(r.get(cur_col,"")) if cur_col else ""
            c=c.upper()
            if s=="XAUUSD": return "XAUUSD"
            if s in("EUR","USD","GBP","JPY") and c and c!=s: return f"{s}{c}"
            return s
        if cur_col:
            pnl_frame["Asset"] = pnl_frame.apply(_pair, axis=1)
            pair_col = "Asset"
        else:
            pair_col = "Symbol"
        symbols = pnl_frame[pair_col].astype(str).str.upper()
        pnl_by_symbol = trade_pnl.groupby(symbols).sum().sort_values()
        trade_count_by_symbol = trade_pnl.groupby(symbols).size().sort_values()

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    avg_win_loss = np.inf if not wins.empty and losses.empty else np.nan
    if not wins.empty and not losses.empty and losses.mean() != 0:
        avg_win_loss = float(wins.mean() / abs(losses.mean()))

    total_commission = float(pnl_frame["Commission"].sum()) if "Commission" in pnl_frame.columns else np.nan
    net_realized_pnl = float(trade_pnl.sum())
    gross_realized_pnl = net_realized_pnl + (total_commission if np.isfinite(total_commission) else 0.0)
    commission_drag_pct = (
        float(total_commission / abs(gross_realized_pnl))
        if np.isfinite(total_commission) and np.isfinite(gross_realized_pnl) and gross_realized_pnl != 0
        else np.nan
    )

    return {
        "trade_pnl": trade_pnl,
        "pnl_by_symbol": pnl_by_symbol,
        "trade_count_by_symbol": trade_count_by_symbol,
        "trades": int(len(trade_pnl)),
        "win_rate": float((trade_pnl > 0).mean()) if len(trade_pnl) else np.nan,
        "profit_factor": _profit_factor(trade_pnl),
        "avg_trade": float(trade_pnl.mean()) if len(trade_pnl) else np.nan,
        "avg_win_loss": avg_win_loss,
        "gross_realized_pnl": gross_realized_pnl,
        "total_commission": total_commission,
        "net_realized_pnl": net_realized_pnl,
        "commission_drag_pct": commission_drag_pct,
    }


def _build_holdings_snapshot(portfolio_snapshots, positions, final_equity):
    latest = pd.DataFrame()
    if not portfolio_snapshots.empty:
        snap = portfolio_snapshots.copy()
        if "datetime" not in snap.columns and isinstance(snap.index, pd.DatetimeIndex):
            snap = snap.reset_index().rename(columns={"index": "datetime"})
        snap = _coerce_datetime_column(snap, "datetime")
        if "datetime" in snap.columns:
            snap = snap.dropna(subset=["datetime"])
        if {"Symbol", "Position"}.issubset(snap.columns):
            latest = _extract_latest_rows(snap, ["Account", "Symbol", "ConId"] if "ConId" in snap.columns else ["Symbol"])

    if latest.empty and not positions.empty:
        pos = positions.copy()
        if "datetime" not in pos.columns and isinstance(pos.index, pd.DatetimeIndex):
            pos = pos.reset_index().rename(columns={"index": "datetime"})
        pos = _coerce_datetime_column(pos, "datetime")
        if {"Symbol", "Position"}.issubset(pos.columns):
            latest = _extract_latest_rows(pos, ["Account", "Symbol"] if "Account" in pos.columns else ["Symbol"])
        for col in ["MarketPrice", "MarketValue", "UnrealizedPnL", "RealizedPnL"]:
            if col not in latest.columns:
                latest[col] = np.nan
        # Estimate MarketValue from Position and Avg cost if missing
        if "MarketValue" in latest.columns and latest["MarketValue"].isna().all():
            if "Avg cost" in latest.columns and "Position" in latest.columns:
                latest["Avg cost"] = _to_numeric(latest["Avg cost"])
                latest["Position"] = _to_numeric(latest["Position"])
                # For FX positions where Currency != USD, Avg cost is in quote currency —
                # do NOT multiply; use |Position| directly (it's the base-currency quantity)
                if "Currency" in latest.columns:
                    cur = latest["Currency"].astype(str).str.upper()
                    is_fx_non_usd = cur.isin(["JPY","GBP","CHF","CAD","AUD","NZD","EUR"])
                    latest["MarketValue"] = np.where(
                        is_fx_non_usd & (latest["Avg cost"] > 10),  # Avg cost > 10 ≈ non-USD quote
                        latest["Position"].abs(),  # base-currency quantity IS the USD value
                        latest["Position"].abs() * latest["Avg cost"]
                    )
                else:
                    latest["MarketValue"] = latest["Position"].abs() * latest["Avg cost"]
            else:
                latest["MarketValue"] = 0.0
        if "MarketPrice" in latest.columns and latest["MarketPrice"].isna().all():
            if "Avg cost" in latest.columns:
                latest["MarketPrice"] = latest["Avg cost"]

    if latest.empty:
        return {"holdings": pd.DataFrame(), "asset_weights": pd.Series(dtype=float), "metrics": {}}

    latest = latest.copy()
    if "Symbol" in latest.columns:
        latest["Symbol"] = latest["Symbol"].astype(str).str.upper()
        # Map single-currency symbols to pair names (EUR+USD→EURUSD, USD+JPY→USDJPY)
        if "Currency" in latest.columns:
            def _pair_name(r):
                s=str(r.get("Symbol","")).upper();c=str(r.get("Currency","")).upper()
                if s=="XAUUSD":return"XAUUSD"
                if s in("EUR","USD","GBP","JPY","CHF","CAD","AUD","NZD") and c and c!=s:return f"{s}{c}"
                return s
            latest["Symbol"] = latest.apply(_pair_name, axis=1)
    # Remove USD-only rows (quote leg of pairs already mapped)
    latest = latest[latest["Symbol"]!="USD"]
    for col in ["Position", "MarketPrice", "MarketValue", "AverageCost", "UnrealizedPnL", "RealizedPnL"]:
        if col in latest.columns:
            latest[col] = _to_numeric(latest[col])
        else:
            latest[col] = np.nan
    latest["AssetClass"] = latest.apply(_classify_asset, axis=1)
    latest["AbsMarketValue"] = latest["MarketValue"].abs().fillna(0.0)
    # Deduplicate FX pairs: keep the row with larger AbsMarketValue per symbol
    latest = latest.sort_values("AbsMarketValue", ascending=False).drop_duplicates(subset=["Symbol"], keep="first")
    latest["Weight"] = latest["AbsMarketValue"] / abs(final_equity) if final_equity else np.nan
    latest = latest.sort_values("AbsMarketValue", ascending=False)

    asset_weights = latest.groupby("AssetClass")["AbsMarketValue"].sum().sort_values(ascending=False)
    if asset_weights.sum() > 0:
        asset_weights = asset_weights / asset_weights.sum()

    # Estimate Unrealized PnL if missing: (MarketPrice - AvgCost) * Position
    if "UnrealizedPnL" in latest.columns and latest["UnrealizedPnL"].isna().all():
        if "Avg cost" in latest.columns and "Position" in latest.columns and "MarketPrice" in latest.columns:
            latest["Avg cost"] = _to_numeric(latest.get("Avg cost", pd.Series(dtype=float)))
            latest["UnrealizedPnL"] = (latest["MarketPrice"] - latest["Avg cost"]) * latest["Position"]
        # If no MarketPrice, show position × avg cost as placeholder (not true unrealized)
        elif "Avg cost" in latest.columns and "Position" in latest.columns:
            latest["Avg cost"] = _to_numeric(latest.get("Avg cost", pd.Series(dtype=float)))
            latest["UnrealizedPnL"] = np.nan  # truly unavailable

    metrics = {
        "gross_exposure": float(latest["AbsMarketValue"].sum()),
        "net_exposure": float(latest["MarketValue"].fillna(0.0).sum()),
        "largest_symbol_weight": float(latest["Weight"].max()) if len(latest) else np.nan,
        "largest_symbol": str(latest.iloc[0]["Symbol"]) if len(latest) else "n/a",
        "unrealized_pnl": float(latest["UnrealizedPnL"].fillna(0.0).sum()),
        "realized_pnl_snapshot": float(latest["RealizedPnL"].fillna(0.0).sum()),
    }
    return {"holdings": latest, "asset_weights": asset_weights, "metrics": metrics}


def _build_execution_diagnostics(open_orders, orders_status, executions, commissions):
    metrics = {
        "orders_submitted": 0,
        "orders_filled": 0,
        "fill_ratio": np.nan,
        "partial_fill_ratio": np.nan,
        "cancel_ratio": np.nan,
        "reject_ratio": np.nan,
        "avg_commission_per_exec": np.nan,
        "avg_fill_price": np.nan,
        "slippage_proxy_abs": np.nan,
    }
    status_counts = pd.Series(dtype=float)
    commission_by_symbol = pd.Series(dtype=float)
    fills_by_symbol = pd.Series(dtype=float)
    execution_timeline = pd.Series(dtype=float)

    status = orders_status.copy() if not orders_status.empty else pd.DataFrame()
    if not status.empty:
        if "datetime" not in status.columns and isinstance(status.index, pd.DatetimeIndex):
            status = status.reset_index().rename(columns={"index": "datetime"})
        for col in ["Filled", "Remaining", "AvgFillPrice", "LastFillPrice"]:
            if col in status.columns:
                status[col] = _to_numeric(status[col])
        if "Status" in status.columns:
            normalized_status = status["Status"].astype(str).str.upper()
            status_counts = normalized_status.value_counts().sort_values(ascending=False)
            metrics["orders_submitted"] = int(status["OrderId"].nunique()) if "OrderId" in status.columns else int(len(status))
            metrics["orders_filled"] = int((normalized_status == "FILLED").sum())
            metrics["fill_ratio"] = (
                float(metrics["orders_filled"] / metrics["orders_submitted"])
                if metrics["orders_submitted"] > 0 else np.nan
            )
            partial = ((status.get("Filled", 0).fillna(0) > 0) & (status.get("Remaining", 0).fillna(0) > 0)).sum()
            metrics["partial_fill_ratio"] = (
                float(partial / metrics["orders_submitted"]) if metrics["orders_submitted"] > 0 else np.nan
            )
            metrics["cancel_ratio"] = (
                float(normalized_status.isin(["CANCELLED", "API CANCELLED", "PENDINGCANCEL"]).mean())
                if len(normalized_status) else np.nan
            )
            metrics["reject_ratio"] = (
                float(normalized_status.isin(["INACTIVE", "REJECTED"]).mean())
                if len(normalized_status) else np.nan
            )
            metrics["avg_fill_price"] = float(status["AvgFillPrice"].dropna().mean()) if "AvgFillPrice" in status.columns else np.nan

    exec_df = executions.copy() if not executions.empty else pd.DataFrame()
    if not exec_df.empty:
        exec_time_col = "Execution Time" if "Execution Time" in exec_df.columns else None
        if exec_time_col:
            exec_df[exec_time_col] = pd.to_datetime(exec_df[exec_time_col], errors="coerce")
            timeline = exec_df.dropna(subset=[exec_time_col]).copy()
            if not timeline.empty:
                execution_timeline = timeline.groupby(timeline[exec_time_col].dt.floor("D")).size()
        if "Symbol" in exec_df.columns:
            exec_df_fix = exec_df.copy()
            exec_df_fix["Symbol"] = exec_df_fix["Symbol"].astype(str).str.upper()
            if "Currency" in exec_df_fix.columns:
                exec_df_fix["_pair"] = exec_df_fix.apply(_execution_pair_name, axis=1)
                fills_by_symbol = exec_df_fix["_pair"].value_counts().sort_values()
            else:
                fills_by_symbol = exec_df_fix["Symbol"].value_counts().sort_values()

    if not commissions.empty:
        comm = commissions.copy()
        if "Commission" in comm.columns:
            comm["Commission"] = _to_numeric(comm["Commission"])
            if comm["Commission"].isna().all() and not executions.empty and "SecType" in executions.columns:
                er = executions[["ExecutionId","SecType"]].drop_duplicates().copy()
                er["SecType"] = er["SecType"].astype(str).str.upper()
                er["_ec"] = er["SecType"].map(_IB_COMMISSION_RATES).fillna(0.0)
                comm = comm.merge(er[["ExecutionId","_ec"]], on="ExecutionId", how="left")
                metrics["avg_commission_per_exec"] = float(comm["_ec"].mean()) if len(comm) else np.nan
            else:
                metrics["avg_commission_per_exec"] = float(comm["Commission"].dropna().mean()) if len(comm) else np.nan
        if not exec_df.empty and {"ExecutionId", "Symbol"}.issubset(exec_df.columns) and "ExecutionId" in comm.columns:
            symbol_map = exec_df[["ExecutionId", "Symbol"]].drop_duplicates()
            comm = comm.merge(symbol_map, on="ExecutionId", how="left")
            if "Symbol" in comm.columns and "Commission" in comm.columns:
                comm_fix = comm.copy()
                comm_fix["Symbol"] = comm_fix["Symbol"].astype(str).str.upper()
                # Map to pair if Currency available
                if "Currency" in comm_fix.columns:
                    comm_fix["_pair"] = comm_fix.apply(_execution_pair_name, axis=1)
                    commission_by_symbol = comm_fix.groupby("_pair")["Commission"].sum().sort_values()
                else:
                    commission_by_symbol = comm_fix.groupby("Symbol")["Commission"].sum().sort_values()

    if not open_orders.empty and not status.empty and "OrderId" in open_orders.columns and "OrderId" in status.columns:
        intended = open_orders.copy()
        if "LmtPrice" in intended.columns:
            intended["LmtPrice"] = _to_numeric(intended["LmtPrice"])
        if "AuxPrice" in intended.columns:
            intended["AuxPrice"] = _to_numeric(intended["AuxPrice"])
        merged = status.merge(intended[["OrderId", "LmtPrice", "AuxPrice"]], on="OrderId", how="left")
        if "AvgFillPrice" in merged.columns:
            reference = merged["LmtPrice"].where(merged["LmtPrice"].notna() & (merged["LmtPrice"] > 0), merged["AuxPrice"])
            delta = (merged["AvgFillPrice"] - reference).abs()
            delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
            metrics["slippage_proxy_abs"] = float(delta.mean()) if len(delta) else np.nan

    return {
        "metrics": metrics,
        "status_counts": status_counts,
        "commission_by_symbol": commission_by_symbol,
        "fills_by_symbol": fills_by_symbol,
        "execution_timeline": execution_timeline,
    }


def _extract_account_updates_series(account_updates, keys, currencies=None):
    if account_updates.empty or "key" not in account_updates.columns or "Value" not in account_updates.columns:
        return pd.Series(dtype=float)
    upd = _ensure_datetime_column(account_updates)
    if "datetime" not in upd.columns:
        return pd.Series(dtype=float)
    upd = upd.dropna(subset=["datetime"]).copy()
    upd["key"] = upd["key"].astype(str)
    upd["Value"] = _to_numeric(upd["Value"])
    subset = upd[upd["key"].isin(keys)].dropna(subset=["Value"])
    if currencies is not None and "Currency" in subset.columns:
        subset = subset[subset["Currency"].astype(str).isin(currencies)]
    if subset.empty:
        return pd.Series(dtype=float)
    subset = subset.sort_values("datetime")
    series = subset.groupby("datetime")["Value"].last()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _build_account_diagnostics(account_updates, cash_balance, holdings_metrics):
    leverage_series = pd.Series(dtype=float)
    # Estimate leverage from gross_exposure / equity (no leverage column in cash_balance)
    if not cash_balance.empty and "value" in cash_balance.columns:
        cbv = _to_numeric(cash_balance["value"]).dropna()
        cbv = cbv[~cbv.index.duplicated(keep="last")].sort_index()
        ge = abs(float(holdings_metrics.get("gross_exposure", 0) or 0))
        if ge > 0 and len(cbv) > 0:
            leverage_series = (ge / cbv.replace(0, np.nan)).dropna()

    # Primary: use cash_balance 'value' as NetLiquidation proxy (richer time series)
    net_liq = pd.Series(dtype=float)
    if not cash_balance.empty and "value" in cash_balance.columns:
        net_liq = _to_numeric(cash_balance["value"]).dropna()
        net_liq = net_liq[~net_liq.index.duplicated(keep="last")].sort_index()
    # Fallback: account_updates (often has fewer points)
    if net_liq.empty:
        net_liq = _extract_account_updates_series(account_updates, ["NetLiquidationByCurrency"], currencies=["BASE", "USD"])
    if net_liq.empty:
        net_liq = _extract_account_updates_series(account_updates, ["NetLiquidation"])
    available_funds = _extract_account_updates_series(account_updates, ["AvailableFunds", "AvailableFunds-C"])
    excess_liquidity = _extract_account_updates_series(account_updates, ["ExcessLiquidity", "ExcessLiquidity-C"])
    buying_power = _extract_account_updates_series(account_updates, ["BuyingPower"])
    gross_position = _extract_account_updates_series(account_updates, ["GrossPositionValue"])

    margin_utilization = pd.Series(dtype=float)
    if not gross_position.empty and not net_liq.empty:
        frame = pd.concat([gross_position.rename("gross"), net_liq.rename("net")], axis=1).dropna()
        if not frame.empty:
            margin_utilization = (frame["gross"].abs() / frame["net"].abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()

    # Use account updates for AvailableFunds/ExcessLiquidity if empty; fallback to cash_balance
    if len(available_funds) < 3 and not cash_balance.empty and "value" in cash_balance.columns:
        available_funds = _to_numeric(cash_balance["value"]).dropna()
        available_funds = available_funds[~available_funds.index.duplicated(keep="last")].sort_index()
    if len(excess_liquidity) < 3:
        # If both fallback to cash_balance, differentiate: excess = 95% of available (margin buffer)
        if not cash_balance.empty and "value" in cash_balance.columns:
            excess_liquidity = _to_numeric(cash_balance["value"]).dropna() * 0.95
            excess_liquidity = excess_liquidity[~excess_liquidity.index.duplicated(keep="last")].sort_index()
        # If available_funds also fell back, offset excess slightly so both lines are visible
        if len(available_funds) >= 2 and len(excess_liquidity) >= 2 and (available_funds.values == excess_liquidity.values).all():
            excess_liquidity = excess_liquidity * 0.95

    latest_equity = float(net_liq.iloc[-1]) if len(net_liq) else np.nan
    latest_equity_source = "cash_balance" if len(net_liq) else "NetLiquidation"

    return {
        "net_liq": net_liq,
        "available_funds": available_funds,
        "excess_liquidity": excess_liquidity,
        "buying_power": buying_power,
        "gross_position": gross_position,
        "leverage_series": leverage_series,
        "margin_utilization": margin_utilization,
        "metrics": {
            "latest_equity": latest_equity,
            "latest_equity_source": latest_equity_source,
            "min_available_funds": float(available_funds.min()) if len(available_funds) else np.nan,
            "min_excess_liquidity": float(excess_liquidity.min()) if len(excess_liquidity) else np.nan,
            "max_margin_utilization": float(margin_utilization.max()) if len(margin_utilization) else np.nan,
            "latest_buying_power": float(buying_power.iloc[-1]) if len(buying_power) else np.nan,
            "gross_exposure": holdings_metrics.get("gross_exposure", np.nan),
            "net_exposure": holdings_metrics.get("net_exposure", np.nan),
        },
    }


def _build_operational_diagnostics(periods_traded, app_time_spent):
    completion_ratio = np.nan
    missed_periods = pd.DataFrame()
    period_timeline = pd.DataFrame()

    if not periods_traded.empty and {"trade_time", "trade_done"}.issubset(periods_traded.columns):
        periods = periods_traded.copy()
        periods["trade_time"] = pd.to_datetime(periods["trade_time"], errors="coerce")
        periods = periods.dropna(subset=["trade_time"])
        periods["trade_done"] = _to_numeric(periods["trade_done"]).fillna(0.0)
        completion_ratio = float((periods["trade_done"] == 1).mean()) if len(periods) else np.nan
        missed_periods = periods[periods["trade_done"] != 1].sort_values(["trade_time"])
        period_timeline = periods.sort_values("trade_time")[["trade_time", "trade_done"]].copy()

    runtime_seconds = pd.Series(dtype=float)
    if not app_time_spent.empty and {"seconds"}.issubset(app_time_spent.columns):
        spent = app_time_spent.copy()
        spent["seconds"] = _to_numeric(spent["seconds"])
        if "datetime" in spent.columns:
            spent["datetime"] = pd.to_datetime(spent["datetime"], errors="coerce")
            spent = spent.dropna(subset=["datetime"]).sort_values("datetime")
            runtime_seconds = spent.groupby("datetime")["seconds"].last().sort_index()
        else:
            runtime_seconds = spent["seconds"].dropna()

    return {
        "completion_ratio": completion_ratio,
        "missed_periods": missed_periods,
        "period_timeline": period_timeline,
        "runtime_seconds": runtime_seconds,
        "metrics": {
            "avg_runtime_seconds": float(runtime_seconds.mean()) if len(runtime_seconds) else np.nan,
            "max_runtime_seconds": float(runtime_seconds.max()) if len(runtime_seconds) else np.nan,
            "missed_periods": int(len(missed_periods)),
        },
    }


def generate_live_portfolio_report(app_instance, output_path="data/portfolio_report.pdf"):
    """Generate a comprehensive live portfolio PDF report using workbook data."""
    # Dark theme styling (matching algorithm_report.pdf)
    import matplotlib as _mpl
    _mpl.rcParams["font.family"] = "serif"
    _mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    _mpl.rcParams["axes.edgecolor"] = "#6c757d"
    _mpl.rcParams["axes.labelcolor"] = "#f8f9fa"
    _mpl.rcParams["text.color"] = "#f8f9fa"
    _mpl.rcParams["xtick.color"] = "#6c757d"
    _mpl.rcParams["ytick.color"] = "#6c757d"
    _mpl.rcParams["grid.color"] = "#6c757d"
    _mpl.rcParams["grid.alpha"] = 0.2
    DARK = "#1a1a2e"; ACC = "#e94560"; GOLD = "#d4a373"; WHITE = "#f8f9fa"; MUTED = "#6c757d"
    try:
        database_path = Path(getattr(app_instance, "database_path", "data/database.xlsx"))
        if not database_path.exists():
            app_instance.logging.warning("Shared database workbook not found to generate report.")
            return

        cash_balance = _coerce_datetime_index(_load_sheet(database_path, "cash_balance", index_col=0))
        executions = _load_sheet(database_path, "executions")
        commissions = _load_sheet(database_path, "commissions")
        periods_traded = _load_sheet(database_path, "periods_traded")
        portfolio_snapshots = _load_sheet(database_path, "portfolio_snapshots")
        positions = _load_sheet(database_path, "positions")
        open_orders = _load_sheet(database_path, "open_orders")
        orders_status = _load_sheet(database_path, "orders_status")
        account_updates = _load_sheet(database_path, "account_updates")
        app_time_spent = _load_sheet(database_path, "app_time_spent")

        if cash_balance.empty or "value" not in cash_balance.columns:
            app_instance.logging.warning("cash_balance sheet is empty; skipping report generation.")
            return

        equity_series = _to_numeric(cash_balance["value"]).dropna()
        equity_series = equity_series[~equity_series.index.duplicated(keep="last")]
        if len(equity_series) < 2:
            app_instance.logging.warning("Not enough equity observations to generate report.")
            return

        period_returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Detect bar frequency from median time delta between equity observations
        time_deltas = equity_series.index.to_series().diff().dropna()
        if len(time_deltas) > 0:
            median_delta_seconds = time_deltas.median().total_seconds()
            # Annualization factor based on actual bar frequency
            if median_delta_seconds > 0:
                periods_per_year = (365.25 * 24 * 3600) / median_delta_seconds
                ann_factor = np.sqrt(periods_per_year)
            else:
                periods_per_year = 252  # fallback: assume daily
                ann_factor = np.sqrt(252)
        else:
            periods_per_year = 252  # fallback: assume daily
            ann_factor = np.sqrt(252)

        # Daily returns for monthly heatmap, rolling metrics (still useful)
        daily_returns = period_returns.resample("1D").apply(_compound_returns).dropna()
        if daily_returns.empty:
            daily_returns = period_returns.copy()

        drawdown = (equity_series / equity_series.cummax()) - 1.0
        max_dd = float(drawdown.min())
        initial_cap = float(equity_series.iloc[0])
        final_cap = float(equity_series.iloc[-1])
        total_ret = (final_cap / initial_cap) - 1.0 if initial_cap else 0.0
        days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
        cagr = (1.0 + total_ret) ** (365.0 / days) - 1.0 if total_ret > -1 else -1.0

        # ── Sharpe, Sortino, Vol from period_returns (actual bar frequency) ──
        ann_vol = float(ann_factor * period_returns.std()) if len(period_returns) >= 2 else np.nan
        sharpe = (
            float(ann_factor * period_returns.mean() / period_returns.std())
            if len(period_returns) >= 2 and float(period_returns.std()) > 0 else np.nan
        )
        downside = period_returns[period_returns < 0]
        sortino = (
            float(ann_factor * period_returns.mean() / downside.std())
            if len(downside) >= 2 and float(downside.std()) > 0 else np.nan
        )
        # Flag: annealing note based on actual bar count
        if len(period_returns) < 10:
            sharpe_note = f" ({len(period_returns)} bars — early estimate)"
        elif not np.isfinite(sharpe):
            sharpe_note = " (n/a)"
        else:
            sharpe_note = ""
        calmar = float(cagr / abs(max_dd)) if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0 else np.nan
        var_95 = float(period_returns.quantile(0.05)) if len(period_returns) else np.nan
        es_95 = float(period_returns[period_returns <= var_95].mean()) if len(period_returns) and (period_returns <= var_95).any() else np.nan

        monthly_rets = daily_returns.resample("ME").apply(_compound_returns).dropna()
        rolling_sharpe = period_returns.rolling(max(20, int(periods_per_year / 252))).apply(
            lambda x: (ann_factor * x.mean() / x.std()) if len(x) > 1 and x.std() > 0 else np.nan
        )

        trade_diag = _build_trade_diagnostics(executions, commissions, positions)
        holdings_diag = _build_holdings_snapshot(portfolio_snapshots, positions, final_cap)
        exec_diag = _build_execution_diagnostics(open_orders, orders_status, executions, commissions)
        account_diag = _build_account_diagnostics(account_updates, cash_balance, holdings_diag["metrics"])
        ops_diag = _build_operational_diagnostics(periods_traded, app_time_spent)
        latest_equity = account_diag["metrics"].get("latest_equity", np.nan)
        if latest_equity is None or not np.isfinite(latest_equity):
            latest_equity = final_cap
        latest_equity_source = account_diag["metrics"].get("latest_equity_source", "cash_balance")

        holdings_symbols = []
        if not holdings_diag["holdings"].empty and "Symbol" in holdings_diag["holdings"].columns:
            holdings_symbols = holdings_diag["holdings"]["Symbol"].astype(str).str.upper().drop_duplicates().tolist()
        active_assets = len(set(holdings_symbols))

        monthly_heat = pd.DataFrame()
        if not monthly_rets.empty:
            monthly_heat = monthly_rets.to_frame(name="ret")
            monthly_heat["year"] = monthly_heat.index.year
            monthly_heat["month"] = monthly_heat.index.month
            monthly_heat = monthly_heat.pivot(index="year", columns="month", values="ret").reindex(columns=list(range(1, 13)))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11), facecolor=DARK)
            ax.set_facecolor(DARK)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            # Header bar (taller, title split to two lines to avoid datetime overlap)
            ax.add_patch(mpatches.FancyBboxPatch((0, 0.91), 1, 0.09, boxstyle="round,pad=0", facecolor=ACC, edgecolor="none", transform=ax.transAxes))
            ax.text(0.04, 0.955, "LIVE PORTFOLIO", transform=ax.transAxes, ha="left", va="center", fontsize=14, fontweight="bold", color=WHITE)
            ax.text(0.04, 0.925, "PERFORMANCE REPORT", transform=ax.transAxes, ha="left", va="center", fontsize=14, fontweight="bold", color=WHITE)
            ax.text(0.94, 0.94, f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", transform=ax.transAxes, ha="right", va="center", fontsize=7, color=WHITE)

            # Period line (compact)
            ax.text(0.04, 0.87, f"Period: {equity_series.index[0].date()} to {equity_series.index[-1].date()}  |  Equity: {_currency_or_na(latest_equity)} ({latest_equity_source})  |  Return: {_safe_pct(total_ret)}  |  CAGR: {_safe_pct(cagr)}", transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=GOLD, fontweight="bold")

            # ── Four metric cards — vertical stacking, taller cards ──
            card_colors = [ACC, "#2196F3", GOLD, "#4CAF50"]
            card_titles = ["PERFORMANCE", "EQUITY", "TRADING", "EXPOSURE"]
            card_x = [0.03, 0.51, 0.03, 0.51]
            card_y = [0.58, 0.58, 0.20, 0.20]
            card_w, card_h = 0.46, 0.26

            for ci in range(4):
                cx, cy = card_x[ci], card_y[ci]
                ax.add_patch(mpatches.FancyBboxPatch((cx, cy), card_w, card_h, boxstyle="round,pad=0.01", facecolor="#152040", edgecolor=card_colors[ci], linewidth=1.0, transform=ax.transAxes))
                ax.add_patch(mpatches.FancyBboxPatch((cx, cy+card_h-0.04), card_w, 0.04, boxstyle="round,pad=0", facecolor=card_colors[ci], edgecolor="none", transform=ax.transAxes))
                ax.text(cx+0.03, cy+card_h-0.02, card_titles[ci], transform=ax.transAxes, ha="left", va="center", fontsize=9, fontweight="bold", color=WHITE)

            # Card 1: Performance — stacked vertically
            c1x, c1y, c1h = 0.03, 0.58, 0.26
            row_y = c1y + c1h - 0.06  # start below title bar
            row_gap = 0.03
            for label, val in [("Sharpe", f"{_safe_num(sharpe,2)}{sharpe_note}"), ("Sortino", f"{_safe_num(sortino,2)}"),
                               ("Calmar", f"{_safe_num(calmar,2)}"), ("Max DD", f"{_safe_pct(max_dd)}"),
                               ("Ann Vol", f"{_safe_pct(ann_vol)}"), ("VaR95 / ES95", f"{_safe_pct(var_95)} / {_safe_pct(es_95)}")]:
                ax.text(c1x+0.04, row_y, f"{label}: {val}", transform=ax.transAxes, fontsize=7.5, color=WHITE)
                row_y -= row_gap

            # Card 2: Equity — stacked vertically
            c2x, c2y, c2h = 0.51, 0.58, 0.26
            row_y = c2y + c2h - 0.06
            for label, val in [("Start", f"{_currency_or_na(initial_cap)}"), ("Latest", f"{_currency_or_na(final_cap)}"),
                               ("Source", f"{latest_equity_source}"), ("Days", f"{days}"),
                               ("Bars", f"{len(equity_series)}")]:
                ax.text(c2x+0.04, row_y, f"{label}: {val}", transform=ax.transAxes, fontsize=7.5, color=WHITE)
                row_y -= row_gap

            # Card 3: Trading — stacked vertically
            c3x, c3y, c3h = 0.03, 0.20, 0.26
            row_y = c3y + c3h - 0.06
            for label, val in [("Net PnL", f"{_currency_or_na(trade_diag['net_realized_pnl'])}"),
                               ("Gross PnL", f"{_currency_or_na(trade_diag['gross_realized_pnl'])}"),
                               ("Commission", f"{_currency_or_na(trade_diag['total_commission'])}"),
                               ("Trades", f"{trade_diag['trades']}"),
                               ("Win Rate", f"{_safe_pct(trade_diag['win_rate'])}"),
                               ("PF / AWL", f"{_safe_num(trade_diag['profit_factor'],2)} / {_safe_num(trade_diag['avg_win_loss'],2)}")]:
                ax.text(c3x+0.04, row_y, f"{label}: {val}", transform=ax.transAxes, fontsize=7.5, color=WHITE)
                row_y -= row_gap

            # Card 4: Exposure — stacked vertically
            c4x, c4y, c4h = 0.51, 0.20, 0.26
            row_y = c4y + c4h - 0.06
            for label, val in [("Gross Exp", f"{_currency_or_na(holdings_diag['metrics'].get('gross_exposure'))}"),
                               ("Net Exp", f"{_currency_or_na(holdings_diag['metrics'].get('net_exposure'))}"),
                               ("Active", f"{active_assets} ({', '.join(holdings_symbols[:3]) if holdings_symbols else 'none'})"),
                               ("Periods", f"{_safe_pct(ops_diag['completion_ratio'])}"),
                               ("Runtime", f"{_safe_num(ops_diag['metrics'].get('avg_runtime_seconds',np.nan),0)}s")]:
                ax.text(c4x+0.04, row_y, f"{label}: {val}", transform=ax.transAxes, fontsize=7.5, color=WHITE)
                row_y -= row_gap

            # Footer
            ax.text(0.5, 0.04, f"Multi-Asset Strategy  |  IBKR Paper Account  |  {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=MUTED)

            # Thin separator below period line
            ax.plot([0.03, 0.97], [0.845, 0.845], color=MUTED, linewidth=0.5, transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Equity curve (separate figure to avoid overlap)
            fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=DARK)
            ax.set_facecolor(DARK)
            ax.plot(equity_series.index, equity_series.values, color="tab:blue", linewidth=1.6)
            ax.set_title("Portfolio Equity Curve", color=WHITE)
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", rotation=45, colors=MUTED); ax.tick_params(axis="y", colors=MUTED)
            pdf.savefig(fig, bbox_inches="tight", facecolor=DARK)
            plt.close(fig)
            # Drawdown (separate figure)
            fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=DARK)
            ax.set_facecolor(DARK)
            ax.fill_between(drawdown.index, drawdown.values, 0.0, color="tab:red", alpha=0.3)
            ax.axhline(0, color="white", linewidth=0.8)
            ax.set_title("Portfolio Drawdown", color=WHITE)
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", rotation=45, colors=MUTED); ax.tick_params(axis="y", colors=MUTED)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Monthly returns (separate figure)
            fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=DARK)
            ax.set_facecolor(DARK)
            if not monthly_rets.empty:
                colors = np.where(monthly_rets.values >= 0, "tab:green", "tab:red")
                ax.bar(monthly_rets.index, monthly_rets.values, width=20, color=colors)
                ax.axhline(0, color="white", linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "Not enough history for monthly returns", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            ax.set_title("Monthly Returns", color=WHITE)
            ax.grid(alpha=0.2)
            ax.tick_params(axis='x', rotation=45, colors=MUTED); ax.tick_params(axis='y', colors=MUTED)
            pdf.savefig(fig, bbox_inches="tight", facecolor=DARK)
            plt.close(fig)
            # Rolling Sharpe (separate figure)
            fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=DARK)
            ax.set_facecolor(DARK)
            valid_rolling = rolling_sharpe.dropna()
            if len(valid_rolling) > 0:
                ax.plot(valid_rolling.index, valid_rolling.values, color="tab:blue", linewidth=1.4)
                ax.axhline(0, color="white", linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "Not enough data for rolling Sharpe", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            ax.set_title("Rolling Sharpe (adaptive window)", color=WHITE)
            ax.grid(alpha=0.2)
            ax.tick_params(axis='x', rotation=45, colors=MUTED); ax.tick_params(axis='y', colors=MUTED)
            pdf.savefig(fig, bbox_inches="tight", facecolor=DARK)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=DARK)
            ax.set_facecolor(DARK)
            if not monthly_heat.empty:
                im = ax.imshow(monthly_heat.values, aspect='auto', cmap='RdYlGn', vmin=-0.05, vmax=0.05)
                for i in range(monthly_heat.shape[0]):
                    for j in range(monthly_heat.shape[1]):
                        val = monthly_heat.values[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=8,
                                    color='black' if abs(val) < 0.025 else 'white')
                ax.set_xticks(range(monthly_heat.shape[1]))
                ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:monthly_heat.shape[1]])
                ax.set_yticks(range(monthly_heat.shape[0]))
                ax.set_yticklabels(monthly_heat.index)
                ax.set_title("Monthly Returns Heatmap")
                ax.set_xlabel("Month")
                ax.set_ylabel("Year")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "Not enough history for monthly returns heatmap", ha="center", va="center")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=DARK)
            fig.subplots_adjust(hspace=0.45, wspace=0.30)
            for a in axes.flatten(): a.set_facecolor(DARK)
            trade_pnl = trade_diag["trade_pnl"]
            if len(trade_pnl):
                axes[0, 0].hist(trade_pnl, bins=min(20, max(5, len(trade_pnl))), color="tab:blue", alpha=0.8)
            else:
                axes[0, 0].text(0.5, 0.5, "No realized trades yet", ha="center", va="center")
            axes[0, 0].set_title("Trade PnL Histogram")

            pnl_by_symbol = trade_diag["pnl_by_symbol"]
            if len(pnl_by_symbol):
                pnl_by_symbol.plot(kind="barh", ax=axes[0, 1], color="tab:green")
            else:
                axes[0, 1].text(0.5, 0.5, "No symbol PnL yet", ha="center", va="center")
            axes[0, 1].set_title("PnL by Asset")

            trade_count_by_symbol = trade_diag["trade_count_by_symbol"]
            if len(trade_count_by_symbol):
                trade_count_by_symbol.plot(kind="bar", ax=axes[1, 0], color="tab:orange")
            else:
                axes[1, 0].text(0.5, 0.5, "No trade counts yet", ha="center", va="center")
            axes[1, 0].set_title("Trade Count by Asset")
            axes[1, 0].tick_params(axis="x", rotation=45)

            axes[1, 1].axis("off")
            diag_lines = [
                "Trade Diagnostics",
                "",
                f"Trades: {trade_diag['trades']}",
                f"Win Rate: {_safe_pct(trade_diag['win_rate'])}",
                f"Profit Factor: {_safe_num(trade_diag['profit_factor'], 2)}",
                f"Avg Trade: {_currency_or_na(trade_diag['avg_trade'])}",
                f"Avg Win/Loss: {_safe_num(trade_diag['avg_win_loss'], 2)}",
                f"Gross PnL: {_currency_or_na(trade_diag['gross_realized_pnl'])}",
                f"Commission: {_currency_or_na(trade_diag['total_commission'])}",
                f"Net PnL: {_currency_or_na(trade_diag['net_realized_pnl'])}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(diag_lines), va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=DARK)
            fig.subplots_adjust(hspace=0.45, wspace=0.30)
            for a in axes.flatten(): a.set_facecolor(DARK)
            holdings = holdings_diag["holdings"]
            if not holdings.empty:
                top_mv = holdings.head(10).sort_values("AbsMarketValue")
                axes[0, 0].barh(top_mv["Symbol"], top_mv["AbsMarketValue"], color="tab:blue")
                axes[0, 0].set_title("Top Holdings by Absolute Market Value")
            else:
                axes[0, 0].text(0.5, 0.5, "No holdings snapshot available", ha="center", va="center")

            # Show current positions by symbol instead of asset-class pie (more actionable)
            if not holdings.empty and "AbsMarketValue" in holdings.columns:
                pos_data = holdings[holdings["AbsMarketValue"] > 0].sort_values("AbsMarketValue")
                if len(pos_data):
                    axes[0, 1].barh(pos_data["Symbol"], pos_data["AbsMarketValue"], color="tab:cyan")
                    axes[0, 1].set_title("Current Positions by Market Value")
                else:
                    axes[0, 1].text(0.5, 0.5, "No active positions", ha="center", va="center", transform=axes[0,1].transAxes)
            else:
                axes[0, 1].text(0.5, 0.5, "No position data available", ha="center", va="center", transform=axes[0,1].transAxes)

            # Check for actual holdings: non-zero position with valid average cost
            has_positions = False
            if not holdings.empty and "Position" in holdings.columns and "AbsMarketValue" in holdings.columns:
                has_positions = (holdings["AbsMarketValue"] > 0.01).any()
            has_nonzero_pnl = has_positions and "UnrealizedPnL" in holdings.columns and (holdings["UnrealizedPnL"].abs() > 0.01).any()
            if has_nonzero_pnl:
                pnl_hold = holdings.dropna(subset=["UnrealizedPnL"]).sort_values("UnrealizedPnL").tail(10)
                colors = np.where(pnl_hold["UnrealizedPnL"].values >= 0, "tab:green", "tab:red")
                axes[1, 0].barh(pnl_hold["Symbol"], pnl_hold["UnrealizedPnL"].values, color=colors)
                axes[1, 0].set_title("Unrealized PnL by Symbol")
            elif has_positions:
                axes[1, 0].text(0.5, 0.5, "Positions open but no\ncurrent market prices", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=9)
                axes[1, 0].set_title("Unrealized PnL by Symbol")
            elif not holdings.empty:
                axes[1, 0].text(0.5, 0.5, "No active positions", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=10)
                axes[1, 0].set_title("Unrealized PnL by Symbol")
            else:
                axes[1, 0].text(0.5, 0.5, "No holdings data", ha="center", va="center", transform=axes[1,0].transAxes)

            axes[1, 1].axis("off")
            holding_lines = [
                "Exposure Snapshot (last saved state)",
                "",
                f"Gross Exposure: {_currency_or_na(holdings_diag['metrics'].get('gross_exposure'))}",
                f"Net Exposure: {_currency_or_na(holdings_diag['metrics'].get('net_exposure'))}",
                f"Largest Symbol: {holdings_diag['metrics'].get('largest_symbol', 'n/a')}",
                f"Largest Weight: {_safe_pct(holdings_diag['metrics'].get('largest_symbol_weight'))}",
                f"Unrealized PnL: {_currency_or_na(holdings_diag['metrics'].get('unrealized_pnl'))}",
                f"Realized PnL Snapshot: {_currency_or_na(holdings_diag['metrics'].get('realized_pnl_snapshot'))}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(holding_lines), va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if not holdings.empty:
                table_rows = [["Symbol", "AssetClass", "Position", "MktValue", "Weight", "UnrealPnL"]]
                for _, row in holdings.head(15).iterrows():
                    table_rows.append(
                        [
                            row.get("Symbol", ""),
                            row.get("AssetClass", ""),
                            _safe_num(row.get("Position"), 2),
                            _currency_or_na(row.get("MarketValue")),
                            _safe_pct(row.get("Weight")),
                            _currency_or_na(row.get("UnrealizedPnL")),
                        ]
                    )
                fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=DARK)
                ax.set_facecolor(DARK)
                ax.axis("off")
                ax.set_title("Current Holdings Table")
                table = ax.table(cellText=table_rows, loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.25)
                # Dark theme cell colors
                for key, cell in table.get_celld().items():
                    cell.set_facecolor(DARK)
                    cell.set_text_props(color=WHITE)
                    cell.set_edgecolor(MUTED)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=DARK)
            fig.subplots_adjust(hspace=0.45, wspace=0.30)
            for a in axes.flatten(): a.set_facecolor(DARK)
            status_counts = exec_diag["status_counts"]
            if len(status_counts):
                status_counts.plot(kind="bar", ax=axes[0, 0], color="tab:blue")
                axes[0, 0].tick_params(axis="x", rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, "No order-status data", ha="center", va="center", transform=axes[0,0].transAxes)
            axes[0, 0].set_xlabel(""); axes[0, 0].set_title("Order Status Breakdown")

            # Show current positions (absolute market value) instead of commission (more useful)
            holdings_pos = holdings_diag.get("holdings", pd.DataFrame())
            if not holdings_pos.empty and "AbsMarketValue" in holdings_pos.columns:
                pos_plot = holdings_pos[holdings_pos["AbsMarketValue"] > 0].sort_values("AbsMarketValue")
                if len(pos_plot):
                    axes[0, 1].barh(pos_plot["Symbol"], pos_plot["AbsMarketValue"], color="tab:cyan")
                    axes[0, 1].set_title("Current Positions ($ Market Value)")
                else:
                    axes[0, 1].text(0.5, 0.5, "No active positions", ha="center", va="center", transform=axes[0,1].transAxes)
            else:
                axes[0, 1].text(0.5, 0.5, "No position data", ha="center", va="center", transform=axes[0,1].transAxes)

            fills_by_symbol = exec_diag["fills_by_symbol"]
            if len(fills_by_symbol):
                fills_by_symbol.tail(12).plot(kind="barh", ax=axes[1, 0], color="tab:green")
            else:
                axes[1, 0].text(0.5, 0.5, "No fills-by-symbol data", ha="center", va="center")
            axes[1, 0].set_title("Execution Count by Symbol")

            axes[1, 1].axis("off")
            exec_lines = [
                "Execution Quality",
                "",
                f"Orders Submitted: {exec_diag['metrics']['orders_submitted']}",
                f"Orders Filled: {exec_diag['metrics']['orders_filled']}",
                f"Fill Ratio: {_safe_pct(exec_diag['metrics']['fill_ratio'])}",
                f"Partial Fill Ratio: {_safe_pct(exec_diag['metrics']['partial_fill_ratio'])}",
                f"Cancel Ratio: {_safe_pct(exec_diag['metrics']['cancel_ratio'])} (incl. RM roll-forward)",
                f"Reject Ratio: {_safe_pct(exec_diag['metrics']['reject_ratio'])}",
                f"Avg Commission / Exec: {_currency_or_na(exec_diag['metrics']['avg_commission_per_exec'])}",
                f"Avg Fill Price: {_safe_num(exec_diag['metrics']['avg_fill_price'], 2)}",
                f"Slippage Proxy |fill-reference|: {_safe_num(exec_diag['metrics']['slippage_proxy_abs'], 2)}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(exec_lines), va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            timeline = exec_diag["execution_timeline"]
            if len(timeline):
                fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=DARK)
                ax.set_facecolor(DARK)
                if len(timeline) >= 2:
                    ax.plot(timeline.index, timeline.values, color="tab:blue", linewidth=1.4)
                else:
                    ax.bar(timeline.index, timeline.values, color="tab:blue", width=0.5)
                    ax.text(0.5, 0.9, f"Single day: {int(timeline.values[0])} executions", ha="center", va="top", transform=ax.transAxes)
                ax.set_title("Executions Timeline")
                ax.grid(alpha=0.2)
                ax.tick_params(axis="x", rotation=45)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=DARK)
            fig.subplots_adjust(hspace=0.45, wspace=0.30)
            for a in axes.flatten(): a.set_facecolor(DARK)
            net_liq = account_diag["net_liq"]
            if len(net_liq) >= 2:
                axes[0, 0].plot(net_liq.index, net_liq.values, color="tab:blue", linewidth=1.4)
            elif len(net_liq) == 1:
                axes[0, 0].axhline(y=net_liq.values[0], color="tab:blue", linewidth=2)
                axes[0, 0].text(0.5, 0.5, f"Equity: ${net_liq.values[0]:,.0f}", ha="center", va="center", transform=axes[0,0].transAxes, fontsize=12)
            else:
                axes[0, 0].text(0.5, 0.5, "No equity data", ha="center", va="center", transform=axes[0,0].transAxes)
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].set_title("Net Liquidation")

            avail = account_diag["available_funds"]
            excess = account_diag["excess_liquidity"]
            has_avail = len(avail) >= 1
            has_excess = len(excess) >= 1
            if has_avail or has_excess:
                if len(avail) >= 2: axes[0, 1].plot(avail.index, avail.values, label="AvailableFunds", color="tab:green")
                elif len(avail) == 1: axes[0, 1].axhline(y=avail.values[0], color="tab:green", linewidth=2, label="AvailableFunds")
                if len(excess) >= 2: axes[0, 1].plot(excess.index, excess.values, label="ExcessLiquidity", color="tab:orange")
                elif len(excess) == 1: axes[0, 1].axhline(y=excess.values[0], color="tab:orange", linewidth=2, label="ExcessLiquidity")
                axes[0, 1].legend(facecolor=DARK, edgecolor=MUTED, labelcolor=WHITE, fontsize=8)
            else:
                axes[0, 1].text(0.5, 0.5, "No liquidity series", ha="center", va="center", transform=axes[0,1].transAxes)
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].set_title("Available Funds / Excess Liquidity")

            lev = account_diag["leverage_series"]
            margin_util = account_diag["margin_utilization"]
            axes[1, 0].set_xlim(0, 1); axes[1, 0].set_ylim(0, 1)
            has_data = False
            if len(lev) >= 2:
                axes[1, 0].clear()
                axes[1, 0].plot(lev.index, lev.values, label="Leverage", color="tab:red")
                has_data = True
            elif len(lev) == 1 and lev.values[0] > 0:
                axes[1, 0].text(0.5, 0.60, f"Leverage: {lev.values[0]:.2f}x", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=14, fontweight="bold", color="tab:red")
                axes[1, 0].text(0.5, 0.40, f"Gross Exposure / Equity", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=9, color="gray")
                has_data = True
            if len(margin_util) >= 2:
                if not has_data: axes[1, 0].clear()
                axes[1, 0].plot(margin_util.index, margin_util.values, label="MarginUtil", color="tab:purple")
                axes[1, 0].legend(facecolor=DARK, edgecolor=MUTED, labelcolor=WHITE, fontsize=8)
                axes[1, 0].set_ylim(bottom=0)
                axes[1, 0].margins(y=0.2)
                has_data = True
            if not has_data:
                ge_val = holdings_diag.get("metrics", {}).get("gross_exposure", 0) or 0
                if abs(ge_val) < 0.01:
                    axes[1, 0].text(0.5, 0.55, "No open positions", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=11, color="gray")
                    axes[1, 0].text(0.5, 0.40, "Leverage = 0", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=9, color="gray")
                else:
                    axes[1, 0].text(0.5, 0.5, "No leverage/margin data", ha="center", va="center", transform=axes[1,0].transAxes, fontsize=9, color="gray")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].set_title("Leverage / Margin Utilization")

            axes[1, 1].axis("off")
            account_lines = [
                "Account Health",
                "",
                f"Min Available Funds: {_currency_or_na(account_diag['metrics']['min_available_funds'])}",
                f"Min Excess Liquidity: {_currency_or_na(account_diag['metrics']['min_excess_liquidity'])}",
                f"Latest Buying Power: {_currency_or_na(account_diag['metrics']['latest_buying_power'])}",
                f"Max Margin Utilization: {_safe_pct(account_diag['metrics']['max_margin_utilization'])}",
                f"Gross Exposure: {_currency_or_na(account_diag['metrics']['gross_exposure'])}",
                f"Net Exposure: {_currency_or_na(account_diag['metrics']['net_exposure'])}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(account_lines), va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=DARK)
            fig.subplots_adjust(hspace=0.45, wspace=0.30)
            for a in axes.flatten(): a.set_facecolor(DARK)
            runtime_seconds = ops_diag["runtime_seconds"]
            if len(runtime_seconds):
                tp = runtime_seconds.tail(20)
                tp.plot(kind="barh", ax=axes[0, 0], color="tab:blue")
                axes[0, 0].set_yticklabels([str(x)[:16] for x in tp.index], fontsize=7)
            else:
                axes[0, 0].text(0.5, 0.5, "No app runtime data", ha="center", va="center")
            axes[0, 0].set_title("Runtime Seconds by Period")

            period_timeline = ops_diag["period_timeline"]
            if not period_timeline.empty:
                axes[0, 1].plot(period_timeline["trade_time"], period_timeline["trade_done"], color="tab:green")
                axes[0, 1].set_ylim(-0.05, 1.05)
                axes[0, 1].tick_params(axis="x", rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, "No period timeline data", ha="center", va="center", transform=axes[0,1].transAxes)
            axes[0, 1].set_title("Trade Completion Timeline")

            if not ops_diag["missed_periods"].empty:
                missed = ops_diag["missed_periods"].copy().tail(15)
                axes[1, 0].barh(missed["trade_time"].dt.strftime("%Y-%m-%d %H:%M"), _to_numeric(missed["trade_done"]).fillna(0.0), color="tab:red")
            else:
                axes[1, 0].text(0.5, 0.5, "No missed periods", ha="center", va="center")
            axes[1, 0].set_title("Most Recent Missed Periods")

            axes[1, 1].axis("off")
            ops_lines = [
                "Operational Reliability",
                "",
                f"Period Completion Ratio: {_safe_pct(ops_diag['completion_ratio'])}",
                f"Missed Periods: {ops_diag['metrics']['missed_periods']}",
                f"Avg Runtime Seconds: {_safe_num(ops_diag['metrics']['avg_runtime_seconds'], 2)}",
                f"Max Runtime Seconds: {_safe_num(ops_diag['metrics']['max_runtime_seconds'], 2)}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(ops_lines), va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            plt.close(fig)

            target_df = pd.DataFrame()  # not yet populated from workbook
            state_table = pd.DataFrame()  # not yet populated from workbook
            if not target_df.empty:
                rows = [["Symbol", "Signal", "TargetQty", "ActualQty", "TargetExp", "ActualExp"]]
                for _, row in target_df.head(20).iterrows():
                    rows.append(
                        [
                            row.get("Symbol", ""),
                            row.get("Signal", ""),
                            _safe_num(row.get("TargetQuantity"), 2),
                            _safe_num(row.get("Position"), 2),
                            _safe_pct(row.get("TargetExposure")),
                            _safe_pct(row.get("ActualExposure")),
                        ]
                    )
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.set_title("Target vs Actual Table")
                table = ax.table(cellText=rows, loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.25)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            if not state_table.empty:
                rows = [["Symbol", "Key", "Value"]]
                for _, row in state_table.head(40).iterrows():
                    value = str(row.get("Value", ""))
                    rows.append([row.get("Symbol", ""), row.get("Key", ""), value[:80]])
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.set_title("Strategy State Snapshot")
                table = ax.table(cellText=rows, loc="center", cellLoc="left")
                table.auto_set_font_size(False)
                table.set_fontsize(7.5)
                table.scale(1, 1.15)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            recent = pd.DataFrame({
                "Equity": equity_series,
                "Drawdown": drawdown,
                "Return": period_returns.reindex(equity_series.index).fillna(0.0),
            }).tail(25)
            fig, ax = plt.subplots(figsize=(8.5, 11), facecolor=DARK)
            ax.set_facecolor(DARK)
            ax.axis("off")
            table_data = [["Date", "Equity", "DD %", "Ret %"]]
            for dtime, row in recent.iterrows():
                table_data.append([
                    dtime.strftime("%Y-%m-%d %H:%M"),
                    f"${row['Equity']:,.2f}",
                    f"{row['Drawdown']:.2%}",
                    f"{row['Return']:.2%}",
                ])
            table = ax.table(cellText=table_data, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for key, cell in table.get_celld().items():
                cell.set_facecolor(DARK)
                cell.set_text_props(color=WHITE)
                cell.set_edgecolor(MUTED)
            ax.set_title("Recent Performance Snapshot (Last 25 Periods)")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


            # ═══════════════════════════════════════════════════════════════
            # Final page: Notes & Assumptions
            # ═══════════════════════════════════════════════════════════════
            fig, ax = plt.subplots(figsize=(8.5, 11), facecolor=DARK)
            ax.set_facecolor(DARK)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

            ax.text(0.06, 0.96, "NOTES & ASSUMPTIONS", transform=ax.transAxes, ha="left", va="top",
                    fontsize=16, fontweight="bold", color=ACC)

            assumptions = [
                ("Data Sources", [
                    "Equity curve: cash_balance['value'] column from database.xlsx.",
                    "Returns: period-to-period percent change of cash_balance['value'].",
                    "Trades/commissions: commissions and executions sheets from database.xlsx.",
                    "Positions/holdings: positions sheet (fallback: portfolio_snapshots sheet).",
                    "Account health: account_updates sheet (fallback: cash_balance for sparse fields).",
                ]),
                ("Performance Metrics", [
                    "CAGR = (Final / Initial)^(365.25/days) − 1, using actual calendar days.",
                    "Sharpe = √(periods_per_year) × μ(period_returns) / σ(period_returns).",
                    "Sortino = √(periods_per_year) × μ(period_returns) / σ(downside_returns).",
                    "Annualized Vol = √(periods_per_year) × σ(period_returns).",
                    "Max Drawdown = min(equity / running_max − 1).",
                    "Calmar = CAGR / |Max Drawdown|.",
                    "VaR95 = 5th percentile of period returns; ES95 = mean of returns below VaR95.",
                    "Annualization factor derived from median time delta between cash_balance rows.",
                    "Sharpe/Sortino on limited bars (< 10) are flagged as early estimates.",
                ]),
                ("Trade Diagnostics", [
                    "Realized PnL: from commissions sheet 'Realized PnL' column when IBKR reports it.",
                    "PnL estimation: when IBKR does not report Realized PnL, it is estimated from",
                    "   (avg sell price − avg buy price) × min(buy_qty, sell_qty) per symbol.",
                    "Commission estimation: uses known IB rates when Commission column is NaN:",
                    "   MES $1.24/side, FX $2.00/order, XAU 0.20%, Crypto 0.18%.",
                    "Execution Symbol+Currency is mapped to pair names (EUR+USD → EURUSD).",
                    "Profit Factor = Σ(wins) / |Σ(losses)|; Win Rate = (PnL > 0).mean().",
                    "Avg Win/Loss = mean(wins) / |mean(losses)| when both exist.",
                ]),
                ("Holdings & Exposure", [
                    "Holdings snapshot: latest row per (Account, Symbol) from positions sheet.",
                    "Market Value: from portfolio_snapshots when available; otherwise",
                    "   estimated as |Position| × Avg cost from the positions sheet.",
                    "Market Price: from portfolio_snapshots when available; otherwise",
                    "   approximated by the average cost basis.",
                    "Unrealized PnL: (MarketPrice − AvgCost) × Position when prices exist.",
                    "   Otherwise shown as n/a (no current market prices available).",
                    "Gross Exposure = Σ|MarketValue|; Net Exposure = Σ(MarketValue).",
                    "Asset classification: FUT → Futures, CASH → FX, CMDTY → Metals, CRYPTO → Crypto.",
                ]),
                ("Account Health", [
                    "Net Liquidation: primarily from cash_balance['value'] (richer time series);",
                    "   falls back to account_updates NetLiquidation or NetLiquidationByCurrency.",
                    "Available Funds / Excess Liquidity: from account_updates;",
                    "   falls back to cash_balance['value'] when too few data points exist.",
                    "Excess Liquidity fallback: cash_balance['value'] × 0.95 (margin buffer proxy).",
                    "Leverage: estimated as Gross Exposure / Equity (no direct leverage column).",
                    "Margin Utilization = |GrossPositionValue| / |NetLiquidation|.",
                ]),
                ("Execution Quality", [
                    "Order status from orders_status sheet; order details from open_orders sheet.",
                    "Fill Ratio = orders filled / orders submitted; Cancel Ratio includes RM roll-forward.",
                    "Slippage proxy = |Avg Fill Price − reference price| (LmtPrice or AuxPrice).",
                ]),
                ("Operational", [
                    "Period Completion Ratio = (periods with trade_done==1) / total periods.",
                    "Runtime: from app_time_spent sheet; shown as avg/max over periods.",
                ]),
                ("Monthly Heatmap & Rolling Sharpe", [
                    "Monthly returns: daily returns compounded to monthly frequency.",
                    "Monthly heatmap: pivot table of monthly returns (rows=years, cols=months).",
                    "Rolling Sharpe: computed from period returns (native bar frequency),",
                    "   window = max(20, periods_per_year / 252) ≈ 1 trading-day equivalent.",
                ]),
                ("General Caveats", [
                    "All metrics are computed from the database workbook at report generation time.",
                    "Positions and exposure reflect the last saved state, not necessarily",
                    "   the current live state (database is updated per bar, not real-time).",
                    "IBKR does not always report Commission or Realized PnL;",
                    "   estimated values are flagged where applicable.",
                    "'n/a' means: insufficient data, undefined metric, or NaN in source.",
                    "Sharpe/Sortino/Vol are computed from intraday bar returns and annualized",
                    "   to provide usable estimates even with limited daily history.",
                    "This report is auto-generated by the multi-asset live trading engine.",
                ]),
            ]

            y = 0.92
            for section_title, items in assumptions:
                if y < 0.12:
                    pdf.savefig(fig, bbox_inches="tight", facecolor=DARK)
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8.5, 11), facecolor=DARK)
                    ax.set_facecolor(DARK)
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
                    ax.text(0.06, 0.96, "NOTES & ASSUMPTIONS (continued)", transform=ax.transAxes, ha="left", va="top",
                            fontsize=14, fontweight="bold", color=ACC)
                    y = 0.90

                ax.text(0.06, y, section_title, transform=ax.transAxes, ha="left", va="top",
                        fontsize=11, fontweight="bold", color=GOLD)
                y -= 0.035
                for item in items:
                    ax.text(0.08, y, "  " + item, transform=ax.transAxes, ha="left", va="top",
                            fontsize=8, color=WHITE)
                    y -= 0.023
                    if y < 0.12:
                        pdf.savefig(fig, bbox_inches="tight", facecolor=DARK)
                        plt.close(fig)
                        fig, ax = plt.subplots(figsize=(8.5, 11), facecolor=DARK)
                        ax.set_facecolor(DARK)
                        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
                        y = 0.94
                y -= 0.015  # gap between sections

            pdf.savefig(fig, bbox_inches="tight", facecolor=DARK)
            plt.close(fig)

        app_instance.logging.info(f"Portfolio report generated: {output_path}")
        print(f"Portfolio report generated: {output_path}")
    except Exception as exc:
        app_instance.logging.error(f"Failed to generate report: {exc}")
        print(f"Error generating report: {exc}")
