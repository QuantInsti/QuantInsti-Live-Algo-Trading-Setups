import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
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


def _build_trade_diagnostics(executions, commissions):
    pnl_frame = commissions.copy() if not commissions.empty else pd.DataFrame()
    if pnl_frame.empty or "Realized PnL" not in pnl_frame.columns:
        return {
            "trade_pnl": pd.Series(dtype=float),
            "pnl_by_symbol": pd.Series(dtype=float),
            "trade_count_by_symbol": pd.Series(dtype=float),
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_trade": np.nan,
            "avg_win_loss": np.nan,
            "gross_realized_pnl": np.nan,
            "total_commission": np.nan,
            "net_realized_pnl": np.nan,
            "commission_drag_pct": np.nan,
        }

    pnl_frame = pnl_frame.copy()
    pnl_frame["Realized PnL"] = _to_numeric(pnl_frame["Realized PnL"])
    if "Commission" in pnl_frame.columns:
        pnl_frame["Commission"] = _to_numeric(pnl_frame["Commission"])
    pnl_frame = pnl_frame.dropna(subset=["Realized PnL"])

    if not executions.empty and {"ExecutionId", "Symbol"}.issubset(executions.columns) and "ExecutionId" in pnl_frame.columns:
        execution_symbol_map = executions[["ExecutionId", "Symbol"]].drop_duplicates()
        pnl_frame = pnl_frame.merge(execution_symbol_map, on="ExecutionId", how="left")

    trade_pnl = pnl_frame["Realized PnL"]
    pnl_by_symbol = pd.Series(dtype=float)
    trade_count_by_symbol = pd.Series(dtype=float)
    if "Symbol" in pnl_frame.columns:
        symbols = pnl_frame["Symbol"].astype(str).str.upper()
        pnl_by_symbol = trade_pnl.groupby(symbols).sum().sort_values()
        trade_count_by_symbol = trade_pnl.groupby(symbols).size().sort_values()

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    avg_win_loss = np.nan
    if not wins.empty and not losses.empty and losses.mean() != 0:
        avg_win_loss = float(wins.mean() / abs(losses.mean()))

    total_commission = float(pnl_frame.get("Commission", pd.Series(dtype=float)).sum()) if "Commission" in pnl_frame.columns else np.nan
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

    if latest.empty:
        return {"holdings": pd.DataFrame(), "asset_weights": pd.Series(dtype=float), "metrics": {}}

    latest = latest.copy()
    if "Symbol" in latest.columns:
        latest["Symbol"] = latest["Symbol"].astype(str).str.upper()
    for col in ["Position", "MarketPrice", "MarketValue", "AverageCost", "UnrealizedPnL", "RealizedPnL"]:
        if col in latest.columns:
            latest[col] = _to_numeric(latest[col])
        else:
            latest[col] = np.nan
    latest["AssetClass"] = latest.apply(_classify_asset, axis=1)
    latest["AbsMarketValue"] = latest["MarketValue"].abs().fillna(0.0)
    latest["Weight"] = latest["AbsMarketValue"] / abs(final_equity) if final_equity else np.nan
    latest = latest.sort_values("AbsMarketValue", ascending=False)

    asset_weights = latest.groupby("AssetClass")["AbsMarketValue"].sum().sort_values(ascending=False)
    if asset_weights.sum() > 0:
        asset_weights = asset_weights / asset_weights.sum()

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
            fills_by_symbol = exec_df["Symbol"].astype(str).str.upper().value_counts().sort_values()

    if not commissions.empty:
        comm = commissions.copy()
        if "Commission" in comm.columns:
            comm["Commission"] = _to_numeric(comm["Commission"])
            metrics["avg_commission_per_exec"] = float(comm["Commission"].dropna().mean()) if len(comm) else np.nan
        if not exec_df.empty and {"ExecutionId", "Symbol"}.issubset(exec_df.columns) and "ExecutionId" in comm.columns:
            symbol_map = exec_df[["ExecutionId", "Symbol"]].drop_duplicates()
            comm = comm.merge(symbol_map, on="ExecutionId", how="left")
            if "Symbol" in comm.columns and "Commission" in comm.columns:
                commission_by_symbol = comm.groupby(comm["Symbol"].astype(str).str.upper())["Commission"].sum().sort_values()

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
    if not cash_balance.empty and "leverage" in cash_balance.columns:
        leverage_series = _to_numeric(cash_balance["leverage"]).dropna()
        leverage_series.index = cash_balance.loc[leverage_series.index].index

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

    latest_equity = float(net_liq.iloc[-1]) if len(net_liq) else np.nan
    latest_equity_source = "NetLiquidation" if len(net_liq) else "cash_balance"

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
    try:
        database_path = Path("data") / "database.xlsx"
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

        ann_vol = float(np.sqrt(252.0) * daily_returns.std()) if len(daily_returns) > 1 else np.nan
        sharpe = (
            float(np.sqrt(252.0) * daily_returns.mean() / daily_returns.std())
            if len(daily_returns) > 1 and float(daily_returns.std()) > 0 else np.nan
        )
        downside = daily_returns[daily_returns < 0]
        sortino = (
            float(np.sqrt(252.0) * daily_returns.mean() / downside.std())
            if len(downside) > 1 and float(downside.std()) > 0 else np.nan
        )
        calmar = float(cagr / abs(max_dd)) if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0 else np.nan
        var_95 = float(daily_returns.quantile(0.05)) if len(daily_returns) else np.nan
        es_95 = float(daily_returns[daily_returns <= var_95].mean()) if len(daily_returns) and (daily_returns <= var_95).any() else np.nan

        monthly_rets = daily_returns.resample("ME").apply(_compound_returns).dropna()
        rolling_sharpe = daily_returns.rolling(63).apply(
            lambda x: (np.sqrt(252.0) * x.mean() / x.std()) if len(x) > 1 and x.std() > 0 else np.nan
        )

        trade_diag = _build_trade_diagnostics(executions, commissions)
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
        active_assets = len(holdings_symbols)

        monthly_heat = pd.DataFrame()
        if not monthly_rets.empty:
            monthly_heat = monthly_rets.to_frame(name="ret")
            monthly_heat["year"] = monthly_heat.index.year
            monthly_heat["month"] = monthly_heat.index.month
            monthly_heat = monthly_heat.pivot(index="year", columns="month", values="ret").reindex(columns=list(range(1, 13)))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            summary_lines = [
                "LIVE PORTFOLIO PERFORMANCE REPORT",
                f"Generated at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Period: {equity_series.index[0].date()} to {equity_series.index[-1].date()}",
                f"Starting Equity In Saved History: {initial_cap:,.2f} USD",
                f"Latest Equity: {latest_equity:,.2f} USD",
                f"Equity Source: {latest_equity_source}",
                f"Latest Saved Balance Value: {final_cap:,.2f} USD",
                f"Total Return: {_safe_pct(total_ret)}",
                f"CAGR: {_safe_pct(cagr)}",
                "",
                f"Sharpe: {_safe_num(sharpe, 2)}",
                f"Sortino: {_safe_num(sortino, 2)}",
                f"Calmar: {_safe_num(calmar, 2)}",
                f"Max Drawdown: {_safe_pct(max_dd)}",
                f"Volatility (annualised): {_safe_pct(ann_vol)}",
                f"Daily VaR95 / ES95: {_safe_pct(var_95)} / {_safe_pct(es_95)}",
                "",
                f"Net Realized PnL: {_currency_or_na(trade_diag['net_realized_pnl'])}",
                f"Gross Realized PnL: {_currency_or_na(trade_diag['gross_realized_pnl'])}",
                f"Total Commission: {_currency_or_na(trade_diag['total_commission'])}",
                f"Commission Drag: {_safe_pct(trade_diag['commission_drag_pct'])}",
                "",
                f"Trades: {trade_diag['trades']}",
                f"Win Rate: {_safe_pct(trade_diag['win_rate'])}",
                f"Profit Factor: {_safe_num(trade_diag['profit_factor'], 2)}",
                f"Avg Trade: {_currency_or_na(trade_diag['avg_trade'])}",
                f"Avg Win/Loss: {_safe_num(trade_diag['avg_win_loss'], 2)}",
                "",
                f"Active Assets: {active_assets}",
                f"Holding Symbols: {', '.join(holdings_symbols) if holdings_symbols else 'n/a'}",
                f"Gross Exposure: {_currency_or_na(holdings_diag['metrics'].get('gross_exposure'))}",
                f"Net Exposure: {_currency_or_na(holdings_diag['metrics'].get('net_exposure'))}",
                f"Period Completion Ratio: {_safe_pct(ops_diag['completion_ratio'])}",
            ]
            ax.text(0.08, 0.97, "\n".join(summary_lines), fontsize=10.5, va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            ax1.plot(equity_series.index, equity_series.values, color="tab:blue", linewidth=1.6)
            ax1.set_title("Portfolio Equity Curve")
            ax1.grid(alpha=0.25)
            ax2.fill_between(drawdown.index, drawdown.values, 0.0, color="tab:red", alpha=0.3)
            ax2.axhline(0, color="black", linewidth=1.0)
            ax2.set_title("Portfolio Drawdown")
            ax2.grid(alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
            if not monthly_rets.empty:
                colors = np.where(monthly_rets.values >= 0, "tab:green", "tab:red")
                ax1.bar(monthly_rets.index, monthly_rets.values, width=20, color=colors)
                ax1.axhline(0, color="black", linewidth=1.0)
            else:
                ax1.text(0.5, 0.5, "Not enough history for monthly returns", ha="center", va="center")
            ax1.set_title("Monthly Returns")
            ax1.grid(alpha=0.2)
            ax2.plot(rolling_sharpe.index, rolling_sharpe.values, color="tab:blue", linewidth=1.4)
            ax2.axhline(0, color="black", linewidth=1.0)
            ax2.set_title("Rolling 63-Day Sharpe")
            ax2.grid(alpha=0.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 8.5))
            if not monthly_heat.empty:
                sns.heatmap(monthly_heat, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
                ax.set_title("Monthly Returns Heatmap")
                ax.set_xlabel("Month")
                ax.set_ylabel("Year")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "Not enough history for monthly returns heatmap", ha="center", va="center")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
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
            axes[1, 0].tick_params(axis="x", rotation=25)

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
            axes[1, 1].text(0.03, 0.97, "\n".join(diag_lines), va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            holdings = holdings_diag["holdings"]
            if not holdings.empty:
                top_mv = holdings.head(10).sort_values("AbsMarketValue")
                axes[0, 0].barh(top_mv["Symbol"], top_mv["AbsMarketValue"], color="tab:blue")
                axes[0, 0].set_title("Top Holdings by Absolute Market Value")
            else:
                axes[0, 0].text(0.5, 0.5, "No holdings snapshot available", ha="center", va="center")

            asset_weights = holdings_diag["asset_weights"]
            asset_weights = pd.to_numeric(asset_weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            asset_weights = asset_weights[asset_weights > 0]
            if len(asset_weights) and float(asset_weights.sum()) > 0:
                axes[0, 1].pie(asset_weights.values, labels=asset_weights.index, autopct="%1.2f%%", startangle=90)
                axes[0, 1].set_title("Current Asset-Class Allocation")
            else:
                axes[0, 1].text(0.5, 0.5, "No asset allocation available", ha="center", va="center")

            if not holdings.empty:
                pnl_hold = holdings.sort_values("UnrealizedPnL").tail(10)
                colors = np.where(pnl_hold["UnrealizedPnL"] >= 0, "tab:green", "tab:red")
                axes[1, 0].barh(pnl_hold["Symbol"], pnl_hold["UnrealizedPnL"], color=colors)
                axes[1, 0].set_title("Unrealized PnL by Symbol")
            else:
                axes[1, 0].text(0.5, 0.5, "No unrealized PnL data", ha="center", va="center")

            axes[1, 1].axis("off")
            holding_lines = [
                "Exposure Snapshot",
                "",
                f"Gross Exposure: {_currency_or_na(holdings_diag['metrics'].get('gross_exposure'))}",
                f"Net Exposure: {_currency_or_na(holdings_diag['metrics'].get('net_exposure'))}",
                f"Largest Symbol: {holdings_diag['metrics'].get('largest_symbol', 'n/a')}",
                f"Largest Weight: {_safe_pct(holdings_diag['metrics'].get('largest_symbol_weight'))}",
                f"Unrealized PnL: {_currency_or_na(holdings_diag['metrics'].get('unrealized_pnl'))}",
                f"Realized PnL Snapshot: {_currency_or_na(holdings_diag['metrics'].get('realized_pnl_snapshot'))}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(holding_lines), va="top", family="monospace")
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
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.set_title("Current Holdings Table")
                table = ax.table(cellText=table_rows, loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.25)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            status_counts = exec_diag["status_counts"]
            if len(status_counts):
                status_counts.plot(kind="bar", ax=axes[0, 0], color="tab:blue")
                axes[0, 0].tick_params(axis="x", rotation=30)
            else:
                axes[0, 0].text(0.5, 0.5, "No order-status data", ha="center", va="center")
            axes[0, 0].set_title("Order Status Breakdown")

            commission_by_symbol = exec_diag["commission_by_symbol"]
            if len(commission_by_symbol):
                commission_by_symbol.plot(kind="barh", ax=axes[0, 1], color="tab:red")
            else:
                axes[0, 1].text(0.5, 0.5, "No commission-by-symbol data", ha="center", va="center")
            axes[0, 1].set_title("Commission by Symbol")

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
                f"Cancel Ratio: {_safe_pct(exec_diag['metrics']['cancel_ratio'])}",
                f"Reject Ratio: {_safe_pct(exec_diag['metrics']['reject_ratio'])}",
                f"Avg Commission / Exec: {_currency_or_na(exec_diag['metrics']['avg_commission_per_exec'])}",
                f"Avg Fill Price: {_safe_num(exec_diag['metrics']['avg_fill_price'], 2)}",
                f"Slippage Proxy |fill-reference|: {_safe_num(exec_diag['metrics']['slippage_proxy_abs'], 2)}",
            ]
            axes[1, 1].text(0.03, 0.97, "\n".join(exec_lines), va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            timeline = exec_diag["execution_timeline"]
            if len(timeline):
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.plot(timeline.index, timeline.values, color="tab:blue", linewidth=1.4)
                ax.set_title("Executions Timeline")
                ax.grid(alpha=0.2)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            net_liq = account_diag["net_liq"]
            if len(net_liq):
                axes[0, 0].plot(net_liq.index, net_liq.values, color="tab:blue")
            else:
                axes[0, 0].text(0.5, 0.5, "No NetLiquidation series", ha="center", va="center")
            axes[0, 0].set_title("Net Liquidation")

            avail = account_diag["available_funds"]
            excess = account_diag["excess_liquidity"]
            if len(avail) or len(excess):
                if len(avail):
                    axes[0, 1].plot(avail.index, avail.values, label="AvailableFunds", color="tab:green")
                if len(excess):
                    axes[0, 1].plot(excess.index, excess.values, label="ExcessLiquidity", color="tab:orange")
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, "No liquidity series", ha="center", va="center")
            axes[0, 1].set_title("Available Funds / Excess Liquidity")

            lev = account_diag["leverage_series"]
            margin_util = account_diag["margin_utilization"]
            if len(lev) or len(margin_util):
                if len(lev):
                    axes[1, 0].plot(lev.index, lev.values, label="Leverage", color="tab:red")
                if len(margin_util):
                    axes[1, 0].plot(margin_util.index, margin_util.values, label="MarginUtil", color="tab:purple")
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, "No leverage/margin series", ha="center", va="center")
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
            axes[1, 1].text(0.03, 0.97, "\n".join(account_lines), va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            runtime_seconds = ops_diag["runtime_seconds"]
            if len(runtime_seconds):
                runtime_seconds.plot(kind="barh", ax=axes[0, 0], color="tab:blue")
            else:
                axes[0, 0].text(0.5, 0.5, "No app runtime data", ha="center", va="center")
            axes[0, 0].set_title("Runtime Seconds by Period")

            period_timeline = ops_diag["period_timeline"]
            if not period_timeline.empty:
                axes[0, 1].plot(period_timeline["trade_time"], period_timeline["trade_done"], color="tab:green")
                axes[0, 1].set_ylim(-0.05, 1.05)
            else:
                axes[0, 1].text(0.5, 0.5, "No period timeline data", ha="center", va="center")
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
            axes[1, 1].text(0.03, 0.97, "\n".join(ops_lines), va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            plt.close(fig)

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
            fig, ax = plt.subplots(figsize=(8.5, 11))
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
            ax.set_title("Recent Performance Snapshot (Last 25 Periods)")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        app_instance.logging.info(f"Portfolio report generated: {output_path}")
        print(f"Portfolio report generated: {output_path}")
    except Exception as exc:
        app_instance.logging.error(f"Failed to generate report: {exc}")
        print(f"Error generating report: {exc}")
