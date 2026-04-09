from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import hashlib
import json

import numpy as np
import pandas as pd

from ibkr_multi_asset import trading_functions as tf


DEFAULT_GLOBAL_PARAMS = {
    "seed": 20260407,
    "portfolio_method": "markowitz_mean_reversion",
    "validation_fraction": 0.25,
    "lookback_grid": [12, 20, 30, 40, 60],
    "entry_z_grid": [0.8, 1.0, 1.25, 1.5, 2.0],
    "exit_z_grid": [0.1, 0.25, 0.5, 0.75],
    "target_vol_grid": [0.10, 0.15, 0.20, 0.25, 0.35],
    "max_weight": 0.35,
    "min_weight": 0.0,
    "cash_buffer": 0.05,
    "covariance_ridge": 1e-6,
}

DEFAULT_ASSET_PARAMS = {
    "lookback": 20,
    "entry_z": 1.25,
    "exit_z": 0.25,
    "target_vol": 0.15,
    "allow_short": True,
}

CRYPTO_LONG_ONLY = {"BTC", "ETH", "SOL", "LTC", "BCH"}
DEFAULT_STOCK_SYMBOLS = {"AAPL", "MSFT", "AMZN"}
STOCK_MEAN_REVERSION_FREQUENCY = "15min"

_OPTIMIZATION_MANIFEST_PATH = Path("data/models/strategy2_optimization_manifest.json")
_OPTIMIZATION_FEATURES_PATH = Path("data/models/strategy2_optimal_features_df.xlsx")


def _configured_stock_symbols() -> set[str]:
    try:
        variables = tf.extract_variables("main.py")
        configured = variables.get("stock_symbols", []) or []
        return {str(symbol).upper() for symbol in configured}
    except Exception:
        return set(DEFAULT_STOCK_SYMBOLS)


def get_asset_runtime_policy(symbol, asset_class=None) -> dict:
    symbol = str(symbol or "").upper()
    asset_class = str(asset_class or "").lower()
    configured_stocks = _configured_stock_symbols()
    if asset_class == "crypto" or symbol in CRYPTO_LONG_ONLY:
        return {
            "session": "24_7",
            "flatten_at_day_end": False,
            "daily_maintenance_utc_start": "00:00",
            "daily_maintenance_minutes": 15,
        }
    if asset_class == "stock" or symbol in configured_stocks or symbol in DEFAULT_STOCK_SYMBOLS:
        return {
            "session": "auto",
            "flatten_at_day_end": False,
            "daily_maintenance_utc_start": "00:00",
            "daily_maintenance_minutes": 0,
        }
    return {
        "session": "weekdays",
        "flatten_at_day_end": False,
        "daily_maintenance_utc_start": "00:00",
        "daily_maintenance_minutes": 15,
    }


def get_asset_frequency(symbol) -> str:
    if str(symbol or "").upper() in (_configured_stock_symbols() or DEFAULT_STOCK_SYMBOLS):
        return STOCK_MEAN_REVERSION_FREQUENCY
    return "5min"


def get_asset_train_span(symbol) -> int:
    return 3500


def _symbols_for_kind(kind: str, fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols=None) -> list[str]:
    kind = str(kind or "").lower()
    mapping = {
        "fx": fx_pairs or [],
        "forex": fx_pairs or [],
        "mes": futures_symbols or [],
        "futures": futures_symbols or [],
        "xau": metals_symbols or [],
        "metals": metals_symbols or [],
        "crypto": crypto_symbols or [],
        "stock": stock_symbols or [],
        "stocks": stock_symbols or [],
        "equity": stock_symbols or [],
        "equities": stock_symbols or [],
    }
    return [str(symbol).upper() for symbol in mapping.get(kind, [])]


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    out = df.copy()
    unnamed = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        out = out.drop(columns=unnamed)

    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
            out = out.dropna(subset=["datetime"]).set_index("datetime")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    lower_cols = {str(col).lower(): col for col in out.columns}
    if {"open", "high", "low", "close"}.issubset(lower_cols):
        normalized = out.rename(columns={lower_cols["open"]: "open", lower_cols["high"]: "high", lower_cols["low"]: "low", lower_cols["close"]: "close"})
        return normalized[["open", "high", "low", "close"]].astype(float).dropna()

    if {"bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"}.issubset(lower_cols):
        raw = out.rename(columns={lower_cols[k]: k for k in lower_cols})
        midpoint = tf.get_mid_series(raw)
        midpoint.columns = [str(col).lower() for col in midpoint.columns]
        return midpoint[["open", "high", "low", "close"]].astype(float).dropna()

    title_cols = {str(col): col for col in out.columns}
    if {"Open", "High", "Low", "Close"}.issubset(title_cols):
        normalized = out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        return normalized[["open", "high", "low", "close"]].astype(float).dropna()

    return pd.DataFrame(columns=["open", "high", "low", "close"])


def _resample_ohlc(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    normalized = _normalize_ohlc(df)
    if normalized.empty:
        return normalized
    source = normalized.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    resampled = tf.resample_df(source, frequency, start="00h00min")
    resampled.columns = [str(col).lower() for col in resampled.columns]
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
    }
    return resampled.rename(columns=rename_map)


def _compute_atr(df: pd.DataFrame, lookback: int) -> pd.Series:
    normalized = _normalize_ohlc(df)
    if normalized.empty:
        return pd.Series(dtype=float)
    close_prev = normalized["close"].shift(1)
    true_range = pd.concat(
        [
            (normalized["high"] - normalized["low"]).abs(),
            (normalized["high"] - close_prev).abs(),
            (normalized["low"] - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(max(2, int(lookback)), min_periods=1).mean()


def _load_symbol_history(symbol: str, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    history_path = Path("data/historical") / f"historical_{str(symbol).upper()}.csv"
    if history_path.exists():
        df = pd.read_csv(history_path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce", format="mixed")
        df = df[~df.index.isna()].sort_index()
        return df
    if fallback is not None:
        fb = fallback.copy()
        if not isinstance(fb.index, pd.DatetimeIndex):
            fb.index = pd.to_datetime(fb.index, errors="coerce", format="mixed")
        fb = fb[~fb.index.isna()].sort_index()
        return fb
    return pd.DataFrame()


def _feature_inventory() -> list[str]:
    return ["ret", "mean", "std", "zscore", "atr"]


def _universe(symbol_specs=None) -> list[str]:
    if symbol_specs:
        return sorted({str(spec.get("symbol", "")).upper() for spec in symbol_specs if spec.get("symbol")})
    return []


def _asset_defaults(symbol: str) -> dict:
    params = dict(DEFAULT_ASSET_PARAMS)
    configured_stocks = _configured_stock_symbols()
    if symbol in CRYPTO_LONG_ONLY:
        params["allow_short"] = False
        params["target_vol"] = 0.20
    elif symbol in configured_stocks or symbol in DEFAULT_STOCK_SYMBOLS:
        params["allow_short"] = False
        params["target_vol"] = 0.18
    elif symbol == "MES":
        params["target_vol"] = 0.12
    elif symbol == "XAUUSD":
        params["target_vol"] = 0.14
    else:
        params["target_vol"] = 0.10
    return params


def _prepare_symbol_frame(history: pd.DataFrame, symbol: str, params: dict | None = None) -> pd.DataFrame:
    params = params or _asset_defaults(symbol)
    df = _normalize_ohlc(history)
    if df.empty:
        return pd.DataFrame()
    df = _resample_ohlc(df, get_asset_frequency(symbol))
    if df.empty:
        return pd.DataFrame()
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)
    lookback = max(5, int(params.get("lookback", 20)))
    rolling_mean = close.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    rolling_std = close.rolling(lookback, min_periods=max(5, lookback // 3)).std().replace(0.0, np.nan)
    zscore = ((close - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr = _compute_atr(df, min(lookback, 20)).bfill().ffill()
    out = df.copy()
    out["ret"] = ret
    out["mean"] = rolling_mean
    out["std"] = rolling_std.fillna(0.0)
    out["zscore"] = zscore
    out["atr"] = atr.fillna(0.0)
    return out.dropna(subset=["close"])


def prepare_base_df(historical_data, data_frequency, ticker, train_span=3500):
    params = _load_optimized_params().get("asset_params", {}).get(str(ticker).upper(), _asset_defaults(str(ticker).upper()))
    return _prepare_symbol_frame(historical_data, str(ticker).upper(), params).tail(int(train_span)).copy()


def _position_series(zscore: pd.Series, entry_z: float, exit_z: float, allow_short: bool) -> pd.Series:
    position = []
    current = 0.0
    for z in zscore.fillna(0.0):
        if current == 0.0:
            if z <= -entry_z:
                current = 1.0
            elif allow_short and z >= entry_z:
                current = -1.0
        elif current > 0:
            if z >= -exit_z:
                current = 0.0
        else:
            if z <= exit_z:
                current = 0.0
        position.append(current)
    return pd.Series(position, index=zscore.index, dtype=float)


def _strategy_returns(frame: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    position = _position_series(
        frame["zscore"],
        float(params.get("entry_z", 1.25)),
        float(params.get("exit_z", 0.25)),
        bool(params.get("allow_short", True)),
    )
    realized = position.shift(1).fillna(0.0) * frame["ret"].fillna(0.0)
    return position, realized


def _annualization_factor(freq: str) -> float:
    periods = max(1, tf.get_periods_per_day(freq))
    return float(periods * 252)


def _score_strategy(returns: pd.Series, freq: str) -> float:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty or np.isclose(float(returns.std(ddof=0)), 0.0):
        return -np.inf
    ann = _annualization_factor(freq)
    sharpe = np.sqrt(ann) * returns.mean() / returns.std(ddof=0)
    return float(sharpe)


def _split_train_validation(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    validation_size = max(20, int(len(frame) * float(validation_fraction)))
    validation_size = min(validation_size, max(1, len(frame) // 2))
    train = frame.iloc[:-validation_size].copy()
    validation = frame.iloc[-validation_size:].copy()
    if train.empty:
        train = frame.iloc[:-1].copy()
        validation = frame.iloc[-1:].copy()
    return train, validation


def _optimize_asset_params(symbol: str, frame: pd.DataFrame) -> tuple[dict, pd.Series]:
    defaults = _asset_defaults(symbol)
    if frame.empty or len(frame) < 50:
        prepared = _prepare_symbol_frame(frame, symbol, defaults)
        _, validation = _split_train_validation(prepared, DEFAULT_GLOBAL_PARAMS["validation_fraction"])
        _, validation_returns = _strategy_returns(validation, defaults)
        return defaults, validation_returns

    train, validation = _split_train_validation(frame, DEFAULT_GLOBAL_PARAMS["validation_fraction"])
    best_score = -np.inf
    best_params = dict(defaults)
    best_validation_returns = pd.Series(dtype=float)

    for lookback in DEFAULT_GLOBAL_PARAMS["lookback_grid"]:
        feature_frame = _prepare_symbol_frame(frame, symbol, {"lookback": lookback})
        if feature_frame.empty:
            continue
        train_feature, validation_feature = _split_train_validation(feature_frame, DEFAULT_GLOBAL_PARAMS["validation_fraction"])
        for entry_z in DEFAULT_GLOBAL_PARAMS["entry_z_grid"]:
            for exit_z in DEFAULT_GLOBAL_PARAMS["exit_z_grid"]:
                if exit_z >= entry_z:
                    continue
                for target_vol in DEFAULT_GLOBAL_PARAMS["target_vol_grid"]:
                    params = {
                        "lookback": int(lookback),
                        "entry_z": float(entry_z),
                        "exit_z": float(exit_z),
                        "target_vol": float(target_vol),
                        "allow_short": bool(defaults["allow_short"]),
                    }
                    _, train_returns = _strategy_returns(train_feature, params)
                    score = _score_strategy(train_returns, get_asset_frequency(symbol))
                    if score > best_score:
                        best_score = score
                        best_params = params
                        _, best_validation_returns = _strategy_returns(validation_feature, params)

    return best_params, best_validation_returns


def _markowitz_weights(validation_returns: Dict[str, pd.Series]) -> dict:
    if not validation_returns:
        return {}
    aligned = pd.DataFrame(validation_returns).dropna(how="all").fillna(0.0)
    if aligned.empty:
        symbols = sorted(validation_returns.keys())
        return {symbol: 1.0 / len(symbols) for symbol in symbols}

    mu = aligned.mean().astype(float)
    cov = aligned.cov().astype(float)
    ridge = float(DEFAULT_GLOBAL_PARAMS["covariance_ridge"])
    cov = cov + np.eye(len(cov)) * ridge

    try:
        inv_cov_mu = np.linalg.solve(cov.values, mu.values)
    except Exception:
        inv_cov_mu = np.ones(len(mu), dtype=float)

    raw = pd.Series(inv_cov_mu, index=mu.index, dtype=float)
    raw = raw.clip(lower=0.0)
    if np.isclose(float(raw.sum()), 0.0):
        raw = mu.clip(lower=0.0)
    if np.isclose(float(raw.sum()), 0.0):
        raw = pd.Series(1.0, index=mu.index, dtype=float)

    weights = raw / raw.sum()
    weights = weights.clip(
        lower=float(DEFAULT_GLOBAL_PARAMS["min_weight"]),
        upper=float(DEFAULT_GLOBAL_PARAMS["max_weight"]),
    )
    if np.isclose(float(weights.sum()), 0.0):
        weights = pd.Series(1.0 / len(weights), index=weights.index, dtype=float)
    else:
        weights = weights / weights.sum()
    return {str(symbol).upper(): float(weight) for symbol, weight in weights.items()}


def _strategy_config_payload(symbol_specs=None, asset_params=None, markowitz_weights=None, optimization_frequency=None, optimization_bucket=None, optimized_at=None):
    symbols = _universe(symbol_specs)
    payload = {
        "strategy": "ibkr-multi-asset-strategy2",
        "seed": int(DEFAULT_GLOBAL_PARAMS["seed"]),
        "symbols": symbols,
        "frequencies": {symbol: get_asset_frequency(symbol) for symbol in symbols},
        "train_spans": {symbol: int(get_asset_train_span(symbol)) for symbol in symbols},
        "feature_inventory": _feature_inventory(),
        "global_params": DEFAULT_GLOBAL_PARAMS,
        "asset_params": asset_params or {symbol: _asset_defaults(symbol) for symbol in symbols},
        "markowitz_weights": markowitz_weights or {},
        "optimization_frequency": str(optimization_frequency or "").strip().lower() or None,
        "optimization_bucket": str(optimization_bucket or "").strip() or None,
        "optimized_at": str(optimized_at) if optimized_at is not None else None,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    payload["config_hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return payload


def strategy_parameter_optimization(symbol_specs=None, optimization_frequency=None, optimization_bucket=None, optimized_at=None):
    symbols = _universe(symbol_specs)
    optimized_params = {}
    validation_returns = {}

    for symbol in symbols:
        history = _load_symbol_history(symbol, fallback=None)
        history = _normalize_ohlc(history).tail(int(get_asset_train_span(symbol))).copy()
        params, val_returns = _optimize_asset_params(symbol, history)
        optimized_params[symbol] = params
        validation_returns[symbol] = pd.Series(val_returns, dtype=float)

    weights = _markowitz_weights(validation_returns)
    payload = _strategy_config_payload(
        symbol_specs=symbol_specs,
        asset_params=optimized_params,
        markowitz_weights=weights,
        optimization_frequency=optimization_frequency,
        optimization_bucket=optimization_bucket,
        optimized_at=optimized_at,
    )
    _OPTIMIZATION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OPTIMIZATION_MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame({"final_features": _feature_inventory()}).to_excel(_OPTIMIZATION_FEATURES_PATH, index=False)
    return payload


def _load_optimized_params():
    if not _OPTIMIZATION_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(_OPTIMIZATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _require_optimized_payload(symbol_specs=None, payload=None):
    payload = payload if isinstance(payload, dict) else _load_optimized_params()
    symbols = _universe(symbol_specs)
    asset_params = payload.get("asset_params", {}) if isinstance(payload.get("asset_params", {}), dict) else {}
    markowitz_weights = payload.get("markowitz_weights", {}) if isinstance(payload.get("markowitz_weights", {}), dict) else {}

    if not symbols:
        return payload

    missing_asset_params = [symbol for symbol in symbols if symbol not in asset_params]
    missing_portfolio_weights = [symbol for symbol in symbols if symbol not in markowitz_weights]

    if missing_asset_params or missing_portfolio_weights:
        missing_parts = []
        if missing_asset_params:
            missing_parts.append(f"asset params for {missing_asset_params}")
        if missing_portfolio_weights:
            missing_parts.append(f"portfolio weights for {missing_portfolio_weights}")
        raise FileNotFoundError(
            "Missing optimized strategy state: "
            + ", ".join(missing_parts)
            + ". Run strategy_parameter_optimization() before trading."
        )

    return payload


def validate_strategy_optimization(symbol_specs=None, optimization_result=None, optimization_frequency=None, optimization_bucket=None):
    if not _OPTIMIZATION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Missing strategy2 optimization manifest. Run strategy_parameter_optimization() before trading."
        )

    stored_payload = _require_optimized_payload(symbol_specs=symbol_specs, payload=_load_optimized_params())
    requested_frequency = str(optimization_frequency or "").strip().lower() or None
    requested_bucket = str(optimization_bucket or "").strip() or None
    stored_frequency = str(stored_payload.get("optimization_frequency") or "").strip().lower() or None
    stored_bucket = str(stored_payload.get("optimization_bucket") or "").strip() or None

    if requested_frequency and stored_frequency != requested_frequency:
        raise ValueError(
            f"Optimized parameters were generated for frequency '{stored_frequency}' instead of '{requested_frequency}'. "
            "Re-run strategy_parameter_optimization() before trading."
        )
    if requested_bucket and stored_bucket != requested_bucket:
        raise ValueError(
            f"Optimized parameters were generated for bucket '{stored_bucket}' instead of '{requested_bucket}'. "
            "Re-run strategy_parameter_optimization() before trading."
        )

    expected_payload = optimization_result if isinstance(optimization_result, dict) else _strategy_config_payload(
        symbol_specs=symbol_specs,
        asset_params=stored_payload.get("asset_params"),
        markowitz_weights=stored_payload.get("markowitz_weights"),
        optimization_frequency=stored_frequency,
        optimization_bucket=stored_bucket,
        optimized_at=stored_payload.get("optimized_at"),
    )
    if stored_payload.get("config_hash") != expected_payload.get("config_hash"):
        raise ValueError(
            "Strategy2 optimization manifest is stale for the current multi-asset configuration. "
            "Re-run strategy_parameter_optimization() before trading."
        )
    return stored_payload


def _live_target(symbol: str, frame: pd.DataFrame, params: dict, portfolio_weight: float) -> dict:
    if frame.empty:
        return {"signal": 0, "leverage": 0.0, "stop_price": np.nan, "take_profit_price": np.nan}

    position, returns = _strategy_returns(frame, params)
    current_signal = int(np.sign(float(position.iloc[-1]))) if not position.empty else 0
    realized_vol = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
    annualizer = np.sqrt(_annualization_factor(get_asset_frequency(symbol)))
    realized_vol_annual = realized_vol * annualizer if realized_vol > 0 else np.nan
    target_vol = float(params.get("target_vol", 0.15))
    vol_scale = 1.0 if not np.isfinite(realized_vol_annual) or realized_vol_annual <= 0 else min(2.0, target_vol / realized_vol_annual)
    leverage = float(max(0.0, portfolio_weight * vol_scale))

    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns else max(close * 0.003, 1e-8)
    if current_signal > 0:
        stop_price = close - 1.5 * atr
        take_profit_price = close + 2.0 * atr
    elif current_signal < 0:
        stop_price = close + 1.5 * atr
        take_profit_price = close - 2.0 * atr
    else:
        stop_price = np.nan
        take_profit_price = np.nan

    return {
        "signal": current_signal,
        "leverage": leverage,
        "stop_price": float(stop_price) if np.isfinite(stop_price) else np.nan,
        "take_profit_price": float(take_profit_price) if np.isfinite(take_profit_price) else np.nan,
    }


def get_signal(app, fx_pairs=None, futures_symbols=None, metals_symbols=None, crypto_symbols=None, stock_symbols=None, leverage=None):
    universe = {
        "fx": _symbols_for_kind("fx", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "mes": _symbols_for_kind("mes", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "xau": _symbols_for_kind("xau", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "crypto": _symbols_for_kind("crypto", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "stocks": _symbols_for_kind("stocks", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
    }
    symbols = [s for group in universe.values() for s in group]
    manifest = _require_optimized_payload(
        symbol_specs=[{"symbol": symbol} for symbol in symbols],
        payload=_load_optimized_params(),
    )
    asset_params = manifest.get("asset_params", {})
    markowitz_weights = manifest.get("markowitz_weights", {})

    history_map: Dict[str, pd.DataFrame] = {}
    targets = {}
    strategy_targets = {}

    for symbol in symbols:
        fallback = app.historical_data if str(symbol).upper() == str(app.ticker).upper() else None
        history = _load_symbol_history(symbol, fallback=fallback)
        frame = _prepare_symbol_frame(history.tail(int(get_asset_train_span(symbol))).copy(), symbol, asset_params.get(symbol, _asset_defaults(symbol)))
        history_map[symbol] = frame
        live_target = _live_target(symbol, frame, asset_params.get(symbol, _asset_defaults(symbol)), float(markowitz_weights.get(symbol, 0.0)))
        targets[symbol] = {"signal": int(live_target["signal"]), "leverage": float(live_target["leverage"])}
        strategy_targets[symbol] = {
            "signal": int(live_target["signal"]),
            "leverage": float(live_target["leverage"]),
            "stop_price": live_target["stop_price"],
            "take_profit_price": live_target["take_profit_price"],
            "sleeve": "markowitz",
            "quantity_mode": None,
            "quantity_step": np.nan,
            "target_quantity": np.nan,
        }

    app.target_weights = {symbol: float(markowitz_weights.get(symbol, 0.0)) for symbol in symbols}
    app.applied_target_weights = app.target_weights.copy()
    app.margin_scale = 1.0
    app.required_capital_frac = float(sum(app.target_weights.values()))
    app.used_capital_frac = float(sum(app.target_weights.values()))
    app.cash_weight = float(max(0.0, 1.0 - sum(app.target_weights.values())))
    app.strategy_targets = strategy_targets

    state_updates = {
        "portfolio": {
            "target_weights": app.target_weights,
            "applied_weights": app.applied_target_weights,
            "margin_scale": 1.0,
            "required_capital_frac": app.required_capital_frac,
            "used_capital_frac": app.used_capital_frac,
            "cash_weight": app.cash_weight,
        },
        "targets": strategy_targets,
        "optimized_asset_params": asset_params,
    }
    app.strategy_state_updates = state_updates
    return {"targets": targets, "state_updates": state_updates}


def set_stop_loss_price(app):
    targets = getattr(app, "strategy_targets", {}) or {}
    symbol = str(getattr(app, "ticker", "")).upper()
    target = targets.get(symbol, {})
    price = float(target.get("stop_price", np.nan)) if target else np.nan
    if np.isfinite(price):
        return price
    history = _load_symbol_history(str(getattr(app, "ticker", "")).upper(), fallback=getattr(app, "historical_data", None))
    frame = _prepare_symbol_frame(history.tail(int(get_asset_train_span(getattr(app, "ticker", "")))).copy(), str(getattr(app, "ticker", "")).upper())
    if frame.empty:
        return np.nan
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    signal = int(np.sign(float(getattr(app, "signal", 0) or 0)))
    if signal > 0:
        return float(close - 1.5 * atr)
    if signal < 0:
        return float(close + 1.5 * atr)
    return np.nan


def set_take_profit_price(app):
    targets = getattr(app, "strategy_targets", {}) or {}
    symbol = str(getattr(app, "ticker", "")).upper()
    target = targets.get(symbol, {})
    price = float(target.get("take_profit_price", np.nan)) if target else np.nan
    if np.isfinite(price):
        return price
    history = _load_symbol_history(str(getattr(app, "ticker", "")).upper(), fallback=getattr(app, "historical_data", None))
    frame = _prepare_symbol_frame(history.tail(int(get_asset_train_span(getattr(app, "ticker", "")))).copy(), str(getattr(app, "ticker", "")).upper())
    if frame.empty:
        return np.nan
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    signal = int(np.sign(float(getattr(app, "signal", 0) or 0)))
    if signal > 0:
        return float(close + 2.0 * atr)
    if signal < 0:
        return float(close - 2.0 * atr)
    return np.nan
