from __future__ import annotations

from pathlib import Path
from typing import Dict
import hashlib
import json

import numpy as np
import pandas as pd

from ibkr_multi_asset import trading_functions as tf

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover
    linkage = None
    leaves_list = None
    squareform = None


DEFAULT_GLOBAL_PARAMS = {
    "seed": 20260505,
    "signal_method": "moving_average_trend",
    "portfolio_method": "hierarchical_risk_parity",
    "leverage_method": "kelly_capped",
    "validation_fraction": 0.25,
    "fast_window_grid": [10, 20, 30, 40, 60],
    "slow_window_grid": [80, 120, 160, 200, 260],
    "atr_window_grid": [10, 14, 20, 30],
    "stop_atr_multiple": 2.0,
    "take_profit_atr_multiple": 3.0,
    "covariance_ridge": 1e-8,
}

DEFAULT_ASSET_PARAMS = {
    "fast_window": 20,
    "slow_window": 120,
    "atr_window": 14,
    "allow_short": True,
}

CRYPTO_LONG_ONLY = {"BTC", "ETH", "SOL", "LTC", "BCH"}
_OPTIMIZATION_MANIFEST_PATH = Path("data/models/strategy_optimization_manifest.json")
_OPTIMIZATION_FEATURES_PATH = Path("data/models/optimal_features_df.xlsx")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_USER_CONFIG_ROOT = Path(__file__).resolve().parents[1]


def _main_variables() -> dict:
    try:
        return tf.extract_variables(str(Path(__file__).resolve().parents[1] / "main.py"))
    except Exception:
        return {}


def _strategy_frequency() -> str:
    variables = _main_variables()
    return str(variables.get("strategy_frequency", "5min")).strip()


def _strategy_train_span() -> int:
    variables = _main_variables()
    raw = variables.get("strategy_optimization_lookback", 3000)
    try:
        return max(300, int(raw))
    except Exception:
        return 3000


def _fixed_max_leverage() -> float:
    variables = _main_variables()
    raw = variables.get("fixed_max_leverage", 3.0)
    try:
        return max(0.0, float(raw))
    except Exception:
        return 3.0


def _configured_long_only_symbols() -> set[str]:
    variables = _main_variables()
    configured = variables.get("long_only_symbols", []) or []
    return {str(symbol).upper() for symbol in configured}


def _configured_optimization_frequency() -> str:
    variables = _main_variables()
    return str(variables.get("optimization_frequency", "daily")).strip().lower()


def get_asset_runtime_policy(symbol, asset_class=None) -> dict:
    symbol = str(symbol or "").upper()
    asset_class = str(asset_class or "").lower()
    if asset_class == "crypto" or symbol in CRYPTO_LONG_ONLY:
        return {
            "session": "24_7",
            "flatten_at_day_end": False,
            "daily_maintenance_utc_start": "00:00",
            "daily_maintenance_minutes": 15,
        }
    return {
        "session": "weekdays",
        "flatten_at_day_end": False,
        "daily_maintenance_utc_start": "00:00",
        "daily_maintenance_minutes": 15,
    }


def get_asset_frequency(symbol) -> str:
    return _strategy_frequency()


def get_asset_train_span(symbol) -> int:
    return _strategy_train_span()


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
            out.index = pd.to_datetime(out.index, errors="coerce", format="mixed")
    out = out[~out.index.isna()].sort_index()

    title_cols = {str(col): col for col in out.columns}
    lower_cols = {str(col).lower(): col for col in out.columns}

    def _candidate_frame(mapping):
        candidate = pd.DataFrame(
            {target: pd.to_numeric(out[source], errors="coerce") for target, source in mapping.items()},
            index=out.index,
        ).dropna()
        return candidate.astype(float)

    if {"Open", "High", "Low", "Close"}.issubset(title_cols):
        candidate = _candidate_frame(
            {"open": title_cols["Open"], "high": title_cols["High"], "low": title_cols["Low"], "close": title_cols["Close"]}
        )
        if not candidate.empty:
            return candidate

    if {"open", "high", "low", "close"}.issubset(lower_cols):
        candidate = _candidate_frame(
            {"open": lower_cols["open"], "high": lower_cols["high"], "low": lower_cols["low"], "close": lower_cols["close"]}
        )
        if not candidate.empty:
            return candidate

    midpoint_cols = {"bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"}
    if midpoint_cols.issubset(lower_cols):
        raw = out.rename(columns={lower_cols[key]: key for key in lower_cols})
        midpoint = tf.get_mid_series(raw)
        midpoint.columns = [str(col).lower() for col in midpoint.columns]
        return midpoint[["open", "high", "low", "close"]].astype(float).dropna()

    return pd.DataFrame(columns=["open", "high", "low", "close"])


def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    unnamed = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        out = out.drop(columns=unnamed)

    if not isinstance(out.index, pd.DatetimeIndex):
        datetime_column = None
        for candidate in ["datetime", "Datetime", "date", "Date", "time", "Time"]:
            if candidate in out.columns:
                datetime_column = candidate
                break
        if datetime_column is not None:
            parsed = pd.to_datetime(out[datetime_column], errors="coerce", format="mixed")
            out = out.loc[parsed.notna()].copy()
            out.index = parsed.loc[parsed.notna()]
            if datetime_column in out.columns:
                out = out.drop(columns=[datetime_column])
        else:
            parsed_index = pd.to_datetime(pd.Series(out.index), errors="coerce", format="mixed")
            if parsed_index.notna().any():
                out.index = parsed_index
            elif len(out.columns) > 0:
                parsed_first = pd.to_datetime(out.iloc[:, 0], errors="coerce", format="mixed")
                if parsed_first.notna().any():
                    out = out.loc[parsed_first.notna()].copy()
                    out.index = parsed_first.loc[parsed_first.notna()]
                    out = out.iloc[:, 1:] if len(out.columns) > 1 else pd.DataFrame(index=out.index)

    if not isinstance(out.index, pd.DatetimeIndex):
        return pd.DataFrame()

    out = out[~out.index.isna()].sort_index()
    return out


def _resample_ohlc(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    normalized = _normalize_ohlc(df)
    if normalized.empty:
        return normalized
    source = normalized.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    resampled = tf.resample_df(source, frequency, start="00h00min")
    resampled.columns = [str(col).lower() for col in resampled.columns]
    return resampled[["open", "high", "low", "close"]].astype(float).dropna()


def _compute_atr(frame: pd.DataFrame, window: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    close_prev = frame["close"].shift(1)
    true_range = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - close_prev).abs(),
            (frame["low"] - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(max(2, int(window)), min_periods=1).mean()


def _load_symbol_history(symbol: str, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    filename = f"historical_{str(symbol).upper()}.csv"
    candidate_paths = [
        Path.cwd() / "data" / "historical" / filename,
        Path.cwd() / "user_config" / "data" / "historical" / filename,
        _PROJECT_ROOT / "data" / "historical" / filename,
        _USER_CONFIG_ROOT / "data" / "historical" / filename,
    ]
    for history_path in candidate_paths:
        if history_path.exists():
            df = pd.read_csv(history_path, index_col=0)
            normalized = _normalize_history_frame(df)
            if not normalized.empty:
                return normalized
    if fallback is not None:
        normalized = _normalize_history_frame(fallback.copy())
        if not normalized.empty:
            return normalized
    return pd.DataFrame()


def _asset_defaults(symbol: str) -> dict:
    params = dict(DEFAULT_ASSET_PARAMS)
    params["allow_short"] = str(symbol).upper() not in (_configured_long_only_symbols() | CRYPTO_LONG_ONLY)
    return params


def _prepare_symbol_frame(history: pd.DataFrame, symbol: str, params: dict | None = None) -> pd.DataFrame:
    params = params or _asset_defaults(symbol)
    frame = _resample_ohlc(history, get_asset_frequency(symbol))
    if frame.empty:
        return frame

    fast_window = max(2, int(params.get("fast_window", 20)))
    slow_window = max(fast_window + 1, int(params.get("slow_window", 120)))
    atr_window = max(2, int(params.get("atr_window", 14)))

    out = frame.copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["fast_ma"] = out["close"].rolling(fast_window, min_periods=fast_window).mean()
    out["slow_ma"] = out["close"].rolling(slow_window, min_periods=slow_window).mean()
    out["trend_spread"] = (out["fast_ma"] / out["slow_ma"] - 1.0).replace([np.inf, -np.inf], np.nan)
    out["atr"] = _compute_atr(out, atr_window)
    out["realized_vol"] = out["ret"].rolling(slow_window, min_periods=max(10, slow_window // 3)).std().fillna(0.0)
    return out.dropna(subset=["fast_ma", "slow_ma", "close"])


def prepare_base_df(historical_data, data_frequency, ticker, train_span=3500):
    payload = _load_optimized_params()
    params = payload.get("asset_params", {}).get(str(ticker).upper(), _asset_defaults(str(ticker).upper()))
    return _prepare_symbol_frame(historical_data, str(ticker).upper(), params).tail(int(train_span)).copy()


def _trend_position_series(frame: pd.DataFrame, params: dict) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    allow_short = bool(params.get("allow_short", True))
    spread = pd.to_numeric(frame.get("trend_spread", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0.0)
    signal = pd.Series(0.0, index=frame.index, dtype=float)
    signal.loc[spread > 0] = 1.0
    if allow_short:
        signal.loc[spread < 0] = -1.0
    return signal


def _strategy_returns(frame: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    position = _trend_position_series(frame, params)
    realized = position.shift(1).fillna(0.0) * frame["ret"].fillna(0.0)
    return position, realized


def _annualization_factor(freq: str) -> float:
    return float(max(1, tf.get_periods_per_day(freq)) * 252)


def _score_strategy(returns: pd.Series, freq: str) -> float:
    series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if series.empty or np.isclose(float(series.std(ddof=0)), 0.0):
        return -np.inf
    ann = _annualization_factor(freq)
    return float(np.sqrt(ann) * series.mean() / series.std(ddof=0))


def _split_train_validation(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    validation_size = max(50, int(len(frame) * float(validation_fraction)))
    validation_size = min(validation_size, max(1, len(frame) // 2))
    train = frame.iloc[:-validation_size].copy()
    validation = frame.iloc[-validation_size:].copy()
    if train.empty:
        train = frame.iloc[:-1].copy()
        validation = frame.iloc[-1:].copy()
    return train, validation


def _optimize_asset_params(symbol: str, history: pd.DataFrame) -> tuple[dict, pd.Series]:
    defaults = _asset_defaults(symbol)
    best_params = dict(defaults)
    best_validation_returns = pd.Series(dtype=float)
    best_score = -np.inf

    for fast_window in DEFAULT_GLOBAL_PARAMS["fast_window_grid"]:
        for slow_window in DEFAULT_GLOBAL_PARAMS["slow_window_grid"]:
            if int(slow_window) <= int(fast_window):
                continue
            for atr_window in DEFAULT_GLOBAL_PARAMS["atr_window_grid"]:
                params = {
                    "fast_window": int(fast_window),
                    "slow_window": int(slow_window),
                    "atr_window": int(atr_window),
                    "allow_short": bool(defaults["allow_short"]),
                }
                frame = _prepare_symbol_frame(history, symbol, params)
                if frame.empty or len(frame) < 100:
                    continue
                _, validation = _split_train_validation(frame, DEFAULT_GLOBAL_PARAMS["validation_fraction"])
                _, validation_returns = _strategy_returns(validation, params)
                score = _score_strategy(validation_returns, get_asset_frequency(symbol))
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_validation_returns = validation_returns

    if best_validation_returns.empty:
        frame = _prepare_symbol_frame(history, symbol, best_params)
        _, validation = _split_train_validation(frame, DEFAULT_GLOBAL_PARAMS["validation_fraction"])
        _, best_validation_returns = _strategy_returns(validation, best_params)

    return best_params, pd.Series(best_validation_returns, dtype=float)


def _inverse_variance_weights(cov: pd.DataFrame) -> pd.Series:
    diagonal = np.diag(cov.values).astype(float)
    diagonal = np.where(diagonal <= 0, np.nan, diagonal)
    inv_diag = 1.0 / diagonal
    inv_diag = np.where(np.isfinite(inv_diag), inv_diag, 0.0)
    weights = pd.Series(inv_diag, index=cov.index, dtype=float)
    if np.isclose(float(weights.sum()), 0.0):
        weights[:] = 1.0
    return weights / weights.sum()


def _cluster_variance(cov: pd.DataFrame, cluster_items: list[str]) -> float:
    subcov = cov.loc[cluster_items, cluster_items]
    weights = _inverse_variance_weights(subcov).values.reshape(-1, 1)
    variance = float((weights.T @ subcov.values @ weights).item())
    return max(variance, 1e-12)


def _hrp_weights(returns_map: Dict[str, pd.Series]) -> dict:
    if not returns_map:
        return {}
    aligned = pd.DataFrame(returns_map).dropna(how="all").fillna(0.0)
    if aligned.empty:
        return {}

    cov = aligned.cov().astype(float)
    cov = cov + np.eye(len(cov)) * float(DEFAULT_GLOBAL_PARAMS["covariance_ridge"])
    corr = aligned.corr().fillna(0.0)

    if linkage is None or leaves_list is None or squareform is None or len(cov.columns) <= 1:
        weights = _inverse_variance_weights(cov)
        return {str(symbol).upper(): float(weight) for symbol, weight in weights.items()}

    distance = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="single")
    ordered = corr.index[leaves_list(linkage_matrix)].tolist()

    weights = pd.Series(1.0, index=ordered, dtype=float)
    clusters = [ordered]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        left_var = _cluster_variance(cov, left)
        right_var = _cluster_variance(cov, right)
        alpha = 1.0 - left_var / (left_var + right_var)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])

    weights = weights / weights.sum()
    return {str(symbol).upper(): float(weight) for symbol, weight in weights.items()}


def _kelly_leverage(portfolio_returns: pd.Series) -> float:
    returns = pd.to_numeric(pd.Series(portfolio_returns), errors="coerce").dropna()
    if len(returns) < 20:
        return 1.0
    mean_return = float(returns.mean())
    variance = float(returns.var(ddof=0))
    if variance <= 0 or not np.isfinite(variance):
        return 1.0
    raw_kelly = max(0.0, mean_return / variance)
    return float(min(_fixed_max_leverage(), raw_kelly))


def _portfolio_validation_returns(weights: dict, validation_returns: Dict[str, pd.Series]) -> pd.Series:
    if not weights or not validation_returns:
        return pd.Series(dtype=float)
    aligned = pd.DataFrame(validation_returns).dropna(how="all").fillna(0.0)
    if aligned.empty:
        return pd.Series(dtype=float)
    vector = pd.Series(weights, dtype=float).reindex(aligned.columns).fillna(0.0)
    return aligned.mul(vector, axis=1).sum(axis=1)


def _strategy_config_payload(symbol_specs=None, asset_params=None, hrp_weights=None, portfolio_leverage_multiplier=1.0, optimization_frequency=None, optimization_bucket=None, optimized_at=None):
    symbols = sorted({str(spec.get("symbol", "")).upper() for spec in (symbol_specs or []) if spec.get("symbol")})
    payload = {
        "strategy": "ibkr-multi-asset-trend-hrp",
        "seed": int(DEFAULT_GLOBAL_PARAMS["seed"]),
        "signal_method": DEFAULT_GLOBAL_PARAMS["signal_method"],
        "portfolio_method": DEFAULT_GLOBAL_PARAMS["portfolio_method"],
        "leverage_method": DEFAULT_GLOBAL_PARAMS["leverage_method"],
        "frequency": _strategy_frequency(),
        "train_span": int(_strategy_train_span()),
        "fixed_max_leverage": float(_fixed_max_leverage()),
        "symbols": symbols,
        "asset_params": asset_params or {symbol: _asset_defaults(symbol) for symbol in symbols},
        "hrp_weights": hrp_weights or {},
        "portfolio_leverage_multiplier": float(portfolio_leverage_multiplier),
        "optimization_frequency": str(optimization_frequency or "").strip().lower() or None,
        "optimization_bucket": str(optimization_bucket or "").strip() or None,
        "optimized_at": str(optimized_at) if optimized_at is not None else None,
        "global_params": DEFAULT_GLOBAL_PARAMS,
        "feature_inventory": ["ret", "fast_ma", "slow_ma", "trend_spread", "atr", "realized_vol"],
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    payload["config_hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return payload


def strategy_parameter_optimization(symbol_specs=None, optimization_frequency=None, optimization_bucket=None, optimized_at=None):
    symbols = sorted({str(spec.get("symbol", "")).upper() for spec in (symbol_specs or []) if spec.get("symbol")})
    cutoff = pd.Timestamp(optimization_bucket) if optimization_bucket is not None else None
    asset_params = {}
    validation_returns = {}

    for symbol in symbols:
        history = _normalize_history_frame(_load_symbol_history(symbol))
        if history.empty:
            asset_params[symbol] = _asset_defaults(symbol)
            validation_returns[symbol] = pd.Series(dtype=float)
            continue
        if cutoff is not None:
            history = history[history.index < cutoff].copy()
        if history.empty:
            asset_params[symbol] = _asset_defaults(symbol)
            validation_returns[symbol] = pd.Series(dtype=float)
            continue
        history = history.tail(int(_strategy_train_span()) + 500).copy()
        params, returns = _optimize_asset_params(symbol, history)
        asset_params[symbol] = params
        validation_returns[symbol] = returns

    weights = _hrp_weights(validation_returns)
    portfolio_returns = _portfolio_validation_returns(weights, validation_returns)
    portfolio_leverage_multiplier = _kelly_leverage(portfolio_returns)
    payload = _strategy_config_payload(
        symbol_specs=symbol_specs,
        asset_params=asset_params,
        hrp_weights=weights,
        portfolio_leverage_multiplier=portfolio_leverage_multiplier,
        optimization_frequency=optimization_frequency,
        optimization_bucket=optimization_bucket,
        optimized_at=optimized_at,
    )
    _OPTIMIZATION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OPTIMIZATION_MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame({"final_features": payload["feature_inventory"]}).to_excel(_OPTIMIZATION_FEATURES_PATH)
    return payload


def _load_optimized_params():
    if not _OPTIMIZATION_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(_OPTIMIZATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_strategy_optimization(symbol_specs=None, optimization_result=None, optimization_frequency=None, optimization_bucket=None):
    payload = optimization_result if isinstance(optimization_result, dict) else _load_optimized_params()
    if not payload:
        raise FileNotFoundError("Missing optimized strategy manifest. Run strategy_parameter_optimization() before trading.")

    symbols = sorted({str(spec.get("symbol", "")).upper() for spec in (symbol_specs or []) if spec.get("symbol")})
    asset_params = payload.get("asset_params", {}) if isinstance(payload.get("asset_params", {}), dict) else {}
    hrp_weights = payload.get("hrp_weights", {}) if isinstance(payload.get("hrp_weights", {}), dict) else {}

    missing_params = [symbol for symbol in symbols if symbol not in asset_params]
    missing_weights = [symbol for symbol in symbols if symbol not in hrp_weights]
    if missing_params or missing_weights:
        raise ValueError(
            "Optimized strategy manifest is incomplete. Missing "
            f"params={missing_params} weights={missing_weights}. Re-run strategy_parameter_optimization()."
        )

    expected_frequency = str(optimization_frequency or "").strip().lower() or None
    if expected_frequency and str(payload.get("optimization_frequency") or "").strip().lower() != expected_frequency:
        raise ValueError("Optimized strategy manifest frequency does not match the requested optimization frequency.")

    expected_bucket = str(optimization_bucket or "").strip() or None
    if expected_bucket and str(payload.get("optimization_bucket") or "").strip() != expected_bucket:
        raise ValueError("Optimized strategy manifest bucket does not match the current trading day bucket.")

    expected_payload = _strategy_config_payload(
        symbol_specs=symbol_specs,
        asset_params=asset_params,
        hrp_weights=hrp_weights,
        portfolio_leverage_multiplier=float(payload.get("portfolio_leverage_multiplier", 1.0)),
        optimization_frequency=payload.get("optimization_frequency"),
        optimization_bucket=payload.get("optimization_bucket"),
        optimized_at=payload.get("optimized_at"),
    )
    if payload.get("config_hash") != expected_payload.get("config_hash"):
        raise ValueError("Optimized strategy manifest is stale for the current strategy configuration.")
    return payload


def _live_target(symbol: str, frame: pd.DataFrame, params: dict, portfolio_weight: float) -> dict:
    if frame.empty:
        return {"signal": 0, "target_weight": 0.0, "stop_price": np.nan, "take_profit_price": np.nan}

    position = _trend_position_series(frame, params)
    signal = int(np.sign(float(position.iloc[-1]))) if not position.empty else 0
    if signal == 0:
        weight = 0.0
    else:
        weight = max(0.0, float(portfolio_weight))

    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    stop_mult = float(DEFAULT_GLOBAL_PARAMS["stop_atr_multiple"])
    tp_mult = float(DEFAULT_GLOBAL_PARAMS["take_profit_atr_multiple"])

    if signal > 0:
        stop_price = close - stop_mult * atr
        take_profit_price = close + tp_mult * atr
    elif signal < 0:
        stop_price = close + stop_mult * atr
        take_profit_price = close - tp_mult * atr
    else:
        stop_price = np.nan
        take_profit_price = np.nan

    return {
        "signal": signal,
        "target_weight": float(weight),
        "stop_price": float(stop_price) if np.isfinite(stop_price) else np.nan,
        "take_profit_price": float(take_profit_price) if np.isfinite(take_profit_price) else np.nan,
    }


def get_signal(app, fx_pairs=None, futures_symbols=None, metals_symbols=None, crypto_symbols=None, stock_symbols=None, leverage=None):
    universe = {
        "fx": _symbols_for_kind("fx", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "futures": _symbols_for_kind("futures", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "metals": _symbols_for_kind("metals", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "crypto": _symbols_for_kind("crypto", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
        "stocks": _symbols_for_kind("stocks", fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols),
    }
    symbols = [symbol for group in universe.values() for symbol in group]
    symbol_specs = [{"symbol": symbol} for symbol in symbols]
    optimization_frequency = str(getattr(app, "optimization_frequency", "") or _configured_optimization_frequency()).strip().lower()
    optimization_bucket = str(getattr(app, "optimization_bucket", "") or "").strip() or None
    payload = _load_optimized_params()
    if payload:
        try:
            payload = validate_strategy_optimization(
                symbol_specs=symbol_specs,
                optimization_result=payload,
                optimization_frequency=optimization_frequency,
                optimization_bucket=optimization_bucket,
            )
        except Exception:
            payload = strategy_parameter_optimization(
                symbol_specs=symbol_specs,
                optimization_frequency=optimization_frequency,
                optimization_bucket=optimization_bucket,
                optimized_at=pd.Timestamp.utcnow().tz_localize(None).replace(microsecond=0).isoformat(sep=" "),
            )
    else:
        payload = strategy_parameter_optimization(
            symbol_specs=symbol_specs,
            optimization_frequency=optimization_frequency,
            optimization_bucket=optimization_bucket,
            optimized_at=pd.Timestamp.utcnow().tz_localize(None).replace(microsecond=0).isoformat(sep=" "),
        )

    asset_params = payload.get("asset_params", {})
    hrp_weights = payload.get("hrp_weights", {})
    portfolio_leverage_multiplier = float(payload.get("portfolio_leverage_multiplier", leverage or 1.0))

    targets = {}
    strategy_targets = {}
    for symbol in symbols:
        fallback = app.historical_data if str(symbol).upper() == str(getattr(app, "ticker", "")).upper() else None
        history = _load_symbol_history(symbol, fallback=fallback).tail(int(_strategy_train_span()) + 100).copy()
        params = asset_params.get(symbol, _asset_defaults(symbol))
        frame = _prepare_symbol_frame(history, symbol, params)
        live_target = _live_target(symbol, frame, params, float(hrp_weights.get(symbol, 0.0)))
        targets[symbol] = {"signal": int(live_target["signal"])}
        strategy_targets[symbol] = {
            "signal": int(live_target["signal"]),
            "target_weight": float(live_target["target_weight"]),
            "stop_price": live_target["stop_price"],
            "take_profit_price": live_target["take_profit_price"],
            "sleeve": "hrp_trend",
            "quantity_mode": None,
            "quantity_step": np.nan,
            "target_quantity": np.nan,
        }

    app.leverage = float(min(_fixed_max_leverage(), max(0.0, portfolio_leverage_multiplier)))
    app.target_weights = {symbol: float(strategy_targets[symbol]["target_weight"]) for symbol in symbols}
    app.applied_target_weights = app.target_weights.copy()
    app.margin_scale = float(app.leverage / max(_fixed_max_leverage(), 1e-8)) if _fixed_max_leverage() > 0 else 0.0
    app.required_capital_frac = float(sum(app.target_weights.values()))
    app.used_capital_frac = float(min(app.leverage, _fixed_max_leverage()) * sum(app.target_weights.values()))
    app.cash_weight = float(max(0.0, 1.0 - sum(app.target_weights.values())))
    app.strategy_targets = strategy_targets

    state_updates = {
        "portfolio": {
            "target_weights": app.target_weights,
            "applied_weights": app.applied_target_weights,
            "portfolio_leverage_multiplier": float(app.leverage),
            "fixed_max_leverage": float(_fixed_max_leverage()),
            "margin_scale": float(app.margin_scale),
            "required_capital_frac": float(app.required_capital_frac),
            "used_capital_frac": float(app.used_capital_frac),
            "cash_weight": float(app.cash_weight),
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
    history = _load_symbol_history(symbol, fallback=getattr(app, "historical_data", None))
    frame = _prepare_symbol_frame(history.tail(int(_strategy_train_span())).copy(), symbol, _asset_defaults(symbol))
    if frame.empty:
        return np.nan
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    signal = int(np.sign(float(getattr(app, "signal", 0.0) or 0.0)))
    if signal > 0:
        return float(close - DEFAULT_GLOBAL_PARAMS["stop_atr_multiple"] * atr)
    if signal < 0:
        return float(close + DEFAULT_GLOBAL_PARAMS["stop_atr_multiple"] * atr)
    return np.nan


def set_take_profit_price(app):
    targets = getattr(app, "strategy_targets", {}) or {}
    symbol = str(getattr(app, "ticker", "")).upper()
    target = targets.get(symbol, {})
    price = float(target.get("take_profit_price", np.nan)) if target else np.nan
    if np.isfinite(price):
        return price
    history = _load_symbol_history(symbol, fallback=getattr(app, "historical_data", None))
    frame = _prepare_symbol_frame(history.tail(int(_strategy_train_span())).copy(), symbol, _asset_defaults(symbol))
    if frame.empty:
        return np.nan
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    signal = int(np.sign(float(getattr(app, "signal", 0.0) or 0.0)))
    if signal > 0:
        return float(close + DEFAULT_GLOBAL_PARAMS["take_profit_atr_multiple"] * atr)
    if signal < 0:
        return float(close - DEFAULT_GLOBAL_PARAMS["take_profit_atr_multiple"] * atr)
    return np.nan
