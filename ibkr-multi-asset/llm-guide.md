# LLM Strategy Guide — IBKR Multi-Asset Setup

> Copy this entire document and paste it into an LLM (Claude, GPT-4, DeepSeek, etc.).
> The LLM will generate a complete `my_strategy.py` file for the IBKR multi-asset live trading setup.

---

## Task

Create a Python file named `my_strategy.py` that plugs into the IBKR Multi-Asset Trading Setup at:

```
QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/
```

The file must implement the strategy interface defined in the reference file `strategy.py` located in the same folder. You may change the strategy's internal logic (signals, portfolio weighting, optimization) freely, but you MUST keep the function signatures and return types identical to `strategy.py`.

After creating the file, the user activates it by editing `main.py`:

```python
strategy_file = "strategies/my_strategy.py"
```

---

## Setup architecture — how the files connect

```
main.py                          strategy.py (your file)
────────                         ──────────────────────
fx_pairs = ["EURUSD"]       ──→  get_signal(app, fx_pairs, ...)
futures_symbols = ["MES"]        set_stop_loss_price(app)
crypto_symbols = ["ETH"]         set_take_profit_price(app)
long_only_symbols = ["ETH"]      strategy_parameter_optimization(...)
strategy_frequency = "5min"      validate_strategy_optimization(...)
strategy_optimization_lookback   get_asset_frequency(symbol)
fixed_max_leverage = 2.0         get_asset_train_span(symbol)
strategy_file = "strategies/     get_asset_runtime_policy(symbol)
                my_strategy.py"  prepare_base_df(data, freq, ticker)
                                      │
                                      ▼
                               engine.py (DO NOT EDIT)
                               ─────────────────────
                               Calls your functions every bar
                               Manages IBKR connection, orders,
                               position tracking, risk checks
```

**What you edit:** `main.py` (universe, frequency) and `my_strategy.py` (signals, weights, optimization).

**What you never touch:** `src/` (engine, IB functions, trading functions) — these are the setup's core.

**Data flow per bar:**
1. Engine fetches 5-min OHLC bars from IBKR for each symbol
2. Engine calls `get_signal()` → you return per-symbol signals (+1/0/-1) and portfolio weights
3. Engine calls `set_stop_loss_price()` per symbol → you return a price
4. Engine calls `set_take_profit_price()` per symbol → you return a price
5. Engine sizes orders using your weights × equity, submits to IBKR
6. Engine monitors stops (including synthetic stops for crypto)

**Data flow on optimization day/week:**
1. Engine calls `validate_strategy_optimization()` → you check if manifest is current
2. If stale, engine calls `strategy_parameter_optimization()` → you run grid search, save manifest
3. Engine uses optimized params from manifest for signal generation

---

## Two ways to create your strategy file

### Path A: Convert from a backtest script

If you have a backtesting script that already implements your strategy logic, share it alongside this guide with your LLM. Tell the LLM:

> "Here is my backtesting script and the LLM strategy guide. Convert my strategy logic from the backtest into `my_strategy.py` following the guide's function signatures. Map my signal generation to `_trend_position_series`, my portfolio construction to the weighting function, and my parameter optimization to `strategy_parameter_optimization`."

**What the LLM will extract from your backtest:**

| Backtest concept | Maps to strategy.py function |
|---|---|
| Per-asset signal indicator (MA crossover, Bollinger, RSI) | `_trend_position_series(frame, params)` |
| Feature computation (ATR, z-score, regime detection) | `_prepare_symbol_frame(history, symbol, params)` |
| Parameter grid (window sizes, thresholds) | `DEFAULT_GLOBAL_PARAMS` grids |
| Per-asset optimized parameters | `DEFAULT_ASSET_PARAMS` |
| Portfolio weighting (HRP, inverse-vol, equal-weight) | Your weighting function inside `get_signal()` |
| Leverage method (Kelly, fixed, target-vol) | Kelly/leverage calculation inside `get_signal()` |
| Stop-loss / take-profit rules | `set_stop_loss_price(app)` / `set_take_profit_price(app)` |
| Long-only rules (crypto) | `_trend_position_series` → clip to ≥ 0 |
| IB constraints (whole-units, fractional) | `strategy_targets` → `quantity_step`, `fractional` |

**Common backtest-to-live adaptations:**
- Replace vectorized pandas operations with bar-by-bar `.iloc[-1]` lookups for live signals
- Replace `pd.Timestamp` date slicing with `app.historical_data.tail(N)` lookback windows
- Replace backtest commission models with IBKR's actual fees (already handled by engine)
- Replace backtest portfolio curve construction with per-symbol `target_weight` allocation
- Add `allow_short` checks for crypto from `long_only_symbols` in `main.py`

### Path B: Create from scratch with the LLM guide

Copy this entire guide into an LLM and describe your strategy in plain language:

> "I want a trend-following strategy with 20/120 MA crossover on 5-minute bars for FX and futures. Crypto is long-only with 20/80 EMA. Portfolio weights are inverse-volatility with a 30% crypto cap. Use Kelly leverage. Optimize MA windows daily from [10,20,30] fast and [80,120,160] slow."

The LLM will generate a complete `my_strategy.py` using all the boilerplate code in this guide, customized with your signal and weighting logic.

---

## Questions the LLM MUST ask before generating code

Whether the student provides a backtest script (Path A) or a plain-language description (Path B), the LLM must verify that every item below is covered. If the student's input is missing any item, the LLM must either ask the student directly or fill the gap with a safe default and flag it with a `# NOTE` comment in the generated code.

### 1. Signal generation

| Question | Safe default if missing |
|---|---|
| What indicator generates entry/exit signals? (MA crossover, Bollinger, RSI, MACD, custom) | 20/120 MA crossover on `trend_spread` |
| What bar frequency is used? (1min, 5min, 15min, 1hour, 1day) | Read from `main.py` → `strategy_frequency` |
| Are there different signal rules per asset class? (FX vs futures vs crypto vs metals vs stocks) | Same MA crossover for all |
| Are short positions allowed? If so, which assets? | Shorts allowed for FX/futures/metals. Crypto/stocks: check `long_only_symbols` from `main.py` |

### 2. Risk management

| Question | Safe default if missing |
|---|---|
| Is there a stop-loss rule? (ATR-based, fixed-pip, percentage, trailing) | ATR-based: `stop_atr_multiple = 2.0` |
| Is there a take-profit rule? (ATR-based, fixed-pip, risk-reward ratio) | ATR-based: `take_profit_atr_multiple = 3.0` |
| Is there a maximum leverage limit? | `fixed_max_leverage` from `main.py` (default 1.0) |
| Is there a daily loss limit or max drawdown circuit breaker? | Not implemented (flag with `# NOTE: no daily loss limit` comment) |
| Is there a per-symbol maximum position size? | No cap (flag with `# NOTE: no per-symbol position cap`) |
| Is there a crypto concentration cap? | 30% of portfolio (IBKR observed limit) |
| Are there correlation-based position reductions? | Not implemented |

### 3. IBKR broker constraints

| Question | Safe default if missing |
|---|---|
| Which assets trade whole-units only (no fractional sizing)? | MES, USDJPY, XAUUSD: `quantity_step = 1.0, fractional = False`. EURUSD: `quantity_step = 0.01`. Crypto: `quantity_step = 1e-8` |
| Which assets are long-only? | Read `long_only_symbols` from `main.py` + `CRYPTO_LONG_ONLY` set |
| Which assets have no native stop orders (need synthetic monitoring)? | PAXOS crypto (BTC, ETH): engine handles synthetic stops automatically |
| Are there venue-specific trading hours? (crypto = 24/7, others = weekdays) | `get_asset_runtime_policy` — crypto returns `"24_7"`, others `"weekdays"` |
| Is there a front-month futures roll? | Engine handles `AUTO_FRONT_MONTH` from `main.py` |
| What are the IBKR commission/fee structures? | Engine handles fees automatically. Strategy doesn't need to model them |

### 4. Portfolio construction

| Question | Safe default if missing |
|---|---|
| How are portfolio weights computed? (equal-weight, inverse-vol, HRP, risk-parity, custom) | Inverse-volatility weighting |
| Is there a minimum or floor weight for any asset? (e.g., XAU floor at 10%) | No floor |
| Is there a maximum weight per asset? | No cap |
| How is portfolio leverage determined? (fixed, Kelly, target-vol) | Kelly criterion capped at `fixed_max_leverage` |
| What is the rebalance frequency? | Every bar (engine's natural cycle) |

### 5. Parameter optimization

| Question | Safe default if missing |
|---|---|
| What parameters are optimized? (MA windows, ATR windows, entry thresholds) | `fast_window_grid: [10,20,30,40]`, `slow_window_grid: [80,120,160,200]`, `atr_window_grid: [10,14,20]` |
| What is the optimization frequency? (daily, weekly) | `optimization_frequency` from `main.py` (default `"daily"`) |
| What is the optimization scoring metric? (Sharpe ratio, CAGR/MaxDD, profit factor) | Sharpe ratio |
| What is the train/validation split? | Last 25% of training data as validation |
| Is there walk-forward or out-of-sample validation? | Simple train/validation split |
| Are parameters optimized per-asset or shared across all? | Per-asset (each symbol gets its own optimized params) |

### 6. Data handling

| Question | Safe default if missing |
|---|---|
| What columns does the historical data have? | `open, high, low, close` (handled by `_normalize_ohlc`) |
| Is stock data adjusted for splits/dividends? | If CSV has `Adj Close`, use it. Flag if not |
| Is futures data continuous (front-month roll adjusted)? | Engine handles automatically |
| Is FX data bid/ask or midpoint? | `_normalize_ohlc` computes midpoint from bid/ask if present |
| What features are computed? (returns, MAs, ATR, volatility, z-scores) | Minimum: `ret`, `fast_ma`, `slow_ma`, `trend_spread`, `atr`, `realized_vol` |

### How the LLM must handle gaps — ASK, don't assume

**Rule: the LLM must never fill a gap silently.** Every function in `my_strategy.py` must contain code that the user explicitly approved or provided. If the user's backtest script or prompt is missing any item below, the LLM must formulate a specific question and wait for the answer before generating that function.

**Per-function completeness check:**

| Function | What must be filled | If missing, LLM asks |
|---|---|---|
| `get_asset_runtime_policy` | Which assets are 24/7 vs weekdays | "Which of your assets trade 24/7 (crypto) vs weekdays only?" |
| `get_asset_frequency` | Bar frequency | "What bar frequency does your strategy use? (1min, 5min, 15min, 1h)" |
| `get_asset_train_span` | Training lookback bars | "How many bars of history does your strategy need for training?" |
| `prepare_base_df` | Feature columns to compute | "What features does your strategy compute? (MAs, ATR, z-scores, volatility, Bollinger bands...)" |
| `strategy_parameter_optimization` | Parameter grid, scoring metric, validation split, schedule (daily/weekly) | "What parameters do you optimize? What grid? Daily or weekly? What scoring metric?" |
| `validate_strategy_optimization` | Validation rules (must match the payload saved by optimization) | This is structural — LLM generates it from the payload format. No user input needed |
| `get_signal` | Signal logic per asset, portfolio weights, leverage method | "What is your entry/exit signal? How are portfolio weights computed? How is leverage determined?" |
| `set_stop_loss_price` | Stop-loss rule (ATR-based, fixed-pip, trailing, or none) | "What is your stop-loss rule? If you don't use stops, say so — the function still returns `np.nan`" |
| `set_take_profit_price` | Take-profit rule (ATR-based, risk-reward, or none) | "What is your take-profit rule? If you don't use take-profits, say so" |

**Domain-specific questions the LLM must ask:**

| Domain | If user is silent on this, LLM asks |
|---|---|
| **Short selling** | "Do you allow short positions? On which assets? (FX: yes, Crypto: typically no)" |
| **IBKR constraints** | "MES and XAUUSD require whole-unit orders. Your backtest may have used fractional sizing — should I enforce whole units? What about your crypto: long-only?" |
| **Risk limits** | "Your backtest may not have daily loss limits or drawdown circuit breakers. Do you want these in the live strategy? If not, I'll note them as not implemented." |
| **Portfolio allocation** | "How do you allocate capital across assets? Equal-weight, inverse-vol, HRP, or fixed percentages?" |
| **Leverage** | "What maximum leverage do you use? How is it calculated — fixed multiplier, Kelly criterion, or target volatility?" |
| **Optimization** | "Does your strategy use fixed parameters or do you re-optimize periodically? If fixed, I'll skip grid search in `strategy_parameter_optimization` and return your fixed values." |
| **Position sizing** | "Do you cap position size per asset? Is there a maximum allocation per symbol?" |
| **Regime/volatility** | "Does your strategy reduce exposure in high-volatility regimes? If so, what threshold?" |

**Format for LLM questions:** The LLM should batch related questions together (2-4 at a time) rather than asking one at a time. After receiving answers, the LLM should confirm: "I now have enough to generate all 9 functions. Shall I proceed?"

**If the user says "I don't have that / I don't know":**

- For stops, take-profits, daily loss limits: implement the function returning `np.nan` or `0.0` and add `# NOTE: not implemented — no stop/take-profit/limit rule provided`
- For signal logic: this is essential. Ask again with more specific prompting ("Even a simple MA crossover is fine — what fast and slow windows?")
- For portfolio weights: suggest inverse-vol as a simple starting point and ask if they want that
- For optimization: if they have fixed params, implement `strategy_parameter_optimization` as a pass-through that stores the fixed params — no grid search needed

---

## What the engine expects from your strategy file

The setup engine calls these 9 functions. Every one must exist with the exact signature shown below. The engine imports your file as `stra` and calls:

- `stra.get_asset_runtime_policy(symbol, asset_class)` — per-asset session rules
- `stra.get_asset_frequency(symbol)` — bar frequency string
- `stra.get_asset_train_span(symbol)` — number of bars for training
- `stra.prepare_base_df(historical_data, data_frequency, ticker, train_span)` — feature engineering
- `stra.strategy_parameter_optimization(symbol_specs, ...)` — parameter search, returns config payload
- `stra.validate_strategy_optimization(symbol_specs, ...)` — validates existing config
- `stra.get_signal(app, fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols, leverage)` — live signals
- `stra.set_stop_loss_price(app)` — per-bar stop price
- `stra.set_take_profit_price(app)` — per-bar take-profit price

---

## Function reference (copy these signatures exactly)

### 1. `get_asset_runtime_policy`

```python
def get_asset_runtime_policy(symbol, asset_class=None) -> dict:
```

**Purpose:** Tell the engine whether this asset trades 24/7 (crypto) or weekdays only.

**Return shape:**
```python
{
    "session": "24_7" | "weekdays",
    "flatten_at_day_end": False,
    "daily_maintenance_utc_start": "00:00",
    "daily_maintenance_minutes": 15,
}
```

**Rules:** Return `"24_7"` for crypto assets, `"weekdays"` for everything else.

---

### 2. `get_asset_frequency`

```python
def get_asset_frequency(symbol) -> str:
```

**Purpose:** Return the bar frequency for this symbol. Read from `main.py` via the helper below.

```python
def get_asset_frequency(symbol):
    return _strategy_frequency()   # reads "strategy_frequency" from main.py
```

---

### 3. `get_asset_train_span`

```python
def get_asset_train_span(symbol) -> int:
```

**Purpose:** Number of bars used for training/optimization. Read from `main.py`.

```python
def get_asset_train_span(symbol):
    return _strategy_train_span()   # reads "strategy_optimization_lookback" from main.py
```

---

### 4. `prepare_base_df`

```python
def prepare_base_df(historical_data, data_frequency, ticker, train_span=3500):
```

**Purpose:** Given raw OHLC data, return a feature DataFrame the engine uses for optimization scoring.

**Input:** `historical_data` is a DataFrame with columns `open, high, low, close`. `ticker` is the symbol string. `train_span` keeps the N most recent bars.

**Return:** A DataFrame with at minimum columns `open, high, low, close` plus any features you compute (returns, moving averages, ATR, volatility, etc.).

---

### 5. `strategy_parameter_optimization`

```python
def strategy_parameter_optimization(symbol_specs=None, optimization_frequency=None, optimization_bucket=None, optimized_at=None):
```

**Purpose:** Run parameter optimization for the full universe. Returns a config payload the engine saves and validates on future runs.

**Input:**
- `symbol_specs`: list of dicts like `[{"symbol": "EURUSD"}, {"symbol": "MES"}, ...]`
- `optimization_frequency`: `"daily"` or `"weekly"` from main.py
- `optimization_bucket`: ISO timestamp marking the optimization window boundary
- `optimized_at`: ISO timestamp of when optimization ran

**Return shape (dict):**
```python
{
    "strategy": "my-custom-strategy",
    "seed": 42,
    "signal_method": "my_signal_name",
    "portfolio_method": "my_portfolio_name",
    "leverage_method": "kelly_capped",
    "frequency": "5min",
    "train_span": 3000,
    "fixed_max_leverage": 1.0,
    "symbols": ["EURUSD", "MES", "XAUUSD", "ETH"],
    "asset_params": {
        "EURUSD": {"fast_window": 20, "slow_window": 120, ...},
        "MES": {"fast_window": 15, ...},
    },
    "my_weights": {"EURUSD": 0.25, "MES": 0.30, ...},
    "portfolio_leverage_multiplier": 1.2,
    "optimization_frequency": "daily",
    "optimization_bucket": "2026-01-15 00:00:00",
    "optimized_at": "2026-01-15 06:30:00",
    "global_params": {...},
    "feature_inventory": ["ret", "fast_ma", "slow_ma", "atr", ...],
    "config_hash": "abc123..."
}
```

**Critical rules:**
- Must save the payload to `data/models/strategy_optimization_manifest.json` (create dirs if needed)
- Must include a `config_hash` computed via SHA-256 over the JSON-sorted payload
- The engine calls `validate_strategy_optimization` with this payload next

**How the engine schedules re-optimization (daily vs weekly):**

The engine re-runs optimization automatically when the "bucket" changes. You control the schedule via `main.py`:

```python
optimization_frequency = "daily"   # re-optimize every trading day
optimization_frequency = "weekly"  # re-optimize every Monday
```

The engine computes an `optimization_bucket` (ISO timestamp at midnight of the current day or Monday of the current week). Each trading period it calls `validate_strategy_optimization()` with this bucket. If the stored manifest has a different bucket (e.g., yesterday vs today), validation fails → the engine calls `strategy_parameter_optimization()` to re-optimize. If the bucket matches, the stored manifest is reused — no re-optimization runs.

Your strategy file only needs to:
1. Accept `optimization_frequency` and `optimization_bucket` in both functions
2. Store both in the payload
3. Compare them during `validate_strategy_optimization()` — if they don't match, raise `ValueError`

The rest is handled by the engine. No scheduling logic needed in your strategy file.

**Implementation pattern:**
1. Iterate over `symbol_specs` to get the symbol list
2. For each symbol, load historical data, optimize parameters, store results
3. Compute portfolio weights from validation returns
4. Compute Kelly leverage multiplier
5. Build the payload dict, compute hash, save to disk

---

### 6. `validate_strategy_optimization`

```python
def validate_strategy_optimization(symbol_specs=None, optimization_result=None, optimization_frequency=None, optimization_bucket=None):
```

**Purpose:** Check whether a previously-saved optimization manifest is still valid for the current universe and trading day. Raise `ValueError` or `FileNotFoundError` if invalid. Return the payload dict if valid.

**Validation checks you must perform:**
- Manifest file exists and is valid JSON
- Every symbol in `symbol_specs` has params and weights in the manifest
- `optimization_frequency` matches
- `optimization_bucket` matches
- `config_hash` matches a freshly-computed hash of the same params

---

### 7. `get_signal` — THE MAIN FUNCTION

```python
def get_signal(app, fx_pairs=None, futures_symbols=None, metals_symbols=None, crypto_symbols=None, stock_symbols=None, leverage=None):
```

**Purpose:** Compute live trading signals for every symbol in the universe. Called once per bar for the full portfolio.

**Attributes you MUST set on `app`:**
```python
app.leverage = 1.0
app.target_weights = {"EURUSD": 0.25, ...}
app.applied_target_weights = app.target_weights.copy()
app.margin_scale = app.leverage / fixed_max_leverage
app.required_capital_frac = sum(app.target_weights.values())
app.used_capital_frac = min(app.leverage, fixed_max_leverage) * sum(app.target_weights.values())
app.cash_weight = max(0.0, 1.0 - sum(app.target_weights.values()))
app.strategy_targets = {
    "EURUSD": {
        "signal": 1,
        "target_weight": 0.25,
        "stop_price": 1.0850,
        "take_profit_price": 1.0950,
        "sleeve": "trend",
        "quantity_mode": None,
        "quantity_step": None,
        "target_quantity": None,
    },
}
app.strategy_state_updates = {
    "portfolio": {...},
    "targets": app.strategy_targets,
    "optimized_asset_params": {...},
}
```

**Return shape:**
```python
return {
    "targets": {"EURUSD": {"signal": 1}, "MES": {"signal": 0}, ...},
    "state_updates": app.strategy_state_updates,
}
```

---

### 8. `set_stop_loss_price`

```python
def set_stop_loss_price(app):
```

**Purpose:** Called per symbol per bar. Return the stop-loss price for the current position. Return `float` or `np.nan`.

---

### 9. `set_take_profit_price`

```python
def set_take_profit_price(app):
```

**Purpose:** Same as stop-loss but for take-profit. Return `float` or `np.nan`.

---

## Required helper functions

```python
import json
from pathlib import Path
from ibkr_multi_asset import trading_functions as tf

_USER_CONFIG_ROOT = Path(__file__).resolve().parents[1]

def _main_variables() -> dict:
    try:
        return tf.extract_variables(str(_USER_CONFIG_ROOT / "main.py"))
    except Exception:
        return {}

def _strategy_frequency() -> str:
    return str(_main_variables().get("strategy_frequency", "5min")).strip()

def _strategy_train_span() -> int:
    return max(300, int(_main_variables().get("strategy_optimization_lookback", 3000)))

def _fixed_max_leverage() -> float:
    return max(0.0, float(_main_variables().get("fixed_max_leverage", 1.0)))

def _configured_long_only_symbols() -> set[str]:
    configured = _main_variables().get("long_only_symbols", []) or []
    return {str(s).upper() for s in configured}
```

---

## Required data functions — FULL IMPLEMENTATIONS

Include these verbatim. They are boilerplate that every strategy file needs.

```python
_OPTIMIZATION_MANIFEST_PATH = Path("data/models/strategy_optimization_manifest.json")
_OPTIMIZATION_FEATURES_PATH = Path("data/models/optimal_features_df.xlsx")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _normalize_ohlc(df):
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
            {"open": title_cols["Open"], "high": title_cols["High"],
             "low": title_cols["Low"], "close": title_cols["Close"]})
        if not candidate.empty: return candidate
    if {"open", "high", "low", "close"}.issubset(lower_cols):
        candidate = _candidate_frame(
            {"open": lower_cols["open"], "high": lower_cols["high"],
             "low": lower_cols["low"], "close": lower_cols["close"]})
        if not candidate.empty: return candidate
    midpoint_cols = {"bid_open", "bid_high", "bid_low", "bid_close",
                     "ask_open", "ask_high", "ask_low", "ask_close"}
    if midpoint_cols.issubset(lower_cols):
        raw = out.rename(columns={lower_cols[key]: key for key in lower_cols})
        midpoint = tf.get_mid_series(raw)
        midpoint.columns = [str(col).lower() for col in midpoint.columns]
        return midpoint[["open", "high", "low", "close"]].astype(float).dropna()
    return pd.DataFrame(columns=["open", "high", "low", "close"])


def _normalize_history_frame(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    unnamed = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if unnamed: out = out.drop(columns=unnamed)
    if not isinstance(out.index, pd.DatetimeIndex):
        datetime_column = None
        for candidate in ["datetime", "Datetime", "date", "Date", "time", "Time"]:
            if candidate in out.columns:
                datetime_column = candidate; break
        if datetime_column is not None:
            parsed = pd.to_datetime(out[datetime_column], errors="coerce", format="mixed")
            out = out.loc[parsed.notna()].copy()
            out.index = parsed.loc[parsed.notna()]
            if datetime_column in out.columns: out = out.drop(columns=[datetime_column])
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


def _resample_ohlc(df, frequency):
    normalized = _normalize_ohlc(df)
    if normalized.empty: return normalized
    source = normalized.rename(columns={"open": "Open", "high": "High",
                                         "low": "Low", "close": "Close"})
    resampled = tf.resample_df(source, frequency, start="00h00min")
    resampled.columns = [str(col).lower() for col in resampled.columns]
    return resampled[["open", "high", "low", "close"]].astype(float).dropna()


def _compute_atr(frame, window):
    if frame.empty: return pd.Series(dtype=float)
    close_prev = frame["close"].shift(1)
    true_range = pd.concat([
        (frame["high"] - frame["low"]).abs(),
        (frame["high"] - close_prev).abs(),
        (frame["low"] - close_prev).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(max(2, int(window)), min_periods=1).mean()


def _load_symbol_history(symbol, fallback=None):
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
            if not normalized.empty: return normalized
    if fallback is not None:
        normalized = _normalize_history_frame(fallback.copy())
        if not normalized.empty: return normalized
    return pd.DataFrame()


def _asset_defaults(symbol):
    params = dict(DEFAULT_ASSET_PARAMS)
    params["allow_short"] = str(symbol).upper() not in (
        _configured_long_only_symbols() | CRYPTO_LONG_ONLY)
    return params


def _prepare_symbol_frame(history, symbol, params=None):
    params = params or _asset_defaults(symbol)
    frame = _resample_ohlc(history, get_asset_frequency(symbol))
    if frame.empty: return frame
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
```

---

## Optimization helpers — FULL IMPLEMENTATIONS

```python
def _annualization_factor(freq):
    return float(max(1, tf.get_periods_per_day(freq)) * 252)


def _score_strategy(returns, freq):
    series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if series.empty or np.isclose(float(series.std(ddof=0)), 0.0): return -np.inf
    ann = _annualization_factor(freq)
    return float(np.sqrt(ann) * series.mean() / series.std(ddof=0))


def _split_train_validation(frame, validation_fraction):
    if frame.empty: return frame.copy(), frame.copy()
    validation_size = max(50, int(len(frame) * float(validation_fraction)))
    validation_size = min(validation_size, max(1, len(frame) // 2))
    train = frame.iloc[:-validation_size].copy()
    validation = frame.iloc[-validation_size:].copy()
    if train.empty:
        train = frame.iloc[:-1].copy(); validation = frame.iloc[-1:].copy()
    return train, validation
```

---

## Payload builder & manifest I/O — FULL IMPLEMENTATIONS

```python
def _strategy_config_payload(symbol_specs=None, asset_params=None, weights_dict=None,
                              portfolio_leverage_multiplier=1.0, optimization_frequency=None,
                              optimization_bucket=None, optimized_at=None):
    symbols = sorted({str(spec.get("symbol", "")).upper()
                     for spec in (symbol_specs or []) if spec.get("symbol")})
    payload = {
        "strategy": "my-custom-strategy",
        "seed": int(DEFAULT_GLOBAL_PARAMS.get("seed", 42)),
        "signal_method": DEFAULT_GLOBAL_PARAMS.get("signal_method", "my_signal"),
        "portfolio_method": DEFAULT_GLOBAL_PARAMS.get("portfolio_method", "my_portfolio"),
        "leverage_method": DEFAULT_GLOBAL_PARAMS.get("leverage_method", "kelly_capped"),
        "frequency": _strategy_frequency(),
        "train_span": int(_strategy_train_span()),
        "fixed_max_leverage": float(_fixed_max_leverage()),
        "symbols": symbols,
        "asset_params": asset_params or {s: _asset_defaults(s) for s in symbols},
        "my_weights": weights_dict or {},
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


def _load_optimized_params():
    if not _OPTIMIZATION_MANIFEST_PATH.exists(): return {}
    try: return json.loads(_OPTIMIZATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    except: return {}
```

---

## Signal computation & live target — FULL IMPLEMENTATION

Customize the signal inside `_trend_position_series`. The rest is boilerplate:

```python
def _trend_position_series(frame, params):
    if frame.empty: return pd.Series(dtype=float)
    allow_short = bool(params.get("allow_short", True))
    spread = pd.to_numeric(frame.get("trend_spread",
        pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0.0)
    signal = pd.Series(0.0, index=frame.index, dtype=float)
    signal.loc[spread > 0] = 1.0
    if allow_short: signal.loc[spread < 0] = -1.0
    return signal


def _strategy_returns(frame, params):
    if frame.empty:
        empty = pd.Series(dtype=float); return empty, empty
    position = _trend_position_series(frame, params)
    realized = position.shift(1).fillna(0.0) * frame["ret"].fillna(0.0)
    return position, realized


def _live_target(symbol, frame, params, portfolio_weight):
    if frame.empty:
        return {"signal": 0, "target_weight": 0.0,
                "stop_price": np.nan, "take_profit_price": np.nan}
    position = _trend_position_series(frame, params)
    signal = int(np.sign(float(position.iloc[-1]))) if not position.empty else 0
    weight = max(0.0, float(portfolio_weight)) if signal != 0 else 0.0
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(
        frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    sm = float(DEFAULT_GLOBAL_PARAMS.get("stop_atr_multiple", 2.0))
    tm = float(DEFAULT_GLOBAL_PARAMS.get("take_profit_atr_multiple", 3.0))
    if signal > 0:
        sp = close - sm * atr; tp = close + tm * atr
    elif signal < 0:
        sp = close + sm * atr; tp = close - tm * atr
    else:
        sp = np.nan; tp = np.nan
    return {"signal": signal, "target_weight": float(weight),
            "stop_price": float(sp) if np.isfinite(sp) else np.nan,
            "take_profit_price": float(tp) if np.isfinite(tp) else np.nan}
```

---

## Stop-loss & take-profit — FULL IMPLEMENTATIONS

```python
def set_stop_loss_price(app):
    targets = getattr(app, "strategy_targets", {}) or {}
    symbol = str(getattr(app, "ticker", "")).upper()
    target = targets.get(symbol, {})
    price = float(target.get("stop_price", np.nan)) if target else np.nan
    if np.isfinite(price): return price
    history = _load_symbol_history(symbol, fallback=getattr(app, "historical_data", None))
    frame = _prepare_symbol_frame(history.tail(int(_strategy_train_span())).copy(),
                                   symbol, _asset_defaults(symbol))
    if frame.empty: return np.nan
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(
        frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    signal = int(np.sign(float(getattr(app, "signal", 0.0) or 0.0)))
    sm = float(DEFAULT_GLOBAL_PARAMS.get("stop_atr_multiple", 2.0))
    if signal > 0: return float(close - sm * atr)
    if signal < 0: return float(close + sm * atr)
    return np.nan


def set_take_profit_price(app):
    targets = getattr(app, "strategy_targets", {}) or {}
    symbol = str(getattr(app, "ticker", "")).upper()
    target = targets.get(symbol, {})
    price = float(target.get("take_profit_price", np.nan)) if target else np.nan
    if np.isfinite(price): return price
    history = _load_symbol_history(symbol, fallback=getattr(app, "historical_data", None))
    frame = _prepare_symbol_frame(history.tail(int(_strategy_train_span())).copy(),
                                   symbol, _asset_defaults(symbol))
    if frame.empty: return np.nan
    close = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1]) if "atr" in frame.columns and np.isfinite(
        frame["atr"].iloc[-1]) else max(close * 0.003, 1e-8)
    signal = int(np.sign(float(getattr(app, "signal", 0.0) or 0.0)))
    tm = float(DEFAULT_GLOBAL_PARAMS.get("take_profit_atr_multiple", 3.0))
    if signal > 0: return float(close + tm * atr)
    if signal < 0: return float(close - tm * atr)
    return np.nan
```


---

## Required universe mapping

```python
def _symbols_for_kind(kind, fx_pairs, futures_symbols, metals_symbols, crypto_symbols, stock_symbols=None):
    kind = str(kind or "").lower()
    mapping = {
        "fx": fx_pairs or [], "forex": fx_pairs or [],
        "mes": futures_symbols or [], "futures": futures_symbols or [],
        "xau": metals_symbols or [], "metals": metals_symbols or [],
        "crypto": crypto_symbols or [],
        "stock": stock_symbols or [], "stocks": stock_symbols or [],
    }
    return [str(s).upper() for s in mapping.get(kind, [])]
```

---

## Data contract — what columns your strategy receives

The engine always provides `app.historical_data` with these columns, guaranteed:

```
Columns:  open | high | low | close
Index:    DatetimeIndex (UTC)
Dtype:    float64 (or float after normalization)
```

**How this guarantee is enforced:**

1. `engine.py` → `_ensure_history_file()` creates CSV with columns `['open', 'high', 'low', 'close']`
2. `setup.py` → `trading_app.__init__()` reads the CSV: `pd.read_csv(..., index_col=0)` → columns are always lowercase
3. IBKR populates bars with `bar.open, bar.high, bar.low, bar.close` → same column names
4. `engine.py` → `_load_symbol_history_for_app()` reads CSV, parses DatetimeIndex, returns `.tail(keep_rows)`
5. Your strategy's `_load_symbol_history` reads the same CSV and calls `_normalize_ohlc` for safety

**What your `_prepare_symbol_frame` can rely on:**
- `frame["close"]` always exists
- `frame["open"], frame["high"], frame["low"]` always exist
- The index is always a `DatetimeIndex`
- After `_prepare_symbol_frame`, you add: `ret`, `fast_ma`, `slow_ma`, `trend_spread`, `atr`, `realized_vol` (plus any custom features)

**What your `_trend_position_series` can rely on:**
- `frame.get("trend_spread")` exists after `_prepare_symbol_frame`
- `frame.get("close")` exists
- Any custom columns you added in `_prepare_symbol_frame` (z-score, Bollinger bands, EMA, volatility percentile)

**When loading from CSV directly** (`_load_symbol_history` fallback): the CSV might have different column formats depending on the data source (Yahoo Finance CSV with `Date,Open,High,Low,Close,Adj Close,Volume` or IBKR CSV with `open,high,low,close`). `_normalize_ohlc` handles all these cases and outputs the standard `open, high, low, close` format.

---

## Asset-specific data considerations

The `_normalize_ohlc` function above handles the most common column formats. Here's what you need to know about each asset class:

| Asset Class | IBKR Data Format | Columns to Expect | Special Handling |
|---|---|---|---|
| **Stocks** | `whatToShow="TRADES"` (unadjusted) or `"ADJUSTED_LAST"` (split/dividend adjusted) | `Open, High, Low, Close` or `open, high, low, close` | If CSV has `Adj Close` column, use it instead of `Close` to avoid gap artifacts from splits/dividends. Add this check before the standard column detection in `_normalize_ohlc` |
| **Futures** | Continuous contract via `AUTO_FRONT_MONTH` roll policy (set in `main.py`) | `Open, High, Low, Close` | The engine handles contract roll automatically. The data arriving in `app.historical_data` is already continuous. No special normalization needed |
| **FX** | `whatToShow="MIDPOINT"` or bid/ask bars | `bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close` | Already handled by `_normalize_ohlc` midpoint computation |
| **Metals** (XAUUSD) | Spot commodity via SMART exchange | `Open, High, Low, Close` | Standard OHLC format. Quantity step is integer (set in `main.py`: `metals_quantity_step = 1.0`) |
| **Crypto** | 24/7 trading via PAXOS | `Open, High, Low, Close` | Standard OHLC. Session returned as `"24_7"`. Long-only by default (set in `main.py`: `long_only_symbols = ["ETH", "BTC"]`) |

**Stock adjusted close fix** — add this as the first check in your `_normalize_ohlc`:

```python
# Before the existing column detection, add:
if {"Date", "Adj Close", "Close"}.issubset(title_cols) or \
   {"date", "adj_close", "close"}.issubset(lower_cols):
    # Use adjusted close for stocks to avoid split/dividend gaps
    adj_col = "Adj Close" if "Adj Close" in title_cols else "adj_close"
    close_col = "Close" if "Close" in title_cols else "close"
    out[close_col] = pd.to_numeric(out[adj_col], errors="coerce")
```

**IBKR live data**: When trading live, the engine fetches historical bars from IBKR via `reqHistoricalData`. The data arrives with standard column names matching the contract type. Your normalization functions don't need to handle IBKR-specific quirks — they're covered by the existing logic.

---

## Data auto-population — no manual downloads needed

The engine fetches and maintains historical data automatically. Here's the lifecycle:

| Stage | What happens |
|---|---|
| **First run** | `engine.py` → `_ensure_history_file()` creates a skeleton CSV at `data/historical/historical_MES.csv` with columns `open, high, low, close` and one dummy row |
| **Every bar** | `engine.py` → `_configure_portfolio_app_for_symbol()` sets `app.historical_data = pd.read_csv(...)` and the engine appends new bars from IBKR's `reqHistoricalData` |
| **End of cycle** | `sf.save_portfolio_cycle_data()` writes `app.historical_data.to_csv(...)` — data persists across restarts |
| **Next restart** | `trading_app.__init__()` reads the CSV: `self.historical_data = pd.read_csv(historical_data_address, index_col=0).tail(keep_rows)` |

**What this means for your strategy file:**
- You never download data. The engine handles IBKR fetching, CSV storage, and bar-by-bar append.
- Your `_load_symbol_history` function reads from the same CSV path and falls back to `app.historical_data`.
- On first run, the CSV has only one dummy row. Your `_prepare_symbol_frame` will return empty until enough bars accumulate. This is normal — the engine retries each bar.
- Keep your `train_span` realistic. A 5-min strategy with `strategy_optimization_lookback = 3000` needs ~10 trading days of data before signals are reliable.

---

## Full `main.py` variable reference

Your strategy can read ALL of these via `_main_variables()`. Use only what you need — the required helpers already cover the most common ones.

| Variable | Type | Default | Used by |
|---|---|---|---|
| `fx_pairs` | list | `["EURUSD"]` | `_symbols_for_kind("fx", ...)` |
| `futures_symbols` | list | `["MES"]` | `_symbols_for_kind("futures", ...)` |
| `metals_symbols` | list | `["XAUUSD"]` | `_symbols_for_kind("metals", ...)` |
| `crypto_symbols` | list | `["ETH"]` | `_symbols_for_kind("crypto", ...)` |
| `stock_symbols` | list | `[]` | `_symbols_for_kind("stocks", ...)` |
| `strategy_frequency` | str | `"5min"` | `_strategy_frequency()` — bar size |
| `strategy_optimization_lookback` | int | `3000` | `_strategy_train_span()` — training bars |
| `fixed_max_leverage` | float | `1.0` | `_fixed_max_leverage()` — leverage cap |
| `long_only_symbols` | list | `["ETH"]` | `_configured_long_only_symbols()` — no-short list |
| `optimization_frequency` | str | `"daily"` | `_configured_optimization_frequency()` — daily/weekly |
| `portfolio_leverage` | float | `1.0` | Fallback leverage (engine uses optimized value from manifest) |
| `host` | str | `"127.0.0.1"` | IBKR connection (read-only, don't modify) |
| `port` | int | `7497` | IBKR port |
| `account` | str | `"YOUR_IBKR_ACCOUNT"` | Account ID |
| `timezone` | str | `"America/Lima"` | Timezone for session detection |
| `trading_day_origin` | str | `"18:00"` | When the trading day starts |
| `metals_quantity_step` | float | `1.0` | XAUUSD whole-unit constraint |
| `futures_roll_policy` | str | `"AUTO_FRONT_MONTH"` | MES contract roll |
| `forex_exchange` | str | `"IDEALPRO"` | FX venue |

**To add a custom variable** (e.g., your own parameter): add it to `main.py`, then read it in your strategy via `_main_variables().get("my_param", default_value)`. The engine passes all `main.py` variables through unchanged.

---

## Portability rules

1. **Do NOT modify engine source files.** Only create/modify files under `user_config/`.
2. **All 9 functions must exist** with the exact signatures documented above.
3. **Internal logic is free.** Change signal generation, portfolio weighting, parameter optimization — anything inside the function body. Just keep the signatures.
4. **Data paths use `Path(__file__).resolve().parents[1]`** — not hardcoded absolute paths.
5. **Optimization manifest** must be saved to `data/models/` inside the `user_config` directory.
6. **Historical data** is loaded from `user_config/data/historical/historical_<SYMBOL>.csv` or received as fallback from the engine via `app.historical_data`.
7. **The `config_hash`** is computed via `hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()`. The engine compares this to detect stale configs.

---

## Long-only crypto

Define this set at the top of your file:

```python
CRYPTO_LONG_ONLY = {"BTC", "ETH", "SOL", "LTC", "BCH"}
```

The `_configured_long_only_symbols()` helper reads `long_only_symbols` from `main.py`. The `_asset_defaults` function merges both to set `allow_short=False` for those symbols.

---

## File structure — your LLM should output

A single `my_strategy.py` file containing (in order):

1. Imports (`numpy`, `pandas`, `json`, `hashlib`, `Path`, `trading_functions`)
2. `DEFAULT_GLOBAL_PARAMS` dict — your strategy's default parameters
3. `DEFAULT_ASSET_PARAMS` dict — per-asset parameter defaults
4. `CRYPTO_LONG_ONLY` set
5. Path constants (`_USER_CONFIG_ROOT`, `_OPTIMIZATION_MANIFEST_PATH`, etc.)
6. Configuration helpers (`_main_variables`, `_strategy_frequency`, etc.)
7. `get_asset_runtime_policy`, `get_asset_frequency`, `get_asset_train_span`
8. Universe helper (`_symbols_for_kind`)
9. Data normalization functions (full code provided above)
10. `prepare_base_df`
11. Optimization helpers (full code provided above)
12. Payload builder & manifest I/O (full code provided above)
13. Signal computation (`_trend_position_series` — YOUR CUSTOM LOGIC HERE)
14. `_strategy_returns`, `_live_target` (boilerplate provided above)
15. Portfolio weighting — YOUR CUSTOM LOGIC (HRP, inverse-vol, equal-weight, etc.)
16. `_optimize_asset_params` — YOUR CUSTOM grid search
17. `strategy_parameter_optimization` (skeleton provided in section 5)
18. `validate_strategy_optimization` (skeleton provided in section 6)
19. `get_signal` (skeleton provided in section 7)
20. `set_stop_loss_price`, `set_take_profit_price` (full code provided above)

The file should be self-contained (no imports from `strategy.py` or other user files) and ~600-900 lines.

---

## What you can customize

| Component | How |
|---|---|
| **Signal logic** | Replace `_trend_position_series` with any indicator: Bollinger bands, RSI, MACD, ML model, etc. |
| **Feature computation** | Add columns to `_prepare_symbol_frame`: z-scores, regime detection, volatility percentiles. |
| **Portfolio weighting** | Replace HRP with equal-weight, inverse-vol, risk parity, Black-Litterman, or a custom allocation. |
| **Leverage** | Replace Kelly with fixed, target-vol, or regime-based scaling. |
| **Stops / take-profits** | Replace ATR-based stops with trailing stops, support/resistance levels, or time-based exits. |
| **Optimization grid** | Change `DEFAULT_GLOBAL_PARAMS` grids to sweep your own parameter ranges. |
| **Long-only rules** | The `_configured_long_only_symbols()` function reads `long_only_symbols` from `main.py`. Enforce it in your signal logic. |
| **Asset-specific params** | `DEFAULT_ASSET_PARAMS` can have different keys per symbol (e.g., different MA windows for FX vs futures). |

---

## Risk management — what the engine does vs what you must do

The engine provides several risk controls automatically. Your strategy file must supply the signals and targets that feed into them.

### Handled automatically by the engine

| Risk Control | Where | How |
|---|---|---|
| **Maximum leverage cap** | `main.py` → `fixed_max_leverage` | `app.leverage` is clamped to this value. You set `app.leverage` in `get_signal`; the engine enforces the cap |
| **Order quantity validation** | `engine.py` → `sf.send_orders()` | The engine validates order quantities against IBKR contract constraints (min size, quantity step, fractional support) |
| **Synthetic stop monitoring (crypto)** | `engine.py` → `_run_deferred_synthetic_monitor_phase` | For assets without native stop orders (PAXOS crypto), the engine polls prices and sends market exits when stops are breached. Your strategy just sets `stop_price` in `strategy_targets` |
| **Portfolio margin check** | `engine.py` → shared account updates | The engine tracks `shared_unlevered_capital` and validates that total required capital does not exceed available equity |
| **Session-based flattening** | `engine.py` → `_should_flatten_at_day_end` | If your `get_asset_runtime_policy` returns `"flatten_at_day_end": True`, the engine closes positions at session end |

### Must be provided by your strategy

| Risk Control | Where | What to implement |
|---|---|---|
| **Per-symbol stop-loss price** | `set_stop_loss_price(app)` | Return a price that limits downside. Typically ATR-based: `close − stop_atr_mult × ATR` for longs |
| **Per-symbol take-profit price** | `set_take_profit_price(app)` | Return a price that locks in gains. Typically ATR-based: `close + tp_atr_mult × ATR` for longs |
| **Portfolio weights** | `get_signal()` → `app.target_weights` | Weight allocation across symbols. Affects position sizing. A concentrated portfolio has higher risk |
| **Long-only enforcement** | `_trend_position_series()` | For symbols in `CRYPTO_LONG_ONLY` or `_configured_long_only_symbols()`, clip signals to `≥ 0` |
| **Leverage scalar** | `get_signal()` → `app.leverage` | Set based on Kelly criterion, target volatility, or a fixed multiplier. The engine caps it at `fixed_max_leverage` |

### Optional risk controls you can add

| Risk Control | Implementation |
|---|---|
| **Daily loss limit** | Track cumulative P&L in `app.strategy_state_updates`. If daily loss exceeds threshold, set all signals to 0 in `get_signal` |
| **Max drawdown circuit breaker** | Read `app.portfolio_snapshots_df` or `app.shared_unlevered_capital`. If drawdown from peak exceeds threshold, reduce `app.leverage` to 0 |
| **Volatility targeting** | Compute rolling portfolio volatility in `get_signal`. Scale `app.leverage` inversely to keep ex-ante vol at a target level |
| **Correlation-based position reduction** | In portfolio weighting, down-weight symbols with high pairwise correlation to reduce concentration risk |
| **Max position size per symbol** | In `_live_target`, cap `target_weight` per symbol (e.g., max 25% per asset) |
| **Crypto concentration cap** | In portfolio weighting, ensure total crypto weight ≤ 30% of portfolio (IBKR observed limit) |
| **Regime-based exposure reduction** | In `_prepare_symbol_frame`, detect high-volatility regimes. Reduce or zero out signals when vol exceeds a percentile threshold |

### Risk attributes in `strategy_targets`

For each symbol, your `strategy_targets` dict can include these risk-related fields:

```python
app.strategy_targets = {
    "BTC": {
        "signal": 1,
        "target_weight": 0.15,
        "stop_price": 85200.0,
        "take_profit_price": 88500.0,
        "sleeve": "crypto_long_only",
        "quantity_mode": None,       # "fixed" for fixed-quantity, None for weight-based
        "quantity_step": 1e-8,       # IBKR minimum increment (crypto = near-continuous)
        "target_quantity": None,     # Computed by engine from weight × equity / price
    },
}
```

The `quantity_mode` and `quantity_step` fields are passed to the engine's order sizing logic. For assets with whole-unit constraints (MES, XAUUSD, USDJPY), set `quantity_step` to the IBKR increment (1.0 for whole units). For fractional assets (EURUSD, crypto), set it to the allowed increment (0.01 for EURUSD, 1e-8 for crypto).

---

## Bar cycle — order of operations

Every bar, the engine executes this sequence. Knowing the order matters if your strategy needs data that was set in a previous step.

```
1. sf.refresh_symbol_market_data(app)     ← Fetches new bar from IBKR, appends to app.historical_data
2. sf.collect_shared_account_snapshot()    ← Gets NetLiq, BuyingPower, margin from IBKR
3. stra.get_signal(app, fx_pairs, ...)    ← YOUR FUNCTION: compute signals + weights
   └─ You set: app.leverage, app.target_weights, app.strategy_targets, app.strategy_state_updates
4. sf.collect_shared_contract_details()    ← Resolves IBKR contract specs
5. sf.collect_shared_broker_snapshot()     ← Gets current positions, open orders, executions
6. For each symbol:
   ├─ _apply_portfolio_target_attrs(app)  ← Copies weights/leverage from step 3
   ├─ stra.set_stop_loss_price(app)       ← YOUR FUNCTION: return stop price
   ├─ stra.set_take_profit_price(app)     ← YOUR FUNCTION: return take-profit price
   └─ sf.send_orders(app)                 ← Size and submit orders to IBKR
7. sf.collect_shared_broker_snapshot()     ← Post-trade verification
8. Synthetic stop monitor sweep            ← Polls crypto prices against stop prices (PAXOS has no native STP)
9. sf.save_portfolio_cycle_data()          ← Persist all data to Excel/CSV/JSON
```

**Key insight:** `get_signal` runs ONCE per bar for the full portfolio — not per symbol. Your function must return signals for ALL symbols in one call. `set_stop_loss_price` and `set_take_profit_price` run per symbol during order preparation (step 6).

---

## `app` object API — what you can read in your strategy functions

The `app` object is the central state container. These attributes are guaranteed to exist when your functions are called.

### Available in all functions (`get_signal`, `set_stop_loss_price`, `set_take_profit_price`)

| Attribute | Type | Description |
|---|---|---|
| `app.ticker` | str | Current symbol being processed (e.g., `"MES"`) |
| `app.historical_data` | DataFrame | Full OHLC history: columns `open, high, low, close`, DatetimeIndex. Updated every bar |
| `app.data_frequency` | str | Bar frequency (e.g., `"5min"`) |
| `app.train_span` | int | Training lookback in bars |
| `app.signal` | float | `+1.0` (long), `-1.0` (short), `0.0` (flat). Set by engine from `get_signal` return value |
| `app.leverage` | float | Portfolio leverage multiplier. YOU set this in `get_signal` |
| `app.optimization_frequency` | str | `"daily"` or `"weekly"` from main.py |
| `app.optimization_bucket` | str | ISO timestamp of current optimization window |
| `app.asset_spec` | dict | Contract metadata: `symbol`, `asset_class`, `exchange`, `currency` |
| `app.allowed_symbols` | list | Full universe symbol list |
| `app.strategy_targets` | dict | Per-symbol targets with signal, weight, stop, take-profit. YOU set this in `get_signal` |
| `app.target_weights` | dict | `{symbol: weight}` — YOU set this in `get_signal` |
| `app.pos_df` | DataFrame | Current positions from IBKR (available after step 5) |
| `app.portfolio_snapshots_df` | DataFrame | Position snapshots with MarketValue, UnrealizedPnL |
| `app.cash_balance` | DataFrame | Account cash and leverage history |

### Available only in `get_signal`

| Attribute | Type | Description |
|---|---|---|
| `fx_pairs` (param) | list | FX symbols from main.py |
| `futures_symbols` (param) | list | Futures symbols from main.py |
| `metals_symbols` (param) | list | Metals symbols from main.py |
| `crypto_symbols` (param) | list | Crypto symbols from main.py |
| `stock_symbols` (param) | list | Stock symbols from main.py |

### Available only in `set_stop_loss_price` and `set_take_profit_price`

| Attribute | Type | Description |
|---|---|---|
| `app.signal` | float | Already populated from `get_signal` return. Use for stop direction |
| `app.strategy_targets` | dict | Has `stop_price` and `take_profit_price` from `get_signal`. Check these first, fall back to ATR calculation |

---

## Strategy state persistence — how to remember things across bars

The strategy can persist state across bars, restarts, and crashes via `app.strategy_state`.

### Reading state (on startup)

```python
# app.strategy_state is a nested dict: {scope: {key: value}}
# Loaded from data/strategy_state.json on startup
daily_pnl = app.strategy_state.get('risk', {}).get('daily_pnl', 0.0)
last_signal = app.strategy_state.get('signals', {}).get('MES', 0)
```

### Writing state (during get_signal)

```python
# Set app.strategy_state_updates — the engine persists this automatically
app.strategy_state_updates = {
    'risk': {'daily_pnl': accumulated_pnl, 'peak_equity': current_peak},
    'signals': {'MES': mes_signal, 'EURUSD': eurusd_signal},
    'portfolio': {'last_rebalance_date': str(today)},
}
```

The engine calls `app.queue_strategy_state(app.strategy_state_updates)` during `sf.send_orders()`. This appends to a temp buffer. At end-of-cycle, `save_portfolio_cycle_data` flushes to `data/strategy_state.json`.

### Writing state directly (anytime)

```python
# For non-get_signal state updates (e.g., in stop-loss):
app.strategy_state.setdefault('execution', {})['last_stop_hit'] = str(now)
app.queue_strategy_state({'execution': {'last_stop_hit': str(now)}})
```

### Persistence guarantees

| Action | Persists |
|---|---|
| `app.strategy_state_updates = {...}` in `get_signal` | ✅ Auto-persisted at end of cycle |
| `app.queue_strategy_state({...})` anywhere | ✅ Persisted at end of cycle |
| `app.strategy_state['key'] = value` without queue | ❌ Not persisted — must use `queue_strategy_state` |

### Use cases for strategy state

- **Daily P&L tracking**: accumulate day's P&L in `strategy_state_updates`
- **Drawdown circuit breaker**: track peak equity, compare to current
- **Regime memory**: remember last regime state for hysteresis
- **Position tracking**: number of consecutive winning/losing trades
- **Parameter warm-start**: store last optimized params across restarts

---

## Troubleshooting — when the generated strategy.py has errors

If the user reports an error after placing `my_strategy.py` and running the setup, the LLM must diagnose systematically. Ask one question at a time, starting from the error message.

### Common errors and their fixes

| Error | Likely cause | LLM should ask |
|---|---|---|
| `ModuleNotFoundError: No module named 'my_strategy'` | File not in `strategies/` folder or `main.py` points to wrong path | "Is `my_strategy.py` in `user_config/strategies/`? Does `main.py` have `strategy_file = 'strategies/my_strategy.py'`?" |
| `AttributeError: module 'my_strategy' has no attribute 'get_signal'` | Missing function or wrong function name | "Does your file define all 9 required functions? Check for typos in function names." |
| `TypeError: get_signal() missing X required positional arguments` | Function signature doesn't match | "Compare your `get_signal` signature with the one in the guide. It must accept all parameters even if you don't use them." |
| `KeyError: 'trend_spread'` or `KeyError: 'close'` | Column not found in DataFrame — missing feature or normalization failed | "What columns does your `_prepare_symbol_frame` produce? Is `_normalize_ohlc` returning the standard `open, high, low, close` columns?" |
| `ValueError: JosGT manifest incomplete` | Optimization manifest missing per-asset params or weights | "Does your `strategy_parameter_optimization` store params for every symbol in the universe? Check the `asset_params` and weights dicts." |
| `FileNotFoundError: data/models/strategy_optimization_manifest.json` | First run — manifest doesn't exist yet | "This is normal on first run. The engine will call `strategy_parameter_optimization` to create it. Does that function complete without errors?" |
| `ImportError: cannot import name 'linkage' from 'scipy.cluster.hierarchy'` | Missing `scipy` dependency (only needed if using HRP) | "Your strategy uses HRP but `scipy` may not be installed. Run `pip install scipy` or switch to inverse-vol weighting." |
| Empty signals / no trades | `_prepare_symbol_frame` returns empty DataFrame | "What bar frequency is your strategy using? Does the historical data have enough bars to compute your indicators (e.g., 260-bar slow MA needs 260 bars)?" |
| `RecursionError` or infinite loop | Optimization grid too large or logic error | "How many parameter combinations does your grid produce? Large grids (e.g., 10×10×10 = 1000 combos × N symbols) may hang. Reduce the grid." |
| `MemoryError` or `numpy.linalg.LinAlgError` | Covariance matrix singular — not enough data or highly correlated returns | "How many symbols vs how many bars of validation returns? Need more bars than symbols. Try reducing the universe or increasing lookback." |

### Debugging workflow the LLM should follow

1. **Read the error:** ask the user to share the full traceback.
2. **Identify the function:** which of the 9 functions raised the error?
3. **Check the function's contract:** compare against the guide's signature and return shape.
4. **Isolate:** ask the user to run a minimal test importing just `my_strategy` and calling one function.
5. **Fix or rebuild:** apply the fix to the strategy file. If the error is in the engine (not the strategy), see next section.

---

## When strategy.py is not enough — modifying source code

The strategy interface covers signal generation, portfolio weighting, optimization, and risk management. But some customizations require changes to the engine or setup code.

### Examples that need source code changes

| Customization | What to modify |
|---|---|
| New order type (e.g., iceberg, pegged, bracket orders) | `setup_functions.py` → `send_orders()` |
| Custom contract resolution (non-standard IBKR contracts) | `ib_functions.py` → `build_contract_from_spec()` |
| Different bar cycle (e.g., skip account snapshot, custom period logic) | `engine.py` → `run_portfolio_setup_loop()` |
| Custom synthetic stop behavior (e.g., different polling interval, trail stops for crypto) | `engine.py` → `_run_deferred_synthetic_monitor_phase` |
| Additional data feeds (e.g., VIX, order book depth) | `setup_functions.py` → `refresh_symbol_market_data()` |
| Custom persistence format (e.g., database instead of Excel/JSON) | `setup_functions.py` → `save_portfolio_cycle_data()` |
| New asset class not covered by `_normalize_symbol_specs` | `engine.py` → `_normalize_symbol_specs()` |

### After modifying source code — rebuild the package

Source code changes only take effect after rebuilding. The setup includes platform-specific rebuild scripts:

| OS | Script |
|---|---|
| **Linux** | `user_config/rebuild_and_run.sh` |
| **macOS** | `user_config/rebuild_and_run_mac.command` |
| **Windows** | `user_config/rebuild_and_run_windows.bat` |

Manual rebuild command (all platforms):

```bash
python -m build
python -m pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
python user_config/main.py
```

### The LLM must warn the user

If the LLM determines that the user's request requires source code changes:

1. **Flag it explicitly:** "This customization requires modifying [filename] in the source code. This goes beyond the strategy interface."
2. **Explain the trade-off:** "You'll need to rebuild the package after every source change. Future updates to the setup may conflict with your changes."
3. **Reference the rebuild script:** "After editing, run the rebuild script for your OS before launching."
4. **Suggest alternatives:** "Can this be achieved within the strategy interface instead? For example, [alternative approach]."

---

## Testing your strategy

1. Place `my_strategy.py` in `user_config/strategies/`
2. Edit `main.py`: `strategy_file = "strategies/my_strategy.py"`
3. Run: `python main.py`

The engine will call `strategy_parameter_optimization` on the first bar, validate the manifest, then call `get_signal` every bar. Check `data/log/` for logs and `data/database.xlsx` for trade records.
