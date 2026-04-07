import importlib.util
import sys
from pathlib import Path


_CURRENT_STRATEGY_MODULE = None
_CURRENT_STRATEGY_FILE = None


def _resolve_strategy_path(strategy_file):
    strategy_name = str(strategy_file or "strategy.py").strip()
    if not strategy_name:
        strategy_name = "strategy.py"
    path = Path(strategy_name)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_strategy_module(strategy_file=None):
    global _CURRENT_STRATEGY_MODULE, _CURRENT_STRATEGY_FILE

    path = _resolve_strategy_path(strategy_file)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    cache_key = f"ibkr_multi_asset.user_strategy.{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(cache_key, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    _CURRENT_STRATEGY_MODULE = module
    _CURRENT_STRATEGY_FILE = str(path)
    return module


def get_strategy_module():
    global _CURRENT_STRATEGY_MODULE
    if _CURRENT_STRATEGY_MODULE is None:
        return load_strategy_module("strategy.py")
    return _CURRENT_STRATEGY_MODULE


def get_strategy_file():
    return _CURRENT_STRATEGY_FILE


class StrategyProxy:
    def __getattr__(self, name):
        module = get_strategy_module()
        return getattr(module, name)


stra = StrategyProxy()
