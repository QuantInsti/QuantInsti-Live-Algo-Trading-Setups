import importlib.util
import sys
import ast
from pathlib import Path


_CURRENT_STRATEGY_MODULE = None
_CURRENT_STRATEGY_FILE = None


def _default_strategy_name():
    cwd = Path.cwd()
    main_path = cwd / "main.py"
    if main_path.exists():
        try:
            tree = ast.parse(main_path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "strategy_file":
                            try:
                                value = ast.literal_eval(node.value)
                            except (ValueError, SyntaxError):
                                continue
                            if isinstance(value, str) and value.strip():
                                return value.strip()
        except Exception:
            pass

    if (cwd / "strategies" / "strategy.py").exists():
        return "strategies/strategy.py"
    return "strategy.py"


def _resolve_strategy_path(strategy_file):
    strategy_name = str(strategy_file or _default_strategy_name()).strip()
    if not strategy_name:
        strategy_name = _default_strategy_name()
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
        return load_strategy_module(_default_strategy_name())
    return _CURRENT_STRATEGY_MODULE


def get_strategy_file():
    return _CURRENT_STRATEGY_FILE


class StrategyProxy:
    def __getattr__(self, name):
        module = get_strategy_module()
        return getattr(module, name)


stra = StrategyProxy()
