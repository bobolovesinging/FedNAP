from os.path import dirname, basename, isfile, join
import glob
import importlib

# Auto-import all trainer modules so that any register_trainer() calls
# in submodules are executed at package import time.  This allows new
# trainer variants (e.g. fedcqd_joint_trainer.py) to auto-register
# themselves without modifying this file or any other code.
_modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in _modules
    if isfile(f) and not f.endswith('__init__.py')
]

for _mod_name in __all__:
    try:
        importlib.import_module(f"{__name__}.{_mod_name}")
    except ImportError as _e:
        # Silently skip modules that fail to import (e.g. missing deps);
        # the core fedcqd_trainer is imported explicitly elsewhere.
        pass
