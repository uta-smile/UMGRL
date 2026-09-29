from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([str(package_dir)]):
    try:
        module = import_module(f"{__name__}.{module_name}")
    except Exception as exc:
        print(f"[models] Skipping {module_name} due to import error: {exc}")
        continue
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):
            globals()[attribute_name] = attribute
