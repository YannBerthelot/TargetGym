"""Units of the params fields, for the generated environment pages.

Each module that defines a params class keeps a ``PARAM_UNITS`` dict for the
fields it adds. Provenance fingerprints skip those dicts (they document a value,
they do not change one), and this helper lives outside the fingerprinted files
for the same reason.
"""

import sys


def param_units(params_cls: type) -> dict[str, str]:
    """Units of a params class's fields, merged over its bases' modules."""
    units: dict[str, str] = {}
    for cls in reversed(params_cls.__mro__):
        module = sys.modules.get(cls.__module__)
        units.update(getattr(module, "PARAM_UNITS", {}))
    return units
