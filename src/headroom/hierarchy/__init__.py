"""The city, borough and dispatch-area hierarchy.

:mod:`headroom.hierarchy.spec` holds the structure and its summing matrix;
:mod:`headroom.hierarchy.build` builds one from the loader's daily counts.
"""

from headroom.hierarchy.spec import Hierarchy

__all__ = ["Hierarchy"]
