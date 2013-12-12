try:
    from _version import version as __version__
except ImportError:
    __version__ = "no-built"

from ._implementation import HasTraits, MetaHasTraits, TraitType, Any, CBytes, Dict, \
    Int, Long, Integer, Float, Complex, Bytes, Unicode, TraitError, \
    Undefined, Type, This, Instance, TCPAddress, List, Tuple, \
    ObjectName, DottedObjectName, CRegExp, Enum, Bool, CBool
