from types import SimpleNamespace
from collections.abc import Mapping, Sequence

def namespace_to_dict(obj):
    """
    Recursively convert a SimpleNamespace (or any object with __dict__) 
    into a nested dict. Other types (lists, primitives, etc.) are left intact.
    """
    # If it's already a dict, recurse into its values
    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}

    # If it's a namespace (or any object with __dict__), recurse its attributes
    if hasattr(obj, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}

    # Otherwise, assume it's a leaf (int, float, list, etc.)—leave as is
    return obj

def nestednamespace_to_dict(ns):
    """
    Recursively convert a SimpleNamespace (or any object with __dict__)
    into a nested dict. Other types (lists, primitives, etc.) are left intact.
    """
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    else:
        return ns

def dict_to_namespace(obj):
    """
    Recursively convert a dict (or list/tuple of dicts) into a namespace
    with dot-accessible attributes.
    """
    if isinstance(obj, Mapping):
        # for dicts, recurse into values and wrap as SimpleNamespace
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        # for lists/tuples, recurse into each element
        return type(obj)(dict_to_namespace(v) for v in obj)
    else:
        # everything else stays as is
        return obj
