from types import SimpleNamespace

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
