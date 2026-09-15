import json
import numpy as np

def dumps(obj, indent = 2):
    json_result = json.dumps(obj, default=_default, indent=indent, ensure_ascii=False)
    return json_result

def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return vars(obj)