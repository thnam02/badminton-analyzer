from pycocotools import mask as _mask

decode = _mask.decode
encode = _mask.encode
frPyObjects = getattr(_mask, "frPyObjects", None)
