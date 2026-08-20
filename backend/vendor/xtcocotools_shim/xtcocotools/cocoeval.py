try:
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover
    COCOeval = None
