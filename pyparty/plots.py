""" Imports/wrapper for separating plots out of utils.  Maintains
backwards compatibility by not changing utils.py"""

from pyparty.utils import grayhist, splot, showim, zoom, zoomshow

from pyparty.utils import multi_axes

def multishow(images, *args, **kwargs):
    if getattr(images, '__iter__'):
        return showim(images, *args, **kwargs)

    if len(images) == 1:
        return showim(images[0], *args, **kwargs)
    
    axes, kwargs = multi_axes(len(images), **kwargs)
    for idx, ax in enumerate(axes):
        showim(images[0], ax, *args, **kwargs)
    return axes