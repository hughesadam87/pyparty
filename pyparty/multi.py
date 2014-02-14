import numpy as np
import logging
from pyparty.tools import Canvas, ParticleManager
import skimage.morphology as morphology

logger = logging.getLogger(__name__) 

class MultiError(Exception):
    """ """

def multi_mask(img, *names, **kwargs):
    """ """
    
    sort = kwargs.pop('sort', True)
    ignore = kwargs.pop('ignore', 0)
    names = list(names)
    
    # Fix late; requires ignore to be a list
    if hasattr(ignore, '__iter__'):
        raise MultiError("Only can ignore one item at a time")
    
    if ignore == 'black':
        ignore = 0
        if img.ndim == 3:
            ignore = (0,0,0)

    elif ignore == 'white':
        ignore = 255    
        if img.ndim == 3:
            ignore = (1,1,1)
    
    unique = np.unique(img)

    if ignore not in unique:
        logger.warn("Ignore set to %s but was not found on image." % ignore)
    else:
        unique = [v for v in unique if v != ignore]

    # Handle various cases of names/values not being the same
    if names:
        if len(names) == len(unique):
            pass
            
        elif len(names) < len(unique):
            logger.warn("length : %s names provided by %s unique"
                       "labels were found" % (len(names), len(unique)) )              
            names.extend(unique)
            names = names[:len(unique)]
        
        else: #len(names) > len(unique)
            logger.warn("length : %s names provided by %s unique"
                       "labels were found" % (len(names), len(unique)) )  
            
    else:
        names[:] = unique[:]
            
    # Make the mask dict as generator
    out = ((str(names[idx]), (img==v)) for idx, v in enumerate(unique))
                            
    if sort:
        try:
            from collections import OrderedDict
        except ImportError:
            raise ImportError('Mask sorting requires OrderedDict form '
                'python.collection package; package is standard in 2.7 and '
                'higher')
        return OrderedDict(out)
    
    return dict(out)
          
def multi_labels(masksdict, as_canvas=True, prefix=None, neighbors=4, **pmankwargs):
        
    # SUPPOSED TO HANDLE LIST AS WELL    
        
    # Should out be dict, or ordered dict...
    outfcn = type(masksdict)
    
    out = []
    for key, mask in masksdict.items():

        labels = morphology.label(mask, neighbors, background=False)                
        
        if prefix:
            if prefix == True:
                prefix = key
            pmankwargs['prefix'] = prefix               
        
        particles = ParticleManager.from_labels(labels, **pmankwargs)

        if as_canvas:
            canvas = Canvas(particles=particles, rez=mask.shape)
            out.append( (key, canvas) )
        else:
            out.append( (key, particles) )

    return outfcn(out)    