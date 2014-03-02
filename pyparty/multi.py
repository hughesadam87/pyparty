import numpy as np
import logging
import skimage.morphology as morphology
import matplotlib.pyplot as plt
from traits.api import HasTraits, List, Instance, Str
from pyparty.tools import Canvas, ParticleManager
from pyparty.utils import _parse_generator, _parse_ax

logger = logging.getLogger(__name__) 

def multi_mask(img, *names, **kwargs):
    """ 
    """
    sort = kwargs.pop('sort', False)
    astype = kwargs.pop('astype', tuple)
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
    return _parse_generator(out, astype)

class MultiError(Exception):
    """ """
    
class MultiKeyError(MultiError):
    """ """

class MultiCanvas(HasTraits):
    """ Basic container for storing multiple canvases"""
   
    # Probably want these are property lists to prevent user fault
    canvii = List(Instance(Canvas))
    names = List(Str)  # Names are unique, maybe enforce through property
    
    def __init__(self, canvii, names):
        
        self.canvii = canvii
        self.names = names
        
        # General trait change to make sure these are same legnth
        if len(self.canvii) != len(self.names):
            raise MultiError("Names and canvii must have same length")
       
    # HANDLE THIS
    def _names_changed(self, newval):
        if len(newval) != len(self.canvii):
            print "NAMES NOT EQUAL LENGTH"
            
        
    @classmethod
    def from_canvas(cls, canvas, *names):
        """ Split a single canvas into multiple canvas by particle type"""
        # Later, optionall exclude ptypes?
        
        #PARSE NAMES()
        
    @classmethod
    def from_labeled(cls, img, *names, **kwargs):
        """ Labels an image and creates multi-canvas, one for each species
        in the image."""
        
        sort = kwargs.pop('sort', False) 
        astype = kwargs.pop('astype', tuple)
        ignore = kwargs.pop('ignore', 0)        
        masks = multi_mask(img, *names, sort=sort,
                           astype=astype, ignore=ignore)

        for idx, mask in enumerate(masks):
            name = names[idx]
            labels = morphology.label(mask, neighbors, background=False)                                 
            particles = ParticleManager.from_labels(labels, 
                            prefix=names[idx], **kwargs)
            canvii.append(Canvas(particles=particles, rez=mask.shape) )
            
        return cls(canvii=canvii, names=names)          

            
    def to_masks(self, astype=tuple):
        """ Return masks as tuple, list, dict or generator.
        
        Parameters
        ----------
        astype : container type (tuple, list, dict) or None
            Return masks in tuple, list etc... if None, generator. 
        """        
        gen_out = ( (self.names[idx], c.pbinary) for idx, c 
                        in enumerate(self.canvii) )
        return _parse_generator(gen_out, astype)
        
        
    def super_map(self):
        """ """
        NotImplemented

                
    def transmute(self, attr=None, as_type=tuple):
        """ Return a container of names, attributes.  
        
        Parameters
        ----------
        attr : str or None
            Value to be returned paired to name.  Must be valid canvas 
            attribute, or None to return full canvas.

        astype : container type (tuple, list, dict) or None
            Return values in tuple, list etc... if None, generator. 
        """
        if not attr:
            gen_out = ( (self.names[idx], c) for idx, c 
                            in enumerate(self.canvii) )            
        else:
            if hasattr(attr, '__iter__'):
                raise NotImplementedError("Please select a single canvas"
                        "attribute or None to return the entire canvas.")            

            gen_out = ( (self.names[idx], getattr(c, attr)) for idx, c 
                        in enumerate(self.canvii) )

        return _parse_generator(gen_out, as_type)        
        
        
    # Better this way than as functions?
    def pie(self, *chartargs, **chartkwargs):
        """
        *autopct*: [ *None* | format string | format function ]
        If not *None*, is a string or function used to label the
        wedges with their numeric value.  The label will be placed inside
        the wedge.  If it is a format string, the label will be ``fmt%pct``.
        If it is a function, it will be called. 
        """
        attr = chartkwargs.pop('attr', None)
        annotate = chartkwargs.pop('annotate', True)     
        autopct = chartkwargs.get('autopct', 'percent')

        axes, chartkwargs = _parse_ax(*chartargs, **chartkwargs)	
        if not axes:
            fig, axes = plt.subplots()       
        
        if attr is None or attr == 'count':
            attr_list = [len(c) for c in self.canvii]
            attr = 'number' # for title
        else:
            attr_list = [sum(getattr(c, attr)) for c in self.canvii]
        
        chartkwargs.setdefault('labels', self.names) # In annotate?              
        chartkwargs.setdefault('shadow', False)      
        
        # Percentage or true values
        if autopct == 'percent':
            chartkwargs['autopct'] = '%1.1f%%' #Label size and position                       

        elif autopct == 'count':
            chartkwargs['autopct'] = \
                lambda(p): '{:.0f}'.format(p * sum(attr_list) / 100)

        elif autopct == 'both':
            def double_autopct(pct):
                total = sum(attr_list)
                val = int(round(pct*total/100.0,0))
                return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)            
            chartkwargs['autopct'] = double_autopct
            
        axes.pie(attr_list, *chartargs, **chartkwargs)
        # This even worth doing?
        if annotate:
            axes.set_title('%s Distribution' % attr.title())
        return axes   
        
        
    def hist(self, *histargs, **histkwargs):
        """ """
        
        annotate = histkwargs.pop('annotate', True)   
        attr = histkwargs.pop('attr', 'area')  
        histkwargs.setdefault('stacked', True)
        histkwargs.setdefault('label', self.names)     
        
        axes, histkwargs = _parse_ax(*histargs, **histkwargs)	
        if not axes:
            fig, axes = plt.subplots()        
        
        attr_list = [getattr(c, attr) for c in self.canvii]            

        axes.hist(attr_list, *histargs, **histkwargs)         
        if annotate:
            axes.set_xlabel(attr.title()) #Capitalize first letter
            axes.set_ylabel('Counts')
            axes.legend()
        return axes

    # Slicing Interface
    def __getitem__(self, keyslice):
        """ Single name lookup; otherwise single or sliced indicies."""
        if hasattr(keyslice, '__iter__'):
            canvii = [self.canvii[idx] for idx in keyslice]              
            names = [self.names[idx] for idx in keyslice]              
        else:
            if isinstance(keyslice, int):
                idx = keyslice   #keyslice is index 
            else: 
                idx = self.names.index(keyslice)  #keyslice is name              

        canvii, names = self.canvii[idx], self.names[idx]
        
        # Can't attr check canvii because it has iter and len()!
        if not hasattr(names, '__iter__'):
            names, canvii = [names], [canvii]
            
        return MultiCanvas(canvii=canvii, names=names)

    def __delitem__(self, keyslice):
        """ Delete a single name, or a keyslice from names/canvas """        
        NotImplemented

    def __setitem__(self, key, canvas):
        """ """
        idx = self.names.index(key)        
        self.pop(idx)
        self.names.insert(idx, key)
        # Traits checks that this is valid type, right? TEST!
        self.canvii.insert(idx, canvas)

    def summary(self):
        """ """
        # Breakdown of c things in names
        NotImplemented

    def pop(self, idx):
        self.names.pop(idx)
        self.canvii.pop(idx)        
        
    def show(layers):
        """ layered verison of show?  Useful?"""
        # Maybe imshow multiplot ax1, ax2
        NotImplemented
        
    def __repr__(self):
        return super(MultiCanvas, self).__repr__()
        
        
if __name__ == '__main__':
    c1=Canvas.random_circles(n=25)
    c2=Canvas.random_circles(n=75)
    mc = MultiCanvas([c1,c2], ['foo','bar'])
    print mc
    print mc.names, mc.canvii
    print mc.transmute(attr='area', as_type=dict)
    print mc.canvii
#    mc.pie(autopct='both', colors=['r','y'])
#    mc.pie_chart(attr=None)
#    plt.show()
        