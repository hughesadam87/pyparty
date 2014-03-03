import operator
import numpy as np
import logging
import skimage.morphology as morphology
import matplotlib.pyplot as plt
from traits.api import HasTraits, List, Instance, Str

from pyparty.tools import Canvas, ParticleManager
from pyparty.utils import mem_address, _parse_generator, _parse_ax, rgb2uint
import pyparty.tools.arraytools as ptools
from pyparty.config import MAXOUT, _PAD

logger = logging.getLogger(__name__) 

def multi_mask(img, *names, **kwargs):
    """ 
    """
    astype = kwargs.pop('astype', tuple)
    ignore = kwargs.pop('ignore', 0)
    names = list(names)
    
    if img.ndim == 3:
        img = rgb2uint(img, warnmsg=True)
    
    # Labels requires on grayscale; npunique also doesn't play nice w/ rgb
    
    if ignore == 'black':
        ignore = 0
        if img.ndim == 3:
            ignore = (0.0,0.0,0.0)

    elif ignore == 'white':
        ignore = 255    
        if img.ndim == 3:
            ignore = (1.0,1.0,1.0)
    
    unique = ptools.unique(img)

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
    
class MultiCanvas(HasTraits):
    """ Basic container for storing multiple canvases"""
   
    # Probably want these are property lists to prevent user fault
    canvii = List(Instance(Canvas)) #MUST BE LISTS FOR SETTING AND SO FORTH
    names = List(Str)  # Names are unique, maybe enforce through property
    
    def __init__(self, canvii, names):
        
        self.canvii = canvii
        self.names = list(names) #Allow tuple input
        
        # NO DUPLICATES IN NAMES!!
        
        # General trait change to make sure these are same legnth
        if len(self.canvii) != len(self.names):
            raise MultiError("Names and canvii must have same length")
       
    # HANDLE THIS
    def _names_changed(self, newval):
        if len(newval) != len(self.canvii):
            logger.warn("NAMES NOT EQUAL LENGTH")
            
        
    @classmethod
    def from_canvas(cls, canvas, *names):
        """ Split a single canvas into multiple canvas by particle type"""
        # Later, optionall exclude ptypes?
        
        #PARSE NAMES()
        
    @classmethod
    def from_labeled(cls, img, *names, **pmankwargs):
        """ Labels an image and creates multi-canvas, one for each species
        in the image."""
        
        ignore = pmankwargs.pop('ignore', 0)        
        neighbors = pmankwargs.pop('neighbors', 4)
        
        name_masks = multi_mask(img, *names, astype=tuple, ignore=ignore)
        canvii = []
        for (name, mask) in name_masks:
            labels = morphology.label(mask, neighbors, background=False)                                 
            particles = ParticleManager.from_labels(labels, 
                            prefix=name, **pmankwargs)
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
        

    # Is there some magic method to bury this in?  Can't find it.  DO NOT USE SORTED
    def sort(self, inplace=False):
        """ Sort by names.  DO NOT USE sorted(multicanvas)!"""
        z = zip(self.names, self.canvii)
        z.sort(key=operator.itemgetter(0))
        names, canvii = map(list, zip(*z)) #unzip into lists instead of tuple
        if inplace:
            self.names, self.canvii = names, canvii
        else:
            return MultiCanvas(names=names, canvii=canvii)
        
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
        """ Pie chart wrapper to matplotlib.
        
        Parameters
        ----------
        attr : Str or None
            Sum of variable to show in the pie.  Generally particle 
            descriptor (eg area).  If None or "count", the particle
            counts by species are used.
            
        annotate : Bool (True)
            Add auto title to piechart.
            
        autopct : str or fcn or None
            Label of pie slices.  Some built in short cuts include
            "count", "percentage", "both".  Note results in no labels.
            See matplotlib.pie for more.

        usetex : bool (True)
            Label of pie slcies use latex rendering.  If matplotlib.rcparams
            usetex = False, then set this to False.
        
        """
        attr = chartkwargs.pop('attr', None)
        annotate = chartkwargs.pop('annotate', True)     
        usetex = chartkwargs.pop('usetex', True)     
        chartkwargs.setdefault('shadow', False)      
        
        if annotate:
            autopct = chartkwargs.get('autopct', 'percent')
            chartkwargs.setdefault('labels', self.names)                        
        else:
            autopct = chartkwargs.get('autopct', None)
            
            
        
        if 'color' in chartkwargs:
            raise MultiError('Found "color" in kwargs; '
                             'mpl.pie requires "colors".')
        # Could also possible check that colors, if iterable, are correct
        # length as self.names; this results in a non-obvious error
    
        axes, chartkwargs = _parse_ax(*chartargs, **chartkwargs)	
        if not axes:
            fig, axes = plt.subplots()       
        
        if attr is None or attr == 'count':
            attr_list = [len(c) for c in self.canvii]
            attr = 'number' # for title
        else:
            attr_list = [sum(getattr(c, attr)) for c in self.canvii]
                
        # Percentage or true values
        if autopct == 'percent':
            if usetex:
                chartkwargs['autopct'] = r'\bf{%.1f\%%}' #Label size and position                      
            else:
                chartkwargs['autopct'] = '%.1f%%' #Label size and position                                      

        elif autopct == 'count':
            if usetex:
                chartkwargs['autopct'] = \
                    lambda(p): r'\bf{:.0f}'.format(p * sum(attr_list) / 100)
            else:
                chartkwargs['autopct'] = \
                    lambda(p): '{:.0f}'.format(p * sum(attr_list) / 100)                    

        elif autopct == 'both':
            
            def double_autopct(pct):
                total = sum(attr_list)
                val = int(round(pct*total/100.0, 0))
                if usetex:
                    texstring = r'{v} ({p:.1f}\%)'.format(p=pct,v=val)   
                    return r'\bf{%s}' % texstring  
                else:
                    return '{v}\n({p:.1f}%)'.format(p=pct,v=val)                                    

            chartkwargs['autopct'] = double_autopct

        axes.pie(attr_list, **chartkwargs)
        if annotate:
            axes.set_title('%s Distribution' % attr.title())
        return axes   
        
        
    def hist(self, *histargs, **histkwargs):
        """ matplotlib histogram wrapper. """
        
        annotate = histkwargs.pop('annotate', True)   
        attr = histkwargs.pop('attr', 'area')  
        histkwargs.setdefault('stacked', True)
        histkwargs.setdefault('label', self.names)  
        histkwargs.setdefault('bins', 10)
        
        if 'colors' in histkwargs:
            raise MultiError('Found "colors" in kwargs; '
                             'mpl.hist requires "color".')        
        
        axes, histkwargs = _parse_ax(*histargs, **histkwargs)	
        if not axes:
            fig, axes = plt.subplots()        
        
        attr_list = [getattr(c, attr) for c in self.canvii]            

        axes.hist(attr_list, **histkwargs)         
        if annotate:
            axes.set_xlabel(attr.title()) #Capitalize first letter
            axes.set_ylabel('Counts')
            axes.set_title('%s Distribution (%s bins)' % 
                           (attr.title(), histkwargs['bins']) )
            axes.legend()
        return axes

    # Slicing Interface
    def __getitem__(self, keyslice):
        """ Single name lookup; otherwise single or sliced indicies."""
        if hasattr(keyslice, '__iter__'):
            canvii = self.canvii.__getitem__(keyslice)    
            names = self.names.__getitem__(keyslice)
        else:
            if isinstance(keyslice, int) or isinstance(keyslice, slice):
                idx = keyslice   #keyslice is index 
            else: 
                idx = self.names.index(keyslice)  #keyslice is name              

        canvii, names = self.canvii[idx], self.names[idx]
        
        # If single item, return Canvas, else, return new MultiCanvas
        # Canonical; best choice, don't change unless good reason
        if not hasattr(names, '__iter__'):
            return canvii
        else:            
            return MultiCanvas(canvii=canvii, names=names)

    def __delitem__(self, keyslice):
        """ Delete a single name, or a keyslice from names/canvas """        

        if isinstance(keyslice, str):
            idx = self.names.index(keyslice)        
            self.pop(idx)
        else:
            raise NotImplementedError("Deletion only supports single entry")

    def __setitem__(self, name, canvas):
        """ """
        if name in self.names:
            idx = self.names.index(name)        
            self.pop(name)
            self.insert(idx, name, canvas)
        else:
            self.names.append(name)
            self.canvii.append(canvas)
        
    def __contains__(self, name):
        if name in self.names:
            return True
        return False
        
    def __len__(self):
        return len(self.names)

    def summary(self):
        """ """
        # Breakdown of c things in names
        NotImplemented

    def pop(self, idx):
        self.names.pop(idx)
        cout = self.canvii.pop(idx)        
        return cout
    
    def insert(self, idx, name, canvas):
        self.names.insert(idx, name)
        self.canvii.insert(idx, canvas)    
    
    @property
    def _address(self):
        """ Property to make easily accesible by multicanvas """
        return mem_address(super(MultiCanvas, self).__repr__())    
        
    def show(layers):
        """ layered verison of show?  Useful?"""
        # Maybe imshow multiplot ax1, ax2
        NotImplemented
        
    def __repr__(self):
        outstring = "%s at %s: " % \
            (self.__class__.__name__, self._address)     
        Ln = len(self)
        
        if Ln == 0:                
            outstring += '0 canvii'

        elif Ln >= MAXOUT:
            outstring +=  '%s canvii (%s ... %s)' % \
                (Ln, self.names[0], self.names[-1])                        
            
        else:
            outstring += '\n'
            for idx, name in enumerate(self.names):
                c = self.canvii[idx]
                cx, cy = c.rez
                outstring += "%s%s:   Canvas (%s) : %s X %s : %s particles\n" \
                    % (_PAD, name, c._address, cx, cy, len(c))             

        return outstring            


if __name__ == '__main__':
    
    # MAKE SMALL TUTORIAL THIS WAY!
    cs=[]; names=[]
    for i in range(5,10):
        names.append('foo_%s'%i)
        cs.append(Canvas.random_circles(n=i))

    mc = MultiCanvas(cs, names)
    mc.sort()
    print mc
    #print mc.names, mc.canvii
    #print mc.transmute(attr='area', as_type=dict)
    #print mc.canvii
    #mc.pie(attr='area', annotate=True, usetex=False)#, colors=['r','y'])
    ##plt.rc('text', usetex=False)
    ##mc.pie(attr=None, autopct='count')
    
    #plt.show()
        