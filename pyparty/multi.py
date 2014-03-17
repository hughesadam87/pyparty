import operator
import copy
import numpy as np
import itertools
import logging
import skimage.morphology as morphology
import matplotlib.pyplot as plt
from traits.api import HasTraits, List, Instance, Str, Dict, Property

from pyparty.tools import Canvas, ParticleManager
from pyparty.utils import _get_ccycle, mem_address, \
    _parse_generator, _parse_ax, rgb2uint, rand_color, multi_axes
import pyparty.tools.arraytools as ptools
from pyparty.config import MAXOUT, PADDING, ALIGN

logger = logging.getLogger(__name__) 

MAXDEFAULT = 10

def _parse_names(names, default_names):
    """ Boilerplate: user enters *names to overwrite X default names.  
    For example, if user enters 2 names but 5 unique labels in an image 
    are found, and vice versa."""

    default_names = list(default_names)
    names = list(names)

    # Handle various cases of names/values not being the same
    if names:
        if len(names) == len(default_names):
            pass
            
        elif len(names) < len(default_names):
            logger.warn("length : %s names provided but %s unique "
                       "labels were found" % (len(names), len(default_names)) )              
            default_names[0:len(names)] = names[:]
            return default_names
        
        else: #len(names) >= len(default_names)
            logger.warn("length : %s names provided but %s unique "
                       "labels were found" % (len(names), len(default_names)) )     
            return names[0:len(default_names)] 

    else:
        names[:] = default_names[:]
        
    return names


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
        
    names = _parse_names(names, unique)
            
    # Make the mask dict as generator
    out = ((str(names[idx]), (img==v)) for idx, v in enumerate(unique))
    return _parse_generator(out, astype)

class MultiError(Exception):
    """ """
    
class MultiKeyError(MultiError):
    """ """    

class MultiCanvas(HasTraits):
    """ Basic container for storing multiple canvases"""
   
    canvii = List(Instance(Canvas)) 
    names = List(Str)  # Names are unique, maybe enforce through property

    # _mycolors does storage.  mycolors calls an update based on changes
    # to names.  Thus _mycolors may be out of sync if names deleted/changed
    _mycolors = Dict()
    mycolors = Property() 

    def __init__(self, canvii=[], names=[], _mycolors=None):
        """ Canvii and names are public.  _mycolors is private; used mainly
        for copying.  Make sure _mycolors is a new dictionary and not a 
        reference to the original. Calling self.mycolors makes a new object
        through the property interface.  Passing self._mycolors is a 
        bad idea (see copy or __getitem__) for correct syntax.
        """
        self.canvii = canvii
        self.names = list(names) #Allow tuple input
        if _mycolors:
            self._mycolors = _mycolors # Should do more type checking
        
    def _names_changed(self, oldnames, newnames):

        # Check duplicates
        if len(newnames) != len(list(set(newnames))):
            raise MultiError("Duplicate names found")

        # Length of names and cavas
        if len(newnames) != len(self.canvii):
            raise MultiError("Names and canvii must have same length.")
    
    
    def _get_mycolors(self):
        """ Updates self._mycolors to reflect any changes to
        names; eg user may have deleted or popped some"""
        return dict((name, color) for name, color in 
                      self._mycolors.items() if name in self.names)


    def _request_plotcolors(self):
        """ Gives a list of colors based on stored colors AND
        plot color cycle."""
        
        mplcolors = _get_ccycle(upto=len(self))
        colors = []
        
        for idx, name in enumerate(self.names):
            try:
                colors.append(self.mycolors[name])
            except KeyError:
                colors.append(mplcolors.pop(0))               
        return colors

            
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
    
    
    def to_canvas(self, mapnames=False, mapcolors=False):
        """ USE MAX RESOLUTION """
        #WARNING IF REZ NOT SAME
        #USERS CAN SET REZ, BACKGROUND ONE CANVAS MADE
        all_rez = [c.rez for c in self.canvii]
        rezmax, rezmin = max(all_rez), min(all_rez)
        if rezmax != rezmin:
            logger.warn('Resolutions vary on canvii from %s to %s; using '
                        'maximum...' % (rezmin, rezmax))
            
        all_particles = []
        mycolors = self.mycolors #Prevent property recompute each iteration

        for idx, (name, c) in enumerate(self.items()):
            particles = c.particles
            if mapcolors:
                try:
                    mycolor = self.mycolors[name]
                except KeyError:
                    pass
                else:
                    for p in particles:
                        p.color = mycolor
                    
            all_particles.append(particles)
                
            if mapnames:
                for p in particles:
                    p.ptype = name

        # Collapse particle lists of lists down to single list
        flat_particles = itertools.chain.from_iterable(all_particles)
        pout = ParticleManager(plist = flat_particles)                
                    
        return Canvas(rez=rezmax, particles=pout)
        
        

    # Is there some magic method to bury this in?  Can't find it.  DO NOT USE SORTED
    def sort(self, inplace=False):
        """ Sort by names.  DO NOT USE sorted(multicanvas)!"""
        z = zip(self.names, self.canvii)
        z.sort(key=operator.itemgetter(0))
        names, canvii = map(list, zip(*z)) #unzip into lists instead of tuple
        if inplace:
            self.names, self.canvii = names, canvii
        else:
            return MultiCanvas(names=names, canvii=canvii, 
                                _mycolors=self.mycolors)
        
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
        

    # ------------------
    # -- Plotting API
    def show(self, *args, **kwargs):
        """ see _show() docstring """
        kwargs['showstyle'] = 'show'
        return self._show(*args, **kwargs)
    
    def patchshow(self, *args, **kwargs):
        """ see _show() docstring """        
        kwargs['showstyle'] = 'patchshow'
        return self._show(*args, **kwargs)        

    def _show(self, *args, **kwargs):
        """ show() and patchshow()** wrap their respective Canvas methods, 
        so any valid arguments (like colormaps, grids etc...) should just
        work.  In addition, multicanvas show methods have the following
        additional keyword arguments:
        
        Parameters
        ----------
        names: bool (False)
            Show multicanvas names at top of plot
            
        colors: bool (True):
            Map stored color to each particle in subplot.

        **kwargs:
            Any valid splot arg (ncols, figsize etc...) or show/patchshow
            args, such as cmap, grid etc...
                    
        If passing a pre-constructed axes/subplots to mc.show(), it must be 
        as a keyword.  As a positional, it will not work! 
        """
        names = kwargs.pop('names', False)
        colors = kwargs.pop('colors', True)
        showstyle = kwargs.pop('showstyle', 'show')
        
        if showstyle not in ['show', 'patchshow']:
            raise MultiError("showstyle must be show or patchshow, "
                             "not %s" % showstyle)
        
        axes, kwargs = _parse_ax(*args, **kwargs)	
                
        if not axes:
            axes, kwargs = multi_axes(len(self), **kwargs)

        if len(axes) < len(self):
            logger.warn("MultiCanvas has %s canvas, but only %s axes recieved"
                        " in show()" % (len(self), len(axes)))
            upperlim = len(axes)

        else:
            upperlim = len(self)
            
        pcolors = self._request_plotcolors()
        
        for idx in range(upperlim):
            c = self.canvii[idx]
            if colors:
                def cmap(p):
                    p.color = pcolors[idx]
                    return p             
                c = c.pmap(cmap)
            
            if names:
                kwargs['title'] = self.names[idx]

            getattr(c, showstyle)(axes[idx], **kwargs)
        return axes
        

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

        usetex : bool (False)
            Label of pie slcies use latex rendering.  If matplotlib.rcparams
            usetex = True, then set this to True.
        
        """
        attr = chartkwargs.pop('attr', None)
        annotate = chartkwargs.pop('annotate', True)     
        usetex = chartkwargs.pop('usetex', False)     
        chartkwargs.setdefault('shadow', False)      
        metavar = chartkwargs.pop('metavar', None)
        
        # If not colors, set colors based on _colors
        if 'colors' not in chartkwargs:
            chartkwargs['colors'] = self._request_plotcolors()        
        
        if annotate:
            autopct = chartkwargs.get('autopct', 'percent')
            chartkwargs.setdefault('labels', self.names)                        
        else:
            autopct = chartkwargs.get('autopct', None)
    
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
            if metavar:
                attr = metavar
            axes.set_title('%s Distribution' % attr.title())
        return axes   

        
    def hist(self, *histargs, **histkwargs):
        """ Matplotlib histogram wrapper. 
    
        Parameters
        ----------
        **annotate:** bool - True
            Add general legend, title, axis labels. 
            
        **attr:** str - "area"
            Particle attribute for data.  (Also a pie chart keyword).
            
        **xlim:** range(start, stop) - None
            Shortcut to set x-limits.  If **xlim=auto**, absolute min and 
            absolute max of data will be used.  
            This crops data AND sets axis limits; to only change plot axes,
            use *axes.set_xlim()*.
            
        Notes
        -----
        We see that the **annotate** option adds a legend, title and axis 
        labels.  The default attribute of the histogram is *area*, 
        corresponding to the **attr** kwargs.  All other valid matplotlib 
        histogram kwargs should work.
        """
        
        annotate = histkwargs.pop('annotate', True)   
        attr = histkwargs.pop('attr', 'area')  
        histkwargs.setdefault('stacked', True)
        histkwargs.setdefault('label', self.names)  
        histkwargs.setdefault('bins', 10)
        metavar = histkwargs.pop('metavar', None)    
        xlim = histkwargs.pop('xlim', None)     
        
        #MPL api asymmetry with pie
        if 'colors' in histkwargs:
            histkwargs['color'] = histkwargs.pop('colors')        
            
        # If not colors, set colors based on _colors
        if 'color' not in histkwargs:
            histkwargs['color'] = self._request_plotcolors()
            
        axes, histkwargs = _parse_ax(*histargs, **histkwargs)	
        if not axes:
            fig, axes = plt.subplots()        
        
        # Get list of arrays for descriptor, slice if xlim
        attr_list = [getattr(c, attr) for c in self.canvii]   
        if xlim:
            if xlim == 'auto':
                xi, xf = min(map(min, attr_list)), max(map(max, attr_list))
            else:
                xi, xf = xlim     
            
            attr_list = [array[(array >= xi) & (array <= xf)] 
                         for array in attr_list]
            axes.set_xlim(xi, xf)

        # Validate attr_list for empy arrays; avoid ambiguous mpl error
        for idx, array in enumerate(attr_list):
            if len(array) == 0:
                raise MultiError('Empty array returned for "%s" attribute'
                    ' on "%s" Canvas.' % (attr, self.names[idx]) )
                
        axes.hist(attr_list, **histkwargs)         
        
        if annotate:
            axes.set_xlabel(attr.title()) #Capitalize first letter
            axes.set_ylabel('Counts')
            if metavar:
                attr = metavar
            axes.set_title('%s Distribution (%s bins)' % 
                           (attr.title(), histkwargs['bins']) )
            axes.legend()
        return axes

    def summary(self):
        """ """
        # Breakdown of c things in names
        NotImplemented

    def copy(self): #, **kwargs):
        """ Copy multicanvas """
        names, canvii = [], []
        names[:] = self.names[:]
        canvii[:] = self.canvii[:]
        return MultiCanvas(names=names, canvii=canvii, 
                             _mycolors = self.mycolors)
        
    def pop(self, idx):
        self.names.pop(idx)
        cout = self.canvii.pop(idx)        
        return cout
    
    
    def append(self, name, canvas):
        if name in self.names:  #Trait handler should handle this
            raise MultiError("%s name already exists" % name)
        self.names.append(name)
        self.canvii.append(canvas)


    def insert(self, idx, name, canvas):
        self.names.insert(idx, name)
        self.canvii.insert(idx, canvas)    
              
    
    def set_colors(self, *colors, **kwcolors):
        """ Set colors through a list or name:color mapping.  If no
        arguments, colors are purged.  'fillnull' kwarg used if
        color contains null values; replaced by random selection.
        """
        fillnull = kwcolors.pop('fillnull', False)
                
        if colors and kwcolors:
            raise MultiError("Please set colors from list or keyword, "
                             "but not both.")
        elif not colors and not kwcolors:
            self._mycolors = {}
            return
        
        if kwcolors:
            names, colors = zip(*kwcolors.items())
            if fillnull:
                colors = [c if c else rand_color(style='hex') for c in colors]                
            
            for idx, name in enumerate(names):
                if name not in self.names:
                    raise MultiKeyError('"%s" not found' % name)
                self._mycolors[name] = colors[idx]

        else: #if *colors
            if fillnull:
                colors = [c if c else rand_color(style='hex') for c in colors]
        
            for idx, name in enumerate(self.names):
                self._mycolors[name] = colors[idx]


    def rename(self, oldname, newname):
        """ Change name; preserves color assignment. """

        idx = self.names.index(oldname)
        self.names.pop(idx)
        self.names.insert(idx, newname)
        
        try:
            self._mycolors[newname] = self._mycolors[oldname]
        except KeyError:
            pass        


    def set_names(self, *names, **namemap):
        if names and namemap:
            raise MultiError("Please set names from list or keyword, "
                             "but not both.")
        
        elif not names and not namemap:
            return
        
        if namemap:
            for oldname, newname in namemap.items():
                self.rename(oldname, newname)
                
        else:  #if *names
            if len(names) > len(self.names):
                logger.warn("%s names passed; only %s canvii stored" %
                            (len(names), len(self.names)))
                            
            for idx, newname in enumerate(names):
                oldname = self.names[idx]
                self.rename(oldname, newname)   
                
    @property
    def mycolors(self):
        """ Returns current colors set by user; new dict to avoid reference
        errors."""
        return dict((k,v) for k, v in self._mycolors.items())
        

    @property
    def _address(self):
        """ Property to make easily accesible by multicanvas """
        return mem_address(super(MultiCanvas, self).__repr__())   


    # ---------------
    # Magic Methods  
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
            return MultiCanvas(canvii=canvii, names=names, 
                               _mycolors=self.mycolors)

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
            self.pop(idx)
            self.insert(idx, name, canvas)
            
        else:
            self.names.append(name)
            self.canvii.append(canvas)
        
        
    def __contains__(self, name_or_canvas):
        """ If name in self.names or canvas in self.canvii"""
        if isinstance(name_or_canvas, str):
            inspect = self.names
        elif isinstance(name_or_canvas, Canvas):
            inspect = self.canvii            
        else:
            raise MultiError("Invalid type %s; please enter a "
                "name (str) or Canvas" % type(name_or_canvas) )
            
        if name_or_canvas in inspect:
            return True
        else:
            return False
        
        
    def __len__(self):
        return len(self.names)
    
    def __iter__(self):
        """ Return names like dictionary"""
        return self.canvii.__iter__()
    
    def items(self):
        return zip(self.names, self.canvii)
    

    def __repr__(self):
        outstring = "%s (%s): " % \
            (self.__class__.__name__, self._address)     
        Ln = len(self)
        
        if Ln == 0:                
            outstring += 'EMPTY'

        elif Ln >= MAXOUT:
            outstring +=  '%s canvii (%s ... %s)' % \
                (Ln, self.names[0], self.names[-1])                        
            
        else:
            SEP_CHARACTER = '-'
            _NEWPAD =  (PADDING-1) * ' '  # REduce CONFIG PADDING by one space
            just_fcn = {
                'l': str.ljust,
                'r': str.rjust,
                'c': str.center}[ALIGN]            
            
            outstring += '\n'
            outrows=[]
            for idx, name in enumerate(self.names):
                c = self.canvii[idx]
                cx, cy = c.rez
                
                col1 = '%s%s' % (_NEWPAD, name)
                col2 = 'Canvas (%s) : %s X %s : %s particles' % \
                    (c._address, cx, cy, len(c))
                outrows.append([col1, SEP_CHARACTER, col2])
         
            widths = [max(map(len, col)) for col in zip(*outrows)]
            outstring = outstring + '\n'.join( [ _NEWPAD.join((just_fcn(val,width) 
                for val, width in zip(row, widths))) for row in outrows] )

        return outstring

    # Class methods
    # ------------
    @classmethod
    def from_canvas(cls, canvas, *names):
        """ Split a single canvas into multiple canvas by particle type"""
        # PARSE NANMES()

        ptypes = canvas.ptypes
        names = _parse_names(names, ptypes)
        
        canvii = []
        for p in ptypes:
            canvii.append(canvas.of_ptypes(p))
        return cls(names=names, canvii=canvii)          

    @classmethod
    def from_labeled(cls, img, *names, **pmankwargs):
        """ Labels an image and creates multi-canvas, one for each species
        in the image."""
        
        ignore = pmankwargs.pop('ignore', 0)        
        neighbors = pmankwargs.pop('neighbors', 4)
        maximum = pmankwargs.pop('maximum', MAXDEFAULT)
        
        name_masks = multi_mask(img, *names, astype=tuple, ignore=ignore)
        if len(name_masks) > maximum:
            raise MultiError("%s labels found, exceeding maximum of %s"
                " increasing maximum may result in slowdown" % maximum)
        
        canvii = []
        for (name, mask) in name_masks:
            labels = morphology.label(mask, neighbors, background=False)                                 
            particles = ParticleManager.from_labels(labels, 
                            prefix=name, **pmankwargs)
            canvii.append(Canvas(particles=particles, rez=mask.shape) )
            
        return cls(canvii=canvii, names=names)          
    
if __name__ == '__main__':
    c1 = Canvas.random_circles(n=100, pcolor='yellow')
    c2 = Canvas.random_triangles(n=100, pcolor='red')
    c3 = c1+ c2
    mc =  MultiCanvas.from_canvas(c3, 'dimer', 'trimer')
    mc.set_colors('r','g')

    mcout = mc[0:2]
    mc.hist(xlim=(2000,50000), attr='equivalent_diameter', bins=30)
    plt.show()
