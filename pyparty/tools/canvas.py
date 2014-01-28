from __future__ import division
import os
import os.path as op
import logging 
import copy
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from traits.api import HasTraits, Array, Instance, Property, Float, Tuple, \
     Int, Function

import skimage.io
import skimage.measure as measure
import skimage.morphology as morphology

import pyparty.background.bg_utils as bgu
from pyparty.shape_models.abstract_shape import ParticleError
from pyparty.tools.thresholding import choose_thresh
from pyparty.tools.manager import ParticleManager, concat_particles
from pyparty.tools.grids import Grid, CartesianGrid
from pyparty.utils import coords_in_image, where_is_particle, to_normrgb, \
     any2rgb, crop, _parse_ax, _parse_path, mem_address, rgb2uint
from pyparty.config import BGCOLOR, BGRES, GRIDXSPACE, GRIDYSPACE, _PAD, \
     GCOLOR, THRESHDEF

# Ensure colors are correctly mapped
BGCOLOR, GCOLOR = to_normrgb(BGCOLOR), to_normrgb(GCOLOR)

logger = logging.getLogger(__name__) 

def rint(x): return int(round(x,0))

#def inplace(method):
    #""" Thought I could decorate methods that have inplace keyword, but turned
    #out to be more than I bargained for.  Decorator syntax is fine, but not correct.
    #"""
    #methodname = method.__name__
    ##variables = method.func_code.co_varnames
    #@wraps(method)
    #def wrapper(obj, *args, **kwargs):
        #print 'method is', method, type(obj), type(method)
        #inplace = kwargs.pop('inplace', False)
        #if inplace:
            #print 'in inplace', args, kwargs
            #getattr(obj, methodname)(*args, **kwargs)
        #else:
            #print 'not inplace. obj is', obj
            #new = Canvas(background=obj.background, particles=obj._particles,
                         #res=obj.rez)
            #return getattr(new, methodname)(*args, **kwargs)
#    return wrapper


def concat_canvas(c1, c2, bg_resolve='c2', **particle_args):
    """ Adds two canvas objects under various conditions """
    bg_resolve = bg_resolve.lower()

    # Choose output background
    if bg_resolve == 'merge':
        raise NotImplementedError
    elif bg_resolve == 'c1':
        bgout = c1.background
        rezout = c1.rez  #UNTESTED
        gridout = c1.grid
    elif bg_resolve == 'c2':
        bgout = c2.background        
        rezout = c2.rez
        gridout = c2.grid
    else:
        raise CanvasAttributeError('"bg_resolve" invalid; must be %s' % bg_valid)

    _pout = concat_particles(c1._particles, c2._particles, **particle_args)
    return Canvas(background=bgout, particles=_pout, rez=rezout, grid=gridout)

class CanvasError(Exception):
    """ """    

class CanvasAttributeError(CanvasError):
    """ """    
    
class CanvasPlotError(CanvasError):
    """ """    
    
class Canvas(HasTraits):
    """  """
    image = Property(Array) #chaching doesnt work

    # ALL INTERNAL REFERENCES SHOULD GO TO _PARTICLES
    particles = Property(Instance(ParticleManager, depends_on='_particles'))
    _particles = Instance(ParticleManager)

    # Background is only a property because for _set validation...
    background = Property()
    _background = Array
        
    # Use this as a listener for grid
    _resolution = Tuple(Int, Int)

    grid = Instance(CartesianGrid)

    def __init__(self, particles=None, background=None, rez=None, grid=None, 
                 _threshfcn=None): #No other traits
        """ Load with optionally a background image and instance of Particle
            Manager"""
        
        if not particles:
            particles = ParticleManager()
        self._particles = particles

        # Set bg before resolution
        self._resolution = BGRES
        if background is None:
            self.reset_background() #sets default color/resolution    
        else:
            self.set_bg(background, keepres=rez, inplace=True) 
            
        if not grid:
            self.reset_grid()
        else:
            self.grid = grid
            
        # _threshfcn through __init__ only really for Cavnas.copy(); not users
        if _threshfcn is None:
            self.set_threshfcn(THRESHDEF)
        else:
            self._threshfcn = _threshfcn
       
    def set_threshfcn(self, fcn_or_string, *args, **kwargs):
        """ Set a binarization function. """
        if isinstance(fcn_or_string, str):
            # Update later
            if args:
                raise CanvasError('Please use keyword args for threshold function')
            self._threshfcn = choose_thresh(fcn_or_string, **kwargs)
            self._threshtype = fcn_or_string
        else:
            self._threshfcn = functools.partial(fcn_or_string, *args, **kwargs)
            self._threshtype = fcn_or_string.__name__
        
    @property
    def threshfcn(self):
        return self._threshtype
    
    @threshfcn.setter
    def threshfcn(self, val):
        raise CanvasAttributeError('Please set thresh function through "set_threshfcn()*')
    
    @threshfcn.setter
    def threshfcn(self):
        raise CanvasError('Please use "set_threshfcn(fcn/str, *args, **kwargs)" to set the binary '
            'function.')

    # Public Methods
    # -------------              
    def reset_background(self):
        """ Restore default background image; restores default RES, redraws
            particles over it."""

        self.rez = BGRES  #must be set first
        self._background = self.color_background        
        self._bgstyle = 'default'
                
                
    def reset_grid(self):
        """ New grid of default x/y spacing; rez is optional """
        xs = ys = 0
        xe, ye = self.rez
        self.grid = CartesianGrid(ystart=ys, xstart=xs, yend=xe, xend=ye,
                              xspacing=GRIDYSPACE, yspacing = GRIDXSPACE)
        

    def clear_canvas(self):
        """ Background image to default; removes ALL particles."""

        self.clear_particles()
        self.reset_background()
        self.reset_grid()

    def clear_particles(self):
        """ Clears all particles from image."""

        self._particles.plist[:] = []

    #   @inplace
    def pmap(self, fcn, *fcnargs, **fcnkwargs):
        """ Maps a function to each particle in ParticleManger; optionally
            can be done in place"""

        #self._particles.map(fcn, *fcnargs, **fcnkwargs)
        inplace = fcnkwargs.pop('inplace', False)
        if inplace:
            self._particles.map(fcn, *fcnargs, **fcnkwargs)
        else:
            cout = Canvas.copy(self)
            cout._particles.map(fcn, *fcnargs, **fcnkwargs)
            return cout    


    def pixelmap(self, fcn, axis=0, *fcnargs, **fcnkwargs):
        """ Image mapper (np.apply_along_axis)

            fcn must be 1d!

            Notes
            -----
            Calls numpy.apply_along_axis, which doesn't acces
            keyword arguments.
        """
        return np.apply_along_axis(fcn, axis, self.image, *fcnargs)


#http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.label
#    @inplace
    def from_labels(self, inplace=False, neighbors=4, bgout=None,
                    bgexclude=None, binary=True, pbinary=True, **pmangerkwds):
        """ Get morphological labels from gray or binary image. """

        if binary:
            if pbinary:
                image = self.pbinary #PBINARY NOT BINARYIMAGE
                if len(self.particles) == 0:
                    logger.warn('from_labels() recieved "pbinary=True", but '
                        'no particles are stored.  Use "pbinary=False" to '
                        'use implicit thresholding function.')
            else:
                image = self.binaryimage
        else:
            image = self.grayimage
            logger.warn('Labels from grayimage can be very slow (fix coming)')
            
        if bgexclude is None: # scikit api doesn't accept None
            labels = morphology.label(image, neighbors)
            
        else:
            # Parse various cases
            if bgexclude == 'w' or bgexclude == 'white':
                if binary:
                    bgexclude = 1
                else:
                    bgexclude = 255                    
            elif bgexclude == 'b' or bgexclude == 'black':
                bgexclude = 0
                        
            labels = morphology.label(image, neighbors, background=bgexclude)

        pout = ParticleManager.from_labels(labels, **pmangerkwds)

        if inplace:
            self._particles = pout
        else:
            cout = Canvas.copy(self)
            cout.particles = pout
            if bgout is not None:
                cout.background = bgout
            return cout

    def patchshow(self, *args, **kwargs):
        """ ...
        args/kwargs include alpha, edgecolors, linestyles 

        Notes:
        Matplotlib API is setup that args or kwargs can be entered.  Order is
        important for args, but the correspond to same kwargs.
        """

        axes, kwargs = _parse_ax(*args, **kwargs)	
        
        title = kwargs.pop('title', None)        
        bgonly = kwargs.pop('bgonly', False)

        grid = kwargs.pop('grid', False)
        gcolor = kwargs.pop('gcolor', None)
        gstyle = kwargs.pop('gstyle', None)
        gunder = kwargs.pop('gunder',False)
        pmap = kwargs.pop('pmap', None)
        
        alpha = kwargs.get('alpha', None)
        edgecolor = kwargs.get('edgecolor', None)
        linewidth = kwargs.get('linewidth', None)
        linestyle = kwargs.get('linestyle', None)

        # Some keywords to savefig; not all supported
        save = kwargs.pop('save', None)
        dpi = kwargs.pop('dpi', None)
        bbox_inches = kwargs.pop('bbox_inches', None)
        
        # GET NOT POP
        cmap = kwargs.get('cmap', None)       
        
        # grid defaults
        if gcolor or gunder or gstyle and not grid:
            grid = True
            
        # If user enters gcolor/gstyle, assume default grid
        if grid and not gcolor:
            gcolor = GCOLOR       
            
        if grid and not gstyle:
            gstyle = 'solid'        
        
        # Corner case, don't touch
        if pmap and cmap and bgonly:
            bgonly = False
        
        if bgonly and not cmap: 
            raise CanvasPlotError('"bgonly" is only valid when a colormap is' 
            ' passed.')

        if cmap:
            bg = self.graybackground

        else:
            bg = self.background
            
        #Overwrite axes image
        if not axes:
            fig, axes = plt.subplots()
        else:
            axes.images=[]
        # DONT PASS ALL KWARGS
        axes.imshow(bg, cmap=cmap)

        # FOR PATICLES IN IMAGE ONLY.
        in_and_edges = self.pin + self.pedge
        
        # PARTICLE FACECOLOR, ALPHA and other PATCH ARGS
        # http://matplotlib.org/api/artist_api.html#matplotlib.patches.Patch
        patches = [p.particle.as_patch(facecolor=p.color, alpha=alpha, 
                    edgecolor=edgecolor, linestyle=linestyle, linewidth=linewidth)
                   for p in in_and_edges]

        # If no particles or grid, just pass to avoid deep mpl exceptiosn
        if patches or grid:
            if patches:
                if pmap:
                    kwargs['cmap'] = pmap               
                if 'cmap' in kwargs and not bgonly:
                    ppatch = PatchCollection(patches, **kwargs) #cmap and Patch Args
                    ppatch.set_array(np.arange(len(patches)))        
        
                # Use settings passed to "patches"
                else:                                                  
                    ppatch = PatchCollection(patches, match_original=True, **kwargs) #
        
            # Grid under particles
            if gunder:
                axes.add_collection(self.grid.as_patch(
                    edgecolors=gcolor, linestyles=gstyle))
                if patches:
                    axes.add_collection(ppatch)
            # Grid over particles
            else:
                if patches:
                    axes.add_collection(ppatch)          
                if grid:
                    axes.add_collection(self.grid.as_patch(
                        edgecolors=gcolor, linestyles=gstyle))    
        
        if title:
            axes.set_title(title)        

        if save:
            path = _parse_path(save)
            plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        
        return axes 


    #http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
    def show(self, *args, **kwargs):
        """ Wrapper to imshow.  Converts to gray to allow color maps.
        
        Notes
        -----
        Differs from patchshow in that the collective image (bg, grid, particles)
        is a series of masks, so they have to be drawn onto a single ndarray and 
        then plotted.  Sicne patchshow is writing patchs, it can plot the background
        separate from the grid and particles, which is slightly easier"""

        # This will pull out "ax", leaving remaing args/kwargs
        axes, kwargs = _parse_ax(*args, **kwargs)
        title = kwargs.pop('title', None)
        save = kwargs.pop('save', None)
        bgonly = kwargs.pop('bgonly', False)
        
        grid = kwargs.pop('grid', False)
        gcolor = kwargs.pop('gcolor', None)
        gunder = kwargs.pop('gunder', False)
        gstyle = kwargs.pop('gstyle', None) #NOT USED
        
        if 'pmap' in kwargs:
            raise CanvasPlotError('"pmap" is only valid for patchshow() method')
        
        # Get the background
        if bgonly: 
            if not kwargs['cmap']:
                raise CanvasPlotError('"bgonly" is only valid when a colormap is' 
                ' passed.')
            bg = kwargs['cmap'](self.graybackground)[... , :3]
            del kwargs['cmap']

        else:
            bg = self.background
                              
        # grid overlay
        if gcolor or gunder and not grid:
            grid = True
            
        # If user enters gcolor, assume default grid
        if grid and not gcolor:
            gcolor = GCOLOR
        
        # Map attributes from grid (centers, corners, grid)
        gattr = np.zeros(bg.shape).astype(bool)  #IE pass
        if grid:
            if not gcolor:
                gcolor = GCOLOR
            if grid == True:
                grid = 'grid'

            # Validate grid keyword
            try:
                gattr=getattr( self.grid, grid.lower() )
            except Exception:
                raise CanvasPlotError('Invalid grid argument, "%s".  Choose from:  '
                    'True, "grid", "centers", "corners", "hlines", "vlines"' 
                    % grid)            
            gcolor = to_normrgb(gcolor)
                                  
        #Draw grid over or under?
        if gunder:
            bg[gattr] = gcolor
            image = self._draw_particles(bg)
        else:
            image = self._draw_particles(bg)
            image[gattr] = gcolor
            
        # GRAY CONVERT
        if 'cmap' in kwargs:
            image = rgb2uint(image)
            
        # Matplotlib
        if axes:
            axes.imshow(image, **kwargs)
        else:     
            axes = plt.imshow(image, **kwargs).axes        

        if title:
            axes.set_title(title)
            
        # Save image array (with grid)
        if save:
            path = _parse_path(save)
            skimage.io.imsave(path, image)
                        
        return axes 

    def _draw_particles(self, image):
        """ Draws particles over any image (ie background, background+grid """
        for p in self._particles:
            rr_cc, color = p.particle.rr_cc, p.color 
            rr_cc = coords_in_image(rr_cc, image.shape)
            image[rr_cc] = color
        return image 


    def _get_image(self):
        """ Creates image array of particles.  Tried fitting to a property or
        Traits events interface to control the caching, but manually choosing
        cache points proved to be easier."""

        # self.background is always a new array; otherwise would use np.copy
        return self._draw_particles(self.background)
        

    # Image Attributes Promoted
    # ------------------
    @property
    def shape(self):
        """ Avoid recomputing image"""
        return (self.rx, self.ry, 3)

    @property
    def ndim(self):
        """ Avoid recomputing image for this purpose """
        return len(self._background)

    @property
    def dtype(self):
        # Same dtype as image
        return self._background.dtype

    @property
    def grayimage(self):
        """ Collapse multi-channel, scale to 255 (via ubyte) """
        return rgb2uint(self.image)

    @property
    def binaryimage(self):
        return self._threshfcn(self.grayimage)
        
    @property
    def graybackground(self):
        return rgb2uint(self.background)

    @property
    def binarybackground(self):
        return self._threshfcn(self.graybackground)

    @property
    def color_background(self):
        """ Generate a colored background at current resolution """
        return bgu.from_color_res(BGCOLOR, self.rx, self.ry)         

    # Promote most common grid attributes    
    @property
    def gcenters(self):
        return self.grid.centers
    
    @property
    def gcorners(self):
        return self.grid.corners
    
    @property
    def ghlines(self):
        return self.grid.hlines
    
    @property
    def gvlines(self):
        return self.grid.vlines      
    
    def gpairs(self, attr):
        return self.grid.pairs(attr)
    
    #GRID LISTENER
    def __resolution_changed(self):        
        # BACKWARDS BECAUSE GRID IS RELATIVE INVERSE
        if self.grid:
            self.grid.xend = self.ry 
            self.grid.yend = self.rx 

    
    # Image Resolution
    @property
    def rez(self):
        """ Resoloution; since x,y is wonky on image, rx really is y dim"""
        return self._resolution

    @rez.setter
    def rez(self, rez):
        rx, ry = rint(rez[0]), rint(rez[1])
        self._resolution = rx, ry
        

    @property
    def rx(self):
        return self.rez[0]

    @rx.setter
    def rx(self, rx):
        self._resolution = ( int(rx), self._resolution[1] )

    @property
    def ry(self):
        return self.rez[1]    

    @ry.setter
    def ry(self, ry):
        self._resolution = ( self._resolution[0], int(ry) )    

    @property
    def pbinary(self):
        """ Returns boolean mask of all particles IN the image, with 
            background removed."""      
        # Faster to get all coords in image at once since going to paint white
        out = np.zeros( self.rez, dtype=bool )
        if self.particles:
            rr_cc = coords_in_image( self._particles.rr_cc_all, self.rez)
            out[rr_cc] = True
        return out  


    def _whereis(self, choice='in'):
        """ Wraps utils.where_is_particles for all three possibilities """
        return [p.name for p in self._particles 
                if where_is_particle(p.rr_cc, self.rez) == choice]


    @property
    def pin(self):
        """ Returns all particles appearing FULLY in the image"""
        return self._particles[self._whereis('in')]

    @property
    def pedge(self):
        """ Returns all particles appearing PARTIALLY in the image"""
        return self._particles[self._whereis('edge')]

    @property
    def pout(self):
        """ Returns all particles appearing FULLY outside the image"""
        return self._particles[self._whereis('out')]    


    @property
    def pixcount(self):
        """ Counts # pixels in image """
        l, w = self.rez
        return int(l * w)

    @property
    def pixarea(self):
        """ What's the best way to get this? """
        return float(np.sum(self.pbinary)) / self.pixcount

    @property
    def pixperim(self):
        """ Wraps measure.perimeter to estimate total perimeter of 
            particles in binary image."""
        return skimage.measure.perimeter(self.pbinary, neighbourhood=4)     

    # Trait Defaults / Trait Properties
    # ---------------------------------
    def _get_particles(self):
        """ Return a NEW INSTANCE of particles manager (ie new particles instead
        of in-memory references)"""
        return ParticleManager(plist=self._particles.plist, 
                               fastnames=self._particles.fastnames)

    def _set_particles(self, particles):
        """ Make a copy of the particles to avoid passing by reference. 
        Note this is implictly controlled by _COPYPARTICLES in config.
        """
        self._particles = ParticleManager(particles.plist, particles.fastnames)    

    def _set_image(self):
        raise CanvasError('Image cannot be set; please make changes to particles'
                          ' and/or background attributes.')

    # BACKGROUND RELATED
    # ------------------

#    @inplace
    def set_bg(self, bg, keepres=False, inplace=False):
        """ Public background setting interface. """
        #print 'IN SET BG', bg, keepres

        #oldres = self.rez                
        #self._update_bg(bg)    

        #if keepres:
            #if keepres == True:
                #self.rez = oldres
            #else:
                #self.rez = keepres

        #else:
            #self.rez = self._background.shape[0:2]   	

        if inplace:
            cout = self
        else:
            cout = Canvas.copy(self)  

        oldres = cout.rez                
        cout._update_bg(bg)    

        if keepres:
            if keepres == True:
                cout.rez = oldres
            else:
                cout.rez = keepres

        else:
            cout.rez = cout._background.shape[0:2]   

        if not inplace:
            return cout


    def zoom_bg(self, *coords, **kwds):
        """ Zoom in on current image and set the zoomed background as the 
        background of a new canvas.  Note that because indicies will always 
        resume at 0, particle positions will not maintain their relative 
        positions.
        """
        # avoid set_bg() because uses size of coords, not coords itself
        inplace = kwds.pop('inplace', False)
        autogrid = kwds.pop('autogrid', True)
        
        if inplace:
            cout = self
        else:
            cout = Canvas.copy(self)          

        cout._background = crop(cout._background, coords)
        cout.rez = cout._background.shape[0:2]        
        if autogrid:
            xmag = self.rx / float(cout.rx) 
            ymag = self.ry / float(cout.ry) 
            
            cout.grid.xdiv = rint(self.grid.xdiv / xmag)
            cout.grid.ydiv = rint(self.grid.ydiv / ymag)

        if not inplace:
            return cout        


    def _get_background(self):
        """ Crop or extend self._background based on self.rx, ry.  Always 
        returns a new object to avoid accidental refernce passing."""
        bgx, bgy = self._background.shape[0:2]
        rx, ry = self.rez
        if bgx == rx and bgy == ry:
            return np.copy(self._background)

        def _smaller(idx1, idx2):
            if idx1 < idx2:
                return idx1
            return idx2

        xs, ys = _smaller(rx, bgx), _smaller(ry, bgy)

        out = np.empty( (rx, ry, 3) )
        out[:] = BGCOLOR #.fill only works with scalar	
        out[:xs, :ys] = self._background[:xs, :ys] #no copy needed	
        return out

    def _set_background(self, bg):
        """ Set background and use new resolution if there is one """
        self.set_bg(bg, keepres=False, inplace=True)

    # REDO THIS WITH COLOR NORM AND STUFF!  Also, dtype warning?
    def _update_bg(self, background):
        """ Parses several valid inputs including: None, Path, Color (of any 
        valid to_normRGB() type, ndarray, (Color, Resolution)."""

        if background is None:
            self._background = self.color_background
            self._bgstyle = 'default'            
            
        elif isinstance(background, Grid):
            self._background = bgu.from_grid(background)
            self._bgstyle = 'grid'

        elif isinstance(background, np.ndarray):
            self._background = background
            self._bgstyle = 'ndarray'

        # colorstring or hex
        elif isinstance(background, basestring):
            self._background = bgu.from_string(background, self.rx, self.ry)
            self._bgstyle = 'file/colorstring'            

        # If not array, color is assume as valid to_norm_rgb(color) arg
        # It will raise its own error if failure occurs
        else:
            self._background = bgu.from_color_res(background, self.rx, self.ry)            
            self._bgstyle = 'color'

        # Float-color convert array       
        self._background = any2rgb(self._background, 'background')

        # IMAGE IS CACHED AT END OF _set_bg()

    # Delegate dictionary interface to ParticleManager
    # -----------
    def __getitem__(self, keyslice):
        """ Employs particle manager interface; however, returns single entry
            as a list to allow slicing directly into get_item[]"""
        return self._particles.__getitem__(keyslice)

        #IF WANT TO RETURN CANVAS ALWAYS SEE BELOW
        #pout = self._particles.__getitem__(keyslice)
        #return Canvas(background=self.background, particles=pout, rez=self.rez)

    def __delitem__(self, keyslice):
        return self._particles.__delitem__(keyslice)    

    def __setitem__(self, key, particle):
        return self._particles.__setitem__(key, particles)

    def __getattr__(self, attr):
        """ Defaults to particle manager """
        try:
            return getattr(self._particles, attr)
        except ParticleError:
            raise CanvasAttributeError('"%s" could not be found on %s, '
                'underlying manager, or on one-or multiple of the '
                'Particles' % (attr, self.__class__.__name__) )

    def __iter__(self):
        """ Iteration is blocked """
        raise CanvasError("Iteration on canvas is ambiguous.  Iterate over "
                          "canvas.image or canvas.particles")

    def __repr__(self):
        _bgstyle = self._bgstyle #REPLACE
        res = '(%s X %s)' % (self.rx, self.ry ) 
        address = mem_address(super(Canvas, self).__repr__())
        
        g=self.grid
        gridstring = "%sxygrid     -->  (%sp X %sp) : (%.1f X %.1f) [pix/tile]" \
            % (_PAD, g.xdiv, g.ydiv, g.xspacing, g.yspacing)
        
                                                         # For sublcassing
        outstring = "%s at %s:\n" % (self.__class__.__name__, address)

        outstring += "%sbackground -->  %s : %s\n" % (_PAD, res, _bgstyle) 

        outstring += "%sparticles  -->  %s particles : %s types\n" % (_PAD, \
            len(self._particles), len(self._particles.ptype_count))
        outstring += gridstring
        
        return outstring

    def __len__(self):
        return self._particles.__len__()

    # Arithmetic Operation
    # --------------------

    def __add__(self, c2):
        return concat_canvas(self, c2, bg_resolve='c2')
    
    def __sub__(self, c2):
        raise CanvasError("%s does not support subtraction" % 
              self.__class__.__name__)
    
    
    # Class methods
    # ------------
    @classmethod
    def copy(cls, obj):
        """ Returns a copied canvas object. """
        newgrid = copy.copy(obj.grid)
        return cls(background=obj.background, particles=obj._particles, 
                        rez=obj.rez, grid=newgrid, _threshfcn=obj._threshfcn)        

    
    # Extend to polygons/other partciles in future
    # May want to refactor into particle manager method actually
    @classmethod
    def random_circles(cls, n=50, rmin=5, rmax=50, background=BGCOLOR, pcolor=None):
        """ Return a canvas populated with n randomly positioned circles.  
        Radius ranges vary randomly between 5 and 50."""
        import random as r
                       
        particles = ParticleManager()            
        # Randomize particle centers within the image default dimensions
        for i in range(n):
            cx, cy = r.randint(0, BGRES[0]), r.randint(0, BGRES[1])
            radius = r.randint(rmin,rmax)
            particles.add('circle', center=(cx,cy), radius=radius, 
                          color=to_normrgb(pcolor))
        
        # Use default resolution and grid
        return cls(background=background, particles=particles)


class ScaledCanvas(Canvas):
    """ Canvas with a "scale" that maps system of coordinates from pixels
        to pre-set units."""
    NotImplemented

if __name__ == '__main__':
    from skimage import draw 
    import matplotlib.pyplot as plt
    
    c=Canvas()
    gt = c.grid.tiles
    gt1d = c.grid.as_tiles(key='flat')
    gt2d = c.grid.as_tiles(key='2d')
    img = c.image

    #for key, tile in gt1d.items():
        #print key, key%3
        #if not key%3:
            #img[tile] = (1,0,0)
        #else:
            #img[tile] = (0,1,0)

    for d in gt:
        img[d] = (0,0,1)
        
    #for d in c.grid.dlines_neg:
        #img[d] = (1,0,0)

    #for key, tile in gt2d.items():
        #kx, ky = key
        #if kx%2:
            #if ky%2:
                #img[tile] = (1,0,0)
            #else:
                #img[tile] = (0,1,0)
        #else:
            #img[tile]=(0,0,1)


    
    #for key in gt.keys()[::2]:
        #img[ gt[key] ] = 255
        
#    for l in c.grid.dlines:
#        img[ l ] = 255

    plt.imshow(img)
    plt.show()