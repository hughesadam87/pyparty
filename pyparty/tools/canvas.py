import os.path as op
import logging 

import numpy as np
import matplotlib.pyplot as plt

import skimage.io
import skimage.color as color
import skimage.measure as measure
import skimage.morphology as morphology
from skimage import img_as_float, img_as_bool, img_as_ubyte
import pyparty.background.bg_utils as bgu

from traits.api import HasTraits, Array, Instance, Property, Float
from manager import ParticleManager, concat_particles

# pyparty imports
from pyparty.utils import coords_in_image, where_is_particle, to_normrgb
from pyparty.config import BGCOLOR, BGRES

logger = logging.getLogger(__name__) 

def concat_canvas(c1, c2, bg_resolve='c2', **particle_args):
    """ Adds two canvas objects under various conditions """
    bg_resolve = bg_resolve.lower()
    
    # Choose output background
    if bg_resolve == 'merge':
        raise NotImplementedError
    elif bg_resolve == 'c1':
        bgout = c1.background
    elif bg_resolve == 'c2':
        bgout = c2.background        
    else:
        raise CanvasAttributeError('"bg_resolve" invalid; must be %s' % bg_valid)
    
    pout = concat_particles(c1._particles, c2._particles, **particle_args)
    return Canvas(background=bgout, particles=pout)
        

class CanvasError(Exception):
    """ Custom exception """ 
    
class CanvasAttributeError(CanvasError):
    """ Custom exception """     

class Canvas(HasTraits):
    """  """
    image = Property(Array, depends_on='_image')
    _image = Array

    # ALL INTERNAL REFERENCES SHOULD GO TO _PARTICLES
    particles = Property(Instance(ParticleManager, depends_on='_particles'))
    _particles = Instance(ParticleManager)
    
    # Background is only a property because for _set validation...
    background = Property()
    _background = Array
    default_background = Property(Array)
    
    def __init__(self, background=None, particles=None): #No other traits to be set
        """ Load with optionally a background image and instance of Particle
            Manager"""
        
        if not particles:
            particles = ParticleManager()
        self._particles = particles
        
        # This does a bunch of parsing for various bg inputs
        self.background = background
                        
    # Public Methods
    # -------------              
    def clear_background(self):
        """ Restore default background image; redraws
            particles over it."""

        self.background = self.default_background        
        
    def clear_canvas(self):
        """ Background image to default; removes ALL particles."""

        self.clear_particles()
        self.clear_background()
        
    def clear_particles(self):
        """ Clears all particles from image."""

        self._particles.plist[:] = []
            
    def pmap(self, fcn, *fcnargs, **fcnkwargs):
        """ Maps a function to each particle in ParticleManger; optionally
            can be done in place"""

        inplace = fcnkwargs.pop('inplace', False)
        if inplace:
            self._particles.map(fcn, *fcnargs, **fcnkwargs)
        else:
            cout = Canvas(background=self.background, particles=self._particles)
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
    def from_labels(self, inplace=False, neighbors=4,
                    background=None, **pmangerkwds):
        """ """
        	
        self._cache_image()
        
        if background: # scikit api doesn't accept None
            background = int(background) 
            labels = morphology.label(self.grayimage, neighbors, background)
        else:
            labels = morphology.label(self.grayimage, neighbors)
            
        pout = ParticleManager.from_labels(labels, **pmangerkwds)
        if inplace:
            self._particles = pout
        else:
            return Canvas(background=self.background, particles=pout)
        
    #http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
    def show(self, *imshowargs, **imshowkwds):
        """ Wrapper to imshow.  Converts to gray to allow color maps."""
        
        axes = imshowkwds.pop('axes', None)   
        title = imshowkwds.pop('title', None)
        
        self._cache_image()

        # cmap is the first argument in imshowargs
        if imshowargs or 'cmap' in imshowkwds:
            image = self.grayimage
        else:
            image = self.image

        if axes:
            axes.imshow(image, *imshowargs, **imshowkwds)
        else:      # matplotlib API asymmetry
            axes = plt.imshow(image, *imshowargs, **imshowkwds).axes           
        
        if title:
            axes.set_title(title)
        return axes
    
        
    # Private methods
    def _cache_image(self):
        """ Creates image array of particles.  Tried fitting to a property or
        Traits events interface to control the caching, but manually choosing
        cache points proved to be easier."""
    
        # Notice this procedure
        image = np.copy(self.background)
        for p in self._particles:
            rr_cc, color = p.particle.rr_cc, p.color 
            rr_cc = coords_in_image(rr_cc, image.shape)
            image[rr_cc] = color
        self._image = image 
    
        
    # Image Attributes Promoted
    # ------------------
    @property
    def shape(self):
        return self.image.shape
    
    @property
    def ndim(self):
        return self.image.ndim
 
    @property
    def dtype(self):
        return self.image.dtype
    
    @property
    def grayimage(self):
        """ Collapse multi-channel, scale to 255 (via ubyte) """
        return img_as_ubyte( color.rgb2gray(self.image) )
    
    @property
    def binaryimage(self):
        return img_as_bool(self.grayimage) # 3--1 channel reduction required
    
    @property
    def graybackground(self):
        return color.rgb2gray(self.background)
    
    @property
    def binarybackground(self):
        return img_as_bool(self.graybackground)
    
    @property
    def pbinary(self):
        """ Returns boolean mask of all particles IN the image, with 
            background removed."""      
        # Faster to get all coords in image at once since going to paint white
        rr_cc = coords_in_image( self._particles.rr_cc_all, self.image.shape)
        out = np.zeros( self.imres, dtype=bool )
        out[rr_cc] = True
        return out  

    
    def _whereis(self, choice='in'):
        """ Wraps utils.where_is_particles for all three possibilities """
        return [p.name for p in self._particles 
                if where_is_particle(p.rr_cc, self.image.shape) == choice]
    
    @property
    def imres(self):
        """ _background shape first 2 dimensions; color-independent """
        return self._background.shape[0:2]
    
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
        l, w = self.imres
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
    
    @property
    def mem_address(self):
        return super(Canvas, self).__repr__() .split()[-1].strip('>')


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
        self._cache_image()
    
    def _get_image(self):
        return self._image
    
    def _set_image(self):
        raise CanvasError('Image cannot be set; please make changes to particles'
            ' and/or background attributes.')
    
    def _get_background(self):
        return self._background
    
    # REDO THIS WITH COLOR NORM AND STUFF!  Also, dtype warning?
    def _set_background(self, background):
        """ Parses several valid inputs including: None, Path, Color (of any 
        valid to_normRGB() type, ndarray, (Color, Resolution)."""
        if background is None:
            self._background = self.default_background   
            
        elif isinstance(background, np.ndarray):
            self._background = background
            
        elif isinstance(background, basestring):
            # String or colorstring        
            self._background = bgu.from_string(background, BGRES[0], BGRES[1])
            
        # color will be normalized by from_color_res
        elif isinstance(background, int) or isinstance(background, float):
            self._background = bgu.from_color_res(background, BGRES[0], BGRES[1])

        elif hasattr(background, '__iter__'):
            try:
                
    
        
        # TYPE CAST THE ARRAY        
        if self._background.ndim == 3:
            logger.debug("self._background is ndim 3; color adjustment not required")
        
        elif self._background.ndim == 2:
            logger.warn('background color has been converted (from grayscale to RGB)')
            self._background = color.gray2rgb(self._background)
            
        else:
            raise CanvasError('Background must be 2 or 3 dimensional array!')
        
        # *****
        # Note sure best way to check float dtype (worth doing?)
        dtold = self._background.dtype
        self._background = img_as_float(self._background)
        if self._background.dtype != dtold:
            logger.warn("Background dtype changed from %s to %s" %
                              (dtold, self._background.dtype))
        
        # To ensure image updates even if show() not called
        self._cache_image()
        
    def _get_default_background(self):
        return bgu.from_color_res(BGCOLOR, BGRES[0])       
    
    
    # Delegate dictionary interface to ParticleManager
    # -----------
    def __getitem__(self, keyslice):
        """ Employs particle manager interface; however, returns single entry
            as a list to allow slicing directly into get_item[]"""
        return self._particles.__getitem__(keyslice)
    
    
    def __delitem__(self, keyslice):
        return self._particles.__delitem__(keyslice)    
    
    def __setitem__(self, key, particle):
        return self._particles.__setitem__(key, particles)
    
    def __getattr__(self, attr):
        """ Defaults to particle manager """
        return getattr(self._particles, attr)
        
    def __iter__(self):
        """ Iteration is blocked """
        raise CanvasError("Iteration on canvas is ambiguous.  Iterate over "
                          "canvas.image or canvas.particles")
    
    def __repr__(self):
        _bgstyle = 'user-array' #REPLACE
        res = '(%s X %s)' % (self.imres[0], self.imres[1] ) 
        _PAD = ' ' * 3
        
        outstring = "Canvas as %s:\n" % self.mem_address

        outstring += "%sbackground -->  %s : %s\n" % (_PAD, res, _bgstyle) 

        outstring += "%sparticles  -->  %s particles : %s types\n" % (_PAD, \
            len(self._particles), len(self._particles.ptype_count))

        return outstring
    
    def __len__(self):
        return self._particles.__len__()
    
    # Arithmetic Operation
    # --------------------
    
    def __add__(self, c2):
        return concat_canvas(self, c2, bg_resolve='c2')

           
class ScaledCanvas(Canvas):
    """ Canvas with a "scale" that maps system of coordinates from p
    ixels
        to pre-set units."""

    
if __name__ == '__main__':

    c=Canvas()
    
    c.add('polygon', orientation=20.)
    c.add('ellipse', orientation=32.0)
    c.add('circle', radius=20, center=(200,200))
    c.add('circle', radius=20, center=(20000,20000))
    
    print c
    
    
    #clab = c.from_labels()
    
    #c.background=30
    #print c.rr_cc
##    c2 = c.from_labels(me_sobel)
    #c2.show()   
    
##    c.show()
    
    
    #c.particles    
    #print c.pin
    
    
    ## Run pyclean
    #try:
        #subprocess.Popen('pyclean .', shell=True, stderr=open('/dev/null', 'w'))
    #except Exception:
        #pass 