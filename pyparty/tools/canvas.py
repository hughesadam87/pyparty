import os.path as op
import logging 

import numpy as np
import skimage.io
import skimage.color as color
import matplotlib.pyplot as plt

from traits.api import HasTraits, Array, Instance, Property, Bool, Float
from manager import ParticleManager

# pyparty imports
from pyparty.skiutils import coords_in_image, where_is_particle
from pyparty.config import BACKGROUND

logger = logging.getLogger(__name__) 

class CanvasError(Exception):
    """ Custom exception """ 

class Canvas(HasTraits):
    """
    blah blah blah

    Parameters
    ----------
    foo : int
        The starting value of the sequence.

    Returns
    -------
    bar : float
        blah blah ...
        blah blah ...

    Raises
    ------
    MyException
       If condition foo is met or not met...
       blah blah ...

    Notes
    -----
    blah blah blah
    blah blah blah

    See Also
    --------
    baz : description of baz
    
    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(...)
        ...

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    ...
    >>> plt.show()

    """

    
    image = Array
    image_shape = Property(depends_on = 'image')
    pixelarea = Property(Float, depends_on = 'image') #area already exists at particle level
    
    background = Property()
    _background = Array
    default_background = Property(Array)

    # Maybe just make mesh a separate tool to draw over top? 
    
    #Particles Interface 
    particles = Property()
    _particles = Instance(ParticleManager)
    
    def __init__(self, background=None, particles=None, *traitargs, **traitkwds):
        """ Load with optionally a background image and instance of Particle
            Manager"""
        
        if particles is None:
            particles = ParticleManager()
        self.particles = particles

        # This should distinguish (or in load_bg) between array vs. path
        if background is not None:
            if isinstance(background, basestring):
                self.load_background(background)
            else:
                self.background = background                
        else:
            self.clear_background()
            
    # Public Methods
    # -------------
              
    def clear_background(self):
        """ Restore default background image; redraws
            particles over it."""

        # This will trigger a _draw_particles()
        self.background = self.default_background        
        
    def clear_canvas(self):
        """ Background image to default; removes ALL particles."""

        self.clear_particles()
        self.clear_background()
        
    def clear_particles(self):
        """ Clears all particles from image."""

        self.particles.plist[:] = []
            
    def pmap(self, fcn, *fcnargs, **fcnkwargs):
        """ Maps a function to each particle in ParticleManger; optionally
            can be done in place"""

        inplace = fcnkwargs.pop('inplace', True)
        if not inplace:
            cout = Canvas(background=self.background, particles=self.particles)
            cout._particles.map(fcn, *fcnargs, **fcnkwargs)
            return cout
        else:
            self.particles.map(fcn, *fcnargs, **fcnkwargs)

        
    def pwrap(self):
        """ Wrapper that passes self.particles to a function """
        raise NotImplementedError("Are there any functions explicitly built" 
             "to run on ParticleManager that return an instance of it?")
        
    def iwrap(self, fcn, *fcnargs, **fcnkwargs):
        """ Wrapper that passes self.image to function"""
        
        return fcn(self.image, *fcnargs, **fcnkwargs)

        
    def imap(self, fcn, axis, *fcnargs, **fcnkwargs):
        """ Image mapper (np.apply_along_axis)

            fcn must be 1d!
            
            Notes
            -----
            Calls numpy.apply_along_axis, which doesn't acces
            keyword arguments.
        """
        return np.apply_along_axis(fcn, axis, self.image, *fcnargs)

    def load_background(self, path):
        """ Load an image from harddrive; wraps skimage.io.imread. 
            os.path.expanduser is called to allow ~/foo/path calls."""
        
        try:
            background = skimage.io.imread(op.expanduser( path) )
        except Exception as EXC:
            raise CanvasError('Background must be an array or a valid file '
                'path: %s' % EXC.message)
   
        # Array will undergo further typechecking in _set property
        self.background = background
        return background   
    
            
    def show(self, axes=None, centers=False, cr=3):
        """ Wrapper to imshow.
            If centers, red circle of radius csize in pixels is drawn showing
            center of shapes."""
        
        self._draw_particles()
        if not axes:
            axes = plt.imshow(self.image)
        else:
            axes.imshow(self.image)
        if centers:
            if cr == 'auto':
                raise NotImplemented
            # Return all the centers, draw a box around them (how)
            # Maybe just use "circle" functionality as we have it 
            # to generate it on the fly
               # - centers array
               # - make into circles array of r (via paritcle manager or raw?)
               #    - maybe have as a second particle manager object? need rr_cc codes
               # - Don't use add(), as you don't want them in manager.
                   # - use draw particles *extra particles 
            raise NotImplementedError
        return axes
    
        
    # Private methods
    def _draw_particles(self):
        """  Overwrites all particles on image.  Tis better to redraw whole
        image rather than just draw pieces that change, since redrawing
        will preserve order of overlapping segments.  For example, if I
        make a circle blue, but it is below another circle, I need to redraw
        the whole image to preserve this."""
    
        # Notice this procedure
        image = np.copy(self.background)
        for p in self.particles:
            rr_cc, color = p.particle.rr_cc, p.color 
            rr_cc = coords_in_image(rr_cc, self.image.shape)
            image[rr_cc] = color
        self.image = image 
    
        
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


    # Trait Defaults
    # ------------
    def _image_default(self):
        return np.copy(self._background)

    # Properties
    # ------------
    
    # Image Properties
    def _get_image_shape(self):
        return self.image.shape
    
    def _get_pixelarea(self):
        """ What's the best way to get this? """
        raise NotImplemented
    
    def _get_background(self):
        return self._background
    
    def _set_background(self, background):
    
        try:
            self._background = background
        except Exception as EXC:
            raise CanvasError('Background must castible to ndarray:' 
               ' To load from file, see load_background()')
        
        if self._background.ndim == 3:
            logger.debug("self._background is ndim 3; color adjustment not required")
        
        elif self._background.ndim == 2:
            logger.warn('background color has been converted (from grayscale to RGB)')
            self._background = color.gray2rgb(self._background)
            
        else:
            raise CanvasError('Background must be 2 or 3 dimensional array!')
        
        self._draw_particles()

    def _get_default_background(self):
        width, height = BACKGROUND['resolution']
        background = np.empty( (width, height, 3) )

        if BACKGROUND['bgcolor'] != (0, 0, 0):
            background[:,:,:] = BACKGROUND['bgcolor']
        return background
        
    
    # Particle Properties
    def _get_particles(self):
        """ Want to hide the particles object, so returns a list of names
            instead"""
        
        # LATER CALL SOME METHOD LIKE PARTICLES.SHOW()
        return self._particles
    
    def _set_particles(self, particleinstance):
        """ For now, only Instance(ParticleManager) is supported. """

        # To add: (more explit type check/error)
        #   - support for other types, like a list of tuples passed directly
        #   - to add particles
        self._particles = particleinstance
    
    # Delegate dictionary interface to ParticleManager
    # -----------
    def __getitem__(self, keyslice):
        """ Employs particle manager interface; however, returns single entry
            as a list to allow slicing directly into get_item[]"""
        out = self.particles.__getitem__(keyslice)
        if not hasattr(out, '__iter__'):
            out = [out]
        return out
    
    def __delitem__(self, keyslice):
        return self.particles.__delitem__(keyslice)    
    
    def __setitem__(self, key, particle):
        return self.particles.__setitem__(key, particles)
    
    def __getattr__(self, attr):
        """ Defaults to particle manager """
        return getattr(self.particles, attr)
        
    def __iter__(self):
        return self.particles.__iter__
           
           
class ScaledCanvas(Canvas):
    """ Canvas with a "scale" that maps system of coordinates from pixels
        to pre-set units."""



    
if __name__ == '__main__':

    c=Canvas()
    c.add('circle', radius=100, center=(0,0))
    c.add('circle', radius=20, center=(200,200))
    c.add('circle', radius=20, center=(20000,20000))
    c._draw_particles()

    c.clear_background()
    
    c.add('dimer', radius_1=50, center=(250, 250), color=(1,0,0),
      overlap=0.0)
    c._draw_particles()
    
    c.add('trimer', radius_1=50, center=(250, 250), color=(1,0,0),
      overlap=0.0)
    c._draw_particles()    
    
    
    
#    c.show()
    
    # Run pyclean
    try:
        subprocess.Popen('pyclean .', shell=True, stderr=open('/dev/null', 'w'))
    except Exception:
        pass 