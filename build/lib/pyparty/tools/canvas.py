import numpy as np
import skimage.io
import skimage.color as color
import matplotlib.pyplot as plt

from traits.api import HasTraits, Array, Instance, Property, Bool, Float
from manager import ParticleManager

CONFIG ={
    'bgcolor' : (1.0, 1.0, 1.0),
    'resolution' : (768, 1024)
    }

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
    area = Property(Float, depends_on = 'image')
    
    background = Array

    # Maybe just make mesh a separate tool to draw over top? 
    
    #Particles Interface 
    particles = Property()
    _particles = Instance(ParticleManager)
    
    def __init__(self, background=None, particles=None, *traitargs, **traitkwds):
        ''' '''

        if not particles:
            particles = ParticleManager()
        self._particles = particles

        # This should distinguish (or in load_bg) between array vs. path
        if background:
            self.load_background(background)
        else:
            self.reset_background()
            
            
    # Main function
    def draw_particles(self):
        ''' 
        Overwrites all particles on image.  Tis better to redraw whole
        image rather than just draw pieces that change, since redrawing
        will preserve order of overlapping segments.  For example, if I
        make a circle blue, but it is below another circle, I need to redraw
        the whole image to preserve this.'''
        
        # Notice this procedure
        image = np.copy(self.background)
        for rr_cc, color in self._particles.rr_cc_color():
            image[rr_cc] = color
        self.image = image 
    
    def clear_particles(self):
        ''' Clears all particles from image.'''
        
        self._particles.clear()
    
    #Image interface
    def load_background(self, path_or_array):
        ''' Load an image from harddrive; wraps skimage.io.imread.
            Image must be eihter gray (2d) or RGB!  HSV or other color
            styles will be assumed to be RGB.  To convert, see 
            skimage.color.   '''
        
        
        # HANDLE CASE OF COLORED TUPLE
        if isinstance(path_or_array, np.ndarray):
            background = path_or_array

        else:
            try:
                background = skimage.io.imread(path_or_array)
            except Exception as EXC:
                raise CanvasError('Background must be an array or a valid file '
                'path: %s' % EXC.message)
        if background.ndim != 3:
            # Should put a warning/info here
            print 'Converting image to RGB'
            background = color.gray2rgb(background)
        self.background = background

    def reset_background(self):
        ''' Clears background image back to default settings; redraws
            particles over it.'''
        width, height = CONFIG['resolution']
        background = np.empty( (width, height, 3) )

        if CONFIG['bgcolor'] != (0, 0, 0):
            background[:,:,:] = CONFIG['bgcolor']
        self.background = background
        
        self.draw_particles()
            
    def show(self, axes=None):
        ''' Wrapper to imshow. '''
        
        self.draw_particles()
        if not axes:
            return plt.imshow(self.image)
        return axes.imshow(self.image)
    
    def add(self, *args, **kwargs):
        ''' Wrapper to self.particles.add_particle '''
        self._particles.add_particle(*args, **kwargs)


    # Trait Defaults
    # ------------
    def _image_default(self):
        return np.copy(self.background)

    # Trait Properties
    # ------------
    
    # Image Properties
    def _get_image_shape(self):
        return self.image.shape
    
    def _get_area(self):
        ''' What's the best way to get this? '''
        raise NotImplemented
    
    # Particle Properties
    def _get_particles(self):
        ''' Want to hide the particles object, since attributes and slicing
            delegate downward anwyay. '''
        
        # LATER CALL SOME METHOD LIKE PARTICLES.SHOW()
        return self.particles
    
    def _set_particles(self, particleinstance):
        ''' For now, only Instance(ParticleManager) is supported. '''

        # To add: (more explit type check/error)
        #   - support for other types, like a list of tuples passed directly
        #   - to add particles
        self._particles = particleinstance
    
    # Delegate dictionary interface to ParticleManager
    # -----------
    def __getitem__(self, keyslice):
        return self._particles.__getitem__(keyslice)
    
    def __delitem__(self, key):
        return self._particles.__delitem__(key)    
    
    def __setitem__(self, key, particle):
        return self._particles.__setitem__(key, particles)
    
    # Delegate slicing interface to ParticleManager
    # -----------
    
    # BE CAREFUL WITH THIS, LOTS OF EXTRA ATTRIBUTES CUZ OF TRAITS
    # Delegate unfound attributes to ParticleManager
    def __getattr__(self, attr, *fcnargs, **fcnkwargs):
        ''' Delegate unfound attriutes to ParticleManager'''
        try:
            return getattr(self._particles, attr)
        except AttributeError:
            raise CanvasError('Could not find attribute "%s" in %s or its'
                ' underlying ParticleManager object'%(attr, self.__class__.__name__))           
           
class ScaledCanvas(Canvas):
    """ Canvas with a "scale" that maps system of coordinates from pixels
        to pre-set units."""
    NotImplemented
    
if __name__ == '__main__':
    c=Canvas()
    print c.particles
    c.add_particle('circle', radius=10, center=(500,500))
    c.add_particle('circle', radius=20, center=(200,200))

    c.show()
    
    # Run pyclean
    try:
        subprocess.Popen('pyclean .', shell=True, stderr=open('/dev/null', 'w'))
    except Exception:
        pass 
    