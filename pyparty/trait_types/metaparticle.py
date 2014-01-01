import logging

from enthought.traits.api import HasTraits, Str, Tuple, Instance, Float

from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.config import PCOLOR
from matplotlib.colors import ColorConverter

logger = logging.getLogger(__name__) 

ERRORMESSAGE = 'a smatplotlib.colors.ColorConverter.to_rgb() argument.  ' +  \
   '[includes a handful of color names (eg "aqua"), ints/floats (0-1) or' + \
   '(0-255) or hexcolorstring (#00FFFF)'

class ColorError(Exception):
    """ Error raised when color is incorrect; mimics HasTraits error"""
    
class MetaParticle(HasTraits):
    """ Stores a particle and metadata for use by ParticleManager.
    
        Notes
        -----
        May be intelligent to store a default index for sorting and restoring
        purposes, but that would cause issue with preserving order when
        inserting (eg adding new particles.)"""

    # TO DO: Use color trait
    name = Str()
    color = Tuple( Float(0), Float(0), Float(1) )
    particle = Instance(Particle)
    
    #Static variable (can't be invoked directly from matplotlib as such)
    converter = ColorConverter()
    
    def __init__(self, *args, **kwargs):
        color = kwargs.pop('color', None)
        super(MetaParticle, self).__init__(*args, **kwargs)
        
        self.color = MetaParticle.to_rgb(color)        
    
    def _color_changed(self, new):
        self.color = MetaParticle.to_rgb(color)
        
    @staticmethod
    def pix_norm(value, imax=255):
        """ Normalize pixel intensity to 255 """
        return float(value) / imax

    #http://matplotlib.org/api/colors_api.html#matplotlib.colors.ColorConverter            
    @staticmethod
    def to_rgb(color):
        """ Returns an RGB image under several input styles; wraps
        matplotlib.color.ColorConvert"""
        if color is None:
            return PCOLOR
        
        elif isinstance(color, str):
            return MetaParticle.converter.to_rgb(color)          
        
        elif isinstance(color, float) or isinstance(color, int):
            if color > 1:
                if color > 255:
                    raise ColorError(ERRORMESSAGE)
                c_old = color
                color = MetaParticle.pix_norm(color)
                logger.warn("Normalizing color intensity %s to %.2f" % 
                            (c_old, color))                
            return MetaParticle.converter.to_rgb(str(color))          

        elif hasattr(color, '__iter__'):
            return tuple(color)
            #r,g,b = color[0:3]
            #if r > 1 or g > 1 or b > 1:
                #color = tuple( (MetaParticle.pix_norm(c) for c in color[0:3] ) )
                #logger.warn("Normalizing color intensity")
            #else:
                #color = tuple(color)                
                
        else:
            raise ColorError(ERRORMESSAGE)
                                
    @property
    def pclass(self):
        return self.particle.__class__.__name__
    
    @property
    def address(self):
        """ Memory address """
        return super(MetaParticle, self).__repr__() .split()[-1].strip('>')           
    
    def __getattr__(self, attr):
        """ """
        if attr in self.__dict__:
            return getattr(self, attr)
        
        elif attr in CUSTOM_DESCRIPTORS:
            return CUSTOM_DESCRIPTORS[attr](self.particle.boxed())
                          
        elif attr in SKIMAGE_DESCRIPTORS:
            return self.particle.ski_descriptor(attr)        
        
        else:
            try:
                return getattr(self.particle, attr)
            except AttributeError:
                raise ParticleError('%s attribute could not be found on %s'
                                % (attr, self) )
    
    def __setattr__(self, attr, value):
        """ Defer attribute calls to to self.particle unless overwriting
            name, color etc... """
        
        # Some bug where color/name aren't showing up in __dict__ during
        # initialization (this gets called before dict fully updated)
        if attr not in ['name', 'color', 'particle']:
            setattr(self.particle, attr, value)
        else:
            self.__dict__[attr] = value