from enthought.traits.api import HasTraits, Str, Tuple, Instance, Float
from matplotlib.colors import ColorConverter

from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.utils import to_normrgb
from pyparty.config import PCOLOR

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
    color = Tuple(PCOLOR)
    particle = Instance(Particle)
    
    #Static variable (can't be invoked directly from matplotlib as such)
    converter = ColorConverter()
    
    def __init__(self, *args, **kwargs):
        color = kwargs.pop('color', None)
        super(MetaParticle, self).__init__(*args, **kwargs)
        
        self.color = to_normrgb(color)        
    
    def _color_changed(self, new):
        self.color = to_normrgb(color)

    # Worth storing?        
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