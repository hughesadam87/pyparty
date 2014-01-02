import copy

from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.utils import to_normrgb
from pyparty.config import PCOLOR
    
class MetaParticle(object):
    """ Stores a particle and metadata for use by ParticleManager.
    
    Notes
    -----
    This enforces initialization/validation of types without 
    using Traits (to be light in memory).  __slots__ prevents users
    from adding attributes accidentally.  Using __slots__ and 
    __setattr__ leads to some funky syntax as described:        
        http://stackoverflow.com/questions/19566419/can-
        setattr-can-be-defined-in-a-class-with-slots
    """       
    __slots__ = ('name', 'color', 'particle')

    def __init__(self, name='', color='', particle=''):
        
        self.name = str(name)
        self.color = to_normrgb(color)        
        if not isinstance(particle, Particle):
            raise ParticleError('MetaParticle requires instance of Particle'
                                ' recieved %s' % type(particle))
        self.particle = particle
    
    # Worth storing?        
    @property
    def address(self):
        """ Memory address """
        return super(object, self).__repr__() .split()[-1].strip('>')           
    
    def __getattr__(self, attr):
        """ """
        if attr in self.__slots__:
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

        if attr == 'color':
            MetaParticle.__dict__['color'].__set__(self, to_normrgb(value))
            
        elif attr =='name':
            MetaParticle.__dict__['name'].__set__(self, str(value))
            
        elif attr == 'particle':
            if not isinstance(value, Particle):
                raise ParticleError('MetaParticle requires instance of Particle'
                                    ' recieved %s' % type(value))

            MetaParticle.__dict__['particle'].__set__(self, value)
        
        else:
            setattr(self.particle, attr, value)
           
def copy_metaparticle(obj):
    """ Make a copy of MetaParticle.  Since MetaParticle uses __slots__,
        copy.copy doesn't work. Deepcopy is required"""
    
    return MetaParticle(obj.name, obj.color, copy.deepcopy(obj.particle) )