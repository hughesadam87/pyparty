from enthought.traits.api import HasTraits, Str, Tuple, Instance, Float

from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
    
class MetaParticle(HasTraits):
    """ Stores a particle and metadata for use by ParticleManager.
    
        Notes
        -----
        May be intelligent to store a default index for sorting and restoring
        purposes, but that would cause issue with preserving order when
        inserting (eg adding new particles.)"""

    # TO DO: Use color trait
    name = Str()
    color = Tuple( Float(0.0), Float(0.0), Float(1.0) )
    particle = Instance(Particle)
    
    @property
    def pclass(self):
        return self.particle.__class__.__name__
    
    @property
    def address(self):
        """ Memory address """
        return super(MetaParticle, self).__repr__() .split()[-1].strip('>')           
    
    def __getattr__(self, attr):
        
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
        
        if attr not in self.__dict__:
            setattr(self.particle, attr, value)
        else:
            self.__dict__[attr] = value
