import copy

from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.utils import to_normrgb
    
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


    # MAKE OUTLINE A 2D FUNCTION THAT RETURNS TRUE/FALSE MASK AFTER 2D INDEXING
    # IS EASY, THEN CAN USE IT IN GENERAL FOR EXAMPLE LIKE TO DRAW OUTLINE OF
    # OF A PBINARY PIC (IE OUTLINE(PBINARY) ), AND USE IT HERE
    #TEXTURES ; still not sure best place for them.  Seems here 
    #@property
    #def outline(self):
        #""" THIS WILL GIVE BORDER; HOWEVER, DOES NOT RE-TRANSLATE CIRCLE TO 
        #CENTER"""
        #from scipy.ndimage.filters import laplace
        #pbox = self.particle.boxed(pad=1)
        #return lap(pbox)        
        
    #@property
#    def tiled(self):
#        """ Eventually make this more in-depth; even a function like outline"""
#        rr, cc = self.particle.rr_cc
#        rspacing = cspacing = 2
#        return tuple(rr[::rspacing], cc[::cspacing])
       
        
    def __getattr__(self, attr):
        """ Attribute can be in self.__slots__, descriptor or particle 
        attribute."""
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
        name, color etc... 
        """
        
        if attr == 'color':
            MetaParticle.__dict__['color'].__set__(self, to_normrgb(value))
            
        elif attr =='name':
            MetaParticle.__dict__['name'].__set__(self, str(value))
            
        elif attr == 'particle':
            if not isinstance(value, Particle):
                raise ParticleError('MetaParticle requires instance of Particle'
                                    ' recieved %s' % type(value))

            MetaParticle.__dict__['particle'].__set__(self, value)
            
            
        elif attr in CUSTOM_DESCRIPTORS or attr in SKIMAGE_DESCRIPTORS:
            raise ParticleError("%s already reserved name for particle:" 
                " descriptor setting is not allowed." % attr)
                                
        
        else:
            setattr(self.particle, attr, value)           
           

# MEMOIZING (eg depends_on) SCREWS UP WHEN COPYING           
def copy_metaparticle(obj):
    """ Make a copy of MetaParticle.  Since MetaParticle uses __slots__,
    copy.copy doesn't work. Deepcopy is required
    """
    return MetaParticle(obj.name, obj.color, copy.deepcopy(obj.particle) )