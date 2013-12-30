#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
from operator import attrgetter
import itertools
import logging
import copy

# 3rd Party Imports
import numpy as np
from traits.api import HasTraits, Instance, Str, Tuple, Float, \
    cached_property, Property, List

# Package Imports
from pyparty.shape_models.api import GROUPEDTYPES, ALLTYPES
from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.config import NAMESEP

logger = logging.getLogger(__name__) 

class ManagerError(Exception):
    """ Particle Manager general Exception """
    
class KeyErrorManager(KeyError):
    """ Particle Manager dictionary interface Exception """

def concat_particles(p1, p2, alternate=False, overwrite=False):
    """ Joins two instances of particle manager.
    
    Attributes
    ----------
    p1, p2 : ParticleManager
    
    alternate : Bool 
       Add particles in alternating order.  If False, all of p2 is added 
       after p1, meaning p2 would paint "over" p1 for particles w/ overlapping
       regions
       
    overwrite : Bool
       If True, particles in p2 with same name as those in p1 will overwrite.
       Otherwise, an error is raised if particles in p1 and p2 share names.
    
    Returns
    -------
    pout : ParticleManager
       Merged particle manager containing all particles of p1 and p2.
       
    Notes
    -----
    When overwriting, the elements are deleted from p1
    """
    p1_temp = copy.copy(p1)
    shared = [name for name in p1.names if name in p2.names]
    if shared:
        if overwrite:
            p1_temp.plist[:] = [p for p in p1 if p.name not in shared]
        else:
            raise ManagerError("%s Duplicate particle names found." % len(shared))
            
    pout = p1_temp.plist + p2.plist

    #http://stackoverflow.com/questions/7529376/pythonic-way-to-mix-two-lists    
    if alternate:
        pout = [p for pout in map(None, p1_temp, p2) for p in pout]
        pout = [p for p in pout if p is not None]
            
    return ParticleManager(plist=pout)


def subtract_particles(p1, p2):
    p1_temp = copy.copy(p1)
    shared = [name for name in p1.names if name in p2.names]    
    p1_temp.plist[:] = [p for p in p1 if p.name not in shared]
    return p1_temp               

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
    
    def __repr__(self):
        """ Puts name into return with memory address:
              (my name) <__main__.Foo  at 0x3002f90>  """
        
        out = super(MetaParticle, self).__repr__() 
        address = out.split()[-1].rstrip('>')
        return '(%s : %s : %s at %s)' % \
           (self.name, self.particle.ptype, self.pclass, address)
    
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
        
        
class ParticleManager(HasTraits):
    """ Container class for creating, storing, managing particles;
        provides API used by higher-level objects.  Provides a property
        interface for easy mapping and slicing.  
    """
            
    plist = List(MetaParticle)
    _namemap = Property(Tuple, depends_on = 'plist')
   
    @cached_property
    def _get__namemap(self):
        """ Store light map of name to index for faster name lookup.  I verified
            that this updates when plist elemnts are updated as well as obj itself.
            """
        return dict( (pobj.name, idx) for idx, pobj in enumerate(self.plist))    
            
    # ADD INDEX KEYWORD TO SUPPORT INSERTIONS
    def add(self, particle, name='', idx=None, color=None,
                      *traitargs, **traitkwargs):
        """ If color not passed, default color is used
            If not idx, put in last entry  """
    
        # Make particle from name and arguments
        if isinstance(particle, basestring):
            particle = self._make_particle(particle, *traitargs, **traitkwargs)

        if not idx:
            idx = len(self)
       
        #Generate key if it does not yet exist
        if not name:
            name = particle.ptype + NAMESEP + str(idx)
            
        if name in self._namemap:
            raise ManagerError('particle %s is already named "%s"' % 
                                 (self._namemap[name], name) )
        
        if color:
            meta = MetaParticle(name=name, color=color, particle=particle)
        else:
            meta = MetaParticle(name=name, particle=particle)
        

        if idx == len(self):
            self.plist.append(meta)
        else:
            self.plist.insert(idx, meta)
                
    def _make_particle(self, ptype='', *traitargs, **traitkwargs):
        """ Instantiate a particle through string specifying class type
            via PARTICLETYPES """
        try:
            pclass = ALLTYPES[ptype]
        except KeyError:
            raise KeyErrorManager('"%s" is not an understood Particle type.  '
                'Choose from: %s' % (ptype, self.available()))
        return pclass(*traitargs, **traitkwargs)
    
    
    def map(self, fcn, *args, **kwargs):
        """ Apply a function to all particles.  Can also use functools.partial"""
        self.plist[:] = [fcn(p, *args, **kwargs) for p in self.plist]
        
    def reverse(self):
        """ In place since list is inplace """
        self.plist.reverse()
    
    def pop(self, idx):
        self.plist.pop(idx)
        
    def index(self, *names):
        """ Return index given names """
        out = tuple(self._namemap[name] for name in names)
        if len(out) == 1:
            out = out[0]
        return out
    
    def _unmask_index(self, idx):
        """ Test if object is an integer, string, or isntance of 
            MetaParticle.  Returns index in any case. """

        if isinstance(idx, int) or isinstance(idx, slice):
            return idx

        if isinstance(idx, basestring):
            return self._namemap[idx]

        if isinstance(idx, MetaParticle):
            return self._namemap[idx.name]
        
        raise ManagerError("Index must be of type int, str or MetaParticle"
                           " received %s" % type(idx) )
        
    # Magic Methods
    # -------------
    def __add__(self, p2):
        return concat_particles(self, p2, alternate=False, overwrite=False)
    
    def __sub__(self, p2):
        """ Removes duplicates between p2, p1, but does not affect p1 particles
            otherwise."""
        return subtract_particles(self, p2)

    def __iter__(self):
        return self.plist.__iter__()
    
    def __getitem__(self, keyslice):
        """ Supports single name lookup; otherwise defers to list delitem"""

        if hasattr(keyslice, '__iter__'):
            # Boolean indexing
            if isinstance(keyslice, np.ndarray):
                boolout = keyslice.astype('bool')
                return [self.plist[idx] for idx, exists in enumerate(boolout) if exists]                
    
            else:    
                return [self.plist[self._unmask_index(idx)] for idx in keyslice]              

        return self.plist[self._unmask_index(keyslice)]

        
    def __delitem__(self, keyslice):
        """ Supports single name deletion; otherwise defers to list delitem.  
            Deletes in place to congruance with list API.
        
        Notes
        -----
        When deleting iteratively, avoid calls to "del" which changes size 
        during iteariton."""

        if hasattr(keyslice, '__iter__'):
            # Boolean indexing
            if isinstance(keyslice, np.ndarray):
                boolout = keyslice.astype('bool')
                self.plist[:] = [self.plist[idx] for idx, exists in
                                 enumerate(boolout) if not exists]
            else:    
                to_remove = [self._unmask_index(idx) for idx in keyslice]
                self.plist[:] = [self.plist[idx] for idx in range(len(self))
                                 if idx not in to_remove]

        # Delete single entry
        else:
            del self.plist[self._unmask_index(keyslice)]
        

    def __setitem__(self, keyorslice, particle):
        """ Not sure how confident I am in implementing this.  Users can add, 
            and remove particles through add()/remove() API.  Allowing them to
            set through slicing could break the api.  For example, setting 
            many particles to the same value and hence the same name..."""
        raise ManagerError('"%s" object does not support item assigment.' % 
                           self.__class__.__name__)


    def __getattr__(self, attr):
        """ Return numpy array subset of self.plist based on attr. Attribute
            can be name, color, descriptor or otherwise common attribute of
            Particle class.  
            
            Notes
            -----
            Implemented similar pandas package to allow for attribute-based 
            indexing.
            
            Examples
            --------
            big_particles = p[p.perims > 50] """
        
        out = tuple(getattr(p, attr) for p in self.plist)
        return np.array(out)
    
    def __len__(self):
        return self.plist.__len__()
    
    # MAKE MORE SUCCINT
    def __repr__(self):
        """ <Particle at Address> [items]

        Examples
        --------
        <ParticleManager at 0x2c6eb30> [(circle_0 : circle ...],
        """
        old = super(ParticleManager, self).__repr__() 
        ctype = self.__class__.__name__   
        address = old.split()[-1].strip('>')   
        prefix = '<%s at %s>' % (ctype, address)
        return ('%s %s' % (prefix, self.plist.__repr__()))

    def in_region(self, *coords):
        """ Get all particles whose CENTERS are within a rectangular region"""
        raise NotImplementedError
    

    # python properties
    # -------------
        
    # SHOULD REMOVE NAMES/PARTICLES/IDXS AFTER FIX SLICING TO RETURN PMANAGER INSTANCE
    # THEN DOING P.NAME SHOULD ATOMATICALLY RETURN NAMES
    @property
    def names(self):
        """ Particles names; so common, worth doing here"""
        return tuple(p.name for p in self.plist)

    @property
    def idxs(self):
        """ Particle indicies"""
        return range(len(self))
    
    @property
    def particles(self):
        """ Particle objects"""
        return tuple(p.particle for p in self.plist)
    
    @property
    def centers(self):
        """ Return center coordinates of all particles."""
        return self._subset('center')
    
    

    # Full attribute container sorted mappers        
    
    def sortby(self, attr='name', inplace=False):
        """ Sort list by attribute/descriptor"""

        plistout = sorted(self.plist, key=attrgetter(attr))

        if inplace:
            self.plist[:] = plistout
        else:
            return ParticleManager(plist=plistout)
        
    
    def _rr_cc_color(self):
        """ Returns rr_cc, color for each particle; useful to canvas """
        return [(p[3].rr_cc, p[2]) for p in self.values]
    
    
    @property
    def ptypes(self):
        """ All unique particle types. """
        return tuple(set( self._subset(ptype) ) )
        
    @property
    def ptype_count(self):
        """ Unique particle types in plist. """
        return ( (typ, self.ptypes.count(typ)) for typ in self.ptypes)
    
    def available(self, subtype=None):
        """ Show all valid particle types.  Subtype is 'simple' 'multi' to
            return groups of particles"""
        if subtype:
            return tuple( sorted(GROUPEDTYPES[subtype.lower()].keys()) )
        else:
            return tuple( (k+':', sorted(v.keys())) 
                          for k, v in GROUPEDTYPES.items() ) 
        
    # Not printing these since ipython does its own printing
    @property
    def stats(self):
        """ Returns a string/TUPLE? of particles and stuff """
        return 

    @property    
    def full_stats(self):
        """ """
        return     

        
    # Class methods
    # ------------
    @classmethod
    def from_labels(cls, labelarray, prefix='label', colorbynum=False):
        """ Create ParticleManager from output of skimage.morphology.label
        array.  Each item is written to a general Particle type."""

        from pyparty.shape_models.io import LabeledParticle

        num = np.unique(labelarray)
        if num[0] == 0:
            num = num[1:]  #Will this ever not be the case?
        
        # Normalize color
        if colorbynum:
            color_norm = float(255)/max(num)       
       
        plist = []
        for idx, label in enumerate(num):
            name = '%s_%s' % (prefix, idx)
            rr_cc = np.where(labelarray==label)
            particle = LabeledParticle(rr_cc, label=label)
            
            if colorbynum:
                cn = color_norm * label      
                color = (0, 0, cn)
                plist.append( MetaParticle(name=name, color=color, 
                                           particle=particle) )
            else:
                plist.append( MetaParticle(name=name, particle=particle) )
      
        return cls(plist=plist)      
        
if __name__ == '__main__':
    p=ParticleManager()
    for i in range(5):
        p.add(particle='circle', radius=i)
        
    p2=ParticleManager()
    for i in range(5):
        p2.add(particle='circle', name='foo'+str(i), radius=i)
    
    
    print len(p), p
    print len(p2), p2
    pout = concat_particles(p,p2, overwrite=False, alternate=False)    
    print len(pout), pout
    
    pminus = p2-p2
    print len(pminus), pminus

    #print p.name
    #print p.perimeter, type(p.perimeter), p.perimeter.dtype
    #print p[p.perimeter > 50]
    #print p.plist
    #del p[p.perimeter > 50]
    #print p.name