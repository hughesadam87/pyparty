from traits.api import *
import operator as oper
from collections import OrderedDict

import numpy as np
from pyparty import Canvas
from pyparty.utils import to_normrgb

def radius(obj):
    return getattr(obj, 'equivalent_diameter')/2.0

def circularity(obj):
    NotImplemented


SHORTCUTS = {'d':'equivalent_diameter',
             'r': radius }  


# DO NOT CHANGE ORDER!
COMP_OPS = OrderedDict([
            ("==",np.equal), ("!=",np.not_equal), ("<",np.less),
            (">",np.greater), (">=",np.greater_equal), ("<=",np.less_equal)
       ] )


LOGIC_OPS = OrderedDict([
       ("&",np.logical_and), ("|",np.logical_or)
       ])

#http://code.activestate.com/recipes/577616-split-strings-w-multiple-separators/
def multisplit(s, sep):
    """ Split a string based on a list of descriptors.  Returns the split
    string as well as the characters found in the split (ie % & ...) """
    
    splitchar = []
    stack = [s]
    for char in sep:
        pieces = []
        for substr in stack:
            pieces.extend(substr.split(char))            
        stack = pieces
        if char in s:
            splitchar.append(char)
    return stack, splitchar


def _mapcolor(desc, c):
    if desc.color:
        def cmap(p):
            p.color = desc.color
            return p 
        c = c.pmap(cmap)
    return c


class ModelError(Exception):
    """ """

# Name is classname/vairable name
class Descriptor(HasTraits):
    
    classifier = Str('(d > 0)') # Or do I want to just error if missing classifier
    _classfcn = Property(depends_on='classifier')
    color = Any(None)
    alias = Str()
    
    def _color_changed(self, old, new):
        if new:
            self.color=to_normrgb(new)
        
    def _parse_masks(self, canvas):
        """ Apply masks to canvas.  Should split the parsing of the operators 
        into python expressions and return,  and have parsed canvas work with 
        those, but for now, wrapping everything into a single method for 
        prototyping."""

        masks = []
        lops = []
        
        # Split on '|' and '&'
        descriptors, LOGIC_CHARS = multisplit(self.classifier, LOGIC_OPS.keys())
                

        for op in descriptors: # op is (d < 50)
            nospace = "".join(op.split()) # Strip whitespace
           
            #( (foo) & (bar) ) | (baz); can't handle nexted paranthessis
            if nospace.count('(') != 1 or nospace.count(')') != 1:
                raise ModelError('Cannot handle nested (or missing) paranethesis expression: "%s"' % op)
            if nospace[0] != '(' or nospace[-1] != ')':
                raise ModelError("Invalid logic function: %s" % op)
            nospace = nospace.strip("(").strip(")")
        
            # Split on comparison operator (<, == etc..)
            comp_split, COMP_CHAR = multisplit(nospace, COMP_OPS.keys())
            if len(comp_split) != 2:
                raise ModelError('Invalid comparison operator: %s---> [ %s ]' % (op, ", ".join(COMP_OPS) ))    
            variable, value = comp_split
            value = float(value)
        
            if len(COMP_CHAR) != 1:
                raise ModelError("Comparison operator list != 1 %s" % COMP_CHAR)
            comp_char = COMP_CHAR[0]
        
            # Return partial function for value attribute
            if variable in SHORTCUTS:
                variable = SHORTCUTS[variable]

            if isinstance(variable, str):
                variable = oper.attrgetter(variable)
        
            # NEED OBJECT ACCESS STARTING HERE!            
            descriptor_array = variable(canvas) #c.area
            
            # Worth doing
            if not isinstance(descriptor_array, np.ndarray):
                raise ModelError('Attribute inspection not return an array')
            
            comp_op = COMP_OPS[comp_char]            
            mask = comp_op(descriptor_array, value)
            masks.append(mask)

        # If multiple expresions (& or |), apply masks compoundly    
        if LOGIC_CHARS:
            ls = [LOGIC_OPS[char] for char in LOGIC_CHARS]
            ms, mf = masks.pop(0), masks.pop(0)
            l = ls.pop(0)
        
            mask_out = l(ms,mf)
            while masks:
                m = masks.pop(0)
                mask_out = l(mask_out, m)
                
        else:
            mask_out = masks[0] 

        return mask_out
        
                        
    def _get__classfcn(self):
        return eval(self.classifier)
  
    
    def map_canvas_mask(self, canvas, mask):
        cp = []    
        cp[:] = canvas.particles[mask]
        cout = Canvas.copy(canvas)
        cout._particles.plist[:] = cp[:]
        return cout          
    
    def in_canvas(self, canvas):
        mask = self._parse_masks(canvas)
        return self.map_canvas_mask(canvas, mask)

    def not_in_canvas(self, canvas):
        mask = np.invert(self._parse_masks(canvas))
        return self.map_canvas_mask(canvas, mask)
    
    def in_and_out_canvas(self, canvas):
        return (self.in_canvas(canvas), self.not_in_canvas(canvas))

    
    
class Model(HasTraits):
    """ Simple container for descriptors; basically a type-checked List of
    Descriptors."""
    
    descriptors = List(Instance(Descriptor))
    colors = Property(depends_on = 'descriptors')
    aliases = Property(depends_on = 'descriptors')

    
    def __init__(self, *descriptors, **traitkwargs):
        self.descriptors = list(descriptors)
        super(Model, self).__init__(**traitkwargs)
    
    def _get_aliases(self):
        aliases = []

        for desc in self.descriptors:
            if desc.alias:
                alias = desc.alias                  
            else:
                alias = desc.__class__.__name__

            if alias in aliases:
                i = 1
                prefix = alias
                while alias in aliases:
                    alias = '%s_%s' % (prefix, i)
                    i += 1            

            aliases.append(alias)                                        
        return aliases

    
    def _get_colors(self):
        """ List of None and colors """
        return [c.color for c in self.descriptors]
        
    
    @classmethod
    def from_textlist(self):
        """ Read descriptors from a text file"""
        raise NotImplementedError
    
    
    # Item interface at descriptor level    
    def __getitem__(self, keyslice):
        return self.descriptors.__getitem__(keyslice)
    
    def __setitem__(self, keyslice, value):
        self.descriptors.__setitme__(keyslice, value)
    
    def __delitem__(self, keyslice):
        self.descriptors.__delitem__(keyslice)

    def __len__(self):
        return len(self.descriptors)
    
    def pop(self, idx):
        return self.descriptors.pop(idx)
    
    def append(self, descriptor):
        self.descriptors.append(descriptor)
        
    def __repr__(self):
        return super(Model, self).__repr__()
    
    def apply_canvas(self, canvas, iterative=False, mapcolors=True):
        
        cout = []
        f, cnull = self.descriptors[0].in_and_out_canvas(canvas)

        if mapcolors:
            f = _mapcolor(self.descriptors[0], f)
        cout.append(f)

        for desc in self.descriptors[1:] :
            if iterative:
                f, cnull = desc.in_and_out_canvas(cnull)
           
            else:
                f, cnull = desc.in_and_out_canvas(canvas)
                
            if mapcolors:
                f = _mapcolor(desc, f)
            cout.append(f)                
            
        return cout, cnull


    # ADD ASTYPE STUFF FOR DICT
    
if __name__ == '__main__':   

    class Small(Descriptor):
        classifier = '(d < 30)'
        color = 'r'

    class SmallCircles(Descriptor):
        classifier = '(d < 30) & (eccentricity == 0.0)'
        color = 'orange'
  #      alias = 'aunp singles'
        
        
    class Large(Descriptor):
        classifier = '(d > 30)'
        color = 'green'


    class LargeTriangles(Descriptor):
        classifier = '(d > 30) & (eccentricity > 0.01)'
        color = 'lime'
  #      alias = 'aunp singles'
    
      
    
    import matplotlib.pyplot as mpl
    from pyparty import splot 
    
             #This way to preserve model name
    models = Model(Small(), SmallCircles(), Large(), LargeTriangles())    
    c = Canvas.random_circles() + Canvas.random_triangles()
    cout, cnull = models.apply_canvas(c, iterative=False)

    print 'hi', c.eccentricity

    axes = splot(1,len(cout)+1)
    for i, c in enumerate(cout):
        c.show(axes[i], title=models.aliases[i])
        
    cnull.show(axes[i+1], title='remaining')

    cnull.show()
    mpl.show()
        
    
    #ax1, ax2, ax3 = splot(1,3)
    #c.show(ax1)
    #f.show(ax2)
    #g.show(ax3)
    #mpl.show()
    