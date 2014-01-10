import os.path as op
import logging 

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np

import skimage.io
import skimage.color as color
import skimage.measure as measure
import skimage.morphology as morphology
from skimage import img_as_float, img_as_bool, img_as_ubyte
import pyparty.background.bg_utils as bgu

from traits.api import HasTraits, Array, Instance, Property, Float
from manager import ParticleManager, concat_particles
from functools import wraps

# pyparty imports
from pyparty.utils import coords_in_image, where_is_particle, to_normrgb, \
     any2rgb, crop, _parse_ax
from pyparty.config import BGCOLOR, BGRES

logger = logging.getLogger(__name__) 



def inplace(method):
    """ Thought I could decorate methods that have inplace keyword, but turned
    out to be more than I bargained for.  Decorator syntax is fine, but not correct.
    """
    methodname = method.__name__
    #variables = method.func_code.co_varnames
    @wraps(method)
    def wrapper(obj, *args, **kwargs):
	print 'method is', method, type(obj), type(method)
	inplace = kwargs.pop('inplace', False)
	if inplace:
	    print 'in inplace', args, kwargs
	    getattr(obj, methodname)(*args, **kwargs)
	else:
	    print 'not inplace. obj is', obj
	    new = Canvas(background=obj.background, particles=obj._particles,
	                 res=obj.rez)
	    return getattr(new, methodname)(*args, **kwargs)
    return wrapper


def concat_canvas(c1, c2, bg_resolve='c2', **particle_args):
    """ Adds two canvas objects under various conditions """
    bg_resolve = bg_resolve.lower()
    
    # Choose output background
    if bg_resolve == 'merge':
        raise NotImplementedError
    elif bg_resolve == 'c1':
        bgout = c1.background
	rezout = c1.rez  #UNTESTED
    elif bg_resolve == 'c2':
        bgout = c2.background        
	rezout = c2.rez
    else:
        raise CanvasAttributeError('"bg_resolve" invalid; must be %s' % bg_valid)
    
    pout = concat_particles(c1._particles, c2._particles, **particle_args)
    return Canvas(background=bgout, particles=pout, rez=rezout)
        

class CanvasError(Exception):
    """ Custom exception """ 
    
class CanvasAttributeError(CanvasError):
    """ Custom exception """     

class Canvas(HasTraits):
    """  """
    image = Property(Array) #chaching doesnt work

    # ALL INTERNAL REFERENCES SHOULD GO TO _PARTICLES
    particles = Property(Instance(ParticleManager, depends_on='_particles'))
    _particles = Instance(ParticleManager)
    
    # Background is only a property because for _set validation...
    background = Property()
    _background = Array
    
    def __init__(self, particles=None, background=None, rez=None): #No other traits
        """ Load with optionally a background image and instance of Particle
            Manager"""
        
        if not particles:
            particles = ParticleManager()
        self._particles = particles
        
        # Set bg before resolution
        self._resolution = BGRES
        if background is None:
            self.reset_background() #sets default color/resolution    
        else:
            self.set_bg(background, rez, inplace=True) 
                        
    # Public Methods
    # -------------              
    def reset_background(self):
        """ Restore default background image; restores default RES, redraws
            particles over it."""

        self._resolution = BGRES  #must be set first
        self._background = self.color_background        
        
    def clear_canvas(self):
        """ Background image to default; removes ALL particles."""

        self.clear_particles()
        self.reset_background()
        
    def clear_particles(self):
        """ Clears all particles from image."""

        self._particles.plist[:] = []
            
 #   @inplace
    def pmap(self, fcn, *fcnargs, **fcnkwargs):
        """ Maps a function to each particle in ParticleManger; optionally
            can be done in place"""

	#self._particles.map(fcn, *fcnargs, **fcnkwargs)
        inplace = fcnkwargs.pop('inplace', False)
        if inplace:
            self._particles.map(fcn, *fcnargs, **fcnkwargs)
        else:
            cout = Canvas(background=self.background, particles=self._particles, 
                          rez=self.rez)
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
#    @inplace
    def from_labels(self, inplace=False, neighbors=4,
                    background=None, **pmangerkwds):
        """ """
        	        
        if background: # scikit api doesn't accept None
            background = int(background) 
            labels = morphology.label(self.grayimage, neighbors, background)
        else:
            labels = morphology.label(self.grayimage, neighbors)
            
        pout = ParticleManager.from_labels(labels, **pmangerkwds)
	
        if inplace:
            self._particles = pout
        else:
            return Canvas(background=self.background, particles=pout,
                          rez=self.rez)
	
	
    def patchshow(self, *args, **kwargs):
	""" ...
	args/kwargs include alpha, edgecolors, linestyles 
	
	Notes:
	Matplotlib API is setup that args or kwargs can be entered.  Order is
	important for args, but the correspond to same kwargs.
	"""
	
        axes, args, kwargs = _parse_ax(*args, **kwargs)	
	cbar = kwargs.pop('cbar', False)
	
	#Slightly quicker to directly access particle?
	# FOR PATICLES IN IMAGE ONLY?
	in_and_edges = self.pin + self.pedge
	patches = [p.particle.as_patch() for p in in_and_edges]
	p = PatchCollection(patches, *args, **kwargs)

	# Dont in an example
	if not axes:
	    fig, axes = plt.subplots()

	# Needed or only one color is used, even with colormap...
	p.set_array(np.arange(len(patches)))
#        p.set_array(np.array ( (p.color for p in in_and_edges) ) )
	axes.add_collection(p)
	
	# SHOULD SHOW TAKE A COLOR BAR?
	if cbar:
	    fig.colorbar(p)
	
	axes.set_xlim([0, self.rx])
	axes.set_ylim([self.ry, 0])    
	return axes #fig, axes?
    
        
    #http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
    def show(self, *args, **kwargs):
        """ Wrapper to imshow.  Converts to gray to allow color maps."""

        # This will pull out "ax", leaving remaing args/kwargs
        axes, args, kwargs = _parse_ax(*args, **kwargs)
        title = kwargs.pop('title', None)

        # cmap is the first argument in args
	if args or 'cmap' in kwargs: 
            image = self.grayimage
        else:
            image = self.image
                      
	if axes:
	    axes.imshow(image, *args, **kwargs)
	else:      # matplotlib API asymmetry
	    axes = plt.imshow(image, *args, **kwargs).axes        
	
	if title:
	    axes.set_title(title)
	return axes 
    	
    
    def _get_image(self):
        """ Creates image array of particles.  Tried fitting to a property or
        Traits events interface to control the caching, but manually choosing
        cache points proved to be easier."""
    
        # self.background is always a new array; otherwise would use np.copy
        image = self.background
        for p in self._particles:
            rr_cc, color = p.particle.rr_cc, p.color 
            rr_cc = coords_in_image(rr_cc, image.shape)
            image[rr_cc] = color
        return image 
    
        
    # Image Attributes Promoted
    # ------------------
    @property
    def shape(self):
	""" Avoid recomputing image"""
        return (self.rx, self.ry, 3)
    
    @property
    def ndim(self):
	""" Avoid recomputing image for this purpose """
	return len(self._background)
 
    @property
    def dtype(self):
	# Same dtype as image
        return self._background.dtype
    
    @property
    def grayimage(self):
        """ Collapse multi-channel, scale to 255 (via ubyte) """
	# img 127 --> ubyte of 123... 
	# Try this:
        #	print c.grayimage.max(), c.image.max() * 255, img_as_uint(lena()).max()
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
    def color_background(self):
        """ Generate a colored background at current resolution """
        return bgu.from_color_res(BGCOLOR, self.rx, self.ry)         
    
    # Image Resolution
    @property
    def rez(self):
        return self._resolution
    
    @rez.setter
    def rez(self, rez):
        rx, ry = rez
        self._resolution = ( int(rx), int(ry) )
        
    @property
    def rx(self):
        return self.rez[0]
    
    @rx.setter
    def rx(self, rx):
        self._resolution = ( int(rx), self._resolution[1] )
    
    @property
    def ry(self):
        return self.rez[1]    
    
    @ry.setter
    def ry(self, ry):
        self._resolution = ( self._resolution[0], int(ry) )    
    
    @property
    def pbinary(self):
        """ Returns boolean mask of all particles IN the image, with 
            background removed."""      
        # Faster to get all coords in image at once since going to paint white
        rr_cc = coords_in_image( self._particles.rr_cc_all, self.rez)
        out = np.zeros( self.rez, dtype=bool )
        out[rr_cc] = True
        return out  

    
    def _whereis(self, choice='in'):
        """ Wraps utils.where_is_particles for all three possibilities """
        return [p.name for p in self._particles 
                if where_is_particle(p.rr_cc, self.rez) == choice]

    
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
        l, w = self.rez
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
    
    def _set_image(self):
        raise CanvasError('Image cannot be set; please make changes to particles'
            ' and/or background attributes.')
    
    # BACKGROUND RELATED
    # ------------------
    
#    @inplace
    def set_bg(self, bg, keepres=False, inplace=False):
        """ Public background setting interface. """
	#print 'IN SET BG', bg, keepres
	
        #oldres = self.rez                
        #self._update_bg(bg)    
    
        #if keepres:
            #if keepres == True:
                #self.rez = oldres
            #else:
                #self.rez = keepres
                
        #else:
            #self.rez = self._background.shape[0:2]   	
    
        if inplace:
            cout = self
        else:
            cout = Canvas(background=self.background, particles=self._particles, 
                          rez=self.rez)    

        oldres = cout.rez                
        cout._update_bg(bg)    
    
        if keepres:
            if keepres == True:
                cout.rez = oldres
            else:
                cout.rez = keepres
                
        else:
            cout.rez = cout._background.shape[0:2]   

        if not inplace:
            return cout

        
    def crop_bg(self, *coords, **kwds):
        """ Crop self._background and set as current background.  Because
        using *args, can't follow with keywords"""
        # avoid set_bg() because uses size of coords, not coords itself
        inplace = kwds.pop('inplace', False)
        if inplace:
            cout = self
        else:
            cout = Canvas(background=self.background, particles=self._particles, 
                          rez=self.rez)            
            
        cout._background = crop(cout._background, coords)
        cout.rx, cout.ry = cout._background.shape[0:2]        
        
        if not inplace:
            return cout        
        
  
    def _get_background(self):
        """ Crop or extend self._background based on self.rx, ry.  Always 
        returns a new object to avoid accidental refernce passing."""
        bgx, bgy = self._background.shape[0:2]
	rx, ry = self.rez
        if bgx == rx and bgy == ry:
            return np.copy(self._background)
	
	def _smaller(idx1, idx2):
	    if idx1 < idx2:
		return idx1
	    return idx2
	
	xs, ys = _smaller(rx, bgx), _smaller(ry, bgy)
                
        out = np.empty( (rx, ry, 3) )
        out[:] = BGCOLOR #.fill only works with scalar	
        out[:xs, :ys] = self._background[:xs, :ys] #no copy needed	
        return out
    
    def _set_background(self, bg):
        """ Set background and use new resolution if there is one """
        self.set_bg(bg, keepres=False, inplace=True)
    
    # REDO THIS WITH COLOR NORM AND STUFF!  Also, dtype warning?
    def _update_bg(self, background):
        """ Parses several valid inputs including: None, Path, Color (of any 
        valid to_normRGB() type, ndarray, (Color, Resolution)."""

        if background is None:
            self._background = self.color_background
                                        
        elif isinstance(background, np.ndarray):
            self._background = background
            
        # colorstring or hex
        elif isinstance(background, basestring):
            self._background = bgu.from_string(background, self.rx, self.ry)
            
        # If not array, color is assume as valid to_norm_rgb(color) arg
        # It will raise its own error if failure occurs
        else:
            self._background = bgu.from_color_res(background, self.rx, self.ry)            
            
        # Float-color convert array       
        self._background = any2rgb(self._background, 'background')
        
        # IMAGE IS CACHED AT END OF _set_bg()
    
    # Delegate dictionary interface to ParticleManager
    # -----------
    def __getitem__(self, keyslice):
        """ Employs particle manager interface; however, returns single entry
            as a list to allow slicing directly into get_item[]"""
	return self._particles.__getitem__(keyslice)
    
        #IF WANT TO RETURN CANVAS ALWAYS SEE BELOW
        #pout = self._particles.__getitem__(keyslice)
	#return Canvas(background=self.background, particles=pout, rez=self.rez)
    
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
        res = '(%s X %s)' % (self.rx, self.ry ) 
        _PAD = ' ' * 3
        
        outstring = "Canvas at %s:\n" % self.mem_address

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
    
    from pyparty.utils import splot
    ax1, ax2 = splot(1,2)
    
    c.show(ax1)
    
    c=Canvas(background='black', rez=(800,800))
    
    c.add('circle', name='top_right', radius=75, phi=100, center=(400,100), color='y')
    c.add('line', color='yellow', center=(300,300), length=200, width=20, phi=30.0)
    c.add('square', color='purple', length=50, center=(200,200), phi=23.0)
    c.add('triangle', color='teal', length=50, center=(250,250), phi=23.0)
    
    c.add('circle', name='bottom_right', radius=20, center=(400,400), color='red')
    c.add('ellipse', name='bottom_left', center=(100,400), xradius=30, yradius=50, color='green', phi=52.0)
    c.add('circle', name='topleft_corner', radius=100, center=(0,0), color=(20,30,50) )
    c.add('circle', name='off_image', radius=50, center=(900,200), color='teal')
    c.add('polygon', name='bowtie', color='orange', phi=50.0)

    clab = c.from_labels() #neighbors = 4 default connectivity

    clab.patchshow()
    
    print c.plist[0].particle.as_patch()
        
    
    for p in c.particles:
	
	p.as_patch()
    c.show()
    #c.patchshow()
##    plt.show()
    #from skimage.data import moon
    #c.set_bg(moon())
    
    #print c
    
    
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