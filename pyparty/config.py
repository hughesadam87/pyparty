import numpy as np

def rint(x): return int(round(x,0))

""" Shared constants in pypart """

# Particle Manager
# ------------

NAMESEP = '_' 
PADDING = 3
ALIGN = 'l'
MAXOUT = 50 #How many rows to show before cutting off

# Attributes to show when particles printed
PRINTDISPLAY = ('name', 'ptype', 'center', 'phi') 

#Should new ParticleManager make new Particles?
_COPYPARTICLES = True 

# __repr__ indentation for Canvas/Grids
_PAD = ' ' * PADDING

# Color-related
# -------------
_8_bit = ('uint8', 255)
#_16_bit = ('uint16', 65536) UNTESTED

# Colors
# -------
COLORTYPE = _8_bit
#PCOLOR = (0.0, 0.0, 1.0) # default particle color
PCOLOR = 'random'
BGCOLOR = 'white'# default background color
GCOLOR = 'red'  #default grid color

# Canvas
# ------
BGRES = (512, 512) # Default canvas background resolution
THRESHDEF = 'img2bool' #Default binarization function

# Grid
# ----

# BACKWARDS BECAUSE GRID IS RELATIVE INVERSE
_points = 15
GRIDXSPACE = rint( BGRES[0] / _points )#pixels / tile (20 tiles on 512 or so)
GRIDYSPACE = rint( BGRES[1] / _points )

# Shape model defaults
# --------------------

# Center particles (to middle of default image BACKGROUND size)
RADIUS_DEFAULT = 20  
# USED BY ALL PARTICLES
CENTER_DEFAULT = tuple(map(lambda x: int(x/2.0), BGRES))

#Ellipse 
XRADIUS = 10
YRADIUS = 20

#SEGMENTS/BEZIER CURVES
XSTART = 206
YSTART = 206
XMID = 256
YMID = 256
XEND = 306
YEND = 306
BEZIERWEIGHT=1.0

#POLYGONS
XVERTS = np.array( (220, 220, 280, 280) ) #NOT CENTERED
YVERTS = np.array( (220, 280, 220, 280) )
RECTLEN = 20
RECTWID = 10 #Not used
LINELEN = 50
LINEWID = 1
