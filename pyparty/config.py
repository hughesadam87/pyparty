import numpy as np

""" Shared constants in pypart """

# Particle Manager
# ------------

NAMESEP = '_' 
PADDING = 3
ALIGN = 'l'
MAXOUT = 50 #How many rows to show before cutting off

# Attributes to show when particles printed
PRINTDISPLAY = ('name', 'ptype', 'radius', 'phi') 

#Should new ParticleManager make new Particles?
_COPYPARTICLES = True 


# Color-related
# -------------
_8_bit = ('uint8', 255)
#_16_bit = ('uint16', 65536) UNTESTED

COLORTYPE = _8_bit
PCOLOR = (0.0, 0.0, 1.0) # default particle color
BGCOLOR = (1.0, 1.0, 1.0) # default background color

# Canvas
# ------
BGRES = (512, 512) # Default canvas background resolution

# Grid
# ----


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
XVERTS = np.array( (220, 220, 280, 280) )
YVERTS = np.array( (220, 280, 220, 280) )
RECTLEN = 20
RECTWID = 10 #Not used
LINEWID = 3
