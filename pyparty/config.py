""" Shared constants in pypart """

# Particle Manager
# ------------
NAMESEP = '_' 
PADDING = 3
ALIGN = 'l'
MAXOUT = 50 #How many rows to show before cutting off


# Color Constants
# ---------------
_8_bit = ('uint8', 255)
#_16_bit = ('uint16', 65536) UNTESTED

PCOLOR = (0.0, 0.0, 1.0)
COLORTYPE = _8_bit

# Canvas
# ------
BGCOLOR = (1.0, 1.0, 1.0)
BGRES = (512, 512)

# Shape model defaults
# --------------------

# Center particles (to middle of default image BACKGROUND size)
RADIUS_DEFAULT = 20  
CENTER_DEFAULT = tuple(map(lambda x: int(x/2.0), BGRES))

#Ellipse 
XRADIUS = 10
YRADIUS = 20

#LINE/BEZIER CURVES
XSTART = 206
YSTART = 206
XMID = 256
YMID = 256
XEND = 306
YEND = 306
BEZIERWEIGHT=1.0