from pyparty.shape_models.basic import Circle, BezierCurve, Ellipse
from pyparty.shape_models.polygons import Line, Polygon, Rectangle, Square, Triangle
from pyparty.shape_models.multi import Dimer, Trimer, Tetramer

# Shapes with alternate constructors ( auto_init() decides based on kwds)
Polygon = Polygon.auto_init
Rectangle = Rectangle.auto_init
Square = Square.auto_init
Line = Line.auto_init
Triangle = Triangle.auto_init


GROUPEDTYPES= \
    {
    'simple':
        {
        'circle': Circle,
        'bezier' : BezierCurve,
        'ellipse' : Ellipse,
        'line' : Line, 
         },
        
    'polygon':
        {
        'polygon': Polygon,  
        'triangle': Triangle,
        'rectangle': Rectangle,
        'square': Square,
        },

    'n-circle':
        {
        'dimer' : Dimer,
        'trimer' : Trimer,
        'tetramer' : Tetramer            
        }
    }

ALLTYPES = dict(
                GROUPEDTYPES['simple'].items() + 
                GROUPEDTYPES['polygon'].items() + 
                GROUPEDTYPES['n-circle'].items() 
               )
