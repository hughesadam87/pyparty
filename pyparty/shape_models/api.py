from pyparty.shape_models.basic import Circle, BezierCurve, Ellipse, \
     Polygon, Line

PARTICLETYPES= \
    {
     'circle': Circle,
     'bezier' : BezierCurve,
     'ellipse' : Ellipse,
     'line' : Line, 
     'polygon' : Polygon
     }

# Eventually, break these up like "CIRCLETYPES", "RECTANGLES" etc...