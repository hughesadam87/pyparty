import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

## Quick curve fitting of BSA paper from nist
x=[10.0 , 30.0 , 60.0]  #Particle diams 10,30,60nm
y=[0.023, 0.017, 0.014]  #BSA per square nm assuming spheres, converted x--> area
cov=[60.0, 44.0, 36.0] #Coverage percentage corresponding to bsa/per square nm (y)

def bsa_count(diams, style='single'):
    ''' Returns bsa molecules per unit surface area given a particle diameter,
    and a fitting style.  Essentially just returns the y value of a fit curve
    given x (diamter).'''

    if style=='single':
        z=np.polyfit(x, y, 1)  
        p=np.poly1d(z)        
        return p(diams)
                        
    elif style=='dual':
        dout=[]

        x1=x[0:2] #Make x[0:2]
        y1=y[0:2]# ditto
        z1=np.polyfit(x1, y1, 1)  
        p1=np.poly1d(z1)         
            
        x2=x[1:3]   #Make x[1:3]
        y2=y[1:3] # ditto
        z2=np.polyfit(x2, y2, 1)  
        p2=np.poly1d(z2)         
                
        for d in diams:
            if d < x[1]:  #If d < 30
                dout.append(p1(d))
            else:
                dout.append(p2(d))
        return dout
         
    else:
        raise AttributeError('syle must be "single" or "dual", not %s'%style)


# IS THIS ACTUALLY IN USE
def _map_cov(bsa_area):
    ''' Given bsa surface area, map this to percent coverage using the fact
    that 0.0386nm-2 is 100% coverage'''
    return 100.0* ( bsa_area / 0.0386)
