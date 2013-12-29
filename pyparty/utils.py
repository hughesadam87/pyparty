import numpy as np

from enthought.traits.api import BaseInt

class IntOrNone ( BaseInt ):
    """ Traits whose value is an integer or None"""
    default_value = None

    # Describe the trait type
    info_text = 'an integer or None'

    def validate ( self, object, name, value ):
        value = super(IntOrNone, self).validate(object, name, value)
        if value is None:
            return value        

        try:
            return int(value)
        except ValueError:
            self.error( object, name, value )

def coords_in_image(rr_cc, shape):
    ''' Taken almost directly from  skimage.draw().  Decided best not to
        do any formatting implicitly in the shape models.
        
        Attributes
        ----------
        rr_cc : len(2) iter
            rr, cc returns from skimage.draw(); or shape_models.rr_cc
            
        shape : len(2) iter
            image dimensions (ie 512 X 512)
        
        Returns
        -------
        (rr, cc) : tuple(rr[mask], cc[mask])

        '''

    rr, cc = rr_cc 
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return (rr[mask], cc[mask])            
            
def where_is_particle(rr_cc, shape):
    ''' Quickly evaluates if particle rr, cc is fully within, partically in,
        or is outside and image.  Does this by comparin shapes, so is fast,
        but does not track which portions are outside, inside or on edge.
        
        Attributes
        ----------
        rr_cc : len(2) iter
            rr, cc returns from skimage.draw(); or shape_models.rr_cc
            
        shape : len(2) iter
            image dimensions (ie 512 X 512)
        
        Returns
        -------
        'in' / 'out' / 'edge' : str
        '''
    
    
    rr_cc_in = coords_in_image(rr_cc, shape)

    # Get dimensions of rr_cc vs. rr_cc_in
    dim_full = ( len(rr_cc[0]), len(rr_cc[1]) )
    dim_in = ( len(rr_cc_in[0]), len(rr_cc_in[1]) ) 
    
    if dim_in == dim_full:
        return 'in'
    elif dim_in == (0, 0):
        return 'out'
    else:
        return 'edge'
    
def rr_cc_box(rr_cc):
    """ Center the rr_cc values in a binarized box."""

    rr, cc = rr_cc
    ymin, ymax, xmin, xmax = rr.min(), rr.max(), cc.min(), cc.max()

    # Center rr, cc mins to 0 index
    rr_trans = rr - ymin
    cc_trans = cc - xmin
    rr_cc_trans = (rr_trans, cc_trans)
    
    dx = xmax-xmin
    dy = ymax-ymin
    
    rect=np.zeros((dy+1, dx+1), dtype='uint8') 
    rect[rr_cc_trans] = 1
    return rect