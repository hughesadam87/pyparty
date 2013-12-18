""" User-based descriptor.   These must accept a binary labeled-array to be
    compatible with Particle() type in shape model."""

def even_or_odd(label_image):
    """ Determines whether or not a labeled image has an even or odd 
        amount of 1's/Trues """
    if np.sum(label_image) % 2 == 0:
        return 'even'
    else:
        return 'odd'