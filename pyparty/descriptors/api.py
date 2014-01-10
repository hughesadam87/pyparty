""" Stores descriptors availabe in scikit, as well as mapping to user-defined
    descriptors. """

from user_descriptors import even_or_odd

class DescriptorError(Exception):
    """ """

# Some of these rely on image intensity: Separate these out into different set of properties.
# Weighted centroid, for example, IS BACKGROUND IMAGE SPECIFIC.  While ares is intrinsice.

#Some properties also suffer from positional problems since we have to translate the coordinates
#To build the fast bounding box.  To fix this, either we don't translate the coordinates and
#instead project the particle onto a full image.  Alternatively, we modify the positional descriptors
# (like centroid) to have a translate, untranslate dectorator.

SKIMAGE_DESCRIPTORS = [
    'area',
#    'bbox',
    'centroid',
    'convex_area',
    'convex_image',
#    'coords',      #
    'eccentricity',
    'equivalent_diameter',
    'extent',
    'filled_area',  # DO I WANT TO KEEP THIS?

#    'filled_image',
#    'image',
    'inertia_tensor',
    'inertia_tensor_eigenvalues',
    'label', 
    'major_axis_length',
#    'min_intensity',
    'minor_axis_length',
    'moments',
    'moments_central', 
    'moments_hu', 
    'moments_normalized',

    'orientation',  

    'perimeter',
    'solidity',
#    'weighted_centroid',   #THESE ALL DEPEND ON IMAGE_INTENSITY OPTION
#    'weighted_moments',
#    'weighted_moments_normalized'
    ]

CUSTOM_DESCRIPTORS = {
    'evenorodd' : even_or_odd
    }

def add_descriptor(name, fcn):
    """ Add a descriptor to CUSTOM_DESCRIPTORS """
    if name in CUSTOM_DESCRIPTORS:
        raise DescriptorError("Descriptor named %s already found!" % name)
    CUSTOM_DESCRIPTORS[name] = fcn

ALL_DESCRIPTORS = SKIMAGE_DESCRIPTORS + CUSTOM_DESCRIPTORS.keys()

# Ensure unique names between SKIMAGE and USER descriptors
if len( list(set(ALL_DESCRIPTORS)) ) != len(ALL_DESCRIPTORS):
    duplicates = [i for i in SKIMAGE_DESCRIPTORS if i in CUSTOM_DESCRIPTORS.keys()]
    raise DescriptorError("Non-unique descriptors found: %s" % duplicates)

    