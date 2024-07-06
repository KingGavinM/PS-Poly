import numpy as _np

from scipy.signal import convolve2d as _convolve2d
from skimage.filters import threshold_otsu as _otsu
from skimage.morphology import skeletonize as _skeletonize
from skimage.measure import label as _label

from . import utilities as _utilities

paths = _np.array([0., 1., 1., 1., 1., 2., 1., 1., 1., 2., 2., 2., 1., 2., 1., 1., 1., 2., 2., 2., 2., 3., 2., 2., 1., 2., 2., 2., 1., 2., 1., 1., 1., 2., 2., 2., 2., 3., 2., 2., 2., 3., 3., 3., 2., 3., 2., 2., 1., 2., 2., 2., 2., 3., 2., 2., 1., 2., 2., 2., 1., 2., 1., 1., 1., 2., 2., 2., 2., 3., 2., 2., 2., 3., 3., 3., 2., 3., 2., 2., 2., 3., 3., 3., 3., 4., 3., 3., 2., 3., 3., 3., 2., 3., 2., 2., 1., 2., 2., 2., 2., 3., 2., 2., 2., 3., 3., 3., 2., 3., 2., 2., 1., 2., 2., 2., 2., 3., 2., 2., 1., 2., 2., 2., 1., 2., 1., 1., 1., 1., 2., 1., 2., 2., 2., 1., 2., 2., 3., 2., 2., 2., 2., 1., 2., 2., 3., 2., 3., 3., 3., 2., 2., 2., 3., 2., 2., 2., 2., 1., 2., 2., 3., 2., 3., 3., 3., 2., 3., 3., 4., 3., 3., 3., 3., 2., 2., 2., 3., 2., 3., 3., 3., 2., 2., 2., 3., 2., 2., 2., 2., 1., 1., 1., 2., 1., 2., 2., 2., 1., 2., 2., 3., 2., 2., 2., 2., 1., 2., 2., 3., 2., 3., 3., 3., 2., 2., 2., 3., 2., 2., 2., 2., 1., 1., 1., 2., 1., 2., 2., 2., 1., 2., 2., 3., 2., 2., 2., 2., 1., 1., 1., 2., 1., 2., 2., 2., 1., 1., 1., 2., 1., 1., 1., 1., 0.])
bundle = _np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
end_patterns = _np.array([1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254])
branch_patterns = _np.array([ 21,  37,  41,  42,  43,  45,  53,  69,  73,  74,  75,  77,  81, 82,  83,  84,  85,  86,  87,  89,  90,  91,  93, 101, 105, 106, 107, 109, 117, 138, 146, 148, 149, 150, 154, 162, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 178, 180, 181, 182, 186, 202, 210, 212, 213, 214, 218, 234])


def otsu(img):
    threshold = _otsu(img)
    mask = img > threshold
    return mask.astype('int')

def threshold(img):
    threshold = _otsu(img)
    mask = (img > threshold).astype('int')

    l_mask = _label(mask)
    polymers = []
    for v in range(_np.max(l_mask)):
        poly = (l_mask == v+1).astype('int')
        w = _np.where(poly == 1)
        bnd = ((_np.min(w[0]), _np.max(w[0])), (_np.min(w[1]), _np.max(w[1])))
        if bnd[0][0] != 0 and bnd[1][0] != 0 and bnd[0][1]+1 != mask.shape[0] and bnd[1][1]+1 != mask.shape[1]: polymers.append(poly)

    return sum(polymers)

def skeletonize(img):
    return _skeletonize(img).astype('int')

k_encode = _np.array([
    [  1, 2, 4],
    [128, 0, 8],
    [ 64,32,16]
])

def encode(img):
    return (img*_convolve2d(img,k_encode,mode='same',boundary='fill')).astype('int')

def mainpoints(img):
    encoded = encode(img)
    
    end_points = _np.zeros(encoded.shape)
    branch_points = _np.zeros(encoded.shape)
    bundle_points = _np.zeros(encoded.shape)

    for iy in range(encoded.shape[0]):
	    for ix in range(encoded.shape[1]):
		    e = encoded[iy,ix]
		    if e in end_patterns: end_points[iy,ix] = 1
		    if e in branch_patterns: branch_points[iy,ix] = 1
		    if bundle[e] == 1: bundle_points[iy,ix] = 1

    branch_points2 = _utilities.relu((img*_convolve2d(branch_points+bundle_points,[[1,1,1],[1,1,1],[1,1,1]],mode='same',boundary='fill') > 0).astype('float')-branch_points-bundle_points)

    return (end_points, branch_points, bundle_points, branch_points2)

def label_bound(img):
    labeled_img = _label(img)

    polymers = []
    bounds = []
    for v in range(_np.max(labeled_img)):
        poly = (labeled_img == v+1).astype('int')
        w = _np.where(poly == 1)
        bnd = ((_np.min(w[0]), _np.max(w[0])), (_np.min(w[1]), _np.max(w[1])))
        polymers.append(poly[bnd[0][0]:bnd[0][1]+1,bnd[1][0]:bnd[1][1]+1])
        bounds.append(bnd)

    return polymers, bounds