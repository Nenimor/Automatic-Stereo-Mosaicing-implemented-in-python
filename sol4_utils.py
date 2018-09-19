import numpy as np
from scipy.misc import imread as imread
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt #TODO
from skimage.color import rgb2gray
import copy #TODO
import os #TODO

#######   CONSTANTS   ########
GAUSSIAN_KERNEL = np.asmatrix(np.array([1,1]))
MIN_KER_SIZE = 3
GRAYSCALE = 1
GRAY_MAX_LVL = 255
MAX_SHRINK_DIM = 16
RESIZING_FACTOR = 2
MASK_THRESHOLD = 0.9

BINOMIAL_BASE = np.matrix(np.array([1, 1])).astype(np.float64)

def read_image(filename, representation):
    '''
    this function reads an image file and converts it into a given representation
    :param filename: string containing the image filename to read
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2)
    :return: an image, represented by a matrix of type np.float64 with
    intensities normalized to the range [0, 1], according to given
    representation.
    '''
    image = imread(filename)
    image = image.astype(np.float64)
    if representation == GRAYSCALE:
        image = rgb2gray(image)
    image /= GRAY_MAX_LVL
    return image

def create_gaussian_vec_ker(vec_size):
    '''
    this function creates a gaussian kernel by convolution of [1 1] with itself, kernel_size times
    :param kernel_size: number of convolution operations to be performed
    :return: tuple of the (gaussian coefficients vector, gaussian kernel)
    '''
    if (vec_size < MIN_KER_SIZE):
        return np.asmatrix(np.array([1]))
    gaussian_vec = GAUSSIAN_KERNEL
    for i in range(vec_size - 2):
        gaussian_vec = conv(GAUSSIAN_KERNEL, gaussian_vec, mode='full')
    gaussian_ker = conv(gaussian_vec, gaussian_vec.reshape((vec_size, 1)), mode='full')
    vec_normalize_factor = np.sum(gaussian_vec)
    ker_normalize_factor = np.sum(gaussian_ker)
    gaussian_vec = gaussian_vec / vec_normalize_factor
    gaussian_ker = gaussian_ker / ker_normalize_factor
    return (gaussian_vec, gaussian_ker)

def shrink_image(img, filter_vec):
    '''
    this function shrinks the image size by the factor of 1/RESIZING_FACTOR(1/2)
    :param img: a given image desired to be shrunk
    :param filter_vec: the filter vec used to blur the image
    :return: the shrunk blurred image
    '''
    blurred_img = convolve(img, np.asmatrix(filter_vec))
    filter_vec = filter_vec.T
    blurred_img = convolve(blurred_img, np.asmatrix(filter_vec))
    shrunk_img = blurred_img[::RESIZING_FACTOR, ::RESIZING_FACTOR]
    return shrunk_img

def blur_spatial(im, kernel_size):
    '''
    this function blurs the given image- im, by convolution with gaussian kernel
    :param im: the image to be blurred
    :param kernel_size: the desired gaussian kernel size
    :return: the blurred image
    '''
    if (kernel_size < MIN_KER_SIZE):
        return im
    gaussian_ker = create_gaussian_vec_ker(kernel_size)[1]
    blurred_img = conv(im, gaussian_ker, mode='same', boundary='wrap')
    return blurred_img

def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    this function creates a gaussian image pyramid
    :param im: the image to create the pyramid from
    :param max_levels: number of max levels in the pyramid
    :param filter_size: the size of the filter used to blur the image
    :return: tuple of the pyramid, and the filter used to create it
    '''
    filter_vec = create_gaussian_vec_ker(filter_size)[0]
    pyr = []
    pyr.append(im)
    for i in range(max_levels - 1):
        shrunk_img = shrink_image(im, filter_vec)
        pyr.append(shrunk_img)
        im = shrunk_img
        rows_dim, cols_dim = np.shape(im)
        if (rows_dim < MAX_SHRINK_DIM * RESIZING_FACTOR or cols_dim < MAX_SHRINK_DIM * RESIZING_FACTOR):
            break
    return pyr, filter_vec

def img_zero_padding(img):
    '''
    this function expands the given image by padding it with zeros in every 2nd pixel, and a row of zeros in every
    2nd row
    :param img: the image desired to be padded
    :return: the padded image
    '''
    new_shape = tuple(2 * np.array(np.shape(img)))
    padded_img = np.zeros(new_shape, dtype=np.float64)
    padded_img[::RESIZING_FACTOR, ::RESIZING_FACTOR] = img
    return padded_img

def expand_img(img, filter_vec):
    '''
    this function expands the image size by the factor of RESIZING_FACTOR(2), by padding it with zeros, and blurring it
    :param img: a given image desired to be expanded
    :param filter_vec: the filter vec used to blur the image
    :return: the expanded image
    '''
    padded_img = img_zero_padding(img)
    filter_vec = RESIZING_FACTOR * filter_vec
    expanded_img = convolve(padded_img, np.asmatrix(filter_vec))
    filter_vec = filter_vec.T
    expanded_img = convolve(expanded_img, np.asmatrix(filter_vec))
    return expanded_img

def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    this function creates a laplacian image pyramid
    :param im: the image to create the pyramid from
    :param max_levels: number of max levels in the pyramid
    :param filter_size: the size of the filter used to blur the image
    :return: tuple of the pyramid, and the filter used to create it
    '''
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    max_levels = len(gaussian_pyr)
    for i in range(max_levels - 1):
        pyr.append(gaussian_pyr[i] - expand_img(gaussian_pyr[i+1], filter_vec))
    pyr.append(gaussian_pyr[max_levels - 1])
    return pyr, filter_vec

def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    this function reconstructs the original image from a given laplacian pyramid of the image
    :param lpyr: a laplacian pyramid of an image
    :param filter_vec: the filter vector used to create the laplacian pyramid
    :param coeff: a vector of coefficients to multiply the given pyramid with
    :return: the original reconstructed image
    '''
    pyr_levels = len(lpyr)
    original_img = lpyr[pyr_levels - 1]
    for i in range(pyr_levels - 2, -1, -1):
        lpyr[i] *= coeff[i]
        original_img = expand_img(original_img, filter_vec) + lpyr[i]
    return original_img

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    this function blends two images into one image by the given mask
    :param im1: grayscale image to be blended
    :param im2: grayscale image to be blended
    :param mask: a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the max_levels parameter used when generating the Gaussian and Laplacian
            pyramids
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: filter_size_mask { is the size of the Gaussian filter(an odd scalar that represents a
            squared filter) which defining the filter used in the construction of the Gaussian pyramid of mask.
    :return:
    '''
    mask = (mask > MASK_THRESHOLD)
    laplac_pyr1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplac_pyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_gauss_pyr = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    laplac_out = []
    max_levels = len(laplac_pyr1)
    for i in range(max_levels):
        new_level = mask_gauss_pyr[i] * laplac_pyr1[i] + (1 - mask_gauss_pyr[i]) * laplac_pyr2[i]
        laplac_out.append(new_level)
    coeff = np.ones(np.shape(laplac_out))
    im_blend = laplacian_to_image(laplac_out, filter_vec, coeff)
    return np.clip(im_blend, 0, 1)
