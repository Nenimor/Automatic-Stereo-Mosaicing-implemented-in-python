# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
from scipy.misc import imsave


import sol4_utils
from sol4_utils import *
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import convolve

#######   CONSTANTS   ########
GAUSSIAN_KERNEL = np.asmatrix(np.array([1, 1]))
MIN_KER_SIZE = 3
GRAYSCALE = 1
GRAY_MAX_LVL = 255
MAX_SHRINK_DIM = 16
RESIZING_FACTOR = 2
MASK_THRESHOLD = 0.9
X_DERIVATIVE_KERNEL = np.asmatrix(np.array([1, 0, -1]))
Y_DERIVATIVE_KERNEL = X_DERIVATIVE_KERNEL.reshape(3, 1)
RESPONSE_FACTOR = 0.04
Z_HOMOG_COORD = 1

SPLIT_ROW_DIM = 3
SPLIT_COL_DIM = 3
DESC_RADIUS = 3 # descriptor radius
BLUR_KERNEL_SIZE = 3


def harris_corner_detector(im):
  """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  x_derivative = conv(im, X_DERIVATIVE_KERNEL, mode='same', boundary='symm')
  y_derivative = conv(im, Y_DERIVATIVE_KERNEL, mode='same', boundary='symm')
  Ix2 = blur_spatial(x_derivative ** 2, BLUR_KERNEL_SIZE)
  Iy2 = blur_spatial(y_derivative ** 2, BLUR_KERNEL_SIZE)
  IxIy = blur_spatial(x_derivative * y_derivative, BLUR_KERNEL_SIZE)
  m_det = (Ix2 * Iy2) - (IxIy * IxIy)
  m_trace = Ix2 + Iy2
  response_values = m_det - RESPONSE_FACTOR * (m_trace ** 2)
  binary_response = non_maximum_suppression(response_values)
  corners_indices = np.argwhere(binary_response.T)
  return corners_indices


def sample_descriptor(im, pos, desc_rad):
  """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
  pos = pos / 4 # transforms coordinates from original image, to coordinates in gaussian pyr lvl 3 image
  sample_dim = 2 * desc_rad + 1
  patches_arr = []
  for point in pos:
    patch = np.zeros((sample_dim, sample_dim))
    x = point[1]
    y = point[0]
    row_of_point = np.arange(x - desc_rad, x + desc_rad + 1)
    for i in range(sample_dim):
      col_of_point = [y - desc_rad + i] * sample_dim
      interp_line = map_coordinates(im, [row_of_point, col_of_point], order=1, prefilter=False)
      patch[i, : ] = interp_line
    mean = np.mean(patch)
    patch -= mean
    norm = np.linalg.norm(patch)
    if (norm):
      patch = patch / norm
    patches_arr.append(np.copy(patch))
  return np.array(patches_arr)


def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  key_points = spread_out_corners(pyr[0], SPLIT_COL_DIM, SPLIT_ROW_DIM, 1)
  points_descriptors = sample_descriptor(pyr[2], key_points, DESC_RADIUS)
  return [key_points, points_descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    desc1_indices = []
    desc2_indices = []
    scores_matrix = np.tensordot(desc1, desc2, [[1, 2], [1, 2]])  # Computes the dot product of all possible descriptor
    # pairs, score is a matrix where element i,j is the dot product of desc1[i] and desc2[j]

    # gets the indices that would sort scores matrix- by rows and by cols
    sorted_rows_score = np.argsort(scores_matrix)
    sorted_cols_score = np.argsort(scores_matrix.T)

    for i in range(np.shape(sorted_rows_score)[0]):
        #checks whether a value is in 2 max values in its row and column, by comparing the 2 last columns of each
        # sorted matrix (by rows and by columns)
        max_in_row_index = sorted_rows_score[i, -1]
        if sorted_cols_score[max_in_row_index, -1] == i or sorted_cols_score[max_in_row_index, -2] == i:
            if (scores_matrix[i, max_in_row_index] > min_score):
                desc1_indices.append(i)
                desc2_indices.append(max_in_row_index)
        sec_max_in_row_index = sorted_rows_score[i,-2]
        if sorted_cols_score[sec_max_in_row_index, -1] == i or sorted_cols_score[sec_max_in_row_index, -2] == i:
            if (scores_matrix[i, sec_max_in_row_index] > min_score):
                desc1_indices.append(i)
                desc2_indices.append(sec_max_in_row_index)
    desc1_indices = np.asarray(desc1_indices, dtype=int)
    desc2_indices = np.asarray(desc2_indices, dtype=int)
    return (desc1_indices, desc2_indices)

def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    row_dim, col_dim = pos1.shape
    z_tag = np.full((row_dim, 1), Z_HOMOG_COORD, dtype=int)
    homog_points = np.hstack((pos1, z_tag))
    trans_points = np.dot(H12, homog_points.T).T
    z_tag = trans_points[:, 2]
    trans_points[:, 0] = trans_points[:, 0] / z_tag
    trans_points[:, 1] = trans_points[:, 1] / z_tag
    trans_2d_points = trans_points[:, 0:2]
    return trans_2d_points

def compute_inliers(points1, points2, inlier_tol, translation_only=False):
    '''
    performs a single iteration of RANSAC algorithm
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: group of inlier indices
    '''
    points_num = len(points1)
    first_point = np.random.randint(0, points_num)
    img1_points = [points1[first_point]]
    img2_points = [points2[first_point]]
    translation_indicator = translation_only
    while not translation_indicator:
        sec_point = np.random.randint(0, points_num)
        if (first_point != sec_point):
            img1_points.append(points1[sec_point])
            img2_points.append(points2[sec_point])
            break
    img1_points = np.asarray(img1_points)
    img2_points = np.asarray(img2_points)
    homography = estimate_rigid_transform(img1_points, img2_points, translation_only)
    trans_img1_points = apply_homography(points1, homography)
    euclidean_dist = (np.linalg.norm(trans_img1_points - points2, axis=1)) ** 2
    inliers = np.argwhere(euclidean_dist < inlier_tol)
    return inliers

def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    greatest_inliers = compute_inliers(points1, points2, inlier_tol, translation_only)
    for i in range(num_iter - 1):
        inliers = compute_inliers(points1, points2, inlier_tol, translation_only)
        if (len(inliers) > len(greatest_inliers)):
            greatest_inliers = inliers
    points1_inliers = np.reshape(points1[greatest_inliers].flatten(), (greatest_inliers.shape[0], 2))
    points2_inliers = np.reshape(points2[greatest_inliers].flatten(), (greatest_inliers.shape[0], 2))
    homography = estimate_rigid_transform(points1_inliers, points2_inliers, translation_only)
    greatest_inliers = greatest_inliers.reshape(np.shape(greatest_inliers)[0],)
    return homography, greatest_inliers


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    x1, y1 = points1.T
    x2, y2 = points2.T

    shift = im1.shape[1]
    concat_img = np.hstack((im1, im2))
    plt.imshow(concat_img, cmap='gray')

    for i in range(points1.shape[0]):
        if i in inliers:
            plt.plot([x1[i], x2[i] + shift], [y1[i], y2[i]], mfc='r', c='y', lw=1, ms=5, marker='.')
        else:
            plt.plot([x1[i], x2[i] + shift], [y1[i], y2[i]], mfc='r', c='b', lw=0.3, ms=5, marker='.')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    homographies_list = [np.eye(3)]
    for i in range(m - 1, -1, -1):
        homography = np.dot(homographies_list[0], H_succesive[i])
        homography /= homography[2, 2]
        homographies_list = [homography] + homographies_list
    for i in range(m, len(H_succesive)):
        homography = homographies_list[-1].dot(np.linalg.inv(H_succesive[i]))
        homography /= homography[2, 2]
        homographies_list.append(homography)
    return homographies_list


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0,0], [0,h], [w,0], [w, h]])
    trans_corners = apply_homography(corners, homography).astype(np.int)
    top_left = [min(trans_corners[:,0]), min(trans_corners[:,1])]
    bottom_right = [max(trans_corners[:, 0]), max(trans_corners[:, 1])]
    return np.array([top_left, bottom_right])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h, w = np.shape(image)
    boundary = compute_bounding_box(homography, w, h)
    top_left = boundary[0]
    bottom_right = boundary[1]
    x_values = np.arange(top_left[0], bottom_right[0])
    y_values = np.arange(top_left[1], bottom_right[1])
    x_values, y_values = np.meshgrid(x_values, y_values)
    x_values = np.ndarray.flatten(x_values).T
    y_values = np.ndarray.flatten(y_values).T
    points_to_trans = np.column_stack((x_values, y_values))
    inv_homog = np.linalg.inv(homography)
    backwarped_points = apply_homography(points_to_trans, inv_homog)
    interp_points = map_coordinates(image, [backwarped_points.T[1], backwarped_points.T[0]], order=1, prefilter=False)
    row_size = bottom_right[1] - top_left[1]
    col_size = bottom_right[0] - top_left[0]
    warped_img = interp_points.reshape((row_size, col_size))
    return warped_img


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0, -1]
  for i in range(1, len(homographies)):
    if homographies[i][0, -1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0, -1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2, :2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2, 2)
  local_max = maximum_filter(image, footprint=neighborhood) == image
  local_max[image < (image.max() * 0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:, 0], centers[:, 1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0, 2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
           (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
  ret = corners[legit, :]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]

  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))

  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()








