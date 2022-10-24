import cv2 as cv
import os
import numpy as np
# работа над изображениями
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.filters import sobel, gaussian, threshold_local, try_all_threshold, threshold_otsu,threshold_mean
from skimage.data import page
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
# импортируем функцию label под другим именем, чтобы не терять её, если появляется переменная label
from skimage.measure import label as sk_measure_label
import cv2


def template_detecting(path_template: str, path_photo: str, ratio_thresh=0.75, number_of_matchers=4) -> bool:
    """
    :param number_of_matchers: matchers for detecting
    :param ratio_thresh:   for Lowe's ratio test
    :param path_template: template path
    :param path_photo:    target photo path
    :return: detect template in target or not
    """

    img_object = cv.imread(path_template, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(path_photo, cv.IMREAD_GRAYSCALE)

    #def to_uint8(img):
    #    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    #img_object = to_uint8(img_object - threshold_local(img_object, 31, method='median'))
    #img_scene = to_uint8(img_scene - threshold_local(img_scene, 31, method='median'))

    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    detector = cv.xfeatures2d.SIFT_create()
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    # -- Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) > number_of_matchers:
        return True
    return False
