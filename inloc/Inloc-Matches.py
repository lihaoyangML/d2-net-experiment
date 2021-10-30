import cv2

import matplotlib.pyplot as plt

import numpy as np

import os

from PIL import Image

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

image_name_1 = "IMG_0731.JPG"
image_name_2 = "IMG_0732.JPG"
image_d2_net_1 = "IMG_0731.JPG.d2-net"
image_d2_net_2 = "IMG_0732.JPG.d2-net"
pair_path = "../../hierarchical_localization/Hierarchical-Localization/datasets/inloc/query/iphone7/"

image1 = np.array(Image.open(os.path.join(pair_path, image_name_1)))
image2 = np.array(Image.open(os.path.join(pair_path, image_name_2)))
feat1 = np.load(os.path.join(pair_path, image_d2_net_1))
feat2 = np.load(os.path.join(pair_path, image_d2_net_2))

matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
print('Number of raw matches: %d.' % matches.shape[0])

keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
np.random.seed(0)
model, inliers = ransac(
    (keypoints_left, keypoints_right),
    ProjectiveTransform, min_samples=4,
    residual_threshold=4, max_trials=10000
)
n_inliers = np.sum(inliers)
print('Number of inliers: %d.' % n_inliers)

inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)

plt.figure(figsize=(15, 15))
plt.imshow(image3)
plt.axis('off')
plt.savefig('0731_0732_result.jpg', bbox_inches='tight', pad_inches=0)