import cv2

import matplotlib.pyplot as plt

import numpy as np

import os

from PIL import Image

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

import time

pair_path = "../../hierarchical_localization/Hierarchical-Localization/datasets/inloc/query/iphone7/"
image_paths = [pair_path + file for file in os.listdir(pair_path) if file.endswith("JPG")]
feat_paths = [pair_path + file for file in os.listdir(pair_path) if file.endswith("d2-net")]

image_list = [np.array(Image.open(image_path)) for image_path in image_paths]
feat_list = [np.load(feat_path) for feat_path in feat_paths]

# select an image to match with other images
query_image = image_list[0]
image_list = image_list[1:]
query_feat = feat_list[0]
feat_list = feat_list[1:]

start = time.time()
matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
matches_list = []
for feat in feat_list:
    matches = match_descriptors(query_feat['descriptors'], feat['descriptors'], cross_check=True)
    matches_list.append(matches)
    
print("Mutual nearest neighbors matching done! Time taken for this step: ", time.time()-start)

start = time.time()
keypoints_query = query_feat['keypoints'][matches[:, 0], : 2]
keypoints_to_check_list = [feat['keypoints'][matches[:, 1], : 2] for feat in feat_list]
np.random.seed(0)
max_n_inliers = 0
best_matched_index = 0
for i, keypoints_to_check in enumerate(keypoints_to_check_list):
    model, inliers = ransac(
        (keypoints_query, keypoints_to_check),
        ProjectiveTransform, min_samples=4,
        residual_threshold=4, max_trials=10000
    )
    n_inliers = np.sum(inliers)
    if n_inliers > max_n_inliers:
        max_n_inliers = n_inliers
        best_matched_index = i
print("Homography fitting done! Time taken for this step: ", time.time()-start)
print('Max number of inliers: %d.' % max_n_inliers)
print("best matched image: ", image_list[best_matched_index])

inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_query[max_n_inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_to_check_list[best_matched_index][max_n_inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(max_n_inliers)]
image3 = cv2.drawMatches(query_image, inlier_keypoints_left, image_list[best_matched_index], inlier_keypoints_right, placeholder_matches, None)

plt.figure(figsize=(15, 15))
plt.imshow(image3)
plt.axis('off')
plt.savefig('0731_all_result.jpg', bbox_inches='tight', pad_inches=0)