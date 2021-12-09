import cv2
import numpy as np
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

cereal = cv2.imread("../DATA/reeses_puffs.png", 0)
plot_image(cereal, "Cereal", cmap_option="gray")

many_cereals = cv2.imread("../DATA/many_cereals.jpg", 0)
plot_image(many_cereals, "Many Cereals", cmap_option="gray")

# ---- Brute-Force Matching with ORB Descriptors ----
# Get Descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(cereal, None)
kp2, des2 = orb.detectAndCompute(many_cereals, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
print(matches[0].distance)
matches = sorted(matches, key=lambda x: x.distance)
cereal_matches = cv2.drawMatches(
    cereal, kp1, many_cereals, kp2, matches[:25], None, flags=2)
# matches[:25] - to just see the first 25 matches
plot_image(cereal_matches, "Cereal Matches in all the Cereals BFMatching ORB")

# It does not look good, lets try another one

# ---- Brute-Force Matching with SIFT Descriptors and Ratio Test ----
# Get Descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(cereal, None)
kp2, des2 = sift.detectAndCompute(many_cereals, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  # find the top 2 matches of a matcher

# if this first match is pretty close in distance to the second match, then overall it's probably a good feature to match on, if not then that descriptor is not a good match, even if the first matcher is a very good match

# Ration Testing : Match1 < 75% Match2
good = []
for match1, match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

sift_matches = cv2.drawMatchesKnn(
    cereal, kp1, many_cereals, kp2, good, None, flags=2)
plot_image(sift_matches, "Cereal Matches in all the Cereals BFMatching SIFT")

# ---- FLANN based Matcher ----
# Get Descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(cereal, None)
kp2, des2 = sift.detectAndCompute(many_cereals, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
matches_mask = [[0, 0] for i in range(len(matches))]

# Ration Testing : Match1 < 75% Match2
for i, (match1, match2) in enumerate(matches):
    if match1.distance < 0.75*match2.distance:
        matches_mask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(
    255, 0, 0), matchesMask=matches_mask, flags=0)

flann_matches = cv2.drawMatchesKnn(
    cereal, kp1, many_cereals, kp2, matches, None, **draw_params)
plot_image(flann_matches, "Cereal Matches in all the Cereals FLANN")

plt.show()
