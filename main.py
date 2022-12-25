import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings


def select_descriptor_method(image, method=None):
    assert method is not None, "需要传入一个提取特征点的方法，可以是 sift， surf， orb， brisk"
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    if method == 'surf':
        descriptor = cv2.SURF_create()
    if method == 'orb':
        descriptor = cv2.ORB_create()
    if method == 'brisk':
        descriptor = cv2.BRISK_create()
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    return keypoints, features


# Brute-Force Matcher
def create_matching_object(method, crossCheck):
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMathcer(cv2.NORM_HAMMING, crossCheck=crossCheck)

    return bf


def key_points_matching(feature_train_img, feature_query_img, method):
    bf = create_matching_object(method, crossCheck=True)
    best_matches = bf.match(feature_train_img, feature_query_img)
    raw_matches = sorted(best_matches, key=lambda x: x.distance)
    print('Raw matches with Brute Force', len(raw_matches))
    return raw_matches


def key_points_matching_KNN(feature_train_img, feature_query_img, method, ratio):
    bf = create_matching_object(method, crossCheck=False)
    raw_matches = bf.knnMatch(feature_train_img, feature_query_img, k=2)
    print('Raw matches with KNN', len(raw_matches))

    knn_matches = []
    for m, n in raw_matches:
        if m.distance < n.distance * ratio:
            knn_matches.append(m)
    return raw_matches


def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    if len(matches) > 4:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.queryIdx] for m in matches])

        (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)
        return matches, H, status
    else:
        return None


if __name__ == '__main__':
    # 读入图片
    feature_extraction_algo = 'sift'
    feature_to_match = 'bf'

    train_photo = cv2.imread('imgs/src_2/2-3.jpg')
    train_photo = cv2.resize(train_photo, (1280, 720))
    train_photo = cv2.cvtColor(train_photo, cv2.COLOR_BGR2RGB)
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

    query_photo = cv2.imread('imgs/src_2/2-4.jpg')
    query_photo = cv2.resize(query_photo, (1280, 720))
    query_photo = cv2.cvtColor(query_photo, cv2.COLOR_BGR2RGB)
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

    # 查看图片
    cv2.imshow("train_photo.jpg", train_photo_gray)
    cv2.waitKey(0)
    cv2.imshow("query_photo.jpg", query_photo_gray)
    cv2.waitKey(0)

    # 提取特征
    keypoints_train_img, features_train_img \
        = select_descriptor_method(train_photo_gray, method=feature_extraction_algo)
    keypoints_query_img, features_query_img \
        = select_descriptor_method(query_photo_gray, method=feature_extraction_algo)

    train_img_with_keypoint = train_photo
    query_img_with_keypoint = query_photo
    for keypoint in keypoints_query_img:
        x, y = keypoint.pt
        size = keypoint.size
        orientation = keypoint.angle
        response = keypoint.response
        octave = keypoint.octave
        class_id = keypoint.class_id
        x = int(x)
        y = int(y)
        cv2.circle(query_img_with_keypoint, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow('query_img_with_keypoint', query_img_with_keypoint)
    cv2.waitKey(0)

    for keypoint in keypoints_train_img:
        x, y = keypoint.pt
        size = keypoint.size
        orientation = keypoint.angle
        response = keypoint.response
        octave = keypoint.octave
        class_id = keypoint.class_id
        x = int(x)
        y = int(y)
        cv2.circle(train_img_with_keypoint, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow('train_img_with_keypoint', train_img_with_keypoint)
    cv2.waitKey(0)

    print("Drawing matched features for", feature_to_match)

    if feature_to_match == 'bf':
        matches = key_points_matching(features_train_img, features_query_img,
                                      method=feature_extraction_algo)
        print(matches[:100])
        mapped_feature_img = cv2.drawMatches(train_photo, keypoints_train_img,
                                             query_photo, keypoints_query_img,
                                             matches[:100], None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_to_match == 'knn':
        matches = key_points_matching_KNN(features_train_img, features_query_img,
                                          method=feature_extraction_algo, ratio=0.75)
        mapped_feature_img = cv2.drawMatches(train_photo, keypoints_train_img,
                                             query_photo, keypoints_query_img,
                                             np.random.choice(matches, 100), None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('img', mapped_feature_img)
    cv2.waitKey(0)

    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)
    if M is None:
        print('stitching error')

    (matches, Homography_Matrix, status) = M
    print(Homography_Matrix)

    width = query_photo.shape[1] + train_photo.shape[1]
    height = query_photo.shape[0] + train_photo.shape[0]

    result = cv2.warpPerspective(train_photo, Homography_Matrix, (width, height))
    result[0:query_photo.shape[0], 0:query_photo.shape[1]]

    cv2.imshow('result', result)
    cv2.waitKey(0)
