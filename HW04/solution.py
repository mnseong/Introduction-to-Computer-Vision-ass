import numpy as np
import cv2
import math
import random


def RANSACFilter(matched_pairs, keypoints1, keypoints2, orient_agreement, scale_agreement):
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    largest_set = []
    for i in range(10):  # 총 10회 반복한다.
        rand = random.randrange(0, len(matched_pairs))  # random한 수를 생성한다.
        choice = matched_pairs[rand]
        orientation = (keypoints1[choice[0]][3] - keypoints2[choice[1]][3]) % (2 * math.pi)  # first-orientation 계산
        scale = keypoints2[choice[1]][2] / keypoints1[choice[0]][2]  # first-scale 계산
        temp = []
        for j in range(len(matched_pairs)):  # 모든 match들에 대하여 계산한다.
            if j is not rand:
                # second-orientation 계산
                orientation_temp = (keypoints1[matched_pairs[j][0]][3] - keypoints2[matched_pairs[j][1]][3]) % (
                            2 * math.pi)
                # second-scale 계산
                scale_temp = keypoints2[matched_pairs[j][1]][2] / keypoints1[matched_pairs[j][0]][2]
                # check degree error += 30도
                if ((orientation - math.pi / 6) < orientation_temp) and (
                        orientation_temp < (orientation + math.pi / 6)):
                    # check scale error += 50%
                    if (scale - scale * scale_agreement < scale_temp and scale_temp < scale + scale * scale_agreement):
                        temp.append([i, j])
        if (len(temp) > len(largest_set)):  # best matches를 선택한다.
            largest_set = temp
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0], matched_pairs[largest_set[i][1]][1])
    ## END
    assert isinstance(largest_set, list)
    return largest_set


def FindBestMatches(descriptors1, descriptors2, threshold):
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    y1 = descriptors1.shape[0]
    y2 = descriptors2.shape[0]

    temp = np.zeros(y2)
    matched_pairs = []  # Match를 모아둘 배열

    for i in range(y1):
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i], descriptors2[j]))  # feature 마다 내적값의 acos()을 구한다.
        compare = sorted(range(len(temp)), key=lambda k: temp[k])  # 가장 작은 순서를 앞으로오게 sort한다.
        if (temp[compare[0]] / temp[compare[1]]) < threshold:  # sorted_arr[0]/sorted_arr[1] thresholed 값보다 작으면
            matched_pairs.append([i, compare[0]])  # best match point로 보고, 인덱스를 기록

    ## END
    return matched_pairs  # 해당 일치하는 feature 들의 index return


def KeypointProjection(xy_points, h):
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    xy_points_out = []  # return할 point를 담을 배열
    homogeneous_xy = []  # homogeneous 로 변환한 point를 담을 배열
    for i in xy_points:
        tempArr_xy = np.transpose([i[0], i[1], 1])  # 각 2차원 포인트를 3차원으로 만든다. 이때 내적에 사용하기 위해 transpose()한다.
        homogeneous_xy.append(
            list(np.inner(h, tempArr_xy)))  # homogeneous 변환 된 점들을 h matrix과 3차원 좌표계로 표현된 행렬의 내적으로 표현된다.

    for j in homogeneous_xy:
        if j[2] != 0:  # 마지막 z 값이 0이 아니면 단순히 x. y 값에 z 값을 나누고, xy_points_out 배열에 추가
            xy_points_out.append([j[0] / j[2], j[1] / j[2]])
        else:
            REPZERO = math.exp(-10)  # 만일 z값이 0 이면, 근사된 값을 지정하여 나눈다.
            xy_points_out.append([j[0] / REPZERO, j[1] / REPZERO])

    xy_points_out = np.asarray(xy_points_out)

    # END
    return xy_points_out


def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol * 1.0

    # START
    inlier_num = 0  # inlier의 개수
    index_array = []  # 랜덤으로 생성되는 점들을 저장할 배열

    for _ in range(num_iter):
        temp = []
        for _ in range(4):  # homography는 임의의 4개의 점이 필요하므로 4개의 랜덤한 점을 만든다.
            temp.append(random.randrange(0, len(xy_ref)))
        index_array.append(temp)

    for i in index_array:
        A_matrix = []  # homography matrix를 만들기 위한 A matrix를 담을 배열
        temp_inlier_num = 0  # 각각의 case에 대해서 inlier개수를 임시로 담아둘 변수

        for j in i:  # j에는 각 round에서 임의로 선택된 4개의 변수를 나타낸다.
            x_i = xy_src[j][0]
            y_i = xy_src[j][1]
            xp_i = xy_ref[j][0]
            yp_i = xy_ref[j][1]
            A_matrix.append([x_i, y_i, 1, 0, 0, 0, -1 * x_i * xp_i, -1 * y_i * xp_i, -1 * xp_i])
            A_matrix.append([0, 0, 0, x_i, y_i, 1, -1 * x_i * yp_i, -1 * y_i * yp_i, -1 * yp_i])

        u, s, v = np.linalg.svd(A_matrix)  # SVD를 사용해서 |Av|^2 가 최소가 되는 v array를 구한다.
        H_matrix = np.reshape(v[8], (3, 3))  # 해당 v 배열을 3x3 크기에 맞게 조정
        H_matrix = (1 / H_matrix[2][2]) * H_matrix  # 마지막 요소가 1이 되도록 정리한다.

        # 이제 이 H_matrix를 사용해서 transformation 한 후 ref image 모든 점에 대해서 거리를 계산
        xy_proj = KeypointProjection(xy_src, H_matrix)  # source 이미지를 transformation 한 것

        for k in range(len(xy_proj)):  # 투영된 점 하나하나를 ref 이미지와 비교
            dist = np.linalg.norm(xy_ref[k] - xy_proj[k])  # 유클리드 거리를 계산하여 dist에 저장
            if (dist <= tol):  # 해당하는 dist가 tol 보다 작으면 inlier로 본다.
                temp_inlier_num += 1

        if temp_inlier_num >= inlier_num:  # inlier 갯수가 최댓값이면
            inlier_num = temp_inlier_num  # 해당 temp_inlier를 inlier_num에 저장
            h = H_matrix  # 그때 맞는 Homography matrix를 ans에 저장

    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(keypoints1, keypoints2,
                          descriptors1, descriptors2,
                          threshold,orient_agreement, scale_agreement):
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
