import numpy as np
import matplotlib.pyplot as plt

# img1 = plt.imread('./data/warrior_a.jpg')
# img2 = plt.imread('./data/warrior_b.jpg')
#
# cor1 = np.load("data/warrior_a.npy")
# cor2 = np.load("data/warrior_b.npy")

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("data/graffiti_a.npy")
cor2 = np.load("data/graffiti_b.npy")


def compute_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    F = None
    ### YOUR CODE BEGINS HERE

    # build matrix for equations in Page 51
    A = []  # A를 담아 줄 빈 matrix
    for i in range(len(x1[0])):  # len(x1[0])의 크기가 점의 개수임으로 for 문의 반복횟수로 설정
        u_o = x1[0][i]  # x1[0][i] -> x=(u,v,1).T 의 u
        v_o = x1[1][i]  # x1[1][i] -> x=(u,v,1).T 의 v
        u_p = x2[0][i]  # x2[0][i] -> x'=(u',v',1).T 의 u'
        v_p = x2[1][i]  # x2[1][i] -> x'=(u',v',1).T 의 v'

        # 해당 값들을 공식에 따라 A 행렬에 요소로 넣어준다.
        A.append([u_o * u_p, v_o * u_p, u_p, u_o * v_p, v_o * v_p, v_p, u_o, v_o, 1])

    # compute the solution in Page 51
    u, s, v = np.linalg.svd(A)  # A 행렬에 대해서 np.linalg.svd(A) 해서 반환되는 세번째 값이 고유 벡터
    f = np.reshape(v[-1], (3, 3))  # 고유 벡터 값들중 맨 마지막 요소가 가장 작은 고유값
    f_u, f_s, f_v = np.linalg.svd(f)  # 이렇게 생성된 f행렬을 가지고 svd에 넣는다.

    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    f_s = np.diag(f_s)[:9]  # f_s 행렬을 대각행렬로 만들어주고,
    f_s[2][2] = 0  # 마지막 (2,2) 요소를 0으로 바꿔준다.

    ### YOUR CODE ENDS HERE
    F = f_u @ f_s @ f_v  # 이렇게 생성된 u,s,v를 내적하여 F' 을 만들고, F에 넣는다.

    return F


def compute_norm_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # reverse normalization
    F = T2.T @ F @ T1

    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    e1_u, e1_s, e1_v = np.linalg.svd(F)  # F 행렬에 대한 svd의 마지막으로 반환되는 값이 1st epipole
    e2_u, e2_s, e2_v = np.linalg.svd(F.T)  # F를 transpose()하여 svd했을때 마지막으로 반환되는 값이 2nd epipole

    # 얻어진 1st, 2nd epipole 들을 마지막 요소로 나눠 정규화
    e1 = e1_v[-1] / e1_v[-1][-1]
    e2 = e2_v[-1] / e2_v[-1][-1]
    ### YOUR CODE ENDS HERE

    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    # 이미지 x축에서의 max를 불러온다.
    x1_max = (np.shape(img1))[1]
    x2_max = (np.shape(img2))[1]

    a1 = []
    a2 = []
    for i in range(len(cor1[0])):
        a1.append((cor1[1][i] - e1[1]) / (cor1[0][i] - e1[0]))  # 첫번째 이미지에서 두 점 사이 기울기를 구한다.
        a2.append((cor2[1][i] - e2[1]) / (cor2[0][i] - e2[0]))  # 두번째 이미지에서 두 점 사이 기울기를 구한다.

    if (e1[0] > x1_max / 2):  # 만일 epipole의 값이 x 사이즈 반틈 보다 크면,
        x1_0 = 0  # x1_0 는 0 , y 축과 만나는 점
    else: # 그렇지 않으면,
        x1_0 = x1_max  # x1_0 은 x1_max 최대 오른쪽

    if (e2[0] > x2_max / 2):  # epipole2 도 동일한 과정을 수행
        x2_0 = 0
    else:
        x2_0 = x2_max

    # x1,x2 에는 epipole과 cor1,cor2 두 점에 쌍으로 된 x 좌표와 두 선이 이루는 직선 상의 한 점 x1_0들을 넣어준다.
    # y1,y2 에는 epipole과 cor1,cor2 두 점에 대한 쌍으로 이뤄진 y 좌표와 x1_0을 지나고 직선상의 한 점들을 넣어준다.
    for i in range(len(cor1[0])):
        x1.append([e1[0], cor1[0][i], x1_0])
        y1.append([e1[1], cor1[1][i], a1[i] * (x1_0 - e1[0]) + e1[1]])

        x2.append([e2[0], cor2[0][i], x2_0])
        y2.append([e2[1], cor2[1][i], a2[i] * (x2_0 - e2[0]) + e2[1]])

    plt.subplot(1, 2, 1)  # row 는 1줄 cols 는 2줄, 그중 첫번째 이미지에 대한 plot
    plt.scatter(cor1[0], cor1[1])  # feature에 대한 point들을 scatter를 이용해서 점을 찍는다.

    for i in range(len(cor1[0])):
        plt.plot(x1[i], y1[i])  # epipole과 feature 하나가 이루는 직선의 방정식 epipolar line을 그린다.

    plt.imshow(img1)

    plt.subplot(1, 2, 2)  # row = 1, cols = 2, 두번째 이미지에 대한 plot
    plt.scatter(cor2[0], cor2[1])  # feature에 대한 point들을 scatter를 이용해서 점을 찍는다.
    for i in range(len(cor1[0])):
        plt.plot(x2[i], y2[i])  # epipole과 feature 하나가 이루는 직선의 방정식 epipolar line을 그린다.

    plt.imshow(img2)

    plt.show()
    ### YOUR CODE ENDS HERE

    return


draw_epipolar_lines(img1, img2, cor1, cor2)
