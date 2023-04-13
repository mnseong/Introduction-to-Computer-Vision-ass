from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""


def gauss1d(sigma):
    x_array_len = math.ceil(sigma * 6)  # x_array의 길이는 (sigma * 6)의 올림 값으로 정의.
    if x_array_len % 2 == 0:  # x_array의 길이가 짝수면 1을 더해 홀수로
        x_array_len += 1

    x_array_end = math.floor(x_array_len / 2)  # x_array 배열의 양끝은 (x_array_len / 2)의 내림값과 같다.
    x_array = np.linspace(-x_array_end, x_array_end, x_array_len)  # 위의 정보로 linspace()를 이용해 x_array를 생성

    gauss_array = np.exp((-x_array ** 2) / (2.0 * (sigma ** 2)))  # Gaussian 함수에 x값과 sigma 값을 대입하여 gauss_array를 만든다.
    gauss_1d = gauss_array / (gauss_array.sum())  # gauss_1d 배열은 전체 합이 1이 되도록 정규화

    return gauss_1d  # 생성된 gauss_1d array return


def gauss2d(sigma):
    gauss_1d = gauss1d(sigma)  # 위에서 정의한 1차원 gaussian 필터에 sigma값을 넣어 1차원 gaussian filter를 만든다.
    gauss_2d = np.outer(gauss_1d, gauss_1d.transpose())  # 2D gaussian filter는 guass_1d 와 이의 전치행렬을 outer() 외적으로 구할 수 있다.
    gauss_2d = gauss_2d / (gauss_2d.sum())
    return gauss_2d  # 생선된 gauss_2d array return


def convolve2d(array, filter):
    array = np.float32(array)  # convolution을 위해 array, filter를 float32로 변환
    filter = np.float32(filter)

    array_row, array_col = np.shape(array)  # array의 row, column 수를 array_row, array_col에 넣는다.
    kernel_array = np.flip(filter)  # convolution을 위해 filter를 상하좌우 flip() 하여 kernel_array로 지정
    kernel_array_row, kernel_array_col = np.shape(kernel_array)  # kernel의 row, column값을 넣어준다.
    padding_size = math.floor(kernel_array_row / 2)  # padding을 하기 전 padding size를 (kernel_array_row / 2)의 내림값으로
    padded_array = np.pad(array, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                          constant_values=0)  # array에 padding을 적용한다.
    padded_array_row, padded_array_col = np.shape(padded_array)  # padding이 적용된 array의 row, column값 return

    tmp_list = []  # 임시로 convolution 결과를 저장할 list

    # 패딩 된 array와, kernel 크기를 고려하여 합성곱 -> 결과를 tmp_list에 하나씩 넣는다.
    for i in range(padded_array_row - kernel_array_row + 1):
        for j in range(padded_array_col - kernel_array_col + 1):
            tmp_list.append((padded_array[i: i + kernel_array_row,
                             j: j + kernel_array_col] * kernel_array).sum())

    # tmp_list는 1차원 list이므로 array로 변경하고, reshape(row,column) 함수로 처음 array와 동일한 형태로 변형시켜준다.
    result = np.array(tmp_list).reshape(array_row, array_col)
    return result  # 생성된 result_array return


def gaussconvolve2d(array, sigma):
    filter = gauss2d(sigma)  # filter는 sigma 값을 위에서 만든 gauss2d() 함수에 넣어 만든다.
    gauss_conv = convolve2d(array, filter)  # gauss2d filter와 array를 convolve2d로 convolution
    return gauss_conv  # 생성된 gauss_conv return


def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    img = np.float32(img)  # image를 'float32' 형으로 변환
    s_x_filter = np.array([[-1, 0, 1],  # sobel x filter
                           [-2, 0, 2],
                           [-1, 0, 1]])
    s_y_filter = np.array([[1, 2, 1],  # sobel y filter
                           [0, 0, 0],
                           [-1, -2, -1]])

    img_dx = convolve2d(img, s_x_filter)  # 'image' x 's_x_filter' Convolution
    img_dy = convolve2d(img, s_y_filter)  # 'image' x 's_y_filter' Convolution

    G = np.hypot(img_dx, img_dy)  # Array G = sqrt((I_x)^2 + (I_y)^2)
    theta = np.arctan2(img_dy, img_dx)  # theta 는 dx, dy의 arc-tangent

    G = G * 255.0 / np.max(G)  # 0~255 사이값을 가지도록 조정

    return G, theta


def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maximum suppressed image.
    """
    G = np.float32(G)  # G,theta를 'float32' 형으로 변환
    theta = np.float32(theta)
    theta = theta * (180.0 / np.pi)

    theta_row, theta_col = theta.shape
    res = np.zeros((theta_row, theta_col))  # theta와 같은 크기의 zero matrix를 만들어준다.

    theta = np.where((((theta > -22.5) & (theta < 22.5))) | (((theta > 157.5) | (theta < -157.5))), 0,
                     theta)  # theta가 다음 조건에서 0
    theta = np.where((((theta > 22.5) & (theta < 67.5))) | (((theta > -157.5) & (theta < -112.5))), 45,
                     theta)  # theta가 다음 조건에서 45
    theta = np.where((((theta > 67.5) & (theta < 112.5))) | (((theta > -112.5) & (theta < -67.5))), 90,
                     theta)  # theta가 다음 조건에서 90
    theta = np.where((((theta > 112.5) & (theta < 157.5))) | (((theta > -67.5) & (theta < -22.5))), 135,
                     theta)  # theta가 다음 조건에서 135

    # NMS process
    for i in range(1, theta_row - 1):
        for j in range(1, theta_col - 1):

            if theta[i][j] == 0:
                if (G[i][j] <= G[i][j + 1]) or (G[i][j] < G[i][j - 1]):
                    res[i][j] = 0  # 최대값이 아니라면 res배열에 0 을 넣어준다.
                else:
                    res[i][j] = G[i][j]  # 최대값이라면 그 값을 res배열에 넣어준다.

            elif theta[i][j] == 45:
                if (G[i][j] < G[i + 1][j - 1]) or (G[i][j] <= G[i - 1][j + 1]):
                    res[i][j] = 0
                else:
                    res[i][j] = G[i][j]

            elif theta[i][j] == 90:
                if (G[i][j] < G[i + 1][j]) or (G[i][j] <= G[i - 1][j]):
                    res[i][j] = 0
                else:
                    res[i][j] = G[i][j]

            else:
                if (G[i][j] < G[i + 1][j + 1]) or (G[i][j] <= G[i - 1][j - 1]):
                    res[i][j] = 0
                else:
                    res[i][j] = G[i][j]

    res = Image.fromarray(res.astype('uint8'))  # res 배열을 'uint8'로 변환 후, image로 변환
    return res


def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    img = np.float32(img)  # image -> 'float32' 형으로 변환

    diff = np.max(img) - np.min(img)
    t_high = np.min(img) + diff * 0.15  # Threshold 의 high value는 img 최소값 + diff * 0.15
    t_low = np.min(img) + diff * 0.03  # Threshold 의 low value는 img 최소값 + diff * 0.03

    img = np.where(img > t_high, 255, img)  # t_high 보다 높은 값들은 255, strong edge
    img = np.where((img < t_high) & (img > t_low), 80, img)  # t_high 보다는 낮지만 t_low 보다 높은 갚은 80, weak edge
    img = np.where(img < t_low, 0, img)  # t_low 보다 낮은 값들은 edge가 아니라고 판단, -> 0

    res = Image.fromarray(img.astype('uint8'))  # image 배열을 'uint8'로 변환해서 image로 다시 변환
    return res


def dfs(img, res, i, j, visited):
    img = np.float32(img)  # Thresholding 처리된 이미지를 불러와서 'float32'로 변환
    img_row, img_col = img.shape  # index error 검출하기 위한 row, col 크기
    if (i < 0) or (i >= img_row) or (j < 0) or (j >= img_col):
        return

    # 시작점 (i, j)은 최초 호출이 아닌 이상, strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i - 1, i + 2):
        for jj in range(j - 1, j + 2):
            if (ii < 0) or (ii >= img_row) or (jj < 0) or (jj >= img_col):
                return
            if (img[ii, jj] == 80) and ((ii, jj) not in visited):
                dfs(img, res, ii, jj, visited)
            return
    return


def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    img = np.float32(img)  # Thresholding 처리된 이미지를 불러와서 'float32'로 변환
    img_row, img_col = img.shape
    res = np.zeros((img_row, img_col))
    visited = []

    # img pixel 값이 255인 지점(intensity value)에 대해서 dfs 수행
    for i in range(img_row):
        for j in range(img_col):
            if img[i, j] == 255.0:
                dfs(img, res, i, j, visited)

    # Strong edge가 아닌 부분은 0
    for i in range(1, img_row - 1):
        for j in range(1, img_col - 1):
            if img[i][j] == 80:
                img[i][j] = 0

    res = Image.fromarray(img.astype('uint8'))  # image 배열을 'uint8'로 변환해서 image로 다시 변환
    return res


"""1. Noise reduction"""
iguana_img = Image.open('iguana.bmp').convert('L')  # 이구아나 사진을 흑백으로 불러온다.
iguana_arr = np.asarray(iguana_img)  # 이를 np.array로 변환
sigma = 1.6
iguana_gau = gaussconvolve2d(iguana_arr, sigma)  # iguana_array를 gaussian kernel을 이용해 Convolution
iguana_gau = iguana_gau.astype('uint8')  # array를 uint8형으로 변환
iguana_gau = Image.fromarray(iguana_gau)  # array -> image
iguana_img.show()
iguana_gau.show()
iguana_gau.save('iguana_gaussian.bmp', 'bmp')


"""2. Finding the intensity gradient of the image"""
iguana_gau = np.asarray(iguana_gau)  # gaussian filter를 적용한 image를 array로 변환
G, theta = sobel_filters(iguana_gau)  # sobel filter를 적용해 얻은 Gradient, theta 값
G = G.astype('uint8')  # Gradient array를 uint8형으로 변환
iguana_sob = Image.fromarray(G)  # Gradient array -> image
iguana_sob.show()
iguana_sob.save('iguana_sobel.bmp', 'bmp')


"""3. Non-Maximum Suppression"""
iguana_nms = non_max_suppression(G, theta)  # 2번에서 얻은 Gradient, theta를 non_max_suppression()의 parameter로 사용
iguana_nms.show()  # non_max_suppression()의 반환이 image, 별도의 변환 필요없음
iguana_nms.save('iguana_nms.bmp', 'bmp')


"""4. Double threshold"""
iguana_nms = np.asarray(iguana_nms)  # 3번에서 얻은 iguana_nms image를 array로 변환
iguana_thr = double_thresholding(iguana_nms)  # iguana_nms 배열을 double_thresholding() 함수에 넣는다.
iguana_thr.show()  # double_thresholing() 함수의 반환이 image, 별도의 변환 필요없음
iguana_thr.save('iguana_double_thresholding.bmp', 'bmp')


"""5. Edge Tracking by hysteresis"""
iguana_thr = np.asarray(iguana_thr)  # 4번에서 얻은 iguana_thr image를 array로 변환
iguana_hys = hysteresis(iguana_thr)  # iguana_thr 배열을 hysteresis()에 넣는다.
iguana_hys.show()  # hysteresis() 함수의 반환이 image, 별도의 변환 필요없음
iguana_hys.save('iguana_hysteresis.bmp', 'bmp')
