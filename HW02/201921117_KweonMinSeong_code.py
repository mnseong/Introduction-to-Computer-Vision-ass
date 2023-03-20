from PIL import Image
import numpy as np
import math

"""implementation Box Filter"""
def boxfilter(n):
    assert n % 2 != 0, "n value cannot be an even number!"  # Boxfilter row column 값은 홀수, assert를 사용하여 예외 처리
    filter_arr = np.full(shape=(n, n), fill_value=1.0 / n ** 2)  # n x n square matrix, 전체 합이 1이 되도록 정의
    return filter_arr  # filter array return


"""implementation 1D Gaussian Filter"""
def gauss1d(sigma):
    x_array_len = math.ceil(sigma * 6)  # x_array의 길이는 (sigma * 6)의 올림 값으로 정의.
    if x_array_len % 2 == 0:  # x_array의 길이가 짝수면 1을 더해 홀수로
        x_array_len += 1

    x_array_end = math.floor(x_array_len / 2)  # x_array 배열의 양끝은 (x_array_len / 2)의 내림값과 같다.
    x_array = np.linspace(-x_array_end, x_array_end, x_array_len)  # 위의 정보로 linspace()를 이용해 x_array를 생성

    gauss_array = np.exp((-x_array ** 2) / (2.0 * (sigma ** 2)))  # Gaussian 함수에 x값과 sigma 값을 대입하여 gauss_array를 만든다.
    gauss_1d = gauss_array / (gauss_array.sum())  # gauss_1d 배열은 전체 합이 1이 되도록 정규화

    return gauss_1d  # 생성된 gauss_1d array return


"""implementation 2D Gaussian Filter"""
def gauss2d(sigma):
    gauss_1d = gauss1d(sigma)  # 위에서 정의한 1차원 gaussian 필터에 sigma값을 넣어 1차원 gaussian filter를 만든다.
    gauss_2d = np.outer(gauss_1d, gauss_1d.transpose())  # 2D gaussian filter는 guass_1d 와 이의 전치행렬을 outer() 외적으로 구할 수 있다.
    gauss_2d = gauss_2d / (gauss_2d.sum())
    return gauss_2d  # 생선된 gauss_2d array return


"""implementation 2D Convolution Function"""
def convolve2d(array, filter):
    array = np.float32(array)  # convolution을 위해 array, filter를 float32로 변환
    filter = np.float32(filter)

    array_row, array_col = np.shape(array)  # array의 row, column 수를 array_row, array_col에 넣는다.
    kernel_array = np.flip(filter)  # convolution을 위해 filter를 상하좌우 flip() 하여 kernel_array로 지정
    kernel_array_row, kernel_array_col = np.shape(kernel_array)  # kernel의 row, column값을 넣어준다.
    padding_size = math.floor(kernel_array_row / 2)  # padding을 하기 전 padding size를 (kernel_array_row / 2)의 내림값으로
    padded_array = np.pad(array, ((padding_size, padding_size), (padding_size, padding_size)), 'constant', constant_values=0) # array에 padding을 적용한다.
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


"""implementation 2D Gaussian Convolution Function"""
def gaussconvolve2d(array, sigma):
    filter = gauss2d(sigma)  # filter는 sigma 값을 위에서 만든 gauss2d() 함수에 넣어 만든다.
    gauss_conv = convolve2d(array, filter)  # gauss2d filter와 array를 convolve2d로 convolution
    return gauss_conv  # 생성된 gauss_conv return


"""
Part 1: Gaussian Filtering
"""
"""1. Box Filter Example"""
print("Output of boxfilter(3): ")
print(boxfilter(3))
# print("Output of boxfilter(4): ")
# print(boxfilter(4))
print("Output of boxfilter(7): ")
print(boxfilter(7))

"""2. 1D Gaussian Filter Example"""
print("Output of gauss1d(0.3): ")
print(gauss1d(0.3))
print("Output of gauss1d(0.5): ")
print(gauss1d(0.5))
print("Output of gauss1d(1): ")
print(gauss1d(1))
print("Output of gauss1d(2): ")
print(gauss1d(2))

"""3. 2D Gaussian Filter Example"""
print("Output of gauss2d(0.5): ")
print(gauss2d(0.5))
print("Output of gauss2d(1): ")
print(gauss2d(1))

"""4-(c) 2D Gaussian Convolution Example"""
original_img_dog = Image.open('./hw2_image/2b_dog.bmp')  # 강아지 사진을 불러온다.
cp_img_dog = original_img_dog.copy()  # original_img_dog를 얕은 복사하여 따로 변수에 저장한다.
sigma = 3  # sigma value=3

cp_img_dog = cp_img_dog.convert('L')  # 우선 dog_im 사진을 흑백으로 만든다.
dog_array = np.asarray(cp_img_dog)  # 흑백화된 사진을 array로 변환한다.

dog_array = gaussconvolve2d(dog_array, sigma)  # 2D Gaussian Convolution을 수행하기위해 gaussconvolve2d 함수를 이용한다.

dog_array = dog_array.astype('uint8')  # 반환된 array를 'uint8'형태로 만들어준다.
cp_img_dog = Image.fromarray(dog_array)  # dog_array로 부터 다시 image로 변환한다.

original_img_dog.show()  # 원본 이미지 출력
cp_img_dog.show()  # 변환된 이미지 출력
cp_img_dog.save('4_c.bmp', 'bmp')  # bmp형식으로 결과물을 저장


"""
Part 2: Hybrid Images
"""
sigma = 4  # part2 에서 동일한 sigma를 사용한다.

"""1. Create low frequency image"""
original_img_dog = Image.open('./hw2_image/2b_dog.bmp')  # 강아지 사진을 불러온다.
cp_img_dog = original_img_dog.copy()  # original_img_dog를 얕은 복사하여 따로 변수에 저장한다.
dog_r, dog_g, dog_b = cp_img_dog.split()  # split method를 이용해 RGB scale로 분리한다.

# RGB scale -> array
dog_r_array = np.asarray(dog_r)
dog_g_array = np.asarray(dog_g)
dog_b_array = np.asarray(dog_b)

# array에 sigma를 적용하여 2d gaussian convolution한다.
dog_r_array = gaussconvolve2d(dog_r_array, sigma).astype('uint8')
dog_g_array = gaussconvolve2d(dog_g_array, sigma).astype('uint8')
dog_b_array = gaussconvolve2d(dog_b_array, sigma).astype('uint8')

# RGB scale -> image
dog_r_array = Image.fromarray(dog_r_array)
dog_g_array = Image.fromarray(dog_g_array)
dog_b_array = Image.fromarray(dog_b_array)

# RGB를 다시 병합한다.
cp_img_dog = Image.merge("RGB", (dog_r_array, dog_g_array, dog_b_array))

original_img_dog.show()  # 원본 이미지 출력
cp_img_dog.show()  # Low pass filter를 거친 이미지 출력
cp_img_dog.save('low_freq_img.bmp', 'bmp')  # bmp형식으로 결과물을 저장

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""2. Create high frequency image"""
original_img_cat = Image.open('./hw2_image/2a_cat.bmp')  # 고양이 사진을 불러온다.
cp_img_cat = original_img_cat.copy()  # original_img_cat을 얕은 복사하여 따로 변수에 저장한다.
cat_r, cat_g, cat_b = cp_img_cat.split()  # split method를 이용해 RGB scale로 분리한다.

# RGB scale -> array
cat_r_array1 = np.asarray(cat_r)
cat_g_array1 = np.asarray(cat_g)
cat_b_array1 = np.asarray(cat_b)

# array에 sigma를 적용하여 2d gaussian convolution한다.
cat_r_array2 = gaussconvolve2d(cat_r_array1, sigma)
cat_g_array2 = gaussconvolve2d(cat_g_array1, sigma)
cat_b_array2 = gaussconvolve2d(cat_b_array1, sigma)

# Original image - Low Frequency Image = High Frequency Image
cat_r_array = cat_r_array1 - cat_r_array2 + 128.0  # Negative value를 방지하기 위해 128을 더해준다.
cat_g_array = cat_g_array1 - cat_g_array2 + 128.0
cat_b_array = cat_b_array1 - cat_b_array2 + 128.0
cat_r_array = cat_r_array.astype('uint8')
cat_g_array = cat_g_array.astype('uint8')
cat_b_array = cat_b_array.astype('uint8')

# RGB scale -> image
cat_r_array = Image.fromarray(cat_r_array)
cat_g_array = Image.fromarray(cat_g_array)
cat_b_array = Image.fromarray(cat_b_array)

# RGB를 다시 병합한다.
cp_img_cat = Image.merge("RGB", (cat_r_array, cat_g_array, cat_b_array))

original_img_cat.show()  # 원본 이미지 출력
cp_img_cat.show()  # High pass filter를 거친 이미지 출력
cp_img_cat.save('high_freq_img.bmp', 'bmp')  # bmp형식으로 결과물을 저장

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""3. Merge above low & high frequency images"""

# 1번에서 생성한 low frequency image의 RGB channel을 다시 array로 변환
dog_r_array = np.asarray(dog_r_array).astype('float32')
dog_g_array = np.asarray(dog_g_array).astype('float32')
dog_b_array = np.asarray(dog_b_array).astype('float32')

# 1번에서 생성한 high frequency image의 RGB channel을 다시 array로 변환
cat_r_array = np.asarray(cat_r_array).astype('float32')
cat_g_array = np.asarray(cat_g_array).astype('float32')
cat_b_array = np.asarray(cat_b_array).astype('float32')

# Hybrid array = Low Frequency Image + High Frequency Image - 128
hybrid_r_array = dog_r_array + (cat_r_array - 128)
hybrid_g_array = dog_g_array + (cat_g_array - 128)
hybrid_b_array = dog_b_array + (cat_b_array - 128)

# 각 array에서 255를 넘는 값을 255로 제한시킨다.
hybrid_r_array = np.where(hybrid_r_array > 255, 255, hybrid_r_array)
hybrid_g_array = np.where(hybrid_g_array > 255, 255, hybrid_g_array)
hybrid_b_array = np.where(hybrid_b_array > 255, 255, hybrid_b_array)

# 각 array에서 0아래의 값을 0으로 제한시킨다.
hybrid_r_array = np.where(hybrid_r_array < 0, 0, hybrid_r_array).astype('uint8')
hybrid_g_array = np.where(hybrid_g_array < 0, 0, hybrid_g_array).astype('uint8')
hybrid_b_array = np.where(hybrid_b_array < 0, 0, hybrid_b_array).astype('uint8')

# hybrid 이미지의 각 RGB array -> image
hybrid_im_r = Image.fromarray(hybrid_r_array)
hybrid_im_g = Image.fromarray(hybrid_g_array)
hybrid_im_b = Image.fromarray(hybrid_b_array)

# hybrid RGB image를 원래대로 병합
hybrid_im = Image.merge("RGB", (hybrid_im_r, hybrid_im_g, hybrid_im_b))

# Hybrid image를 출력하고, 이를 bmp형식으로 결과물을 저장
hybrid_im.show()
hybrid_im.save('hybrid_img.bmp', 'bmp')
