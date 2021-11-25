import cv2
import numpy as np
from PIL import Image
import math


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N <= 2:
        return dft(x)
    else:
        e = fft(x[::2])
        o = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([e + terms[:int(N / 2)] * o,
                               e + terms[int(N / 2):] * o])


def pixel_log255(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            unorm_image[i, j] = (255 / math.log10(256)) * math.log10(1 + (255 / pxmax) * unorm_image[i, j])
            # unorm_image[i, j] = ((unorm_image[i, j] - pxmin) / (pxmax - pxmin)) * 255

    norm_image = unorm_image
    return norm_image


def center_image(image):
    rows = image.shape[0]
    cols = image.shape[1]
    centered_image = np.zeros((rows, cols))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            centered_image[i, j] = image[i, j] * ((-1) ** ((i - 1) + (j - 1)))

    return centered_image


def tf_completion(htf):
    q1 = np.rot90(htf, 2)
    q2 = np.rot90(htf, 1)

    q_upper = np.concatenate((q1, q2), axis=1)

    q3 = np.rot90(htf, 3)
    q4 = htf

    q_lower = np.concatenate((q3, q4), axis=1)

    return np.vstack((q_upper, q_lower))


def ilpf(shape, d0):
    h_size = int(shape / 2)
    htf = (h_size, h_size)
    htf = np.zeros(htf)

    for r in range(h_size):
        for c in range(h_size):
            if math.sqrt(c * c + r * r) <= d0:
                htf[r][c] = 1.

    htfn = tf_completion(htf)

    return htfn


def ihpf(shape, d0):
    h_size = int(shape / 2)
    s = (h_size, h_size)
    htf = np.zeros(s)

    for r in range(h_size):
        for c in range(h_size):
            if math.sqrt(c * c + r * r) > d0:
                htf[r][c] = 1.

    htfn = tf_completion(htf)

    return htfn


def blpf(shape, d0, n):
    h_size = int(shape / 2)
    s = (h_size, h_size)
    htf = np.zeros(s)

    for r in range(h_size):
        for c in range(h_size):
            duv = math.sqrt(c * c + r * r)
            duvdo = (duv / d0) ** 2 * n
            htf[r][c] = 1 / (1 + duvdo)

    htfn = tf_completion(htf)

    return htfn


def bhpf(shape, d0, n):
    h_size = int(shape / 2)
    s = (h_size, h_size)
    htf = np.zeros(s)

    for r in range(h_size):
        for c in range(h_size):
            if r == 0 and c == 0:
                duv = 1
            else:
                duv = math.sqrt(c * c + r * r)

            doduv = (d0 / duv) ** 2 * n

            htf[r][c] = 1 / (1 + doduv)
            htf[r][c] = 1 / (1 + doduv)

    htfn = tf_completion(htf)

    return htfn


if __name__ == '__main__':

    path = r'b512.jpg'
    img = cv2.imread(path, 0)

    rows = img.shape[0]
    cols = img.shape[1]

    img = center_image(img)
    img_width = rows * cols

    flatten_image = np.zeros(shape=(1, img_width))
    flatten_image = img.flatten()
    fft_image = fft(flatten_image)
    fft_image_2d = np.reshape(fft_image, (rows, cols))

    h = ilpf(rows, 30)

    fft_image_2d = fft_image_2d * h

    fft_image_2d = np.fft.ifft2(fft_image_2d)

    fft_image_2d = abs(fft_image_2d)
    fft_image_2d = np.rot90(fft_image_2d, -1)
    # norm_image = pixel_log255(fft_image_2d)

    im = Image.fromarray(fft_image_2d)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("fft.jpg")
