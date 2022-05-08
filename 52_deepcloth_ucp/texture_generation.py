"""
https://stackoverflow.com/questions/51646185/how-to-generate-a-paper-like-background-with-opencv
"""

import cv2
import numpy as np


BG_COLOR = (140, 175, 232)  # rgb b
BG_SIGMA = 3
MONOCHROME = 3
WIDTH = 256
HEIGHT = 256
TURBULENCE = 2


def blank_image(width=WIDTH, height=HEIGHT, background=BG_COLOR):
    """
    It creates a blank image of the given background color
    """
    img = np.full((height, width, MONOCHROME), background, np.uint8)
    return img


def add_noise(img, sigma=BG_SIGMA):
    """
    Adds noise to the existing image
    """
    width, height, ch = img.shape
    n = noise(width, height, sigma=sigma)
    img = img + n
    return img.clip(0, 255)


def noise(width=WIDTH, height=HEIGHT, ratio=1, sigma=BG_SIGMA):
    """
    The function generates an image, filled with gaussian nose. If ratio parameter is specified,
    noise will be generated for a lesser image and then it will be upscaled to the original size.
    In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
    uses interpolation.

    :param ratio: the size of generated noise "pixels"
    :param sigma: defines bounds of noise fluctuations
    """
    mean = 0
    assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(
        width, ratio
    )
    assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(
        height, ratio
    )

    h = int(height / ratio)
    w = int(width / ratio)

    result = np.random.normal(mean, sigma, (w, h, MONOCHROME))
    if ratio > 1:
        result = cv2.resize(
            result, dsize=(width, height), interpolation=cv2.INTER_LINEAR
        )
    return result.reshape((width, height, MONOCHROME))


def texture(image, sigma=BG_SIGMA, turbulence=TURBULENCE):
    """
    Consequently applies noise patterns to the original image from big to small.

    sigma: defines bounds of noise fluctuations
    turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
    value - the more iterations will be performed during texture generation.
    """
    result = image.astype(float)
    cols, rows, ch = image.shape
    ratio = cols
    while not ratio == 1:
        result += noise(cols, rows, ratio, sigma=sigma)
        ratio = (ratio // turbulence) or 1
    cut = np.clip(result, 0, 255)
    return cut.astype(np.uint8)


def texture_generation(texture_value: tuple, width: int, height: int) -> np.ndarray:
    texture_img = add_noise(
        texture(blank_image(background=texture_value), sigma=BG_SIGMA), sigma=BG_SIGMA
    )
    texture_img = cv2.resize(texture_img, (height, width), interpolation=cv2.INTER_AREA)
    return texture_img


if __name__ == "__main__":
    cv2.imwrite(
        "texture.jpg",
        texture(
            blank_image(background=BG_COLOR), sigma=BG_SIGMA, turbulence=TURBULENCE
        ),
    )
    cv2.imwrite(
        "texture-and-noise.jpg",
        add_noise(
            texture(blank_image(background=BG_COLOR), sigma=BG_SIGMA), sigma=BG_SIGMA
        ),
    )

    cv2.imwrite("noise.jpg", add_noise(blank_image(WIDTH, HEIGHT), sigma=BG_SIGMA))
