import numpy as np
import glob
from scipy import misc


# prints an array of strings
def read_images(path):
    files = glob.glob(path + "/*.png")
    images = []
    for f in files:
        images.append(misc.imread(f))
    return images


def min_color_component_image(image):
    width = len(image)
    height = len(image[0])
    min_colors = [255, 255, 255]

    for y in range(0, height):
        for x in range(0, width):
            min_colors = np.minimum(min_colors, image[y][x])

    return min_colors


def min_color_images(images):
    min_colors = []
    for image in images:
        min_colors.append(min_color_component_image(image))
    return min_colors

def feature_matrix():
    positives = read_images('images/positives')
    negatives = read_images('images/negatives')

    min_colors_positive = np.array(min_color_images(positives))
    min_colors_negative = np.array(min_color_images(negatives))

    red, green, blue = 0, 1, 2
    min_colors_positive_red = min_colors_positive[:,red]
    print(min_colors_positive_red)

feature_matrix()