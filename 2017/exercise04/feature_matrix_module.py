import numpy as np

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


def num_pixels_low_red_green_images(images):
    num_dark_pixels = []
    for image in images:
        num_dark_pixels.append( num_pixels_low_red_green_component_image(image))
    return num_dark_pixels


def num_pixels_low_red_green_component_image( image ):
    num_of_pi = 0
    threshold = 120
    for i in range(len( image )):
        for j in range(len(image[0])):
            red_val = image[i][j][0]
            green_val = image[i][j][1]
            if  red_val < threshold and green_val < threshold:
                num_of_pi += 1
    return num_of_pi


def find_dark_pixel_spots_in_images(images):
    dps = []
    for image in images:
        dps.append( find_dark_pixel_spots_in_image( image ) )
    return dps


def find_dark_pixel_spots_in_image(image):
    num_dps = 0
    threshold = 100
    for i in range( len(image) ):
        for j in range( len(image[0])):
            green_val = image[i][j][1]
            if i > 2 and i < 21 and j > 2 and j < 21:
                north_px_val = image[i-3][j][1]
                east_px_val = image[i][j+3][1]
                south_px_val = image[i+3][j][1]
                west_px_val = image[i][j-3][1]
                if green_val < 100 and north_px_val > 100 and east_px_val > 100 and south_px_val > 100 and west_px_val > 100:
                    num_dps += 1
    return num_dps


def find_very_low_green_values_in_images(images):
    vlv = []
    for image in images:
        vlv.append( find_very_low_green_values_in_image(image))
    return vlv

def find_very_low_green_values_in_image(image):
    num_vlv = 0
    threshold = 75
    for i in range( len(image)):
        for j in range( len(image[0])):
            if image[i][j][1] < threshold:
                num_vlv +=1
    return num_vlv


def build_feature_matrix(positives, negatives):
    all_images = positives + negatives

    all_min_colors = np.array(min_color_images(all_images))
    red, green, blue = 0, 1, 2
    first_column_min_colors_red = all_min_colors[:,red]
    first_column_min_colors_red = np.reshape( first_column_min_colors_red, (-1,1))

    second_column_min_colors_green = all_min_colors[:,green]
    second_column_min_colors_green = np.reshape( second_column_min_colors_green, (-1,1))

    third_column_num_px_low_red_green = np.array( num_pixels_low_red_green_images(all_images))
    third_column_num_px_low_red_green = np.reshape( third_column_num_px_low_red_green, (-1,1))

    forth_column_dps = np.array(find_dark_pixel_spots_in_images(all_images) )
    forth_column_dps = np.reshape( forth_column_dps, (-1,1))

    fifth_column_low_green = np.array(find_very_low_green_values_in_images(all_images))
    fifth_column_low_green = np.reshape( fifth_column_low_green, (-1,1))

    feature_matrix = np.concatenate( (first_column_min_colors_red, second_column_min_colors_green), axis=1)
    feature_matrix = np.concatenate( (feature_matrix, third_column_num_px_low_red_green), axis=1)
    feature_matrix = np.concatenate( (feature_matrix, forth_column_dps), axis=1)
    feature_matrix = np.concatenate( (feature_matrix, fifth_column_low_green), axis=1)

    return feature_matrix

def vectorYLabel():
    positiveY = np.ones(30)
    negativeY = np.zeros(30)
    return np.append(positiveY,negativeY)
