import numpy as np
import glob
from scipy import misc


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


def build_feature_matrix(postives, negatives):
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

def generate_mean_vectors(positives, negatives):
    positive_mean_vector = []
    positive_min_colors = np.array(min_color_images(positives))
    positive_min_colors_red = positive_min_colors[:,0]
    positive_min_colors_green = positive_min_colors[:,1]
    positive_num_low_red_green = num_pixels_low_red_green_images(positives)
    positive_num_dps = find_dark_pixel_spots_in_images(positives)
    positive_num_very_low_green = find_very_low_green_values_in_images(positives)

    positive_mean_vector.append( np.mean(positive_min_colors_red) )
    positive_mean_vector.append( np.mean(positive_min_colors_green) )
    positive_mean_vector.append( np.mean(positive_num_low_red_green) )
    positive_mean_vector.append( np.mean(positive_num_dps) )
    positive_mean_vector.append( np.mean(positive_num_very_low_green) )

    negative_mean_vector=[]
    negatives_min_colors = np.array(min_color_images(negatives))
    negatives_min_colors_red = negatives_min_colors[:,0]
    negatives_min_colors_green = negatives_min_colors[:,1]
    negatives_num_low_red_green = num_pixels_low_red_green_images(negatives)
    negatives_num_dps = find_dark_pixel_spots_in_images(negatives)
    negatives_num_very_low_green = find_very_low_green_values_in_images(negatives)

    negative_mean_vector.append( np.mean(negatives_min_colors_red) )
    negative_mean_vector.append( np.mean(negatives_min_colors_green) )
    negative_mean_vector.append( np.mean(negatives_num_low_red_green) )
    negative_mean_vector.append( np.mean(negatives_num_dps) )
    negative_mean_vector.append( np.mean(negatives_num_very_low_green) )

    return positive_mean_vector, negative_mean_vector


def pdf_multivariate_gauss(x, mean_vector, cov_matrix):
    part1 = 1/ ( ((2 * np.pi)**(len(mean_vector)/2)) * (np.linalg.det(cov_matrix)**(1/2))  )
    part2 = (-1/2) * ((x - mean_vector).T.dot(np.linalg.inv(cov_matrix)).dot((x - mean_vector)))
    return float(part1 * np.exp(part2))


positives = read_images('images/positives')
negatives = read_images('images/negatives')
feature_matrix = build_feature_matrix(positives, negatives)
positive_mean_vector, negative_mean_vector = generate_mean_vectors(positives, negatives)
feature_matrix_transposed = np.transpose(feature_matrix)
cov_matrix = np.cov(feature_matrix_transposed)

image_num = 5
image_values = np.array(feature_matrix[image_num,:])
gauss_negative = pdf_multivariate_gauss(image_values, negative_mean_vector, cov_matrix)
gauss_positive = pdf_multivariate_gauss(image_values, positive_mean_vector, cov_matrix)
px = gauss_negative * 0.5 + gauss_positive * 0.5
pyx = (gauss_negative * 0.5) / px

print("Image {} is negative with a probability of: {}".format(image_num, pyx))
