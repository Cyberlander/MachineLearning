from scipy import misc
import numpy as np
from scipy.stats import multivariate_normal

negatives = [None]*30
positives = [None]*30

#negative Beispiele einlesen
for i in range(0,30):
    if (i<9):
        indexImage = i + 1;
        negative_string = "negatives/n0" + str(indexImage) + ".png"
        #print(negative_string)
        negatives[i] = misc.imread(negative_string)
    else:
        indexImage = i + 1;
        negative_string = "negatives/n" + str(indexImage) + ".png"
        #print(negative_string)
        negatives[i] = misc.imread(negative_string)

print(negatives)
#positive Beispiele einlesen
for j in range(0,30):
    if (j<9):
        jj = j + 1;
        positive_string = "positives/p0" + str(jj) + ".png"
        positives[j] = misc.imread(positive_string)
    else:
        jj = j + 1
        positive_string = "positives/p" + str(jj) + ".png"
        positives[j] = misc.imread(positive_string)



#feature 1 brightness
brightness_images_sum = [None]*30

#brightness for an image
#

"""
Alle nötigen Funktionen für die RGB-Features
"""
def red_component_image(image_matrix):
    red_component_matrix = []
    red_component_matrix = np.array(red_component_matrix)

    for i in range(0,24):
        for j in range(0,24):
            value = image_matrix[i][j][0]
            red_component_matrix = np.append(red_component_matrix,value)
    return red_component_matrix


"""
berechnet für ein Bild (als Matrix dargestellt) und einer
gegebenen Farbkomponente(R,G oder B) den kleinsten Farbwert
der gegebenen Komponente
"""
def min_color_component_image(image,color):
    if color < 0:
        color = 0
    if color > 2:
        color = 2

    min_color = 255

    for i in range(0,24):
        for j in range(0,24):
            value = image[i][j][color]
            if value < min_color:
                min_color = value
    return min_color


"""
Berechnet die Anzahl der Pixel von einem Bild,
die niedrige Grün- und Rotwerte aufweisen
"""
def number_of_pixels_with_low_red_green_value(image):
    number_of_pixels = 0

    for i in range(0,24):
        for j in range(0,24):
            redvalue = image[i][j][0]
            greenvalue = image[i][j][1]
            if greenvalue < 120 and redvalue < 120:
                number_of_pixels = number_of_pixels + 1
    return number_of_pixels

def number_of_pixels_with_low_green_value(image):
    number_of_pixels = 0

    for i in range(0,24):
        for j in range(0,24):
            redvalue = image[i][j][0]
            greenvalue = image[i][j][1]
            if greenvalue < 75:
                number_of_pixels = number_of_pixels + 1
    return number_of_pixels


def find_dark_pixel_spots_in_image(image):
    dark_pixel_spot = 0
    for i in range(0,24):
        for j in range(0,24):

            greenvalue = image[i][j][1]
            if i > 2 and i < 21 and j > 3 and j < 21:
                green_value_north = image[i-3][j][1]
                green_value_south = image[i+3][j][1]
                green_value_east = image[i][j-3][1]
                green_value_west = image[i][j+3][1]
                if greenvalue < 100 and green_value_north > 100 and green_value_south > 100 and green_value_east > 100 and green_value_west > 100:
                    dark_pixel_spot = dark_pixel_spot + 1
    return dark_pixel_spot





"""
colors: 0 = red, 1 = green, 2 = blue
"""
def mean_color_component_image(image_matrix,color):
    if color < 0:
        color = 0
    if color > 2:
        color = 2
    red_component_matrix = np.array([])

    for i in range(0,24):
        for j in range(0,24):
            value = image_matrix[i][j][color]
            red_component_matrix = np.append(red_component_matrix,value)
    return np.mean(red_component_matrix)


"""
Berechnet für alle Pixel eines Bildes den RGB-Durchschnittswert
"""
def mean_rgb_components(image_matrix):
    rgb_mean_matrix = np.array([])
    for i in range(0,24):
        for j in range(0,24):
            value = np.mean(image_matrix[i][j])
            rgb_mean_matrix = np.append(rgb_mean_matrix,value)
    return np.mean(rgb_mean_matrix)



def all_color_means(images,color):
    if color < 0:
        color = 0
    if color > 2:
        color = 2
    images_red_component =  []
    for i in range(0,len(images)):
        value = mean_color_component_image(images[i],color)
        images_red_component = np.append(images_red_component,value)
    images_red_component = np.array(images_red_component)
    return images_red_component

"""
Berechnet für alle gegebenen Bilder den kleinsten Farbwert
"""
def all_min_color(images,color):
    min_colors = []
    for i in range(0,len(images)):
        value = min_color_component_image(images[i],color)
        min_colors = np.append(min_colors,value)
    min_colors = np.array(min_colors)
    return min_colors


"""
Berechnet für alle gegebenen Bilder die Anzahl der Pixel
mit niedrigen Rot- und Grünwerten
"""

def all_low_red_green_pixel_number(images):
    low_pixels = []
    for i in range(0,len(images)):
        value = number_of_pixels_with_low_red_green_value(images[i])
        low_pixels = np.append(low_pixels,value)
    return low_pixels

def all_low_green_pixel_number(images):
    low_pixels = []
    for i in range(0,len(images)):
        value = number_of_pixels_with_low_green_value(images[i])
        low_pixels = np.append(low_pixels,value)
    return low_pixels

def all_dark_pixel_spots(images):
    dark_pixel_spots = []
    for i in range(0,len(images)):
        value = find_dark_pixel_spots_in_image(images[i])
        dark_pixel_spots = np.append(dark_pixel_spots,value)
    return dark_pixel_spots

def all_mean_rgb_components(images):
    means = []
    for i in range(0,len(images)):
        value = mean_rgb_components(images[i])
        means = np.append(means,value)
    return means
"""
Ende der nötigen Funktionen für RGB-Features
"""

"""
notiere: Minimum vom Red- und Grünwert sind ideale Features,
vor allem der Grünwert ist äußerst aussagekräftig
weiteres Feature: Anzahl der Pixel mit niedrigem Rot/Grün-Wert
"""


"""
60x5 Matrix, bestehend aus den 5 Features jedes Bildes
"""


first_column = np.array([])
first_column = np.append(first_column,all_min_color(positives,0))
first_column = np.append(first_column,all_min_color(negatives,0))
first_column = np.reshape(first_column,(-1,1))

second_column = np.array([])
second_column = np.append(second_column,all_min_color(positives,1))
second_column = np.append(second_column,all_min_color(negatives,1))
second_column = np.reshape(second_column,(-1,1))

third_column = np.array([])
third_column = np.append(third_column,all_low_red_green_pixel_number(positives))
third_column = np.append(third_column,all_low_red_green_pixel_number(negatives))
third_column = np.reshape(third_column,(-1,1))

fourth_column = np.array([])
fourth_column = np.append(fourth_column,all_dark_pixel_spots(positives))
fourth_column = np.append(fourth_column,all_dark_pixel_spots(negatives))
fourth_column = np.reshape(fourth_column,(-1,1))

fifth_column = np.array([])
fifth_column = np.append(fifth_column,all_low_green_pixel_number(positives))
fifth_column = np.append(fifth_column,all_low_green_pixel_number(negatives))
fifth_column = np.reshape(fifth_column,(-1,1))

feature_matrix = np.ones(shape=(60,1))
feature_matrix = np.concatenate((feature_matrix,first_column),axis=1)
feature_matrix = np.concatenate((feature_matrix,second_column),axis=1)
feature_matrix = np.concatenate((feature_matrix,third_column),axis=1)
feature_matrix = np.concatenate((feature_matrix,fourth_column),axis=1)
feature_matrix = np.concatenate((feature_matrix,fifth_column),axis=1)
#print(feature_matrix)

feature_matrix = feature_matrix[:,1:]
features_positive_matrix = feature_matrix[:30,:]
features_negative_matrix = feature_matrix[30:,:]

feature_matrix_transposed = np.transpose(feature_matrix)
cov_matrix = np.cov(feature_matrix_transposed)

mean_positives = [np.mean(all_min_color(positives,0)),np.mean(all_min_color(positives,1)),
np.mean(all_low_red_green_pixel_number(positives)),np.mean(all_dark_pixel_spots(positives)),
np.mean(all_low_green_pixel_number(positives))]

mean_negatives = [np.mean(all_min_color(negatives,0)),np.mean(all_min_color(negatives,1)),
np.mean(all_low_red_green_pixel_number(negatives)),np.mean(all_dark_pixel_spots(negatives)),
np.mean(all_low_green_pixel_number(negatives))]


def pdf_multivariate_gauss(x,mean_vector,cov_matrix):
    part1 = 1/ ( ((2 * np.pi)**(len(mean_vector)/2)) * (np.linalg.det(cov_matrix)**(1/2))  )
    part2 = (-1/2) * ((x - mean_vector).T.dot(np.linalg.inv(cov_matrix)).dot((x - mean_vector)))
    return float(part1 * np.exp(part2))


#py = 0.5


#var = multivariate_normal(mean_negatives, cov_matrix)
#print(var.pdf(feature_matrix[5,:]))


#print(pdf_multivariate_gauss(np.array(feature_matrix[5,:]),mean_negatives,cov_matrix))

#print(pdf_multivariate_gauss(np.array(feature_matrix[5,:]),mean_positives,cov_matrix))



px = pdf_multivariate_gauss(np.array(feature_matrix[5,:]),mean_negatives,cov_matrix)*0.5 + pdf_multivariate_gauss(np.array(feature_matrix[5,:]),mean_positives,cov_matrix)*0.5
#print(px)
pyx = (pdf_multivariate_gauss(np.array(feature_matrix[5,:]),mean_negatives,cov_matrix)*0.5)/px
print(pyx)
#print(np.mean(all_mean_rgb_components(positives)))
#print(np.mean(all_mean_rgb_components(negatives)))

#print(np.mean(all_dark_pixel_spots(positives)))
#print(np.mean(all_dark_pixel_spots(negatives)))

#print(np.mean(all_low_red_green_pixel_number(positives)))
#print(np.mean(all_low_red_green_pixel_number(negatives)))



#print(min_color_component_image(positives[0],0))
#print(np.mean(all_min_color(positives,1)))
#print(np.mean(all_min_color(negatives,1)))
