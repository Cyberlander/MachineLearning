from scipy import misc
import numpy as np

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
Ende der nötigen Funktionen für RGB-Features
"""

"""
notiere: Minimum vom Red- und Grünwert sind ideale Features,
vor allem der Grünwert ist äußerst aussagekräftig
"""
#print(min_color_component_image(positives[0],0))
print(np.mean(all_min_color(positives,2)))
print(np.mean(all_min_color(negatives,2)))
