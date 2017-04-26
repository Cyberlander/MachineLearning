import numpy as np
import glob
from scipy import misc


# prints an array of strings
def read_images( path ):
    files = glob.glob( path + "/*.png")
    images = []
    for f in files:
        images.append( misc.imread( f ) )
    return images


negatives = read_images( "negatives" )
positives = read_images( "positives" )

def min_color_component_image(image):
    min_colors = [255, 255, 255]
    for i in range(0,24):
        for j in range(0,24):
            value_red = image[i][j][0]
            value_green = image[i][j][1]
            value_blue = image[i][j][2]
            if value_red < min_colors[0]:
                min_colors[0] = value_red
            if value_green < min_colors[1]:
                min_colors[1] = value_green
            if value_blue < min_colors[2]:
                min_colors[2] = value_blue
    return min_colors

def min_color_images( images ):
    min_colors = []
    for image in images:
        min_colors.append( min_color_component_image( image ))
    return min_colors

def min_green_color( images ):
    min_green = []
    for image in images:
        min_green.append( min_color_component_image( image )[1])
    return min_green

def min_read_color( images ):
    min_red = []
    for image in images:
        min_red.append( min_color_component_image( image )[0])
    return min_red


print( min_read_color( negatives ) )
print( min_read_color( positives ) )
