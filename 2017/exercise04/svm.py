import numpy as np
import glob
from scipy import misc
import feature_matrix_module
from sklearn import svm

def read_images( path ):
    # getting all file names with png extension
    files = glob.glob( path +  "/*.png")
    images = []
    for f in files:
        images.append( misc.imread( f ) )
    return images

positives = read_images("positives")
negatives = read_images("negatives")


feature_matrix = feature_matrix_module.build_feature_matrix( positives, negatives )
yLabel = feature_matrix_module.vectorYLabel()


training_set = feature_matrix[:59,:]
test_set = feature_matrix[0,:]
test_set2 = feature_matrix[feature_matrix.shape[0]-1,:]

def linear_kernel_check( feature_marix, classifier_linear_kernel ):
    output = []
    for i in range( feature_matrix.shape[0]):
        row = feature_matrix[i]
        row = row.reshape(1,-1)
        prediction = classifier_linear_kernel.predict( row )

        output.append( prediction[0] )
    return output

classifier_linear_kernel = svm.SVC(kernel='linear', C = 1.0)
classifier_linear_kernel.fit(feature_matrix,yLabel)
print(linear_kernel_check(feature_matrix, classifier_linear_kernel))
