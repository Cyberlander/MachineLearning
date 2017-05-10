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

clf = svm.SVC(kernel="sigmoid", C=1,gamma=0.07)
clf.fit(feature_matrix,yLabel)
print(clf.predict(test_set))
