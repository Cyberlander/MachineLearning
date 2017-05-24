import csv
import numpy as np
import matplotlib.pyplot as plt


data = []

with open("dataCircle.txt", "r") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=" ")
    for row in csvreader:
        data = data + row


def data_cleaning(data):
    arraysize = len(data)
    newData = []
    iterations = 0
    for i in data:
        iterations = iterations + 1
        if i != '':
            newData.append(i)
    return newData


def transform_data_to_matrix(data):
    data = np.array(data, dtype=np.float64)
    data = np.reshape(data, (-1, 3))
    return data


def initialize_point_importance(points):
    count_points = len(points)


def show_plot(positive_points, negative_points):
    x_positive_points = positive_points[:, 0]
    y_positive_points = positive_points[:, 1]
    x_negative_points = negative_points[:, 0]
    y_negative_points = negative_points[:, 1]
    plt.plot(y_negative_points, x_negative_points,
             "ro", x_positive_points, y_positive_points, "b+")

    plt.show()


data_circle = transform_data_to_matrix(data_cleaning(data))
points = data_circle[:, :-1]
class1 = points[0:51, ]
class2 = points[51:, ]
show_plot(class1, class2)
print(initialize_point_importance(points))
