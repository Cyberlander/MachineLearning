import numpy as np
import csv
import matplotlib.pyplot as plt


class AdaBoost:

    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = np.ones(self.N) / self.N
        self.RULES = []
        self.ALPHA = []
        self.HORIZONTAL_LINES = []
        self.VERTICAL_LINES = []

    def train(self, iterations):
        for i in range(iterations):
            self.train_iteration(-10, -10, 10, 10)

    def train_iteration(self, min_x, min_y, max_x, max_y):
        min_error = 2
        best_rule = None
        horizontal = True
        pos = 0
        for y_pos in range(min_y + 1, max_y - 1):
            rule = lambda x, y_pos=y_pos: x[1] < y_pos
            error = self.add_rule(rule, test=True)

            if error < min_error:
                min_error = error
                best_rule = rule
                horizontal = True
                pos = y_pos

        for x_pos in range(min_x + 1, max_x - 1):
            rule = lambda x, x_pos=x_pos: x[0] < x_pos
            error = self.add_rule(rule, test=True)

            if error < min_error:
                min_error = error
                best_rule = rule
                horizontal = False
                pos = x_pos

        if horizontal:
            self.HORIZONTAL_LINES.append(pos)
        else:
            self.VERTICAL_LINES.append(pos)

        self.add_rule(best_rule)

    def add_rule(self, func, test=False):
        errors = np.array([t[1] != func(t[0]) for t in self.training_set])
        e = (errors * self.weights).sum()
        if test:
            return e

        # print("rule error ", e)
        alpha = 0.5 * np.log((1 - e) / e)
        w = np.zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1:
                w[i] = self.weights[i] * np.exp(alpha)
            else:
                w[i] = self.weights[i] * np.exp(-alpha)

        self.weights = w / w.sum()
        self.RULES.append(func)
        self.ALPHA.append(alpha)

    def print_result(self):
        misclassfied = 0
        for (point, value) in self.training_set:
            classifier_results = [
                alpha * rules(point) for alpha, rules in zip(self.ALPHA, self.RULES)]
            correct_class = np.sign(value) == np.sign(sum(classifier_results))
            if not correct_class:
                misclassfied += 1

        print("Misclassified %d points" % misclassfied)


def import_training_set(file):
    training_set = []
    with open('dataCircle.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            values = [float(x) for x in values]
            training_set.append(((values[0], values[1]), values[2]))

    return training_set


def show_plot(positive_points, negative_points, horizontal_lines, vertical_lines):
    print(horizontal_lines)
    print(vertical_lines)
    for pos in horizontal_lines:
        plt.axhline(pos)

    for pos in vertical_lines:
        plt.axvline(pos)

    x_positive_points = [x[0] for x in positive_points]
    y_positive_points = [x[1] for x in positive_points]
    x_negative_points = [x[0] for x in negative_points]
    y_negative_points = [x[1] for x in negative_points]
    plt.plot(y_negative_points, x_negative_points,
             "ro", x_positive_points, y_positive_points, "b+")

    plt.show()


if __name__ == '__main__':
    training_set = import_training_set('dataCircle.txt')
    adaBoost = AdaBoost(training_set)
    adaBoost.train(iterations=4)
    adaBoost.print_result()

    positive_set = [point[0] for point in training_set if 1.0 == point[1]]
    negative_set = [point[0] for point in training_set if 0.0 == point[1]]
    show_plot(positive_set, negative_set,
              adaBoost.HORIZONTAL_LINES, adaBoost.VERTICAL_LINES)
