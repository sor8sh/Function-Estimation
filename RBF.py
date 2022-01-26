import numpy as np
import matplotlib.pyplot as plt


class RBF(object):

    def __init__(self, input_dimension, center_num):
        self.input_dimension = input_dimension
        self.center_num = center_num
        self.centers = None
        self.weights = None

    def distance(self, training_data):
        dist = np.zeros((training_data.shape[0], self.center_num))
        i = 0
        for data in training_data:
            j = 0
            for center in self.centers:
                dist[i, j] = gaussian_distance(center, data)
                j += 1
            i += 1
        return dist

    def predict(self, test_data):
        output = np.dot(self.distance(test_data), self.weights)
        return output

    def fit(self, training_data, targets):
        random_args = np.random.permutation(training_data.shape[0]).tolist()
        self.centers = [training_data[arg] for arg in random_args][:self.center_num]
        dist = self.distance(training_data)
        self.weights = np.dot(np.linalg.pinv(dist), targets)


def gaussian_distance(center, data_point):
    return np.exp(-1 * np.linalg.norm(center - data_point) ** 2)


x = np.linspace(-10, 10, 500)
y = np.power(x, 2)
model = RBF(input_dimension=1, center_num=75)
model.fit(x, y)

test_x = np.linspace(-2, 2, 100)
test_y = np.power(test_x, 2)
predicted_y = model.predict(test_x)

plt.plot(test_x, test_y, 'b-', label='real')
plt.plot(test_x, predicted_y, 'r-', label='fit')
plt.legend(loc='upper right')
plt.title('RBF')
plt.savefig('RBF_estimation.png')
plt.show()
