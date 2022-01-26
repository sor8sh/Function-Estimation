from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# Training set
train_xx = np.linspace(-10, 10, 500)
train_x = [[x] for x in train_xx]
train_yy = np.power(train_xx, 2)
train_y = [y for y in train_yy]

# neural net model
reg = MLPRegressor(hidden_layer_sizes=(75,), solver='lbfgs')
reg.fit(train_x, train_y)

# test prediction
test_xx = np.linspace(-2, 2, 100)
test_x = [[t] for t in test_xx]
test_yy = np.power(test_xx, 2)

predict = reg.predict(test_x)

plt.plot(test_xx, test_yy, 'b-', label='real')
plt.plot(test_xx, predict, 'r-', label='fit')
plt.legend(loc='upper right')
plt.title('MLP')
plt.savefig('MLP_estimation.png')
plt.show()
