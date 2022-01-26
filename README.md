# Function Estimation with MLP and RBF

> This repository is made for the AI course project - Mar 2018.

**Dependencies:**
- [scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

---

Two machine learning models to estimate function `y = x^2`.

#### MLP:
- Using `MLPRegressor` from scikit-learn
- hidden_layer_sizes = 75
- solver = LBFGS

![MLP](/results/MLP_estimation.png)

#### RBF:
- Implemented with only using NumPy, by calculating `gaussian_distance`.
- center_num = 75

![RBF](/results/RBF_estimation.png)
