import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(3)
n = 40
X = np.hstack((norm.rvs(loc=2, size=n, scale=2), norm.rvs(loc=8, size=n, scale=3)))
y = np.hstack((np.zeros(n),np.ones(n)))
plt.figure(figsize=(10, 4))
plt.xlim((-5, 20))
plt.scatter(X, y, c=y)
plt.xlabel("feature value")
plt.ylabel("class")
plt.grid(True, linestyle='-', color='0.75')
plt.savefig("E:/wulingfei/logistic_classification/logistic_classify1.png", bbox_inches="tight")
