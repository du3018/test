#-----preprare data----------
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target
#-----fit data--------------------
logclf = LogisticRegression(C=1e5)
logclf.fit(X, Y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logclf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(8, 6))
pl.pcolormesh(xx, yy, Z, cmap=plt.get_cmap('Spectral'))
# Plot also the training points
for t,marker,col in zip(xrange(3),"o^D","rgb"):
    plt.scatter(X[Y == t,0], X[Y == t,1], marker=marker,c=col)
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')
pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
plt.savefig("E:/wulingfei/logistic_classification/logistic_classify4.png", bbox_inches="tight")
