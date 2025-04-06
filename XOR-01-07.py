# XOR problem using MLPClassifier from sklearn
# This code demonstrates how to use the MLPClassifier from sklearn to solve the XOR problem.
# The XOR problem is a classic example of a non-linear classification problem.
# The MLPClassifier is a multi-layer perceptron classifier that can learn non-linear decision boundaries.
# The code uses the logistic activation function and stochastic gradient descent (SGD) as the solver.
# The code also visualizes the decision boundary learned by the MLPClassifier.

from sklearn.neural_network import MLPClassifier 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
f=np.array([[0,0],[1,1],[0,1],[1,0]]) 
g=np.array([0,0,1,1]) 
clf=MLPClassifier(hidden_layer_sizes=2,activation='logistic', solver='sgd') 
clf.fit(f,g) 

x_min, x_max = f[:, 0].min() - .5, f[:, 0].max() + .5 
y_min, y_max = f[:, 1].min() - .5, f[:, 1].max() + .5 

xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] 

Z = Z.reshape(xx.shape) 

sns.set_theme(style='white') 

plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.3) 
for i in range(4):   
    if g[i]==0:
        plt.scatter(f[i][0],f[i][1],s=200,c='r',edgecolor='k')   
    else: 
        plt.scatter(f[i][0],f[i][1],s=200, c='b', edgecolor='k') 

plt.title('XOR') 
plt.show()