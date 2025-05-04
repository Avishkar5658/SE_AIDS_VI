'''Demonstrate the Perceptron learning law with its decision regions'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# load iris dataset
iris = load_iris()

# extract sepal length and petal length features
X = iris.data[:, [0, 2]]
y = iris.target

# setosa is class 0, versicolor is class 1
y = np.where(y == 0, 0, 1)
# initialize weights and bias
w = np.zeros(2)
b = 0

# set learning rate and number of epochs
lr = 0.1
epochs = 50
# define perceptron function
def perceptron(x, w, b):
# calculate weighted sum of inputs
z = np.dot(x, w) + b
# apply step function
return np.where(z >= 0, 1, 0)

# train the perceptron
for epoch in range(epochs):
for i in range(len(X)):
x = X[i]
target = y[i]
output = perceptron(x, w, b)
error = target - output
w += lr * error * x
b += lr * error

# plot decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
Z = perceptron(np.c_[xx.ravel(), yy.ravel()], w, b)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

#
p
l
o
t
d
a
t
a
p
o
i
n
t
s
p
l
t.s
c
a
t
t
e
r
(
X
[:, 0
], X
[:, 1
], c
=
y, c
m
a
p
=
p
l
t.c
m.P
a
i
r
e
d
)
p
l
t.x
l
a
b
e
l
('S
e
p
a
l
l
e
n
g
t
h')
p
l
t.y
l
a
b
e
l
('P
e
t
a
l
l
e
n
g
t
h')
p
l
t.t
i
t
l
e
('P
e
r
c
e
p
t
r
o
n
d
e
c
i
s
i
o
n
r
e
g
i
o
n
s')
p
l
t.s
h
o
w
(
)
