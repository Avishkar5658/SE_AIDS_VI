'''Perceptron Neural Network to recognize even and odd numbers. Given numbers are in 
ASCII from 0 to 9.'''


import numpy as np 
class Perceptron:
def init (self, input_size, lr=0.1): 
self.W = np.zeros(input_size + 1) 
self.lr = lr
def activation_fn(self, x): 
return 1 if x >= 0 else 0
def predict(self, x):
x = np.insert(x, 0, 1) 
z = self.W.T.dot(x)
a = self.activation_fn(z) 
return a
def train(self, X, Y, epochs): 
for _ in range(epochs):
for i in range(Y.shape[0]): 
x = X[i]
y = self.predict(x) 
e = Y[i] - y
self.W = self.W + self.lr * e * np.insert(x, 0, 1) 
X = np.array([
[0,0,0,0,0,0,1,0,0,0], # 0
[0,0,0,0,0,0,0,1,0,0], # 1
[0,0,0,0,0,0,0,0,1,0], # 2
[0,0,0,0,0,0,0,0,0,1], # 3
[0,0,0,0,0,0,1,1,0,0], # 4
[0,0,0,0,0,0,1,0,1,0], # 5
[0,0,0,0,0,0,1,1,1,0], # 6
[0,0,0,0,0,0,1,1,1,1], # 7
[0,0,0,0,0,0,1,0,1,1], # 8
[0,0,0,0,0,0,0,1,1,1], # 9
])
Y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# Create the perceptron and train it 
perceptron = Perceptron(input_size=10) 
perceptron.train(X, Y, epochs=100)
# Test the perceptron on some input data 
test_X = np.array([
[0,0,0,0,0,0,1,0,0,0], # 0
[0,0,0,0,0,0,0,1,0,0], # 1
[0,0,0,0,0,0,0,0,1,0], # 2
[0,0,0,0,0,0,0,0,0,1], # 3
[0,0,0,0,0,0,1,1,0,0], # 4
[0,0,0,0,0,0,1,0,1,0], # 5
[0,0,0,0,0,0,1,1,1,0], # 6
[0,0,0,0,0,0,1,1,1,1], # 7
[0,0,0,0,0,0,1,0,1,1], # 8
[0,0,0,0,0,0,0,1,1,1], # 9
])
for i in range(test_X.shape[0]):
x = test_X[i]
y = perceptron.predict(x)
print(f'{x} is {"even" if y == 0 else "odd"}')
