'''Python program to show back propogation network for XOR function with binary input and output'''


import numpy as np

class XORNetwork:
def __init__(self):
self.W1=np.random.randn(2,2)
self.b1=np.random.randn(2)
self.W2=np.random.randn(2,1)
self.b2=np.random.randn(1)
def sigmoid(self,x):
return 1/(1 +np.exp(-x))
def sigmoid_derivative(self,x):
return x*(1-x)
def forward(self,X):
#Performtheforwardpass
self.z1=np.dot(X,self.W1)+self.b1
self.a1=self.sigmoid(self.z1)
self.z2=np.dot(self.a1,self.W2)+self.b2
self.a2=self.sigmoid(self.z2)
return self.a2
def backward(self, X, y,output):
# Perform the backward
self.output_error=y - output
self.output_delta=self.output_error*self.sigmoid_derivative(output)
self.z1_error=self.output_delta.dot(self.W2.T)
self.z1_delta=self.z1_error*self.sigmoid_derivative(self.a1)
self.W1 +=X.T.dot(self.z1_delta)
self.b1+=np.sum(self.z1_delta,axis=0)
self.W2+=self.a1.T.dot(self.output_delta)
self.b2+=np.sum(self.output_delta,axis=0)
def train(self,X,y,epochs):
#Trainthenetworkforagivennumberolepochs
for i in range(epochs):
output=self.forward(X)
self.backward(X,y,output)
def predict(self,X):
#Makepredictionsforagivensetofinputs
return self.forward(X)

#CreateanewXORNetworkinstance
xor_nn=XORNetwork()

#Definetheinputandoutputdatasetsfor
X=np.array([[0,0], [0,1], [1,0],[1, 1]])
y=np.array([[0],[1],[1],[0]])

#Trainthenetworkfor10000epochs
xor_nn.train(X,y,epochs=10000)

#Makepredictionsontheinputidalaset
predictions=xor_nn.predict(X)

#Printhepredictions
print(predictions)


