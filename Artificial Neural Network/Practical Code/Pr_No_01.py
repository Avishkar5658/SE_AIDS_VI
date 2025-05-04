
'''Activation function that are being used in neural network.'''


import numpy as np

import matplotlib.pyplot as plt

def sigmoid(x):
return 1/(1+np.exp(-x))

def relu(x):
return np.maximum(0,x)

def tanh(x):
return np.tanh(x)

def softmax(x):
return np.exp(x)/np.sum(np.exp(x))

x = np.linspace(-10,10,100)

fig,axs =plt.subplots(2,2,figsize=(8,8))
axs[0,0].plot(x,sigmoid(x))
axs[0,0].set_title('Sigmoid')
axs[0,1].plot(x,relu(x))
axs[0,1].set_title('ReLU')
axs[1,0].plot(x,tanh(x))
axs[1,0].set_title('Tanh')
axs[1,1].plot(x,softmax(x))
axs[1,0].set_title('softmax')

fig.suptitle('Common Activation Function')

for ax in axs.flat:
ax.set(xlabel ='x',ylabel = 'y')

plt.subplots_adjust(left =0.1,bottom=0.1, right =0.9,wspace=0.4,hspace=0.4)

plt.show()