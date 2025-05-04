
'''ANDNOT Function using MaCulloch-Pitts Neural Network'''

import numpy as np

def linear_threshold_gate(dot, T): '''Returns the binary threshold output'''
if dot >= T:
return 1
else:
return 0

input_table = np.array([
[0,0], # both no
[0,1],
[1,0],
[1,1]
])
print(f'input table:\n{input_table}')

weights = np.array([1,-1])
dot_products = input_table @ weights
T = 1
for i in range(0,4):
activation = linear_threshold_gate(dot_products[i], T)
print(f'Activation: {activation}')
