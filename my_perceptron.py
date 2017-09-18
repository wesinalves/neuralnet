'''
simple implementation of rosemblat's perceptron
Wesin Alves
'''
import numpy as np
##set parameters
num_inputs = 4
lr = 0.1
bias = np.random.normal()

# input signals
inputs = [np.array([int(y) for y in bin(x).lstrip("0b").zfill(num_inputs)])  for x in range(2**num_inputs)]
print("Shape of input:")
for x in inputs:
	print(x)


##weight array
weights = np.random.randn(num_inputs)
## forward function
def forward(W,x):
	return 1 if np.dot(W,x) + bias > 0 else 0

# set target function (NAND)
def target(x):
	return int (not (x[0] and x[1]))	

## backward
number_of_errors = -1
while number_of_errors !=0:
	number_of_errors = 0
	print ("Beginning iteration")
	print ("Bias: {:.3f}".format(bias))
	print ("Weights:", ", ".join(
                "{:.3f}".format(wt) for wt in weights))

	for x in inputs:
		error = target(x) - forward(weights, x)
		if error:
			number_of_errors += 1
			bias = bias + lr * error
			weights = weights + lr * error * x
	
	print ("Number of errors:", number_of_errors, "\n")





