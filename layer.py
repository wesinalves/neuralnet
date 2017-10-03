'''
High level neural net implemantations
by Wesin Alves

List of methods:
- setInputs(self,inputs)

- setOutputs(self,outputs)

- setBias(self,bias)

- initWeights(self, inputSize)

- forward(self) # propagete inputs

- backward(self,error) # backpropagate erros

- update(self,delta) # upadte weights and bias

Math equations:

W: wheights matrix
i: inputs 
b: biases
W_new: wheights updated
b_new: biases updated

inputSignal = W x i + b // x is a dot product
outputSignal = sig(inputSignal) // sig is a sigmoid function implemented by computeActivation method in Neuron class
delta = gradient * error // gradient is a computed in computeGradient method in Neuron class; error is layer type dependent; * is a hadamard product
W_new = W + lr * inputSignal x delta // lr is learningRate
b_new = b + lr * delta

'''
import numpy as np
import neuron 

class Layer:
	def __init__(self, size, activationFunction, learningRate):
		self.size = size
		self.lr = learningRate
		self.activationFunction = activationFunction
		self.bias = np.random.randn(1,size)
		self.neurons = neuron.Neuron()
		print('creating layer...')

	def setInputs(self, inputs):
		self.inputs = inputs

	def setOutputs(self, outputs):
		self.outputs = outputs

	def setBias(self, bias):
		self.bias = bias

	def initWeights(self, inputSize):
		self.weights = np.random.randn(inputSize, self.size)

	def forward(self):
		inputSignal = np.dot(self.inputs, self.weights) + self.bias
		self.neurons.setInputSignal(inputSignal)
		outputSignal = self.neurons.computeActivation(self.activationFunction)
		return outputSignal
	
	def backward(self, error):
		self.neurons.setOutputSignal(self.outputs)
		grad = 	self.neurons.computeGradient(self.activationFunction)
		delta = grad * error
		return delta
	def update(self, delta):
		self.weights = self.weights + self.lr * self.inputs.T.dot(delta)
		self.bias = self.bias + self.lr * delta
		

#### Test Class ####
inputs = np.random.randn(1,3)
l = Layer(4,'sigm',learningRate=0.2)
print(len(inputs[0]))
l.initWeights(len(inputs[0]))
l.setInputs(inputs)
print('####### inpus ###########')
print(inputs)
print('####### weights ###########')
print(l.weights)
print('####### bias ###########')
print(l.bias)
outputSignal = l.forward()
print('################# Forward method #########')
print(outputSignal)
print('################## backward ##############')
l.setOutputs(outputSignal)
delta = l.backward(-0.12)
print(delta)
print(inputs.T)
l.update(delta)
print('####### weights updated###########')
print(l.weights)
print('####### bias updated###########')
print(l.bias)