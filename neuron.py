'''
High level neural net implemantations
by Wesin Alves

List of methods
- getOutputSignal(self)
- setOutputSignal(self, outputsig)
- setInputSignal(self, inputsig)
- getInputSignal(self)
- computeActivation(self,activationType)
- computeGradient(self,activationType)

'''
import numpy as np 

class Neuron:

	def __init__(self):
		print('building neurons...')


	def getOutputSignal(self):
		return self.outputSignal		

	def setOutputSignal(self, outputsig):
		self.outputSignal = outputsig

	def setInputSignal(self, inputsig):
		self.inputSignal = inputsig

	def getInputSignal(self):
		return self.inputSignal

	def computeActivation(self,activationType):
		return{
			'sigm': 1 / (1 + np.exp(-self.inputSignal)),
			'tanh': np.tanh(self.inputSignal),
			'relu': np.maximum(0,self.inputSignal)
		}.get(activationType, self.inputSignal) #linear function is default if x not found

	def computeGradient(self,activationType):
		return{
			'sigm': self.outputSignal*(1-self.outputSignal),
			'tanh': 1 - self.outputSignal**2,
			'relu': 1 if self.outputSignal.all() > 0 else 0 # i need return a array!!!
		}.get(activationType, self.outputSignal) #linear function is default if x not found
    	


########### Test Neuron Class ############
#n = np.random.randn(5)
inputs = np.array([ 1.12484367, -0.04094506,  0.30312216, -0.3156011,  -0.65810963])
out = np.random.randn(5)
new_neuron = Neuron()
new_neuron.setInputSignal(inputs)
new_neuron.setOutputSignal(out)

print('neron input>>>>>>>>>>>>')
print(new_neuron.getInputSignal())
print('neron outputs>>>>>>>>>>>>')
print(new_neuron.getOutputSignal())

new_neuron.setOutputSignal(new_neuron.computeActivation('sigm'))
print('Sigmoid Activation>>>>>>>>')
print(new_neuron.getOutputSignal())

new_neuron.setOutputSignal(new_neuron.computeActivation('tanh'))
print('TanH Activation>>>>>>>>')
print(new_neuron.getOutputSignal())

new_neuron.setOutputSignal(new_neuron.computeActivation('relu'))
print('Relu Activation>>>>>>>>')
print(new_neuron.getOutputSignal())

###################################################################

new_neuron.setOutputSignal(new_neuron.computeGradient('sigm'))
print('Sigmoid gradiente>>>>>>>>')
print(new_neuron.getOutputSignal())

new_neuron.setOutputSignal(new_neuron.computeGradient('tanh'))
print('TanH gradient>>>>>>>>')
print(new_neuron.getOutputSignal())

new_neuron.setOutputSignal(new_neuron.computeGradient('relu'))
print('Relu gradient>>>>>>>>')
print(new_neuron.getOutputSignal())
