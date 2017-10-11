'''
High level neural net implemantations
Neural Net is a tool bio inspired in brain work. Neural nets is composed by neurons in layers, weights to connect inputs signal to neurons, and outputs.
Like dendrites, neural net has weights values struture that receiveis the input signal from external world. 
Like Axion, neural net has output struture for send signal for external world or other layer neuron.
by Wesin Alves

list of attributes
- layers 
- error
- learningRate
- inputTrain
- inputTest
- target



list of methods
- train(self, epochs)
- loss(self)
- evaluate(self)

'''
import numpy as np
import layer 

class NeuralNet:
	def __init__(self, inputTrain, target, learningRate):
		self.inputTrain = inputTrain
		self.target = target
		self.learningRate = learningRate		
		print('creating neural net...')

	def configLayers(self, config):
		self.layer = {}
		for i in range(config['numLayers']):
			self.layer[i] = layer.Layer(config['size'][i], config['activate'][i], self.learningRate)
			self.layer[i].initWeights(config['isize'][i])

	def loss(self,output,target):
		return (np.sum(target - output)**2)/(2*len(target))

	def train(self,epochs):
		cost = 0
		for e in range(epochs):
			for i in range(len(self.inputTrain)):
				#foward
				print('########### forward ###############')
				for l in range(len(self.layer)):
					layer = self.layer[l]					
					if l > 0:
						layer.setInputs(self.layer[l-1].getOutputs())											
					else:
						layer.setInputs( np.array([self.inputTrain[i]]) )
					
					outputSignal = layer.forward()
					layer.setOutputs(outputSignal)
					#print('layer {0}>>>>>> '.format(l))
									
				cost = self.loss(self.target[i],outputSignal)

				print('######## backward ###############')
				#backward
				for l in range(len(self.layer),0,-1):
					layer = self.layer[l-1]
					
					if l == len(self.layer):
						delta = layer.backward(-(np.array([target[i]]) - outputSignal))
					else:
						grad = np.dot(self.layer[l].delta,self.layer[l].weights.T)
						delta = layer.backward(grad)
					layer.update(delta)
					#print('layer {0}>>>>>> '.format(l-1))
				
				#total_cost = total_cost + partial_cost
				print(cost)

			print(e)	


	

			                                                                                                                                                                          

			



########TEST CLASS ################
inputTrain = np.random.randn(10,3)
target =  np.random.randint(2,size=(10,2))
lr = 0.3
config = {'numLayers':2,'size':[2,2],'isize':[len(inputTrain[0]),len(target[0])], 'activate':['sigm','sigm']}

nnet = NeuralNet(inputTrain,target,lr)
nnet.configLayers(config)
'''
print('print layers*******************')
for i in range(config['numLayers']):
	print(nnet.layer[i].weights)
	print('****************')
'''
nnet.train(1000)

