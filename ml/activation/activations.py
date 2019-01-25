from math import exp
from ..test_shape import wrapper
import numpy as np
def sigmoid(X,prime=False):
	"""an activation function which outputs the value between (0,1)"""
	if prime:
		return sigmoid(X) * (1.0 - sigmoid(X))
	else:	
		return 1.0/(1.0+np.exp(-X))
def sigmoidDerivative(X):
	"""  input : array of features
	     output : Sigmoid derivative of the input  """

	return stable_sigmoid(X) * (1.0 - stable_sigmoid(X))

def tanh(X):
	"""an activation function which outputs the value between (-1,1)"""
	if isinstance(X,np.ndarray):
		return (2.0/(1.0+np.exp(-(2*X))))-1
	else:
		X=np.array(X)
		return tanh(X)
def relu(X,prime=False):
	if prime:
		X = np.array(X)
		grads= 1.0*(X>0)
		grads[grads==0]=1e-1
		return grads
	if isinstance(X,np.ndarray):
		return np.maximum(X,0)
	else:
		X = np.array(X)
		return relu(X)

def reluDerivative(X,ep=1e-1):
	X = np.array(X)
	grads= 1.0*(X>0)
	grads[grads==0]=1e-1
	return grads
def stable_sigmoid(X,prime=False):
	if prime:
		return stable_sigmoid(X) * (1.0 - stable_sigmoid(X))	
	if isinstance(X,(list,np.ndarray)):
		res=[]
		for i in X:
			#for now remove the condition
			if i>=0:
				z=exp(-i)
				res.append(z)
			else:
				z=exp(i)
				res.append(z/(1.0+z))
		return np.asarray(res)
	else:
		if X>=0:
				z=exp(-X)
				return 1.0/(1.0+z)
		else:
				z=exp(X)
				return z/(1.0+z)
def softmax(X):
	y = X - np.max(X)
	exp = np.exp(y)
	return exp/np.sum(exp)
def identity():pass
