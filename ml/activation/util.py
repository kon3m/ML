from math import exp
import numpy as np
def sigmoid(X,prime=None):
	"""an activation function which outputs the value between (0,1)"""
	if isinstance(X,np.ndarray):
		if prime:
			return sigmoid(X) * (np.ones(len(X))-sigmoid((X)))

		else:
			return 1.0/(1.0+np.exp(-X))
	else:
		X=np.array(X)
		return sigmoid(X)
		#return 1.0/(1.0+np.exp(-X))
def tanh(X):
	"""an activation function which outputs the value between (-1,1)"""
	if isinstance(X,np.ndarray):
		return (2.0/(1.0+np.exp(-(2*X))))-1
	else:
		X=np.array(X)
		return tanh(X)
def softmax(X):
	"""an activation function which outputs the value between (0,1)"""
	_X=np.full(X.shape,np.inf)
	for i in range(len(X)):
		_X[i]=exp(X[i])/(np.sum(np.exp(X),axis=0))
	return _X
def relu(X,ep=1e-1,prime=False):
	if isinstance(X,np.ndarray):
		if prime :
			grads= 1.0 * (X > 0)
			grads[grads==0]=1e-1
			return grads
		else: return np.maximum(ep*Xadded,X)
		
		
	
	
