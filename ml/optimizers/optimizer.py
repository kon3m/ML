import numpy as np
from ml import losses
from ml import epoch_an
from ml.activation import activations
from ml.layer import layer
class Optimizer:
	def __init__(self,X,Y,lr_rate=1e-3,epochs=10,layers=None,batch_size=32,loss="cost"):
		self.X=X
		self.Y=Y
		self.lr=lr_rate
		self.epochs=epochs
		self.batch_size=batch_size
		self.data=np.asarray(list(zip(X,Y)))
		self.loss=loss
		self.graph_layers=layers
		self.with_bias=self.graph_layers[-1].set_bias
	def make_batches(self,batch_size):
		"""makes batches

			INPUT:
				batch_size	=	batch_size
		"""
		ite=0
		w=lambda x:(x,x+batch_size) if (x+batch_size)<len(self.data) else (x,x+(len(self.data)-x))
		q=lambda x:w(x) if x==0 else w(x[1])
		size_index_list=[]
		while(1):
			ite=q(ite)
			if ite==(len(self.data),len(self.data)):break
			else:size_index_list.append(ite)
		self.batches=np.array([self.data[_[0]:_[1]] for _ in size_index_list])
		np.random.shuffle(self.batches)
	def optimize(self,cost="cost"):
		self.count=0
		self.make_batches(self.batch_size)
		self.loss=losses.cost
		for i in range(self.epochs):
			print("\t\t\t\trunning epoch:%d"%(i+1))
			for j in self.batches:
				loss=self.update_on_batch(j,batch_size=self.batch_size)
				#print("loss after a batch ",np.sum(loss))
	def update_on_batch(self,batch,batch_size=16):
		trainable_layers=self.graph_layers[1:]
		batch_loss=0
		batch_grads_w=[np.zeros(i.get_weights.shape) for i in trainable_layers]
		if self.with_bias:
			batch_grads_b=[np.zeros(i.get_biases.shape) for i in trainable_layers]
		for ex in batch:
			self.count+=1
			single_grads,cpl,loss=self.update_on_single_example(ex[0],ex[1])
			batch_loss+=loss
			batch_grads_w=[a+b for a,b in zip(batch_grads_w,single_grads)]
			if self.with_bias:
				batch_grads_b=[a+b for a,b in zip(batch_grads_b,cpl)]
		for i in range(len(trainable_layers)):
			trainable_layers[i].weights-=(self.lr/batch_size)*batch_grads_w[i]
			if self.with_bias:
				trainable_layers[i].biases-=(self.lr/batch_size)*batch_grads_b[i]
		return batch_loss
	def update_on_single_example(self,X,Y):
		gradient_updates_w=[]
		gradient_updates_b=[]
		costs_per_layer=[]
		pred=X
		trainable_layers=self.graph_layers[1:]
		for i in trainable_layers:
			pred=i(pred)
		print("predicted",np.argmax(pred))
		network_loss=losses.cost(pred,Y)
		print("network_loss:",network_loss)
		trainable_layers.reverse()
		rev=trainable_layers
		for i in range(len(rev)):
				if not costs_per_layer:
					loss=(network_loss)*(rev[i].get_activation(rev[i].get_weighted_sum,prime=True))
					grad=np.dot(loss,rev[1].get_activations.T)
					print("grad1:",grad)
					gradient_updates_w.append(grad)
					gradient_updates_b.append(loss)
					rev[i].loss=loss
					costs_per_layer.append(loss)	
				else:
					loss=np.dot(rev[i-1].get_weights.T,rev[i-1].loss)*(rev[i].get_activation(rev[i].get_weighted_sum,prime=True))
					print("grad2:",np.sum(loss))
					costs_per_layer.append(loss)
					rev[i].loss=loss
					gradient_updates_b.append(loss)
					if isinstance(self.graph_layers[-(i+2)],layer.Input):
						grad=np.dot(loss,X.T)
					else:
						grad=np.dot(loss,rev[i+1].get_activations.T)
					gradient_updates_w.append(grad)
		# if self.with_bias:
		return  reversed(gradient_updates_w),reversed(costs_per_layer),network_loss