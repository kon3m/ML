import os
from imageio import imread
import numpy as np
SUPPORTED_FORMATS=("png","jpg","PNG","JPG")
IMG_SAVE_PATH = "~/Desktop/ML/results/"
def checkfit(X,Y):
	if not isinstance(X,np.ndarray):
		X=np.array(X)
	if not isinstance(Y,np.ndarray):
		Y=np.array(Y)
	if len(X)!=len(Y):
		raise ValueError("The length of the features vector and target vector does not match")
	elif len(X.shape)!=2:
			raise ValueError("all the values in the feature vector do not have same dimensions")
	else:return X,Y
def one_hot_encoding(y):
	"""creates one-hot vector for the target variables
	for example,if a training set consists of multiple classes y=(0,1,2...n)
	one-hot vector for y(0)=[1,0,0,..n],y(2)=[0,0,1,0,..n]"""
	no_=y.shape[0]
	class_set=set([_[0] for _ in y if isinstance(_,(list,np.ndarray))])
	temp=np.full([no_,len(class_set)],0.0)
	for i in range(len(y)):
		temp[i][y[i]]=1
	return temp
def Flatten(X):
	return np.resize(X,[X.shape[0],X.shape[1],1])
def normalize(X):
	"""	This function normalizes the feature values:
		new_x=(x-min(x))/(max(x)-min(x))"""
	mi=np.min(X,axis=0)
	ma=np.max(X,axis=0)
	r=np.zeros(X.shape)
	r= np.divide((X-mi),(ma-mi))
	return r
def split_train_test(X,Y,per=20):
			
	"""splits training and testing data
	X=feature vector
	Y=target vector
	per=percentage of data the training set should take
	Example:
		from sklearn.datasets import load_iris
		a=load_iris()
		train_f,train_l,test_f,test_l=split_train_test(a.data,a.target)
	"""
	#if checkfit(X,Y):
	class_set=set(Y)
	l=list(Y)
	count=l.count(list(class_set)[0])
	dim_1=int((per/100)*count)*len(list(class_set))
	dim_2=X.shape[1]
	n_train_f=np.full([dim_1,dim_2],np.inf)
	n_test_f=np.full([len(X)-dim_1,dim_2],np.inf)
	n_train_t=np.full([dim_1,],0)
	n_test_t=np.full([len(X)-dim_1,],0)	
	f_i=0
	ins=f_i
	s_i=int((per/100)*count)
	ind=s_i
	temp=0
	for i in range(len(class_set)):
		n_train_f[ins:ins+ind]=X[f_i:s_i]
		n_test_f[temp:(count-ind)+temp]=X[s_i:f_i+count]
		n_train_t[ins:ins+ind]=Y[f_i:s_i]
		n_test_t[temp:(count-ind)+temp]=Y[s_i:f_i+count]
		f_i+=count
		s_i+=count
		ins+=ind
		temp+=count-ind
		return n_train_f,n_train_t,n_test_f,n_test_t

class nx:
	"""a class for custom methods in numpy"""
	def __init__(self,ob):
		if not isinstance(ob,np.ndarray):
			raise "expected np.array object"
		else:
			self.ob=ob
	def add_col(self,col_no,val=np.inf):
		"""this method returns an numpy.ndarray
		which consists of an extra column (with a same value) to the original array object passed in __init__.
		col_no=column number to add
		val=value to be passed"""
		try: new_shape=self.ob.shape[0],self.ob.shape[1]+1
		except IndexError:
			new_shape=self.ob.shape[0],1
		b=np.full(new_shape,np.inf)
		for i in range(len(self.ob)):
			temp=np.insert(self.ob[i],col_no,val)
			b[i]=temp
		return b
class Preprocess:
	"""
	Class for preprocessing, mainly deals with images and their conversion to arrays,checks for valid directory and other paths
	
	INPUT : root path(optional)
	"""
	def __init__(self,root=None):
		self.root_path=root
	def check_dir(self,path):
		"""check on the directory
		
		path=root path to the directory

		"""
		p=path
		if not path.endswith("/"):
			raise ValueError("Provided Invalid Path")
		if not os.path.exists(path):
			raise ValueError("Path does not exists")
		else:
			file_l=os.listdir(path)
			self.formats=checkformat(file_l)
			class_list=list(self.formats.keys())
			self.classes=class_list
		self.root_path=p
	def checkformat(self,path=None):
		"""check format of the images in the directory
		INPUT PARAMS:
			path= path to class directories
		"""
		if not path:raise ValueError("Path not provided")
		formats={}
		dir_list=os.listdir(self.root_path)
		for i in dir_list:
			new_path=os.path.join(self.root_path,i)
			j=os.listdir(new_path)
			if not j[0].endswith(SUPPORTED_FORMATS):
				raise ValueError("FILE FORMAT NOT SUPPORTED")
			else:formats[i]=j[0][-3:]
			return True
	def direc_to_array(self,path=None,shape=[28,28,3],grey_scale=True):
		"""This function converts images present in the specified path to arrays.
		

		INPUT:
		path   =    path to the directory of the images(if not given while creating an instance)
		
		OUTPUT:
		
		arr,tar   =    where	arr=features vector [instance of np.ndarray]
					tar=Target vecor	[instance of np.ndarray] 		
		EXAMPLE:
			>>>path="path_to/image_dir/"
			>>>i=Preprocess(root=path)
			>>>feat,target  =  i.direc_to_array()
				or 
			>>>i=Preprocess()
			>>>i.direc_to_array(path=path)
			>>>feat,target  =  i.img_to_array()
		"""
		arr=[]
		tar=[]
		self.class_dict={}
		if path: 
			if not os.path.exists(path):
				raise ValueError("Provided invalid path")
			else:self.root_path=path
		if self.checkformat(self.root_path):
			class_list=sorted(os.listdir(self.root_path))	
			for i in range(len(class_list)):
				self.class_dict[i]=class_list[i]
				new_path=os.path.join(self.root_path,class_list[i])
				new_dir_list=os.listdir(new_path)
				for j in range(len(new_dir_list)):
					img=self.img_to_array(os.path.join(new_path,new_dir_list[j]),shape=shape)
					if grey_scale:
						img=self.rgb2grey(img).flatten()
						img=img.reshape(img.shape[0],1)
						arr.append(img)
					else:arr.append(img)
				target=categorical(np.full([len(new_dir_list),],i),len(class_list))
				tar.extend(target)
		return np.asarray(arr),np.asarray(tar)

	def img_to_array(self,path,shape=None):
		"""This method converts single image to array
		
		INPUT	:
		
			path  	:  path to the image

		OUTPUT	:
			array form of the image(matrix of pixels)
		
		EXAMPLE:
			This is a method of the class Preprocess so we need to create an instance.
			>>>i=Preprocess()
			>>>path="Desktop/images/../.jpg"
			>>>X=i.img_to_array(path)
		"""

		
		if not os.path.exists(path):
				raise ValueError("Provided invalid path")
		if shape:return np.resize(imread(path),shape)
		else:return imread(path)
	
	def rgb2grey(self,img):
		""" INPUT : Array of rgb image
		    OUTPUT :Array of Greyscale image
		"""
		r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
		grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return grey



def sub_mean_ch(X):
	"""
		removes per channel mean
	"""
	temp_images=[]
	X=X.astype(np.float32)
	for i in X:
		i[:,:,0] -= 103.939
		i[:,:,1] -= 116.779
		i[:,:,2] -= 123.68
		i = i.transpose((2,0,1))
		temp_images.append(i)
	return np.array(temp_images).astype(np.float32)
def categorical(target,no_classes):
	"""
		assigning binary vectors for each class
		
		INPUT:
			target=target vector
			no_class=number of classes
	
	"""
	target=np.asarray(target,dtype="int32")
	if not no_classes:no_classes=np.max(target)+1
	T=np.zeros((len(target),no_classes))
	T[np.arange(len(target)),target]=1
	return T
	
	               
				
