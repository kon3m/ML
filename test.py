from ml.preprocess.util import Preprocess,normalize,Flatten
from ml.layer.layer import *
from ml.graph import *
from ml.preprocess.util import normalize,sub_mean_ch
import numpy as np

path="./ml/dataset/train/"
t_path="./ml/dataset/test/"
i=Preprocess(path)
X,Y=i.direc_to_array()
test_x=normalize(X[0])
X=normalize(X)
p=Graph()
p.add(Input(784))
p.add(DNN(1024,activation="relu",set_bias=False))
p.add(DNN(1024,activation="relu",set_bias=False))
p.add(DNN(1024,activation="relu",set_bias=False))
p.add(DNN(1024,activation="relu",set_bias=False))
p.add(DNN(10,activation="relu",set_bias=False))
p.get_graph
p.train(X=X,Y=Y,lr=0.01,epochs=30,optimizer="sgd")
j=Preprocess(t_path)
X,Y=i.direc_to_array()
# X=Flatten(normalize(X))
outs=p.predict(X)
print(outs)
