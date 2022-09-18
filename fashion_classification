#import libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax, linear, sigmoid
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
plt.style.use('./deeplearning.mplstyle')

#upload data


data= np.loadtxt("Untitled Folder/fashion-mnist_train.csv", delimiter=',')
X_train=data[:,1:]
y_train=data[:,0]
lable=["T-shirt", "Trouser","Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
data2=np.loadtxt("Untitled Folder/fashion-mnist_test.csv", delimiter=',')
X_=data2[:,1:]
Y_=data2[:,0]
X_cv,X_test,y_cv,y_test=train_test_split(X_,Y_,test_size=0.50,random_state=1)

#visualising data

fig, ax=plt.subplots(4,4,figsize=(6,6))
fig.tight_layout(pad=0.13, rect=[0,0.03,1,0.91])
for i, ax in enumerate(ax.flat):
    index=np.random.randint(X_train.shape[0])
    X_img=X_train[index].reshape((28,28))
    ax.imshow(X_img,cmap='gray')
    ax.set_axis_off()
    ax.set_title(lable[int(y_train[index])])
    
#creat nural network structure

    tf.random.set_seed(1234)
model=Sequential([
    Dense(25,input_dim=784,activation="relu",name='l1',kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(15,activation="relu",name='l2',kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(10,activation="linear",name='l3')
],name="model1")
model.summary()

#define loss function and the optimizer type

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.001)
)

#train the model parameters

model.fit(
    X_train,y_train,
    epochs=40
)

#define function to get the error

def get_err(y,yhat):
    m=len(y)
    err1=0
    for i in range (m):
        if y[i]!=yhat[i]:
            err1+=1
    err=err1/m
    return(err)
  
#define function to get the prediction

def evaluate_yhat(X):
    m=X.shape[0]
    yhat=np.zeros(m)
    y1=np.zeros(10)
    for i in range(m):
        y1=model.predict(X[i].reshape(1,784))
        yhat[i]=np.argmax(y1)
    return(yhat)

yhat=evaluate_yhat(X_)
print(yhat[:20])

#visualising the predicted inferences by using 46 random examples of test data 

fig,ax=plt.subplots(4,4,figsize=(7,7))
fig.tight_layout(pad=0.13,rect=[0,0.03,1,0.91])
m2=X_.shape[0]
m2-=1
for i, ax in enumerate(ax.flat):
    index2=np.random.randint(m2)
    X_img_pred=X_[index2].reshape((28,28))
    ax.imshow(X_img_pred,cmap='gray')
    ax.set_axis_off()
    ax.set_title(lable[int(yhat[index2])])
