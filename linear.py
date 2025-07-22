import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
X=np.linspace(1,10,100)
Y=2*X+10+np.random.randn(X.shape[0])
print(X)
print(Y)
X.shape
X.reshape(-1,1)
X.shape
model=Sequential()
model.add(Dense(1,input_dim=1,activation='linear'))
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
model.fit(X,Y,epochs=10,verbose=1)
pred=model.predict(X)
plt.scatter(X,Y,label="original data")
plt.plot(X,pred)
plt.show()







#import libraries
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#load the data
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#print(X_train.shape)
y_test
#normalize the data

#categorical
#print(y_train[0])
#y_train=y_train.astype('float32')/255.0
#y_test=y_test.astype('float32')/255.0

#one-hot encoding
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#build the architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

#compile the model
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#train the model
model.fit(X_train,y_train,epochs=1,batch_size=64)

#evaluate
model.evaluate(X_test,y_test)

#predictons
sample_images=X_test[:5]
sample_labels=y_test[:5]
predictions=model.predict(sample_images)
print(predictions)
result=np.argmax(predictions,axis=1)
print(result)

for i in sample_images:
  plt.subplot(1,5,i+1)
  plt.imshow(sample_images[i])
  plt.title(f"actual label:{sample_labels[i]}\npredicted label:{result[i]}")


plt.show()
