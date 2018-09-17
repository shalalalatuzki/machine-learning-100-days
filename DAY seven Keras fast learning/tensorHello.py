import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
# 载入数据
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# 对数据进行归一化
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
# 建立模型
model=tf.keras.models.Sequential()#一个前向反馈模型
model.add(tf.keras.layers.Flatten())#input layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#测试模型
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss)
print(val_acc)

model.save('epic_num_reader.model')
new_model=tf.keras.models.load_model('epic_num_reader.model')
predictions=new_model.predict(x_test)
# print(predictions)

print(np.argmax(predictions[0]))
print(predictions[0])
print(y_test[0])









