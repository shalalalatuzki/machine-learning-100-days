import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
#导入数据集
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values# -1 last column of data frame (id)
#iloc是pandas用索引读取数据的方法
y=dataset.iloc[:,3].values
#print(x)
#处理丢失数据
imputer =Imputer(missing_values="NaN",strategy="mean",axis=0)
#print(x)
imputer=imputer.fit(x[:,1:3]);
#print(x)
x[:,1:3]=imputer.transform(x[:,1:3]);
#print(x)
#解析分类数据 将分类数据(非数值数据)数值化 使用onehotencoder将数值映射到欧式空间
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
#print(x)
# 创造虚拟变量 将离散型特征进行one-hot编码，为了让距离计算更合理
onehotencoder=OneHotEncoder(categorical_features=[0]);
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
# print(x)
# print(y)
# 分离数据集为训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#random_state填0，每次随机数组不同，=1时每次随机数组相同
print(x_train)
#特征标准化
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
print(x_train)








