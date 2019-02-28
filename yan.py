import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

import matplotlib.pyplot as plt
trainset = pd.read_csv('train1.csv',';')
testset = pd.read_csv('test1.csv',';')

#Подготовка данных
X = trainset.iloc[:, 0:100].values 
y = trainset.iloc[:, 100].values 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_val = testset.iloc[:, 0:100].values
X_val = sc.transform(X_val)

'''
Стандартизация формульно
Среднее значение = 0, дисперсия =1. Вычитаем среднее значение и делим
на стандартное отклонение
mean = X_train.mean(axis = 0) axis = 0 - расчитываем сред знач и станд откл
                                        не по всему тензору а для каждого признака
std = X_train.std(axis = 0)
X_train -= mean
X_train /= std
X_test -=mean
X_test /=std

мне кажется что формульно надо искать к каждому, а не вычитать
среднее трейна из теста, и делить аналогично. это как-то странно
но лучше всего нормализовать сначала весь датасет а потом делить

mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)
mean1 = X_test.std(axis = 0)
std1 = X_test.std(axis = 0)
X_train -= mean
X_train /= std
X_test -= mean1
X_test /=std1

#Стандартизация при помощи библиотеки. (Значения получились одинаковыми)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)
'''
#Создание сети
from keras.layers import Dropout
model = Sequential()
model.add(Dense(units = 128, activation = 'relu', input_dim = (X_train.shape[1])))
model.add(Dropout(rate = 0.1))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
#model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
''' 
mae - mean absolute error - средняя абсолютная ошибка
mse - mean squared error - средняя квадратичная ошибка '''

model.fit(X_train, y_train, epochs = 100, batch_size = 10)
mse, mae = model.evaluate(X_test, y_test)
pred = model.predict(X_val)
pred1 = model.predict(X_test)
print(pred1[1][0],y_test[1])
print(pred1[1034][0],y_test[1034])


 ''' 
f = open('answer.tsv', 'r')
text = f.read()
print(text)
f.close
'''
with open('answer.tsv','w') as out:
    for i in pred:                                                                                      
        print(str(i).replace('[','').replace(']',''), file = out)             
f.close     

with open('ytest.tsv','w') as out:
    for i in y_test:                                                                                      
        print(str(i).replace('[','').replace(']',''), file = out)     