#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing the libraries
import numpy as n 
import pandas as p
import sklearn.datasets #scikitlearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score # used to determine how many correct predictions our model is making


# In[2]:


# Data collection and processing
cancer_dataset=sklearn.datasets.load_breast_cancer()


# In[3]:


# loading a data to panda data frame
data_frame=p.DataFrame(cancer_dataset.data,columns=cancer_dataset.feature_names) #feature_name is nothing just the name of columns


# In[6]:


# print the first 5 rows of the data_frame
data_frame.head()


# In[7]:


#adding the target column to the data_frame it will display 0 for malignant and 1 being bening
data_frame['label']= cancer_dataset.target


# In[9]:


#printing last 5 rows of the dataframe
data_frame.tail()


# In[11]:


#basic analysis and processing
#number of rows and columns in the dataset
data_frame.shape

#output will be (569,31)
# 569 being the number of rows and 31 being the number or rows


# In[12]:


#getting some information about the data
data_frame.info()


# In[13]:


#checking for missing values in dataset
data_frame.isnull().sum()

#it will tell you how many missing values are there in each column


# In[14]:


#statistical measures about the data
data_frame.describe()


# In[15]:


#checking the distribution of the target variable
data_frame['label'].value_counts()


# In[16]:


data_frame.groupby('label').mean()


# In[18]:


#seperating the features(columns except label) and target(label)
# droping a column axis value 1
# fropping a row axis value 0
x=data_frame.drop(columns='label',axis=1)
y=data_frame['label']


# In[19]:


print(x)


# In[20]:


print(y)


# In[28]:


#splitting the data into training data and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[29]:


print(x.shape,x_train.shape,x_test.shape)


# 

# In[30]:


#model training and using logistic regression where logistic regression helps in binary classification problem where we have just 2 classes
#here that is 0 and 1 (malignant.....)
model =LogisticRegression()


# In[31]:


#training the logistic regression model using the training data
model.fit(x_train,y_train)


# In[32]:


#model evaluation
#accuracy score
#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)


# In[33]:


print("Accuracy on training data= ",training_data_accuracy)
# will return 0.95 is 95% accurate


# In[34]:


#accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(y_test,x_test_prediction)


# In[37]:


print("Accuracy on test data= ",test_data_accuracy)
# returns 0.92 is 92% accurate


# In[42]:


#building a predictive system
input_data=(20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575,0.1374,0.205,0.4,0.1625,0.2364,0.07678)
#change the input data to a numpy array
input_data_as_num_arr=n.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_re=input_data_as_num_arr.reshape(1,-1)

predict=model.predict(input_data_re)
print(predict)
if predict[0]==0:
    print('The cancer is Malignant')
else:
    print('The cancer is Benign')
# prints 0 for malignant


# In[43]:


#TEST 2
#building a predictive system
input_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
#change the input data to a numpy array
input_data_as_num_arr=n.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_re=input_data_as_num_arr.reshape(1,-1)

predict=model.predict(input_data_re)
print(predict)
# prints 1 for bening
if predict[0]==0:
    print('The cancer is Malignant')
else:
    print('The cancer is Benign')


# In[44]:


#TEST 3
#building a predictive system
input_data=(13.05,19.31,82.61,527.2,0.0806,0.03789,0.000692,0.004167,0.1819,0.05501,0.404,1.214,2.595,32.96,0.007491,0.008593,0.000692,0.004167,0.0219,0.00299,14.23,22.25,90.24,624.1,0.1021,0.06191,0.001845,0.01111,0.2439,0.06289)
#change the input data to a numpy array
input_data_as_num_arr=n.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_re=input_data_as_num_arr.reshape(1,-1)

predict=model.predict(input_data_re)
print(predict)
# prints 1 for bening
if predict[0]==0:
    print('The cancer is Malignant')
else:
    print('The cancer is Benign')

