# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\FULL-TENSORFLOW-NOTES-AND-DATA (1)\Tensorflow-Bootcamp-master\02-TensorFlow-Basics'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# # TensorFlow Regression Example
#%% [markdown]
# ## Creating Data

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


#%%
# 1 Million Points
x_data = np.linspace(0.0,10.0,1000000)


#%%
noise = np.random.randn(len(x_data))


#%%
# y = mx + b + noise_levels
b = 5

y_true =  (0.5 * x_data ) + 5 + noise


#%%
my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)


#%%
my_data.head()


#%%
my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')

#%% [markdown]
# # TensorFlow
# ## Batch Size
# 
# We will take the data in batches (1,000,000 points is a lot to pass in at once)

#%%
import tensorflow as tf


#%%
# Random 10 points to grab
batch_size = 8

#%% [markdown]
# ** Variables **

#%%
m = tf.Variable(0.5)
b = tf.Variable(1.0)

#%% [markdown]
# ** Placeholders **

#%%
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

#%% [markdown]
# ** Graph **

#%%
y_model = m*xph + b

#%% [markdown]
# ** Loss Function **

#%%
error = tf.reduce_sum(tf.square(yph-y_model))

#%% [markdown]
# ** Optimizer **

#%%
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

#%% [markdown]
# ** Initialize Variables **

#%%
init = tf.global_variables_initializer()

#%% [markdown]
# ### Session

#%%
with tf.Session() as sess:
    
    sess.run(init)
    
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        
        sess.run(train,feed_dict=feed)
        
    model_m,model_b = sess.run([m,b])


#%%
model_m


#%%
model_b

#%% [markdown]
# ### Results

#%%
y_hat = x_data * model_m + model_b


#%%
my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')

#%% [markdown]
# ## tf.estimator API
# 
# Much simpler API for basic tasks like regression! We'll talk about more abstractions like TF-Slim later on.

#%%
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]


#%%
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

#%% [markdown]
# ### Train Test Split
# 
# We haven't actually performed a train test split yet! So let's do that on our data now and perform a more realistic version of a Regression Task

#%%
from sklearn.model_selection import train_test_split


#%%
x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)


#%%
print(x_train.shape)
print(y_train.shape)

print(x_eval.shape)
print(y_eval.shape)

#%% [markdown]
# ### Set up Estimator Inputs

#%%
# Can also do .pandas_input_fn
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)


#%%
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)


#%%
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False)

#%% [markdown]
# ### Train the Estimator

#%%
estimator.train(input_fn=input_func,steps=1000)

#%% [markdown]
# ### Evaluation

#%%
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)


#%%
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)


#%%
print("train metrics: {}".format(train_metrics))
print("eval metrics: {}".format(eval_metrics))

#%% [markdown]
# ### Predictions

#%%
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)


#%%
list(estimator.predict(input_fn=input_fn_predict))


#%%
predictions = []# np.array([])
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])


#%%
predictions


#%%
my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'r')

#%% [markdown]
# # Great Job!

