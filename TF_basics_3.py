# import tensorflow as tf

# print(tf.__version__)

# hello = tf.constant('hello ')
# world = tf.constant('world ')

# print(hello)

# with tf.Session() as sess:
#     result = sess.run(hello+world)

# print(result)


# tensor_1 = tf.constant(1)
# tensor_2 = tf.constant(2)

# tensor_1 + tensor_2

# with tf.Session() as sess:
#     result = sess.run(tensor_1+tensor_2)

# print(result)
# # sess

# # sess.close()

# const = tf.constant(10)
# fill_mat = tf.fill((4,4),10)
# myzeros = tf.zeros((4,4))
# myones = tf.ones((4,4))
# myrandn = tf.random_normal((4,4),mean=0,stddev=0.5)
# myrandu = tf.random_uniform((4,4),minval=0,maxval=1)

# my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

# sess = tf.InteractiveSession()

# for op in my_ops:
#     print(op.eval())
#     print('\n')

# # with tf.Session() as sess:

# a = tf.constant([ [1,2],
#                   [3,4] ])

# a.get_shape()

# b = tf.constant([[10],[100]])

# b.get_shape()

# result = tf.matmul(a,b)

# result.eval()

# # or run
# sess.run(result)


# n1 = tf.constant(1)
# n2 = tf.constant(2)

# n3 = n1 + n2

# with tf.Session() as sess:
#     result = sess.run(n3)
# print(result)


# # When you start TF, a default Graph is created, you can create additional graphs easily:
# print(tf.get_default_graph())

# g = tf.Graph()
# print(g)

# graph_one = tf.get_default_graph()
# graph_two = tf.Graph()

# graph_one is tf.get_default_graph()
# graph_two is tf.get_default_graph()

# with graph_two.as_default():
#     print(graph_two is tf.get_default_graph())

# graph_two is tf.get_default_graph()

# ######## Ch. 28
# sess = tf.InteractiveSession()
# my_tensor = tf.random_uniform((4,4),0,1)
# my_var = tf.Variable(initial_value=my_tensor)
# # print(my_var)
# # #### Note! You must initialize all global variables!
# init = tf.global_variables_initializer()
# init.run()
# sess.run(my_var)

import numpy as np
import tensorflow as tf

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

import matplotlib.pyplot as plt
plt.plot(x_data,y_label,'*')

m = tf.Variable(0.39)
b = tf.Variable(0.2)

error = 0

for x,y in zip(x_data,y_label):
    
    y_hat = m*x + b  #Our predicted value
    
    error += (y-y_hat)**2 # The cost we want to minimize (we'll need to use an optimization function for the minimization!)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 1
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])

final_slope
final_intercept

x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(x_data,y_label,'*')