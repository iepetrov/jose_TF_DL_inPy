import tensorflow as tf

print(tf.__version__)

hello = tf.constant('hello ')
world = tf.constant('world ')

print(hello)

with tf.Session() as sess:
    result = sess.run(hello+world)

print(result)


tensor_1 = tf.constant(1)
tensor_2 = tf.constant(2)

tensor_1 + tensor_2

with tf.Session() as sess:
    result = sess.run(tensor_1+tensor_2)

print(result)
# sess

# sess.close()

const = tf.constant(10)
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4),mean=0,stddev=0.5)
myrandu = tf.random_uniform((4,4),minval=0,maxval=1)

my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

sess = tf.InteractiveSession()

for op in my_ops:
    print(op.eval())
    print('\n')

# with tf.Session() as sess:

a = tf.constant([ [1,2],
                  [3,4] ])

a.get_shape()

b = tf.constant([[10],[100]])

b.get_shape()

result = tf.matmul(a,b)

result.eval()

# or run
sess.run(result)