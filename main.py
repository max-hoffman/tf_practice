# bin/bash python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# config
batch_size = 100
learning_rate = .5
training_epochs = 10
logs_path = "./log/"

with tf.name_scope('input'):
  # batch can be any size, but will have 784 pixels (flattened image)
  x = tf.placeholder(tf.float32, shape=[None, 784])
  # target/output: 10 classes
  y_ = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('weights'):
  # model parameters change, so we want tf.Variables
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

with tf.name_scope('softmax'):
  # prediction
  y = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('cross-entropy'):
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

with tf.name_scope('train'):
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


with tf.name_scope('Accuracy'):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# track cost and accuracy
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge summaries for convenience
summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

# start session to initialize tensorflow variables
with tf.Session() as sesh:
  sesh.run(tf.initialize_all_variables())
  # writer = tf.summary.FileWriter("./", sesh.graph)
  writer = tf.summary.FileWriter(logs_path , sesh.graph)

  for epoch in range(training_epochs):
    batch_count = int(mnist.train.num_examples/batch_size)
    for  i in range(batch_count):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      _, summary = sesh.run([train_op, summary_op], feed_dict={ x: batch_x, y_: batch_y })
      writer.add_summary(summary, epoch * batch_count + i)
    if epoch % 5 ==0:
      print("Epoch: ", epoch)
  
  print("Accuracy: ", accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels }))
  print("done")
