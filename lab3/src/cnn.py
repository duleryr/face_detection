import tensorflow as tf
import numpy as np

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name="W")


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=[shape])
  return tf.Variable(initial, name="B")

def create_convolutional_layer(input_tensor, num_input_channels, 
        conv_filter_size, num_filters, name):  
    
    with tf.name_scope(name):
        weights = weight_variable(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        bias = bias_variable(num_filters)
        
        ## Creating the convolutional layer
        layer = tf.nn.conv2d(input=input_tensor,
                filter=weights,
                strides=[1, 1, 1, 1],
                padding='SAME')
        
        layer += bias
        
        ## We shall be using max-pooling.  
        layer = tf.nn.max_pool(value=layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')
        ## Output of pooling is fed to Relu which is the activation function for us.
        layer = tf.nn.relu(layer)
        return layer

def create_flatten_layer(layer, name):

    with tf.name_scope(name):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])
        return layer

def create_fc_layer(input_tensor, num_inputs,num_outputs,
        use_relu, name):

    with tf.name_scope(name):
        #Let's define trainable weights and biases.
        weights = weight_variable(shape=[num_inputs, num_outputs])
        biases = bias_variable(num_outputs)
        
        layer = tf.matmul(input_tensor, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        
        return layer

# N is the roi size
def construct_cnn(N):

    # Define the CNN by its parameters
    num_channels = 3
    filter_size_conv1 = 3
    num_filters_conv1 = 32
    filter_size_conv2 = 3
    num_filters_conv2 = 32
    filter_size_conv3 = 3
    num_filters_conv3 = 64
    fc_layer_size = 128
    num_classes = 2

    # Input variables
    x_hold = tf.placeholder(tf.float32,shape=[None,N,num_channels], name="x")
    y_true = tf.placeholder(tf.float32,shape=[None,2], name="labels")

    xt = tf.reshape(x_hold,[-1,N,N,num_channels])
    tf.summary.image("input_batch", xt)

    # Define CNN
    layer_conv1 = create_convolutional_layer(input_tensor=xt,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1,
        name="Conv1")
    
    layer_conv2 = create_convolutional_layer(input_tensor=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2,
        name="Conv2")
    
    layer_conv3= create_convolutional_layer(input_tensor=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3,
        name="Conv3")

    layer_flat = create_flatten_layer(layer_conv3,"Flat1")

    layer_fc1 = create_fc_layer(input_tensor=layer_flat,
        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
        num_outputs=fc_layer_size,
        use_relu=True, name="Fc1")
    
    layer_fc2 = create_fc_layer(input_tensor=layer_fc1,
        num_inputs=fc_layer_size,
        num_outputs=num_classes,
        use_relu=False,name="Fc2")

    # Define Post-processing
    y_pred = tf.nn.softmax(layer_fc2,name="y_pred")
    
    with tf.name_scope("accuracy"):
        y_pred_cls = tf.argmax(y_pred,dimension=1)
        y_true_cls = tf.argmax(y_true,dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)

    with tf.name_scope("train"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                            labels=y_true)
        # since we are dealing with mutually exclusive labels, we could try
        # another cost function
        # After trial: doesn't work, minimizes the cost function, but reduces the accuracy
        #y_true_exclusive = tf.cast(y_true[:,0],dtype=tf.int32)
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2,
        #                                                    labels=y_true_exclusive)
        #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=layer_fc2,
        #                                                    targets=y_true,pos_weight=2)
        cost = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("cost",cost)
        # OPTIONAL: decayed learning rate
        starter_learning_rate = 1e-4
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10, 0.96, staircase=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
        optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(cost,global_step=global_step)
    
    merged = tf.summary.merge_all()
    return y_pred, y_true, x_hold, optimizer, accuracy, merged, cost, learning_rate
