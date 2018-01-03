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

    # Depth, i.e. number of channels
    D = 3
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
    x_hold = tf.placeholder(tf.float32,shape=[None,N,D], name="x")
    #y_hold = tf.placeholder(tf.float32,shape=[None,2], name="labels")
    y_true = tf.placeholder(tf.float32,shape=[None,2], name="labels")
    #keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    xt = tf.reshape(x_hold,[-1,N,N,D])
    #xt, y_hold = tf.train.shuffle_batch([xt,y_hold], batch_size=100,capacity=20,min_after_dequeue=10, enqueue_many=True)
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
    
    with tf.name_scope("train"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                            labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    with tf.name_scope("accuracy"):
        y_pred_cls = tf.argmax(y_pred,dimension=1)
        y_true_cls = tf.argmax(y_true,dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ##Layer 1 - 5x5 convolution
    # length(tensor_output) = 4*length(tensor_input)
    #with tf.name_scope("Conv1"):
    #    w1 = weight_variable([5,5,D,4])
    #    b1 = bias_variable([4])
    #    c1 = tf.nn.relu(tf.nn.conv2d(xt,w1,strides=[1,1,1,1],padding='SAME')+b1)
    #    o1 = c1

    ##Layer 2 - 3x3 convolution
    #with tf.name_scope("Conv2"):
    #    w2 = weight_variable([3,3,4,16])
    #    b2 = bias_variable([16])
    #    c2 = tf.nn.relu(tf.nn.conv2d(o1,w2,strides=[1,1,1,1],padding='SAME')+b2)
    #    o2 = c2

    ##Layer 3 - 3x3 convolution
    #with tf.name_scope("Conv3"):
    #    w3 = weight_variable([3,3,16,32])
    #    b3 = bias_variable([32])
    #    c3 = tf.nn.relu(tf.nn.conv2d(o2,w3,strides=[1,1,1,1],padding='SAME')+b3)
    #    o3 = c3

    ## dim = input_dim * 1st-layer * 2nd-layer * 3rd-layer
    #dim = N*N*4*4*2
    #    
    ##Fully connected layer - 600 units
    #with tf.name_scope("Fc1"):
    #    of = tf.reshape(o3,[-1,dim])
    #    w4 = weight_variable([dim,600])
    #    b4 = bias_variable([600])
    #    o4 = tf.nn.relu(tf.matmul(of,w4)+b4)

    #with tf.name_scope("Dropout"):
    #    o4 = tf.nn.dropout(o4, keep_prob)

    ##Output softmax layer - 2 units
    #with tf.name_scope("Softmax"):
    #    w5 = weight_variable([600,2])
    #    b5 = bias_variable([2])
    #    y = tf.nn.softmax(tf.matmul(o4,w5)+b5, name="final_result")

    tf.Session().run(tf.initialize_all_variables())

    merged = tf.summary.merge_all()
    #return y,x_hold,y_hold,keep_prob,merged
    return y_pred, y_true, x_hold, optimizer, accuracy, merged
