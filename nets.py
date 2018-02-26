import tensorflow as tf 

def encoding3d(x,args):
    #    inp = x
#    x = tf.log1p(x)
    print('This is the network shape!')
    print(x.get_shape()) 
    x = tf.layers.conv3d(x, filters = 24, kernel_size = (7,7,7),
            strides = (1,1,1), padding = 'same') 
    x = tf.nn.relu(x)
    x = tf.nn.max_pool3d(x,ksize = [1,2,2,1,1], strides = [1,2,2,1,1], padding = 'SAME')
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dropout(x, args['dropout'], training=args['training'])
    y = x
    print(x.get_shape()) 
    x = tf.layers.conv3d(x, filters = 64, kernel_size = (5,5,5),
            strides = (1,1,1), padding = 'same') 
    x = tf.nn.relu(x)
    x = tf.nn.max_pool3d(x,ksize = [1,2,2,1,1], strides = [1,2,2,1,1], padding = 'SAME')
#    x = tf.layers.batch_normalization(x)
    x = tf.layers.dropout(x, args['dropout'], training=args['training'])

    # I added this! 
    x = tf.layers.conv3d(x, filters = 64, kernel_size = (5,5,5),
            strides = (1,1,1), padding = 'same') 
    x = tf.nn.relu(x)
#    x = tf.layers.batch_normalization(x)
 
    z = x
    print(x.get_shape()) 
    x = tf.layers.conv3d(x, filters = 128, kernel_size = (3,3,3),
            strides = (1,1,1), padding = 'same') 
    x = tf.nn.relu(x)
    x = tf.nn.max_pool3d(x,ksize = [1,2,2,1,1], strides = [1,2,2,1,1], padding = 'SAME')
#    x = tf.layers.batch_normalization(x)

    # I added this! 
    x = tf.layers.conv3d(x, filters = 128, kernel_size = (3,3,3),
            strides = (1,1,1), padding = 'same') 
    x = tf.nn.relu(x)
    print(x.get_shape()) 

    # smallest right now, now increase size
    x = tf.layers.conv3d_transpose(x, filters = 64, kernel_size = (3,3,3),
            strides = (2,2,1), padding = 'same', use_bias = False) 
    x = tf.nn.relu(x)

    # I added this! 
    x = tf.layers.conv3d_transpose(x, filters = 64, kernel_size = (3,3,3),
            strides = (1,1,1), padding = 'same', use_bias = False) 
    x = tf.nn.relu(x)

    x = x + z 
    print(x.get_shape()) 
    x = tf.layers.conv3d_transpose(x, filters = 24, kernel_size = (5,5,5),
            strides = (2,2,1), padding = 'same', use_bias = False) 
    x = tf.nn.relu(x)

    # I added this! 
    x = tf.layers.conv3d_transpose(x, filters = 24, kernel_size = (5,5,5),
            strides = (1,1,1), padding = 'same', use_bias = False) 
    x = tf.nn.relu(x)
    x = x + y 
    print(x.get_shape()) 
    x = tf.layers.conv3d_transpose(x, filters = 1, kernel_size = (7,7,7),
        strides = (2,2,1), padding = 'same', use_bias = False ) 
    print(x.get_shape()) 

    x = tf.reduce_sum(x,axis = 3)
    print(x.get_shape()) 

    # if you want output RF data:
#    x = tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_mean(x, axis=3)), axis=3, keep_dims=True))
    '''
    # condense channels
    x = tf.layers.conv3d(x, filters = 24, kernel_size = (1,1,8),
            strides = (1,1,1), padding = 'valid') 
    #    x = tf.nn.relu(x)
    print(x.get_shape()) 
    x = tf.layers.conv3d(x, filters = 24, kernel_size = (1,1,9),
            strides = (1,1,1), padding = 'valid') 
    #    x = tf.nn.relu(x)
    x = tf.squeeze(x, axis=3)
    '''

    # final deconvolution
     
#    x = tf.expm1(x)
    return x

