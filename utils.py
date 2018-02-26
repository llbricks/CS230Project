import numpy as np
import scipy.misc
import os
import time
import tensorflow as tf
import scipy.io as sio
from shutil import copyfile
import matplotlib.pyplot as plt

def load_dataset(args, verbose = False ):

    X_train, y_train = load_from_mat(args['train_dataset'])
    X_val, y_val = load_from_mat(args['val_dataset'])

    if verbose:
        print('shape of X_train', X_train.shape)
        print('shape of y_train', y_train.shape)
        print('shape of X_val', X_val.shape)
        print('shape of y_val', y_val.shape)

    return X_train ,y_train, X_val, y_val
    
def create_dataset(args):

    # load the data in numpy
    X_train ,y_train, X_val, y_val = load_dataset(args, verbose = True)

#    X_train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
#    y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train)
#    train_dataset = tf.data.Dataset.zip((X_train_dataset,Y_train_dataset))
    handle = tf.placeholder(tf.string, shape=[])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val,y_val))

    if args['add_clean']:
        #X_train2, y_train2 = make_clean_training(y_train)
        #X_train = np.concatenate((X_train,X_train2),axis = 0)
        #y_train = np.concatenate((y_train,y_train2),axis = 0)
        pass

    iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
    X, Y = iterator.get_next()

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_initializable_iterator()

    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())

    #    train_iter = train_dataset.make
    #    X_train, Y_train = train_iter.get_next() 
    #    val_iter = val_dataset.make
    #    X_val, Y_val = val_iter.get_next() 

    return X, Y, train_handle, val_handle

def calc_cost(args,y,pred):

    num_channels = pred.shape[3] 

    # calculate loss based on args['log_loss'] and number of channels
    if num_channels>1:
        y = tf.expand_dims(y,axis=4)
        print('Using channel loss')
        if args['log_loss']:
            logy = tf.log(tf.clip_by_value(y,1E-15,1E15))
            logpred = tf.log(tf.clip_by_value(pred,1E-15,1E15))
            loss = tf.losses.mean_squared_error(logy, logpred)
        else:
            loss = tf.losses.mean_squared_error(y, pred)

    else:
        print('Using bmode loss') 
        if args['log_loss']:
            loss = tf.losses.mean_squared_error(tf.log(y), tf.log(pred))
        else:
            loss = tf.losses.mean_squared_error(y,pred)

    tvloss = tf.reduce_sum(tf.image.total_variation(pred))

    # Define loss and optimizer
    kernel_list = [v for v in tf.trainable_variables() if 'kernel' in v.name]
    l1_loss, l2_loss = 0, 0
    for v in kernel_list:
        l1_loss += tf.reduce_mean(tf.abs(v))
        l2_loss += tf.nn.l2_loss(v)

    cost = loss + args['reg1'] * l1_loss + args['reg2'] * l2_loss

    return cost,loss 


def make_save_dirs(args):

    # make folder for this experiment
    if args['run_name'] == '':
        args['run_name'] = time.strftime("%Y%m%d")
    else:
        args['run_name'] = time.strftime("%Y%m%d") + '_' + args['run_name'] 

    # check if this folder exists, if not make it 
    # also make the model name based on the model folders in that directory
    experiment_path = os.path.join(args['save_dir'],args['run_name'])
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        args['model_name'] = 'model_0'
    else:
        models_list = os.listdir(experiment_path)
        nums = []
        for model in models_list:
            if ('model_' in model ) and (os.path.isdir(os.path.join(experiment_path,model))):
                nums.append(int(model[6:]))
        args['model_name'] = 'model_{}'.format(max(nums)+1)
    args['model_path'] = os.path.join(experiment_path,args['model_name'])
    os.makedirs(args['model_path'])
    print(args['model_path'])

    return args

def make_tensorboard_summaries(args):
    # Add hyperparameters to Tensorboard summary
   with tf.name_scope('hyperparameters'):
        tf.summary.scalar('learning_rate', args['lr'])
        tf.summary.scalar('reg1', args['reg1'])
        tf.summary.scalar('reg2', args['reg2'])
        tf.summary.scalar('batch_size', args['batch_size'])
        tf.summary.scalar('dropout', args['dropout'])
   '''
   with tf.name_scope('loss'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2_err', msqerr)
        tf.summary.scalar('l1_loss', l1_loss)
        tf.summary.scalar('l2_loss', l2_loss)
   ''' 

def save_ckpt(saver,args,sess,epoch):

    if epoch == args['n_epochs']:
        save_path = saver.save(sess, args['model_path'],'epoch_{}.ckpt'.format(epoch))
        print("Model saved in file: %s" % save_path)
    elif (epoch) % args['tb_interval'] == 0:
        save_path = saver.save(sess, os.path.join(args['model_path'],'epoch_{}.ckpt'.format(epoch)))
        print("Model saved in file: %s" % save_path)

   

def save_model_settings(args):

    copyfile('nets.py', os.path.join(args['model_path'],'arch.py'))
    np.save(os.path.join(args['model_path'],'args'),args)
 
def save_network_mat(sess,args,mpdict):
    '''Takes input dict of parameters and saves these values , along with all kernals and biases of the network
    these parameters are saved in a .mat file in the run's folder under modelparams.mat
    ''' 
    param_list = [v for v in tf.trainable_variables() if 'kernel' in v.name or 'bias' in v.name]
    for p in param_list:
        name = re.sub('/', '_', p.name)
        name = re.sub(':0', '', name)
        value = sess.run(p)
        mpdict.update({name: value})
    sio.savemat(os.path.join(args['model_dir'],'modelparams.mat'), mpdict)

def load_from_mat(dataset_path):
    import h5py
    if dataset_path != '':

        try: 
            data = sio.loadmat(dataset_path)
        except:
            data = h5py.File(dataset_path,'r')
#            img = np.array(data['img'])
#            ref = np.array(data['ref'])
#            img = np.transpose(data['img'].value.view(np.complex64),(0,2,3,1))
            ref = np.expand_dims(np.squeeze(np.array(data['ref'])),3)
            img = np.squeeze(data['img'].value.view(np.complex64))

        return img,ref

def preprocess_data(data):

    '''
    szi = img.shape
    szr = ref.shape
    img = img.reshape(szi[0], np.prod(szi[1:]))
    ref = ref.reshape(szr[0], np.prod(szr[1:]))

    # Normalize img, ref by their RMS
#        img = img.reshape((img.shape[0], -1))
#        ref = ref.reshape((ref.shape[0], -1))
    img /= np.sqrt(np.mean(np.square(abs(img)),axis=1,keepdims=True))
    ref /= np.sqrt(np.mean(np.square(ref),axis=1,keepdims=True))
#        print(img)
    print('---------------')

    img = img.reshape(szi)
    ref = ref.reshape(szr)

    img /= np.linalg.norm(img, ord='fro', axis=(1,2),keepdims=True)
    ref /= np.linalg.norm(ref, ord='fro', axis=(1,2),keepdims=True)

    print('mean:',np.mean(img.flatten()))
    for dim in range(1,len(img.shape)):
        img_flat = np.mean(np.real(img),axis = dim).flatten()
        print('var:',np.var(img_flat))
        plt.hist(img_flat)
        plt.show()

    '''

    # Mean subtraction
    mean = np.mean(data, axis = 0)
    data -= mean
    # Normalization
    std = np.std(data, axis = 0)
    data /= std

    return data, mean, std

def dataset_from_tfRecords(path,args):

    feature = {'Y': tf.FixedLenFeature([],tf.string),'X': tf.FixedLenFeature([],tf.string)}

    filename_queue = tf.train.string_input_producer([path], num_epochs = args['n_epochs'])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # decoder
    features = tf.parse_single_example(serialized_example, features = feature)
    Y_ = tf.decode_raw(features['Y'], tf.float32)
    Y_ = tf.reshape(Y_, [64,64,1])
    X_ = tf.decode_raw(features['X'], tf.float32)
    X_ = tf.reshape(X_, [64,64,128,2])

    _X, _Y = tf.train.shuffle_batch([X_, Y_], batch_size = args['batch_size'], capacity = 30, 
            num_threads = 1, min_after_dequeue = 10)

    return _X,_Y

def apply_crop(img,crop):
    # crop is a list [starty,endy,startx,endx] none if no crop
    if crop[1] is not None: 
        img = img[0:crop[1]]
    if crop[0] is not None:
        img = img[crop[0]:]
    if crop[3] is not None:
        img = img[:,0:crop[3]]
    if crop[2] is not None:
        img = img[:,crop[2]:]
    return img

