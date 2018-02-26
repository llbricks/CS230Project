from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math
import config
import scipy.io as sio
import re
import nets
import copy
import utils
from shutil import copyfile
import matplotlib.pyplot as plt


def main(args):

    # turn off tensorflow verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = utils.make_save_dirs(args)
    args['y_shape'] = (64,64,1)
    args['x_shape'] = (64,64,128,2)

    # Load data
    utils.save_model_settings(args)
    utils.make_tensorboard_summaries(args)

    loss_train = np.zeros(args['n_epochs']+1)
    loss_val = np.zeros(args['n_epochs']+1)
    cost_train = np.zeros(args['n_epochs']+1)

    # Launch the graph
    with tf.Session() as sess:

        X_train, y_train= utils.dataset_from_tfRecords(args['train_dataset'],args)
#        X_val, y_val= utils.dataset_from_tfRecords(args['val_dataset'],args)


        # tf Graph input
        is_training = tf.placeholder(tf.bool)
        which_dataset = tf.placeholder(tf.string)

        x = X_train 
        y = y_train

        # Construct model
        if args['arch'] == '3d':
            pred = nets.encoding3d(x,args)

        # Define loss and optimizer
        cost,loss = utils.calc_cost(args,y,pred)
        optimizer = tf.train.AdamOptimizer(learning_rate=args['lr']).minimize(cost)


        saver = tf.train.Saver(max_to_keep = 0 )

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        iters_per_epoch = np.floor(18000/args['batch_size'])
        iters_per_epoch = 100 
        iters_per_epoch_val = np.floor(2000/args['batch_size'])
        loss_epoch_train = []
        cost_epoch_train = []
        args['n_epochs'] *= 100
        for iteration in range(int(iters_per_epoch*args['n_epochs'])):

            loss_, cost_, _, = sess.run([loss, cost, optimizer], 
                    feed_dict={is_training: True,which_dataset: 'train'})
            loss_epoch_train.append(loss_)
            cost_epoch_train.append(cost_)

            if iteration%iters_per_epoch == 0:
                epoch = int(np.floor(iteration/iters_per_epoch)) + 1

                # store and show training loss
                loss_train[epoch] = sum(loss_epoch_train)/len(loss_epoch_train)
                cost_train[epoch] = sum(cost_epoch_train)/len(loss_epoch_train)
                print('Train: \tLoss = %6.6f\tCost = %6.6f' % (cost_train[epoch],
                    loss_train[epoch]))

                '''
                # store and show val loss
                loss_epoch_val = []
                for i in range(iter_per_epoch_val):
                    loss_, = sess.run(loss, feed_dict={is_training: True, which_dataset: 'val'})
                    loss_epoch_val.append(loss_)
                loss_val[epoch] = mean(loss_epoch_val)
                print('Val: \tLoss = %6.6f' % (loss_val[epoch]))
                '''

                loss_epoch_train = []
                cost_epoch_train = []
                utils.save_ckpt(saver,args,sess,epoch)

        # Save all of the model parameters as a .mat file!
        utils.save_network_mat(sess,args,{'loss_train': loss_train, 'cost_train': cost_train})

def get_batch(x,y,ridx,batch_num,args):

    n = x.shape[0]
    idx1 = batch_num * args['batch_size']
    idx2 = min(idx1 + args['batch_size'], n)
    batch_idx = ridx[idx1:idx2]
    x_batch = x[batch_idx, :]
    y_batch = y[batch_idx, :]

    return x_batch, y_batch

def load_batches(dataset_path, args):

    for i,filename in enumerate(os.listdir(dataset_path)[:1]):

        if '.mat' in filename:

            print(os.path.join(dataset_path,filename))
            X_, y_ = load_from_mat(os.path.join(dataset_path,filename), args['n_train'])

            if i == 0:
                X = X_
                y = y_

            else:
                X = np.concatenate((X,X_),axis=0)
                y = np.concatenate((y,y_),axis=0)

    return X,y

def load_from_mat(dataset_path, nimgs=0):
    import h5py
    if dataset_path != '':
        if nimgs == 0:
            img = np.array(h5py.File(dataset_path)['img'])
            ref = np.array(h5py.File(dataset_path)['ref'])
        else:
            img = np.array(h5py.File(dataset_path)['img'][:nimgs])
            ref = np.array(h5py.File(dataset_path)['ref'][:nimgs])
        szi = img.shape
        szr = ref.shape
#        print('(%d,%d,%d,%d,%d) tensor loaded.' % (szi[0], szi[1], szi[2], szi[3], szi[4]))
        # Normalize img, ref by their RMS
        img = img.reshape(szi[0], np.prod(szi[1:]))
        ref = ref.reshape(szr[0], np.prod(szr[1:]))
        img /= np.sqrt(np.mean(np.square(img),axis=1,keepdims=True))
        ref /= np.sqrt(np.mean(np.square(ref),axis=1,keepdims=True))
#        print(np.min(img))
#        print(np.max(img))
        img = img.reshape(szi)
        ref = ref.reshape(szr)
        img /= np.linalg.norm(img, ord='fro', axis=(1,2),keepdims=True)
        ref /= np.linalg.norm(ref, ord='fro', axis=(1,2),keepdims=True)
        return img, ref[:,:,:,0]
    else:
        return None


def make_clean_training(y_train):

    X_train = np.expand_dims(np.tile(y_train, (1,1,1,16)),axis = 4)
    X_train = np.concatenate((X_train,np.zeros_like(X_train)),axis = 4)

    return X_train, y_train 

if __name__ == '__main__':
    a = config.parser()
    main(a)


