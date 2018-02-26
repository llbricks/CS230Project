from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import logging as LOG
import math
import time
import config
import scipy.io as sio
import scipy
import re
from shutil import copyfile
from testing import QualityMetrics as QM
#from testing import MakeImages as MI

'''
def infer(model_ckpt,input_file,pred_file):

    copyfile(os.path.join(model_dir,'arch.py'),'infer_net.py')

    imageMaker_noTGC = MI.InferWrapper()

    if infer_set == 'phantom1':
        acq_numbers = [1,2,3,4,5]
        
        Acq = QM.QualityMetrics()
        for acq_number in acq_numbers:
          imageMaker_noTGC(model_dir, acq_number)
          Acq(acq_number, imageMaker_noTGC.pred, imageMaker_noTGC.crop)

    if infer_set == 'invivo':
        imageMaker_noTGC()

    Acq.saveMetrics(tgc=False, save_path=model_dir)
'''

def main(args,X_test):

    # turn off tensorflow verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, X_test.shape[1], X_test.shape[2], X_test.shape[3], X_test.shape[4]])

    # Construct model
    import infer_net
    args['training'] = False
    pred = infer_net.encoding3d(x,args)

    # Evaluate model
    print('loading data...')

    # Create a Saver
    saver = tf.train.Saver()

    # Use batches to fit everything in GPU memory
    nbatches = (X_test.shape[0]-1) // args['batch_size']
    pred_ = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[2],1),dtype=np.float32)
    print(pred_.shape)

    # Force gpu to use only half of available GPU memory
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    tf.set_random_seed(1)
    np.random.seed(1)

    # Launch the graph
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    with tf.Session() as sess:
        saver.restore(sess, args['model_ckpt'])
        for b in range(nbatches+1):
            idx1 = b * args['batch_size']
            idx2 = min(idx1 + args['batch_size'], X_test.shape[0])
            pred__ = np.array(sess.run([pred], feed_dict={x:X_test[idx1:idx2]}))
            if pred__.shape[4]>1:
                pred_[idx1:idx2] = np.squeeze(np.sum(pred__,axis = 4),axis = 0)
            else: 
                pred_[idx1:idx2] = pred__

        pred_ = pred_.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]) # Get rid of innermost dimension
#        sio.savemat(args.pred_file, {'pred':pred_})
    return np.squeeze(pred_)

def scan_convert(pred):
    interp_data = h5py.File('/data/dhyun/data/fromDuke/Verasonics/P4-2/data/20110525/focdata/jjd_liver1.mat')
    thetai = interp_data['thetai']
    depthi = interp_data['depthi']
    scmask = interp_data['scmask']
    img = scipy.interpolate.interp2d(thetai,depthi,pred)
    return img

'''
def load_from_mat3d(dataset_path):
    import h5py
    if dataset_path != '' :
        # img = np.array(h5py.File(dataset_path)['img'])
        img = sio.loadmat(dataset_path)['img']
        sz = img.shape

        if img.ndim == 3:
            img = img.reshape(1, 1, sz[0], sz[1], sz[2])
        if img.ndim == 4:
            img = img.reshape(1, sz[0],sz[1],sz[2],sz[3])
        sz = img.shape
        print('(%d,%d,%d,%d,%d) tensor loaded.' % (sz[0],sz[1],sz[2],sz[3],sz[4]))
        # Normalize img by its RMS
        img = img.reshape(sz[0], np.prod(sz[1:]))
        img /= np.sqrt(np.mean(np.square(img),axis=1,keepdims=True))
        img = img.reshape(sz) * .8
        # img /= np.sqrt(np.mean(np.square(img)))
        # img /= np.linalg.norm(img, ord='fro', axis=(1,2),keepdims=True)
        # img = img.reshape(sz[0], np.prod(sz[1:]))
        # img /= np.linalg.norm(img, axis=1,keepdims=True)
        # img = img.reshape(sz)
        return img
    else:
        return None
'''

if __name__ == '__main__':
    a = config.parser()
    main(a)
