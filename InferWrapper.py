import testing.QualityMetrics as QM
import numpy as np
import tensorflow as tf
from shutil import copyfile
import scipy.io as sio
import scipy.misc
import os
import re
import infer
import config
import h5py
import utils
import matplotlib.pyplot as plt

class InferWrapper():

    def __init__(self, model_dir, epoch = None):

        # copy the architecture file over for inference
        copyfile(os.path.join(model_dir,'arch.py'),'infer_net.py')
        self.model_path = model_dir
        self.get_ckpt_file(epoch)

    def __call__(self, infer_set):
        ''' infer_set should be name of the dataset to infer on 
        options = 'phantom1' and 'invivo'
        '''

        Acq = QM.QualityMetrics()
        self.infer_set = infer_set
        if self.ckpt_file is not None:
            if infer_set == 'phantom1':
                acq_numbers = [1,2,3,4,5]
                
                self.crop = [24,312,None,None] 
                for acq_number in acq_numbers:
                    
                    self.infer_path = '/data/llbricks/datasets/20170926_cirs_targets/processed/Acq0{}.mat'.format(acq_number)
                    self.save_name = '{}_{}'.format(infer_set,acq_number)
                    self.infer()
                    self.save_pred()
                    tf.reset_default_graph()
                    print('starting first metrics calc ------------------')
                    Acq(self.save_name, self.pred, self.crop)
                self.save_metrics(Acq)

            if infer_set == 'invivo':
                self.crop = [24,None,None,None] 
                self.save_name = infer_set
                self.infer_path = '/data/dhyun/data/fromDuke/Verasonics/P4-2/data/20110525/focdata/jjd_liver1_16ch.mat'
                self.infer()
                self.scan_convert('/data/dhyun/data/fromDuke/Verasonics/P4-2/data/20110525/focdata/jjd_liver1.mat')
                self.save_pred()
                tf.reset_default_graph()

            if infer_set == 'invivo2':
                acq_numbers = [1,2,3,4]
                self.crop = [24,200,None,None]
                for acq_number in acq_numbers:
                    self.infer_path = '/data/llbricks/datasets/121217_L12-3V_invivo/Reprocessed/processed/Acq0{}.mat'.format(acq_number)
                    print(self.infer_path)
                    self.save_name = '{}_{}'.format(infer_set,acq_number)
                    self.infer()
                    self.save_pred()
                    tf.reset_default_graph()

    def change_epoch(self, epoch = None):
        ''' changes epoch of the model in model_dir 
        if no epoch provided, max number is chosen '''
        self.get_ckpt_file(epoch)

    def infer(self):

        self.preprocess_data()
        print('IT SHOULD LOAD THE ARGS FROM THE CKPT')
        try:
            args = np.load(os.path.join(self.model_path,'args.npy')).item()
        except:
            args = config.parser()

        print(self.ckpt_file)
        args['model_ckpt'] = self.ckpt_file
        self.pred = infer.main(args,self.X_test)
        # Normalize
        #self.pred = self.pred/np.median(self.pred.flatten())

    def preprocess_data(self):
        ''' makes self.X_test based on what's in infer_path'''

        # set multiplier for number of dimension reductions in the network
        reshape_multiplier = 8 

        # Load fout from the aperture growth-corrected dataset
        try:
            self.img = sio.loadmat(self.infer_path)['fout']
        except:
            read = h5py.File(self.infer_path,'r')
            self.img = np.transpose(read['fout'].value.view(np.complex64),(2,1,0))

        # Apply crop, if specified
        self.img = utils.apply_crop(self.img,self.crop)

        # Make bmode image and store as class instance
        bimg = np.abs(np.mean(self.img, 2))
        bimg = bimg/np.max(bimg.flatten())
        self.bimg = bimg
#        plt.imshow(np.log(self.bimg), cmap='gray')
#        plt.show()
        
        # Crop image to dimensions that are multiples of reshape_multiplier
        n_shp = np.shape(self.img)
        self.img = self.img[0:reshape_multiplier*(n_shp[0]//reshape_multiplier),  
                0:reshape_multiplier*(n_shp[1]//reshape_multiplier)]
        
        # Reshape based on number of channels
        print('Warning: It might be reading the channels wrong')
        # This should be read from args cause this naming convention has been changed
        try:
            num_channels = int(re.search('_[0-9]{1,3}ch', self.model_path).group(0)[1:-2])
        except:
            num_channels = int(np.load(os.path.join(self.model_path,'args.npy')).item()['x_shape'][-2])
        self.img = np.reshape(self.img, (np.shape(self.img)[0], np.shape(self.img)[1], 
                            int(np.shape(self.img)[2]/num_channels), num_channels))
        self.img = np.sum(self.img, 2)
        print('num channels',self.img.shape)

        # convert complex data to another dimension of real and imaginary
        self.img = np.expand_dims(self.img,axis = 3)
        self.img = np.concatenate((np.real(self.img), np.imag(self.img)), axis=3)

        # Normalize img by its RMS
        print('right now we normalize image by its RMS')
        print('shouldnt this rms be determined by the network/training?')
        self.img /= np.sqrt(np.mean(np.square(self.img)))

        # expand dimension on axis = 0 for batch size
        self.X_test = np.expand_dims(self.img,axis = 0)

        print('shape of self.X_test: {}'.format(self.X_test.shape))

    def save_pred(self):

        # make filename
        save_filename= os.path.join(self.model_path,
                '{}_epoch{}'.format(self.save_name,self.epoch))

        # save as .mat
        sio.savemat(save_filename+'.mat', {'pred':self.pred})

        # Postprocess for png 
        dynamic_range = [-25, 15]
        self.pred_bmode = np.clip(20*np.log10(np.clip(self.pred,1E-10,1E15)), dynamic_range[0], dynamic_range[1])

        # save as .png
        scipy.misc.imsave(save_filename+'.png',self.pred_bmode)

    def save_metrics(self,Acq):

        # make filename
        save_filename= os.path.join(self.model_path,
                '{}_metrics_epoch{}.json'.format(self.save_name,self.epoch))
        # 
        Acq.saveMetrics(tgc=False, save_path=save_filename)

    def scan_convert(self,convert_data_path):
        fout = np.swapaxes(np.array(h5py.File(self.infer_path)['fout']),0,2).astype(np.complex64)
        interp_data = h5py.File(convert_data_path)

        thetai = np.array(interp_data['thetai']).astype(np.float).T
        depthi = np.array(interp_data['depthi']).astype(np.float).T
        scmask = np.array(interp_data['scmask']).T
        scmask[np.isnan(scmask)] = 0
        self.pred = self.pred.T

        shp = self.pred.shape
        interp = scipy.interpolate.RectBivariateSpline(range(shp[0]),range(shp[1]),self.pred)
        scan_converted = np.zeros_like(thetai)
        for x in range(scan_converted.shape[0]):
            for y in range(scan_converted.shape[1]):
                if scmask[x,y]:
                    scan_converted[x,y] = interp(thetai[x,y], depthi[x,y])
#        scan_converted = np.multiply(scan_converted,scmask)
        self.pred = scan_converted

    def get_ckpt_file(self, epoch):

        # if epoch not defined, take highest numbered epoch
        if epoch is None: 
            # if model.ckpt exists, use it
            if os.path.isdir(os.path.join(self.model_path,'model.ckpt')):
                self.ckpt_file = os.path.join(self.model_path,'model.ckpt')
                self.epoch = 'max'

            # otherwise use the highest model number
            else: 
                ckpt_list = []
                for f in os.listdir(self.model_path): 
                    if os.path.isfile(os.path.join(self.model_path, f)):
                        if ('model_' in f) or ('epoch_' in f):
                            try:
                                ckpt_list.append(int(re.search('[0-9]{1,3}',f).group(0)))
                            except:
                                pass
                if len(ckpt_list) < 1:
                    print('Theres no checkpoints in this model!')
                    self.ckpt_file = None
                else:
                    self.epoch = str(max(ckpt_list))
#                        ckpt_file = 'model_{0}.ckpt'.format(max(ckpt_list))
#                        self.ckpt_file = os.path.join(self.model_path,ckpt_file)
                    ckpt_file = 'epoch_{0}.ckpt'.format(max(ckpt_list))
                    self.ckpt_file = os.path.join(self.model_path,ckpt_file)

        # if epoch given, use closest model to that #
        else:
            self.ckpt_file = os.path.join(self.model_path,'model_{}.ckpt'.format(epoch))
            print('Were still not checking if the model of that epoch exists!!')
            self.epoch = str(epoch)



