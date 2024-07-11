## Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
import sys
import matlab.engine
from sklearn.preprocessing import StandardScaler,MinMaxScaler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## load models
sys.path.insert(0,"../stylegan2-ada-pytorch")
import dnnlib
import legacy
from models.VAE_bn_utils import *
from models.DCGAN_utils import *
from plot_result import *
##
class proxy_model():
    ## choose model type (DCGAN/StyleGAN/VAE/PCA), model path, case number (1/2/3)
    def __init__(self, model, model_path) -> None:
        self.model = model
        self.model_path = model_path
    ## load models
    def load_model(self, z_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.z_dim = z_dim
        if self.model == 'DCGAN':
            self.netG = Generator(1,1,z_dim,48).to(self.device)
            self.netG.load_state_dict(torch.load(self.model_path))
            
        elif self.model == 'VAE':
            self.netG = VAE(z_dim).to(self.device)
            self.netG.load_state_dict(torch.load(self.model_path))

        elif self.model == 'styleGAN' or self.model == 'styleGAN_w':
            with dnnlib.util.open_url(self.model_path) as fp:
                self.netG = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(self.device) # type: ignore
        elif self.model =='pixel':
            self.netG = None        
        else:
            print("Model not found")
            sys.exit()
            return 
        try:
            self.netG.eval()
        except:
            pass
    def generate(self, z):
        ## input z is numpy array, dz is the dim of latent space

        ## output img is numpy array, shape (n,64,64)
        z = torch.from_numpy(z).float().to(self.device)
        if self.model == 'DCGAN':
            img = self.netG(z.unsqueeze(2).unsqueeze(3))
            img = img.cpu().detach().numpy()
            img = img*0.5+0.5
            img = np.clip(img,0,1).squeeze()
            return img
        elif self.model == 'VAE':
            img = self.netG.decode(z)
            img = img.cpu().detach().numpy()
            img = np.clip(img,0,1).squeeze()
            return img
        elif self.model == 'styleGAN':
            img = self.netG(z, None, truncation_psi=1, noise_mode='const')
            img = img.cpu().detach().numpy()
            img = img*0.5+0.5
            # img = img.squeeze()
            # img = img*0.5+0.5
            img = np.clip(img,0,1).squeeze()
            return img
        elif self.model == 'pixel':
            img = z.cpu().detach().numpy()
            img = img.reshape(-1,64,64)
            img = np.clip(img,0,1).squeeze()
            return img
        else:
            print("Model not found")
            return
##
class ESMDA():  
    def __init__(self, out_path ,casenum,well_number, real_num = 100,iter_num=4) -> None:
        self.iter_num = iter_num
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.real_num = real_num
        self.casenum = casenum
        self.well_number = well_number
    ## forward simulation
    def mat2np(self,mat):
        mat = [np.array(item).T for item in mat]
        return mat

    def forward_sim(self, img):
        img = matlab.double(img.tolist())
        p_obs, p_map = self.eng.sim_forward(img,nargout=2)
        p_obs = np.array(p_obs, dtype ='float')
        p_map = np.array(p_map, dtype ='float')
        return p_obs,p_map    
    ## initialize simulations (run reference model)

    def normalize(self, data, data2):
        # output = (data/2e-5)
        output = data
        # output = (data - np.min(data2))/(np.max(data2)-np.min(data2))
        return output
    def init_sim(self):
        # save values
        self.eng = matlab.engine.start_matlab()
        self.z_list =[]
        self.p_obs_list =[]
        self.p_map_list = []
        self.image_list=[]
        p_true,obstrue,trueperm,perm_obs,loc = self.eng.init(self.casenum,self.out_path,self.well_number, nargout=5)
        [p_true,obstrue,trueperm,perm_obs,loc] = self.mat2np([p_true,obstrue,trueperm,perm_obs,loc])
        # record true values
        self.p_map_true = p_true
        self.p_obs_true = obstrue
        self.perm_map_true = trueperm
        self.perm_obs_true = perm_obs
        self.loc = (loc-1).astype(int)
        ###
        self.per_map_true_ = trueperm.reshape([64,64]).T
   
        # generate initial ensemble
        return 
    def init_esmda(self, model, model_path, z_dim,select_target = False):
        # generate initial ensemble (n*d_z)
        self.z_dim = z_dim
        self.model = proxy_model(model, model_path)
        self.model.load_model(z_dim)
        self.add_list = []
        self.error_list =[]
        np.random.seed(169)
        if self.model.model == 'VAE':
            Sigma = 1
        elif self.model.model == 'styleGAN':
            Sigma = 0.5
        else: 
            Sigma = 0.5

        if select_target:
            if self.model.model == 'VAE':
                z_samples_all =np.load('./Trained_models/VAE_init/z'+str(z_dim)+'_output.npy')
            elif self.model.model == 'styleGAN':
                z_samples_all = np.array([np.random.normal(0, 0.5, size=self.real_num*5) for i in range(self.z_dim)]).T # (n,d_z)
                # z_samples_all = np.array([np.random.uniform(-.5, .5, size=self.real_num*5) for i in range(self.z_dim)]).T # (n,d_z)
            elif self.model.model == 'pixel':
                z_samples_all = np.load('./Trained_models/img_mat.npy')[:,:]
            else:
                z_samples_all = np.array([np.random.normal(0, Sigma, size=self.real_num*5) for i in range(self.z_dim)]).T # (n,d_z)
            image_pred = self.model.generate(z_samples_all)
            samples = image_pred[:,self.loc[0,:],self.loc[1,:]]   
            distances = np.linalg.norm(samples - self.per_map_true_[self.loc[0,:],self.loc[1,:]], axis=1)
            # distances = np.linalg.norm(samples - self.perm_obs_true, axis=1)  
            sorted_indices = np.argsort(distances)
            closest_indices = sorted_indices[:self.real_num]  
            self.z_samples = z_samples_all[closest_indices]   
            self.img = image_pred[closest_indices]
        else:
            if self.model.model == 'VAE':
                z_samples_all =np.load('./Trained_models/VAE_init/z'+str(z_dim)+'_output.npy')
                self.z_samples = z_samples_all[:self.real_num]
            elif self.model.model == 'pixel':
                self.z_samples  = np.load('./Trained_models/img_mat.npy')[:self.real_num,:]

            else:    
                self.z_samples = np.array([np.random.normal(0, Sigma, size=self.real_num) for i in range(self.z_dim)]).T # (n,d_z)
            self.img = self.model.generate(self.z_samples)

        self.image_list.append(self.img.copy())
        self.z_list.append(self.z_samples.copy())  
        ## run the initial ensemble
        self.p, self.pmap = self.forward_sim(self.img)
        self.p_map_list.append(self.pmap)
        self.p_obs_list.append(self.p)  
        ## normalize
        # self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler(feature_range=(0,0.5))
        # self.scaler.fit(np.vstack([self.p,self.p_obs_true]))
        ## get normalized observation 
        self.p_obs_norm = self.normalize(self.p,self.p_obs_true)
        self.p_obs_true_norm = self.normalize(self.p_obs_true,self.p_obs_true)
        # self.p_obs_norm = self.scaler.transform(self.p)
        # self.p_obs_true_norm = self.scaler.transform(self.p_obs_true)
        self.obsmatrix = np.tile(self.p_obs_true_norm,(self.real_num,1))
        self.errmatrix = np.multiply(self.obsmatrix,0.005*np.ones_like(self.obsmatrix))
        self.errmatrix = np.multiply(self.errmatrix, np.random.normal(0,1,(self.obsmatrix.shape)))
        self.error_list.append(np.linalg.norm(self.obsmatrix+-self.p_obs_norm))
        return
    def run_sim(self,alpha):
        ## run the ensemble

        sim_matrix = self.p_obs_norm
        R = np.cov(self.errmatrix.T)
        CD = np.diag(R)
        CDD =np.cov(sim_matrix.T)
        DA = self.z_samples - np.mean(self.z_samples, axis=0)
        DF = sim_matrix - np.mean(sim_matrix, axis=0)
        CMD = np.divide(DA.T@DF,(self.real_num-1))
        d_new = self.obsmatrix + np.random.normal(0,1,(self.obsmatrix.shape))*(alpha*CD)**(0.5)
        add_ =(CMD@np.linalg.pinv(CDD+alpha*CD)@(d_new - sim_matrix).T).T
        self.add_list.append(add_)

        self.z_samples = self.z_samples + add_
        self.img = self.model.generate(self.z_samples) # run generator
        self.p, self.pmap = self.forward_sim(self.img) # run forward simulation
        self.p_obs_list.append(self.p)
        self.p_map_list.append(self.pmap)
        ## ------------------- update ensemble ------------------- ##
        # self.scaler.fit(np.vstack([self.p,self.p_obs_true]))
        # self.p_obs_norm = self.scaler.transform(self.p)
        self.p_obs_norm = self.normalize(self.p,self.p_obs_true)
        # self.p_obs_true_norm = self.scaler.transform(self.p_obs_true)
        # self.obsmatrix = np.tile(self.p_obs_true_norm,(self.real_num,1))
        ## ------------------- update ensemble ------------------- ##
        self.image_list.append(self.img.copy())
        self.z_list.append(self.z_samples.copy())
        self.error_list.append(np.linalg.norm(self.obsmatrix+-self.p_obs_norm))
        return 
    
    def esmda_ending(self):
        self.eng.quit()
        np.save(self.out_path + '/p_obs_list.npy',self.p_obs_list)
        np.save(self.out_path + '/p_map_list.npy',self.p_map_list)
        np.save(self.out_path + '/z_list.npy',self.z_list)
        np.save(self.out_path + '/image_list.npy',self.image_list)
        np.save(self.out_path + '/p_obs_true.npy',self.p_obs_true)
        np.save(self.out_path + '/p_map_true.npy',self.p_map_true)
        np.save(self.out_path + '/error_list.npy',self.error_list)
        np.save(self.out_path + '/well_loc.npy',self.loc)
        return
    def plot_result(self):
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        font = {'family': 'serif', 'size': 18}
        plt.rc('font', **font)
        plot_perm_mean(self.image_list, self.loc,self.out_path)
        plot_perm_std(self.image_list, self.loc,self.out_path)
        self.p_opt, self.p_ind = rank_performance_mean(self.p_obs_list[-1])
        plot_p_well(self.p_obs_list,self.p_obs_true[0], self.out_path,self.p_ind)
        plot_samples_paper(self.out_path,self.image_list,self.p_ind)
        plot_obj(self.error_list,self.out_path)
        plot_true_perm_map(self,self.out_path)
    def esmda_forward(self, params):
        # self.alpha_list = np.array([1/4]*4)
        self.alpha_list = np.array([9.33,7,4,2])
        self.init_sim()
        self.init_esmda(params['model'], params['model_path'], params['z_dim'], params['select_target'])
        for i in self.alpha_list:
            self.run_sim(i)
        self.esmda_ending()
        self.plot_result()
        print("ESMDA finished")
        return
    