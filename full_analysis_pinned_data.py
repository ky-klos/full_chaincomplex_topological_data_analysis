import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import random
#import matplotlib.image as mpimg
#from matplotlib.legend_handler import HandlerLine2D
import math
#from scipy.optimize import curve_fit
from numpy import pi
import colorcet as cc
from matplotlib.cm import get_cmap
from colorsys import hls_to_rgb
import matplotlib.colors

#from skimage.io import imread
#import matplotlib.pyplot as plt
#%matplotlib inline ~ magic function backend for IPython : output of plotting
#commands displayed inline within frontends directly below code, that it produced

#from sklearn.model_selection import train_test_split

#from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#import random

import h5py

#import pickle
import pandas as pd

#from matplotlib.collections import LineCollection

#import seaborn as sb

#import plotly.graph_objects as go

from observables_analysis_class_nov_24 import Observables
from filtration_function import TopologicalAnalysis

#import time

import os

#from observables_calculation_nov_24 import UnetGenerator

#from observables_calculation_nov_24 import ResidualBlock

#from observables_calculation_nov_24 import AdaIN


from observables_calculation_nov_24 import calculation_observables,output_analysis_basis

from filtration_function_output_data_faster_july_24 import get_edge_state_graphs_sim_gen

from topological_analysis_just_measures_faster_october_24 import get_topological_measures_mean

from topological_analysis_just_persisten_homology_october_24 import get_persistent_homology

##from training_september_25_fixed_sum_fourier import CircularUpscaleConv2d, ResidualBlock, AdaIN, UnetGenerator, CircularPad2d, SumPool2d, InceptionConv2d_mini




generator_input_dict = [{'downsample':'avg', 'upsample':'conv_transposed', 'depth':0, 'activation_function': ['leaky_Relu',0.01], 'max_feature':4

}]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)

#['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'avg', 'conv_transposed', 0, 'leaky_Relu',0.01,'instance',0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0)]]
temps = [0.1,0.2,0.3]#,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
attrib = [[2,16,1000,0.2,]]
#attribute_list = [[0,32,1000,0.1], [2,32,1000,0.1],[2,32,1000,0.2],[2,32,1000,0.3]] #]#,
lattice_size = 16
defect_nmbs = [2]#,4,6]
extra_infos =
               ##[None, 0, 0.5, False,'{}_avg_{}'.format(5,'3_3_fourier_log_10')]]#, ['instance', 0, 0.5, True]] #_avg_gen_fourier_3x5kernel
#get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/', simulation_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',generated_data_attributes=generated_data_atributes_list[0], nmb_label=0,data_type='training_data',temperature=0.1,nmb_defects=attrib[0],noise_size = 1.0, training_data_nmb=100000,lattice_size=attrib[1], new_data= False, further_simulated=False,device='cpu',samplesize = attrib[2])
#get_topological_measures_mean(output_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',generated_data_attributes=generated_data_atributes_list[0],basis_sample_size=attrib[2],number_defects=attrib[0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/', device='cuda', temperature=0.1,samplesize=100,lattice_size=attrib[1],fixed_data=True,data_type_list = ['training_data'], name_idxs = None)

e = extra_infos[0]

generator_info = {}
generator_info['res_critic'] = True
generator_info['res_gen'] = False
generator_info['c_depth'] = 0
generator_info['extra_infos'] =  [None, 0, 0.5, False,'{}_avg_{}'.format(5,'3_5_fourier_log_10')]
generated_data_atributes_list=[
         ['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'sum', 'conv_transposed', 0, 'leaky_Relu',0.01,'instance',0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0)]]#(3, 0.4)
#t= 0.1
##'asymm_{}'.format([1,2]),
#calculation_observables(res_critic= True, extra_info = extra_infos[0],larger=False,defect_nmb=2,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='zero_temp',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./AdaIn/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception/'.format(*generated_data_atributes_list[0],0, *e), epochs=[95], generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = 0.1, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
#calculation_observables(res_critic= True, extra_info = extra_infos[0],larger=False,defect_nmb=4,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='zero_temp',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./AdaIn/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception/'.format(*generated_data_atributes_list[0],0, *e), epochs=[95], generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = 0.1, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
#calculation_observables(res_critic= True, extra_info = extra_infos[0],larger=False,defect_nmb=6,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='zero_temp',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./AdaIn/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception/'.format(*generated_data_atributes_list[0],0, *e), epochs=[95], generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = 0.1, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
#get_edge_state_graphs_sim_gen(res_criritc= r_c,extra_info =e,output_data_path='./observables_output/new/',data_path='./observables_output/new/', simulation_data_path='./observables_output/new/',generated_data_attributes=generated_data_atributes_list[0], nmb_label=10,data_type='zero_temp',temperature=t,nmb_defects=2,noise_size = 1.0, training_data_nmb=350000,lattice_size=lattice_size, new_data= False, further_simulated=False,device='cuda',samplesize = 1000)
#get_edge_state_graphs_sim_gen(res_criritc= r_c,extra_info =e,output_data_path='./observables_output/new/',data_path='./observables_output/new/', simulation_data_path='./observables_output/new/',generated_data_attributes=generated_data_atributes_list[0], nmb_label=10,data_type='zero_temp',temperature=t,nmb_defects=4,noise_size = 1.0, training_data_nmb=350000,lattice_size=lattice_size, new_data= False, further_simulated=False,device='cuda',samplesize = 1000)
#get_edge_state_graphs_sim_gen(res_criritc= r_c,extra_info =e,output_data_path='./observables_output/new/',data_path='./observables_output/new/', simulation_data_path='./observables_output/new/',generated_data_attributes=generated_data_atributes_list[0], nmb_label=10,data_type='zero_temp',temperature=t,nmb_defects=6,noise_size = 1.0, training_data_nmb=350000,lattice_size=lattice_size, new_data= False, further_simulated=False,device='cuda',samplesize = 1000)

#get_persistent_homology(res_criritc= r_c,extra_info= e,output_data_path = './observables_output/new/',generated_data_attributes= generated_data_atributes_list[0],training_data_nmb=350000, simulated_data_path='./observables_output/new/', device='cuda', temperature=t,defect_number=2,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['zero_temp'], name_idxs = [3])
#get_persistent_homology(res_criritc= r_c,extra_info= e,output_data_path = './observables_output/new/',generated_data_attributes= generated_data_atributes_list[0],training_data_nmb=350000, simulated_data_path='./observables_output/new/', device='cuda', temperature=t,defect_number=4,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['zero_temp'], name_idxs = [8])
#get_persistent_homology(res_criritc= r_c,extra_info= e,output_data_path = './observables_output/new/',generated_data_attributes= generated_data_atributes_list[0],training_data_nmb=350000, simulated_data_path='./observables_output/new/', device='cuda', temperature=t,defect_number=6,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['zero_temp'], name_idxs = [24])


def fixed_idx(basic_sample_size, reduced_samplesize, temperature, new=True,data_type='full_pinned',defect_nmb=2):
    if new :
        less_index = random.sample([i for i in range(basic_sample_size)],reduced_samplesize)
        with open('topological_idx_{}_{}_{}_{}.txt'.format(data_type,reduced_samplesize,temperature,defect_nmb), 'w') as f:
            for i in less_index:
                f.write(f"{i}\n")
        return less_index
    else:
        less_idx = []
        file =open('topological_idx_{}_{}_{}_{}.txt'.format(data_type,reduced_samplesize,temperature,defect_nmb))
        for i in range(reduced_samplesize):
            less_idx.append(int(file.readline()[:-1]))
            
        print(less_idx)
        return less_idx
    
##for t in temps[4:]:
##    less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = t, new=False)
##    get_topological_measures_mean(less_index =less_index,res_criritc= r_c,extra_info = e,output_data_path='./',generated_data_attributes=generated_data_atributes_list[0],basis_sample_size=1000,number_defects=2,noise_size = 1.0, training_data_nmb=350000,simulated_data_path='./', device='cuda', temperature=t,samplesize=100,lattice_size=lattice_size,fixed_data=True,data_type_list = ['training_data'], name_idxs = None)
            
            
##for t in temps:
##    less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = t, new=False)
##    get_persistent_homology(less_index=less_index,res_criritc= r_c,extra_info= e,output_data_path = './',generated_data_attributes= generated_data_atributes_list[0],training_data_nmb=350000, simulated_data_path='./', device='cuda', temperature=t,defect_number=2,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs = None)
    
            

#import argparse
#parser = argparse.ArgumentParser(description='Process some integers.')
#args = parser.parse_args()
#parser.add_argument('--name_idxs', type=int, nargs='+', help='an integer for name idxs to calculate')
def full_analysis(data_type, sim_type,defect_nmb,temperatures,generator_info=None,generated_data_atributes_list=None,full_sample_size =1000,tda_sample_size=100,epochs=[-1]):
    epochs = epochs
    basis_sample_size = full_sample_size
    samplesize = tda_sample_size

   
    for t_idx,t in enumerate(temperatures):

                
                ##less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = t, new=True,data_type='new_specific_pinned',defect_nmb=d)
                ##less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = t, new=True,data_type='new_full_pinned',defect_nmb=d)
                
                
                
                #batch_list = np.delete(np.array([i for i in range(1000)]), less_index, None).reshape(9,-1)
                
                #print(batch_list)
                
                sim_path = './'###'./defect_nmb_{}_{}/'.format(d,'new_specific_pinned')
                
                #for b_int,batch in enumerate(batch_list):
                ##spin_dict, defect_dict = output_analysis_basis(batch=False,simulated_data_path='./',temperature=t,lattice_size=lattice_size,data_type='specific_pinned',compare_defect_lattice=False,device=device,samplesize=samplesize,defect_nmb=d,larger=False)
                #less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = t, new=True)
                #calculation_observables(res_critic= r_c, res_gen=False,extra_info = e,larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='generated_data',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                #calculation_observables(res_critic= r_c, res_gen=False,extra_info = e,larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='generated_data',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                ###spin_dict, defect_dict = output_analysis_basis(batch=False,simulated_data_path='./',temperature=t,lattice_size=lattice_size,data_type='specific_pinned',compare_defect_lattice=False,device=device,samplesize=samplesize,defect_nmb=d,larger=False)
                ##calculation_observables(res_critic= r_c, res_gen=False,extra_info = e,sim_data_type='new_specific_pinned',larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='training_data',output_data_path='./',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=True,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  =sim_path, temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                
                calculation_observables(res_critic= r_c, res_gen=False,extra_info = e,sim_data_type='new_specific_pinned',larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='training_data',output_data_path='./',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=True,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  =sim_path, temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                
                ##calculation_observables(res_critic= r_c, res_gen=False,extra_info = e,sim_data_type='new_specific_pinned',larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='training_data',output_data_path='./',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=True,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  =sim_path, temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                ##calculation_observables(res_critic= r_c, res_gen=True,extra_info = e,larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='generated_data',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                #calculation_observables(res_critic= r_c, res_gen=True,extra_info = e,larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='generated_data',output_data_path='./observables_output/new/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./WGAN_checks/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*generated_data_atributes_list[0],0, *e), epochs=epochs, generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                ###get_edge_state_graphs_sim_gen(res_criritc= r_c,extra_info =e,output_data_path='./',data_path='./', simulation_data_path=sim_path,generated_data_attributes=att, nmb_label=10,data_type='training_data',temperature=t,sim_data_type='new_specific_pinned',nmb_defects=d,noise_size = 1.0, training_data_nmb=350000,lattice_size=lattice_size, new_data= False, further_simulated=False,device='cuda',samplesize = 1000)
                ##if t == 0.4 :
                ##    idx_distance = [6,7,8,9,10,11,12,13]
                ##else:
                ##    idx_distance = None
                ####get_topological_measures_mean(extra_idx = 0,less_index =less_index,res_criritc= r_c,extra_info = e,output_data_path='./',generated_data_attributes=generated_data_atributes_list[0],basis_sample_size=10000,number_defects=d,noise_size = 1.0, training_data_nmb=350000,simulated_data_path=sim_path, device='cuda', temperature=t,sim_data_type='specific_pinned_wo_sort',samplesize=100,lattice_size=lattice_size,fixed_data=True,data_type_list = ['training_data'], name_idxs =  None)#calculation_observables(extra_info =['instance', 0, 0.5, True],larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='generated_data',output_data_path='./observables_output/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./AdaIn/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}/res_critic/'.format(*generated_data_atributes_list[0],0, 'instance', 0, 0.5, True), epochs=[120], generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)
                ##get_persistent_homology(extra_idx = 0,less_index=less_index,res_criritc= r_c,extra_info= e,output_data_path = './',generated_data_attributes= generated_data_atributes_list[0],training_data_nmb=350000, simulated_data_path='./', device='cuda', temperature=t,defect_number=d,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs =  None)


generator_info = {}
generator_info['res_critic'] = True
generator_info['res_gen'] = False
generator_info['c_depth'] = 0
generator_info['extra_infos'] =  [None, 0, 0.5, False,'{}_avg_{}'.format(5,'3_5_fourier_log_10')]
generated_data_atributes_list=[
         ['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'sum', 'conv_transposed', 0, 'leaky_Relu',0.01,'instance',0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0)]]#(3, 0.4)