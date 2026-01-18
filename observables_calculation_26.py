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
##from helper_functions import transformation_pos
##from training_september_25_fixed_sum_fourier import CircularUpscaleConv2d, ResidualBlock, AdaIN, UnetGenerator, CircularPad2d, SumPool2d, InceptionConv2d_mini

#import time

import os
import kornia

##from XY_model_slow_sep_23_numba import XYSystem


spin_configurations = []
defect_configurations_real = []
temperatures=[]
generator_epochs = []
vortex_numbers= []
lattice_size = 16
#temp = 0.1
gen_temp = 0
#magnetisations = []
mag_suscept = []
#energys = []
specific_heat_list = []
helicity_modulus_list = []
correlation_function = []
temperature = []
#current_per_link_list = []
#energy_per_link_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)



def load_in_generator_network(data_path,epoch,temperature_label,input_defects,nmb_tlable = 10, upscaling = 'conv_transposed',downscaling='avg', activationfunction = 'leaky_Relu',activationfunction_parameter = 0.01, max_feat_map = 2, extra_depth = 2,noise_factor= 1.0):

    #gen_model = UnetGenerator(input_size=16, input_chan=1,embedding_dim=100, activation_function = activationfunction ,activationfunction_parameter=activationfunction_parameter,num_downs=4, calc_depth=extra_depth, max_feature_map= max_feat_map, downsampling = downscaling , noise_factor = 10.0,batchnorm=False, upsampling= upscaling
    #                ,numb_temp_labels = 0)
    load_model = torch.load(data_path 
                            #+ path_name
                            + 'Generator_check_{}.pth'.format(epoch),map_location=torch.device(device), weights_only = False)
    load_model.change_noise_factor(noise_factor)
    #gen_model.load_state_dict(load_model.state.dict())
    #print(temperature_label)
    input_defects = input_defects.float().view(-1,1,lattice_size,lattice_size)
    generated_data_list = []
    for b in range(0,int(input_defects.shape[0]),10):
        generated_data_sub = load_model(input_defects[b:b+10],labels=temperature_label[b:b+10])
        #print(generated_data_sub.shape)
        generated_data_list.append(generated_data_sub)
    #print(load_model)

    #print('data loaded')
    generated_data = torch.cat(generated_data_list,dim=0)
    #print('check_size',generated_data.shape)
    return generated_data

##test with defect_position x= 6, y= 6 und x = 6 y= 8

""" 
defect_pos_test = torch.tensor([[[6,6],[6,8]],[[6,6],[6,8]]])
defect_lattice_test = torch.zeros((2,lattice_size,lattice_size))
defect_lattice_test[0,6,6] = 1
defect_lattice_test[0,6,8] = 1
defect_lattice_test[1,6,6] = 1
defect_lattice_test[1,6,8] = 1

all_lattice_indices = torch.tensor([[[i,j] for j in range(lattice_size)] for i in range(lattice_size)]).view(1,lattice_size,lattice_size,2)


all_lattice_indices_batches = torch.repeat_interleave(all_lattice_indices,2,dim=0)

print(all_lattice_indices_batches.size())
print(defect_pos_test.size())
print(defect_pos_test.size())



distance_defect_spin_side_1 = torch.sqrt((torch.pow((all_lattice_indices_batches[:,:,:,0] - defect_pos_test[:,0,0].view(-1,1,1)),2)+torch.pow((all_lattice_indices_batches[:,:,:,1] - defect_pos_test[:,0,1].view(-1,1,1)),2)))> lattice_size/4.0
distance_defect_spin_side_2 = torch.sqrt((torch.pow((all_lattice_indices_batches[:,:,:,0] - defect_pos_test[:,1,0].view(-1,1,1)),2)+torch.pow((all_lattice_indices_batches[:,:,:,1] - defect_pos_test[:,1,1].view(-1,1,1)),2)))> lattice_size/4.0

distance_defect_spin_side = torch.logical_and(distance_defect_spin_side_1,distance_defect_spin_side_2)

print(all_lattice_indices_batches)
print(defect_lattice_test)
print(distance_defect_spin_side_1)
print(distance_defect_spin_side_2)
print(distance_defect_spin_side)

"""
def vortex_number(defects,lattice_size):
    defect_array = defects.reshape(lattice_size*lattice_size)
    vortex_counter = 0
    for vortex in defect_array:
        if vortex == 1 or vortex == -1:
            vortex_counter += 1

    return vortex_counter

def defect_position(defects,number_of_defects, lattice_size):

    position_vector = np.zeros((number_of_defects,3))

    counter = 0


    for i in range(lattice_size):
        for j in range(lattice_size):
            if defects[i,j] != 0:
                position_vector[counter,0] += i
                position_vector[counter,1] += j
                position_vector[counter,2] += np.copy(defects[i,j])
                counter +=1

    #print(position_vector)

    return position_vector

def analysitcally_calculation_tensor(spin_position,defects,lattice_size):#tensor

    vortex_numb = torch.sum(torch.abs(defects.view(lattice_size*lattice_size)),dim=0)

    #print(vortex_numb)

    #defect_positions = defect_position(defects = defects,number_of_defects=vortex_numb, lattice_size= lattice_size)

    analytic_solution_list = []

    #print(spin_position.shape)

    defect_pos_tensor = torch.nonzero(defects,as_tuple=True)

    #print(defect_pos_tensor[1][0].item())
    #print(spin_position[:,:,1])
    #print(defects[defect_pos_tensor[0][0],defect_pos_tensor[1][0]])

    analytic_solution = sum([defects[defect_pos_tensor[0][v],defect_pos_tensor[1][v]]*torch.arctan2(spin_position[:,:,1].float()-(defect_pos_tensor[1][v]+0.5),spin_position[:,:,0]-(defect_pos_tensor[0][v]+0.5)) for v in range(int(vortex_numb.item()))])

    print(analytic_solution)

    #torch.arctan2(np.asarray(spin_position[:,:,1]-(defect_positions[v,1]+0.5)), np.asarray(spin_position[i,j,0]-(defect_positions[v,0]+0.5))



    #analytic_solution = np.zeros([lattice_size,lattice_size])

    ##for i in range(lattice_size):#should be x direction
    ##    for j in range(lattice_size):#should be y direction

    ##        analystic_angles = [(defect_positions[v,2])*np.arctan2(np.asarray(spin_position[i,j,1]-(defect_positions[v,1]+0.5)), np.asarray(spin_position[i,j,0]-(defect_positions[v,0]+0.5))) for v in range(vortex_numb)]

     ##       analytic_angle = np.sum(analystic_angles)

     ##       analytic_solution_list.append(analytic_angle)

    ##analytic_solution = np.asarray(analytic_solution_list).reshape(lattice_size,lattice_size)

    return analytic_solution

def analysitcally_calculation(spin_position,defects,lattice_size):

    vortex_numb = vortex_number(defects,lattice_size)

    #print(vortex_number)

    defect_positions = defect_position(defects = defects,number_of_defects=vortex_numb, lattice_size= lattice_size)

    #print(defect_positions)

    analytic_solution_list = []

    #print(spin_position.shape)

    #analytic_solution = np.zeros([lattice_size,lattice_size])

    for i in range(lattice_size):#should be x direction
        for j in range(lattice_size):#should be y direction
            #print(defect_positions[v,2])

            #print(defect_positions[0,1].shape)

            #print(spin_position[i,j,1].shape)



            analystic_angles = [(defect_positions[v,2])*np.arctan2(np.asarray(spin_position[i,j,1]-(defect_positions[v,1]+0.5)), np.asarray(spin_position[i,j,0]-(defect_positions[v,0]+0.5))) for v in range(vortex_numb)]

            #print(analystic_angles)

            analytic_angle = np.sum(analystic_angles)

            analytic_solution_list.append(analytic_angle)

    analytic_solution = np.asarray(analytic_solution_list).reshape(lattice_size,lattice_size)
    #print(analytic_solution)

    return analytic_solution


#defect_lattice = torch.zeros(lattice_size,lattice_size)
#defect_lattice[0,5,14] = 1
#defect_lattice[0,5,1] = -1
#defect_lattice[2,4] = -1
#defect_lattice[2,6] = 1
#lattice_size=16
#lattice_spacing= 1
#spin_positions = torch.tensor([[[i*lattice_spacing,j*lattice_spacing] for j in range(lattice_size)] for i in range(lattice_size)])

#tensor_analytic_solution = analysitcally_calculation_tensor(spin_position=spin_positions,defects=defect_lattice,lattice_size=lattice_size)

#tensor_analytic_solution_v2 = torch.from_numpy(analysitcally_calculation(spin_position=spin_positions.numpy(),defects=defect_lattice.numpy(),lattice_size=lattice_size))

#print('check', tensor_analytic_solution-tensor_analytic_solution_v2)


def testing_correlation_function(simualted_data_path='/localscratch/kklos/XY_Cluster_rust_sim_data/',temperature=0.1, lattice_size = 16):
    simul_spin = pd.read_csv(simualted_data_path+'new_cluster_spin_data_observables_{}_{}_0.csv'.format(temperature, lattice_size), delimiter=',', skipinitialspace= True, header = 0)

def check_defect_type_config(simulated_data_path,temperature, lattice_size,data_type='full_pinned',defect_nmb=2,transformed= False,device='cpu', samplesize = 1000,compare_defect_lattice=False,reference_temp = 0.1,new=True):
    with torch.no_grad():
        
        zeroth_spin_dict = {}
        zeroth_defect_dict = {}
        
        if new:
            
            if data_type == 'full_pinned':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_testing_data_gan_defect_distance_select_compare_0_rotation_0_later_rot_1_latticesize_16_defectnmb_2_samplesize350000_temperature_{}.h5'.format(temperature), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_distances = torch.from_numpy(np.array(h5f['distances'])).to(device).view(-1)


                    print('check input',sim_distances.shape)

                defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                check_observables = Observables(lattice_size=lattice_size, temperature = temperature)
                


                
                #print(torch.any(sim_distances == 5.0))
                for dist in torch.unique(sim_distances):
                    current_defect_distances = sim_defects[sim_distances==dist]
                    #print(current_defect_distances.shape)
                    current_spin_distances = sim_spins[sim_distances==dist]

                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    sample_idx = random.sample(spin_data_index,samplesize)
                    distances_dictionary_spins[dist.item()] = current_spin_distances[sample_idx]
                    distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                    sizes.append(int(sim_spins[spin_data_index].size(0)))
                    
                if transformed:
                
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)
                        
            if data_type == 'new_full_pinned':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_1_dist_sorted_0_defect_nmb_{}.h5'.format(data_type,temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)



                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]


                defect_number_tensor = torch.sum(torch.abs(reduced_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)
                defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                topological_analysis_tool.set_full_defect_positions(defect_lattices=reduced_defects,maximal_defect_nmb=maximal_defect_nmb)
                defect_distances = topological_analysis_tool.return_full_defect_distances()
                
                ##print(defect_distances)
                print(torch.unique(defect_distances,return_counts= True))
                
                
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                check_observables = Observables(lattice_size=lattice_size, temperature = temperature)
                


                
                #print(torch.any(sim_distances == 5.0))
                for dist in torch.unique(defect_distances):
                    current_defect_distances = reduced_defects[defect_distances==dist]
                    #print(current_defect_distances.shape)
                    current_spin_distances = reduced_spins[defect_distances==dist]

                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    if int(current_defect_distances.size(0)) < samplesize:
                        continue
                    else:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist.item()] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                        sizes.append(int(reduced_spins[spin_data_index].size(0)))
                        
                if transformed:
                
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)
                        
            elif data_type == 'real_data':#need to have 10,000 sample size
                
                temperature_idx = int((temperature*10)-1.)
                batch = 10000
                lazy_count = 200000*temperature_idx
                    
                spin_data_index = [i for i in range(int(200000))]
                sample_idx = random.sample(spin_data_index,samplesize)                    
                distances_dictionary_spins_list = []
                distances_dictionary_defects_list = []
                sizes = []
                with open('observables_test_idx_{}_{}_{}_{}.txt'.format(data_type,samplesize,temperature,defect_nmb), 'w') as f:
                    for i in sample_idx:
                        f.write(f"{i}\n")
                print(sample_idx)
                zeroth_spin_dict = None
                zeroth_defect_dict = None
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_1_dist_sorted_0_defect_nmb_{}.h5'.format(data_type,'full',defect_nmb), 'r') as h5f:
                    
                    ##sim_temperatures = torch.from_numpy(np.array(h5f['temperatures'])[200000*temperature_idx:200000*temperature_idx+200000]).to(device).view(-1)
                    ##print(torch.all(sim_temperatures==(temperature*10)))
                    sim_spins = torch.from_numpy(np.array(h5f['spins'][lazy_count:lazy_count+200000])).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'][lazy_count:lazy_count+200000])).view(-1,lattice_size,lattice_size)
                    print(sim_spins.shape)
                    ##sim_distances = torch.zeros(sim_spins.size(0),dvice=device)
                    
                distances_dictionary_spins[0.] = sim_spins[sample_idx].to(device)
                distances_dictionary_defects[0.] = sim_defects[sample_idx].to(device)
                    ##for i in range(200000*temperature_idx,200000*temperature_idx+200000,batch):
                        
                print('test')
                        
                        
                        

                        
                        #print('check input',sim_distances.shape)
                        
                        ##defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                        ##maximal_defect_nmb=torch.max(defect_number_tensor)
                        

                        ##check_observables = Observables(lattice_size=lattice_size, temperature = temperature)
                        ##for dist in torch.unique(sim_distances):
                        ##    current_defect_distances = sim_defects[sim_distances==dist]
                            #print(current_defect_distances.shape)
                        ##    current_spin_distances = sim_spins[sim_distances==dist]
                            

                            ##sizes.append(int(sim_spins[spin_data_index].size(0)))

                            ##distances_dictionary_spins[dist.item()][] = current_spin_distances[sample_idx]
                            ##distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                            
                            
                if transformed:
                        
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)

                ##for key in distances_dictionary_spins:
                ##    distances_dictionary_spins[key] = torch.stack(distances_dictionary_spins_list,dim=0)     
                ##    distances_dictionary_defects[key] = torch.stack(distances_dictionary_defects_list,dim=0) 
            elif data_type == 'specific_pinned' or data_type == 'new_larger_specific_pinned' or data_type == 'new_specific_pinned':#need to have 10,000 sample size
                
                with h5py.File(simulated_data_path+'/Training_data/'+'latticesize_{}_full_{}_data_gan_{}_rotation_0_dist_sorted_0_defect_nmb_{}.h5'.format(lattice_size,data_type,temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_defects = torch.from_numpy(np.array(h5f['zeroth_defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_spins = torch.from_numpy(np.array(h5f['zeroth_spins'])).to(device).view(-1,lattice_size,lattice_size)
                    


                    print('check input',sim_spins.shape)
                    
                print(sum(torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1)!=0))
                
                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]
                reduced_zeroth_spins = zeroth_spins[reduction_idx]
                reduced_zeroth_defects = zeroth_defects[reduction_idx]

                defect_number_tensor = torch.sum(torch.abs(reduced_zeroth_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                topological_analysis_tool.set_full_defect_positions(defect_lattices=reduced_zeroth_defects,maximal_defect_nmb=maximal_defect_nmb)
                defect_distances = topological_analysis_tool.return_full_defect_distances()
                
                ##print(defect_distances)
                print(torch.unique(defect_distances,return_counts= True))
                
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                check_observables = Observables(lattice_size=lattice_size, temperature = temperature)
                for dist in torch.unique(defect_distances):
                    current_defect_distances = reduced_defects[defect_distances==dist]
                    #print(current_defect_distances.shape)
                    current_spin_distances = reduced_spins[defect_distances==dist]
                    
                    current_zeroth_defect = reduced_zeroth_defects[defect_distances==dist]
                    current_zeroth_spin =reduced_zeroth_spins[defect_distances==dist]
                    
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    
                    if int(current_defect_distances.size(0)) < samplesize:
                        continue
                    else:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist.item()] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                        
                        zeroth_spin_dict[dist.item()] = current_zeroth_spin[sample_idx]
                        zeroth_defect_dict[dist.item()] = current_zeroth_defect[sample_idx]
                        sizes.append(int(reduced_spins[spin_data_index].size(0)))
                    
                if transformed:
                
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)
                        
            elif data_type == 'specific_pinned_wo_sort' or data_type == 'larger_specific_pinned_wo_sort':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_1_dist_sorted_0_defect_nmb_{}.h5'.format('specific_pinned',temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_defects = torch.from_numpy(np.array(h5f['zeroth_defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_spins = torch.from_numpy(np.array(h5f['zeroth_spins'])).to(device).view(-1,lattice_size,lattice_size)
                    
                    
                print(sum(torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1)!=0))
                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]
                reduced_zeroth_spins = zeroth_spins[reduction_idx]
                reduced_zeroth_defects = zeroth_defects[reduction_idx]
                spin_data_index = [i for i in range(int(reduced_spins.size(0)))]
                sample_idx = random.sample(spin_data_index,samplesize)  
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                
                distances_dictionary_spins[0.] = reduced_spins[sample_idx]
                distances_dictionary_defects[0.] = reduced_defects[sample_idx]
                        
                zeroth_spin_dict[0.] = reduced_zeroth_spins[sample_idx]
                zeroth_defect_dict[0.] = reduced_zeroth_defects[sample_idx]
            elif data_type == 'new_specific_pinned_wo_sort':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_0_dist_sorted_0_defect_nmb_{}.h5'.format('new_specific_pinned',temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_defects = torch.from_numpy(np.array(h5f['zeroth_defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_spins = torch.from_numpy(np.array(h5f['zeroth_spins'])).to(device).view(-1,lattice_size,lattice_size)
                    
                    
                print(sum(torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1)!=0))
                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]
                reduced_zeroth_spins = zeroth_spins[reduction_idx]
                reduced_zeroth_defects = zeroth_defects[reduction_idx]
                spin_data_index = [i for i in range(int(reduced_spins.size(0)))]
                sample_idx = random.sample(spin_data_index,samplesize)  
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                
                distances_dictionary_spins[0.] = reduced_spins[sample_idx]
                distances_dictionary_defects[0.] = reduced_defects[sample_idx]
                        
                zeroth_spin_dict[0.] = reduced_zeroth_spins[sample_idx]
                zeroth_defect_dict[0.] = reduced_zeroth_defects[sample_idx]



        else:
            with h5py.File(simulated_data_path+'/Training_data/'+'full_training_data_gan_{}_corrected.h5'.format(temperature), 'r') as h5f:
                sim_spins = torch.from_numpy(h5f['spins'][350000:700000]).to(device).view(-1,lattice_size,lattice_size)
                sim_defects = torch.from_numpy(h5f['defects'][350000:700000]).to(device).view(-1,lattice_size,lattice_size)


                print('check input')

            defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)
            print(maximal_defect_nmb)
                    ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
                    ##print(unique_defect_lattices.size())
                    ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)

            topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


            topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
            defect_distances = topological_analysis_tool.return_full_defect_distances()

            print(defect_distances)

            distances_dictionary_spins = {}
            distances_dictionary_defects= {}


            if compare_defect_lattice:
                with h5py.File(simulated_data_path+'/observables_output/'+'defect_for_analysis_full_distances_{}_old.h5'.format(reference_temp,samplesize), 'r') as h5f:

                    name_list = np.array(h5f['distances'])

                for dist_idx, dist in enumerate(name_list):

                


                    with h5py.File(simulated_data_path+'/observables_output/'+'defect_for_analysis_full_distances_{}_old.h5'.format(reference_temp,samplesize), 'r') as h5f:#_sample_{}

                            #sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        sim_basis_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                            #full_spin_list.append(sim_spins)
                            #full_defect_list.append(sim_basis_defects)

                    distances_dictionary_spins[dist] = []
                    distances_dictionary_defects[dist] = []

                    spin_list_unique = []
                    defect_list_unique = []

                    unique_defects, unique_defects_count = torch.unique(sim_basis_defects.view(-1,lattice_size,lattice_size),dim=0, return_counts= True)

                        #print(dist, torch.sum(unique_defects_count))

                    current_defect_distances = sim_defects[defect_distances==name_list[dist_idx]]
                    current_spin_distances = sim_spins[defect_distances==name_list[dist_idx]]

                    ##print('size', current_defect_distances.shape)
                    ##print('size', current_spin_distances.shape)


                    ##print(torch.sum(unique_defects_count))

                    ##print(unique_defects_count[0])

                    if dist_idx == 0:

                        spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]

                        sample_idx = random.sample(spin_data_index,samplesize)

                        distances_dictionary_spins[dist] = current_spin_distances[sample_idx]

                        distances_dictionary_defects[dist] = current_defect_distances[sample_idx]

                    else:

                        size_test = []

                        other_size_test = []



                        for unique_idx,basis_defect in enumerate(unique_defects):

                                #print(current_defect_distances==basis_defect)

                            unique_indexing = torch.all((current_defect_distances==basis_defect).view(-1, lattice_size*lattice_size),dim=1)

                            unique_input_spins = current_spin_distances[unique_indexing]

                            unique_input_defects = current_defect_distances[unique_indexing]

                                #print(unique_idx)

                                #print('spins', unique_input_spins.shape)

                            ##other_size_test.append(unique_input_defects.size(0))
                                #print('unique_spins', unique_input_spins[:unique_defects_count[unique_idx]].shape)

                            ##size_test.append(unique_defects_count[unique_idx])
                            ##if unique_defects_count[unique_idx] > unique_input_spins.size(0):
                            ##    print('Error',unique_defects_count[unique_idx], unique_input_spins.size(0))


                            spin_list_unique.append(unique_input_spins[:unique_defects_count[unique_idx]].view(-1,lattice_size,lattice_size))
                            defect_list_unique.append(unique_input_defects[:unique_defects_count[unique_idx]].view(-1,lattice_size,lattice_size))



                        distances_dictionary_spins[dist] = torch.cat(spin_list_unique,dim=0)
                        distances_dictionary_defects[dist] =  torch.cat(defect_list_unique,dim=0)
                        ##print('sum test', sum(size_test))
                        ##print('sum test', sum(other_size_test))
                        ##print(torch.cat(spin_list_unique,dim=0).shape)




                #print('works?')



            
            else:

                    #print(defect_distances.size())


                    ##make  uniform distributed data of the following type:
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                for dist in torch.unique(defect_distances):
                        ##print(dist)
                    current_defect_distances = sim_defects[defect_distances==dist]
                    current_spin_distances = sim_spins[defect_distances==dist]
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    sample_idx = random.sample(spin_data_index,samplesize)
                    distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                    distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                    sizes.append(int(sim_spins[spin_data_index].size(0)))

                print('defects',sizes)


        return distances_dictionary_spins, distances_dictionary_defects, zeroth_spin_dict, zeroth_defect_dict
        
    
def defect_distance_distributions(defect_distances,correspon_defects, samplesize):
    
    unique_distances, unique_distance_nmb = torch.unique(defect_distances, return_counts= True)
    
    new_unique_distance_nmb = torch.clone(unique_distance_nmb)
    
    new_distance_with_counts = {}
    
    for d_idx,d in unique_distances:
        if unique_distance_nmb[d_idx] < samplesize:
            if (d_idx-1) > 0:
                lower_uniq_nmb = new_unique_distance_nmb[d_idx-1]
    
        
def check_defect_type_config_larger_lattice(simulated_data_path,temperature, lattice_size,batch = True,device='cuda', samplesize = 1000, defect_number = 2):
    
    

    with torch.no_grad():
        
        
        
        if batch:
            
            distances_dictionary_spins = {}
            distances_dictionary_defects = {}
            
            batchsize = 51000
            
            for b in range(10):
            
            
                with h5py.File(simulated_data_path+'Training_data/'+'full_training_data_gan_{}_T_{}_lattice_{}_defect.h5'.format(temperature,lattice_size,defect_number), 'r') as h5f:
                ##with h5py.File(simulated_data_path+'lattice_{}_num_pairs{}_temp{}.h5'.format(lattice_size,defect_number//2,temperature), 'r') as h5f:
                    sim_spins = torch.from_numpy(h5f['spins'][b*batchsize:b*batchsize+batchsize]).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(h5f['defects'][b*batchsize:b*batchsize+batchsize]).to(device).view(-1,lattice_size,lattice_size)


                    print('check input', sim_defects.shape)

                    defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                    maximal_defect_nmb=torch.max(defect_number_tensor)
                    print(maximal_defect_nmb)
                        ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
                        ##print(unique_defect_lattices.size())
                        ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)
                    if  maximal_defect_nmb >= 2:


                        topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                        topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
                        defect_distances = topological_analysis_tool.return_full_defect_distances()

                    else:
                        defect_distances = torch.zeros([sim_defects.size(0)], device=device)

                    print('distances', torch.unique(defect_distances, return_counts= True))


                        #print(defect_distances.size())


                        ##make  uniform distributed data of the following type:

                    sizes = []
                    for dist in torch.unique(defect_distances):
                        #print(dist)
                        
                        current_defect_distances = sim_defects[defect_distances==dist]
                        current_spin_distances = sim_spins[defect_distances==dist]
                        spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                        if int(current_defect_distances.size(0)) > 100:
                            sample_idx = random.sample(spin_data_index,100)
                            if distances_dictionary_spins.get(dist.item()) is not None:
                                print('check')
                                distances_dictionary_spins[dist.item()].append(current_spin_distances[sample_idx])
                                distances_dictionary_defects[dist.item()].append(current_defect_distances[sample_idx])

                            else:

                                distances_dictionary_spins[dist.item()] = [current_spin_distances[sample_idx]]
                                distances_dictionary_defects[dist.item()] = [current_defect_distances[sample_idx]]
                            sizes.append(int(sim_spins[spin_data_index].size(0)))
                            
                    print(list(distances_dictionary_spins.keys()))

            
            
        else:
                    


            #with h5py.File(simulated_data_path+'full_training_data_gan_{}_T_{}_lattice_{}_defect.h5'.format(temperature,lattice_size,defect_number), 'r') as h5f:
            with h5py.File(simulated_data_path+'full_data_gan_{}_rotation_{}_dist_sorted_{}_defect_nmb_{}.h5'.format(temperature,0,0,defect_number), 'r') as h5f:
                sim_spins = torch.from_numpy(h5f['spins'][:]).to(device).view(-1,lattice_size,lattice_size)
                sim_defects = torch.from_numpy(h5f['defects'][:]).to(device).view(-1,lattice_size,lattice_size)
                


                #print('check input', sim_defects.shape)

                defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                print(maximal_defect_nmb)
                    ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
                    ##print(unique_defect_lattices.size())
                    ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)
                if  maximal_defect_nmb >= 2:


                    topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                    topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
                    defect_distances = topological_analysis_tool.return_full_defect_distances()

                else:
                    defect_distances = torch.zeros([sim_defects.size(0)], device=device)

                print('distances', torch.unique(defect_distances, return_counts= True))


                    #print(defect_distances.size())


                    ##make  uniform distributed data of the following type:
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                #if defect_number == 4:
                    
                #reduced_defect_distances
                for dist in torch.unique(defect_distances):
                    #print(dist)
                    
                    current_defect_distances = sim_defects[defect_distances==dist]
                    current_spin_distances = sim_spins[defect_distances==dist]
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    if int(current_defect_distances.size(0)) >= samplesize:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                        sizes.append(int(sim_spins[spin_data_index].size(0)))

                #print('defects',sizes)


        return distances_dictionary_spins, distances_dictionary_defects
    
    

def output_analysis_basis(simulated_data_path,temperature,lattice_size,data_type='full_pinned',transformed=False,batch= True,compare_defect_lattice=False,device='cpu',samplesize=1000, larger=True,defect_nmb=2,new=True):

    if larger:
        spin_dict, defect_dict= check_defect_type_config_larger_lattice(batch = batch,simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,device=device,samplesize=samplesize,defect_number=defect_nmb)

    else:

        spin_dict, defect_dict, zeroth_spin_dict, zeroth_defect_dict = check_defect_type_config(transformed= transformed,simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,data_type=data_type,defect_nmb=defect_nmb,device=device,samplesize=samplesize,compare_defect_lattice=compare_defect_lattice,new=new)
    #name_list = ['full', 'no_defect', 'small_defects', 'mid_defects', 'large_defects']

    output_spin_dict = {}
    output_defect_dict = {}
    print('check',[dist for dist in spin_dict])

    with h5py.File(simulated_data_path + 'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(data_type,temperature,lattice_size,defect_nmb,samplesize),'w') as h5f:

        dataset_distances = h5f.create_dataset('distances', shape=(int(len(spin_dict)),), dtype='float')
        if batch:
            dataset_distances[:] = np.array([dist for dist in spin_dict])
        else:
            dataset_distances[:] = np.array([dist for dist in spin_dict])
            

        for dist in spin_dict:
            
            if batch:
                output_spin_dict[dist] = torch.cat(spin_dict[dist], dim = 0)
                output_defect_dict[dist] = torch.cat(defect_dict[dist], dim = 0)
            else:
                output_spin_dict[dist] = spin_dict[dist]
                output_defect_dict[dist] = defect_dict[dist]
                
            if transformed:
                samplesize = int(samplesize/2)
                
                            
            dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            
            if data_type == 'specific_pinned' or data_type == 'new_larger_specific_pinned' or data_type == 'specific_pinned_wo_sort' or data_type == 'new_specific_pinned' or data_type == 'new_specific_pinned_wo_sort':
                dataset_zeroth_spins_compare = h5f.create_dataset('spin_lattices_zeroth{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
                dataset_zeroth_defects_compare = h5f.create_dataset('defects_lattices_zeroth{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            
            if batch:
                
                print(torch.cat(spin_dict[dist], dim = 0).shape)
            
                dataset_spins_compare[:] = torch.cat(spin_dict[dist], dim = 0).view(-1,lattice_size*lattice_size).cpu().numpy()
                
                dataset_defects_compare[:] = torch.cat(defect_dict[dist],dim=0).view(-1,lattice_size*lattice_size).cpu().numpy()
                
            else:
                
                dataset_spins_compare[:] = spin_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                
                dataset_defects_compare[:] = defect_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                
                if data_type == 'specific_pinned' or data_type == 'new_larger_specific_pinned'or data_type == 'specific_pinned_wo_sort'or data_type == 'new_specific_pinned' or data_type == 'new_specific_pinned_wo_sort':
                    
                    dataset_zeroth_spins_compare[:] = zeroth_spin_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                
                    dataset_zeroth_defects_compare[:] = zeroth_defect_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                

                
    return output_spin_dict, output_defect_dict


#for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#    output_analysis_basis(simulated_data_path='./',temperature=t,lattice_size=16,transformed=True,batch= False,compare_defect_lattice=False,device='cpu',samplesize=1000, larger=False,defect_nmb=2,new=True)

def return_defect_distances(maximal_distance):
    distances = []
    for i in range(maximal_distance):
        for j in range(maximal_distance):
            dist = torch.sqrt(i**2+j**2)
            distances.append(dist)

    return distances

##compare_defect_lattice means here that we take the identical defect_positions for different temperatures
def calculation_observables(res_critic,res_gen,extra_info,generator_input_dict,data_type,output_data_path,gen_folder= None,sim_data_type='full_pinned',larger=False,defect_nmb=2,compare_defect_lattice=False,shifte_to_side_test=False,c_depth=0,noise_size = 1.0, training_data_nmb=350000,nmb_label = 1,maximal_distance= 5,new_data=False,data_path='/localscratch/kyklos/Pytorch_cuda_code/iona/iona_oct_23/Full_training_oct_23/', epochs=[0,5,10,15,20,25,30], generated_data_atributes=[64,0.01,0,5,1,0],simulation_data_path  ='/localscratch/kyklos/Pytorch_cuda_code/iona/iona_oct_23/Full_training_oct_23/full_training_data_gan_low_temp_2_corrected.h5', temperature = 0.1, lattice_size= 16,further_simulated=False,samplesize=1000,rotation=False,later_rotation=True):

    with torch.no_grad():

        
        if new_data :
            if lattice_size == 32:
                batch = True
                device_check = device
            else:
                batch = False
                device_check = device
            spin_dict, defect_dict = output_analysis_basis(batch=batch,simulated_data_path=simulation_data_path,temperature=temperature,lattice_size=lattice_size,data_type=sim_data_type,compare_defect_lattice=compare_defect_lattice,device=device_check,samplesize=samplesize,defect_nmb=defect_nmb,larger=larger)
            full_spin_list = []
            full_defect_list = []
            name_list = []
            for dist in spin_dict:
                full_spin_list.append(spin_dict[dist])
                full_defect_list.append(defect_dict[dist])
                name_list.append(dist)
                
            print('data finished')
            
                
        
        else:
            full_spin_list = []
            full_defect_list = []
            
            if defect_nmb > 2 or lattice_size > 16 :
                #with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(sim_data_type,temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                    name_list = np.array(h5f['distances'])
                            
                    for dist in name_list:
                        sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                        full_spin_list.append(sim_spins)
                        full_defect_list.append(sim_defects)
                        
                        ##this should 
                
            else:

            #name_list = return_defect_distances(maximal_distance=maximal_distance)
                #with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_sample_{}.h5'.format(temperature,samplesize), 'r') as h5f:

                with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                ##with h5py.File(simulation_data_path+'full_training_data_gan_defect_distance_select_compare_{}_rotation_{}_later_rot_{}_latticesize_{}_defectnmb_{}_samplesize{}_temperature_{}.h5'.format(int(compare_defect_lattice),int(rotation),int(later_rotation),lattice_size,defect_nmb,training_data_nmb,temperature), 'r') as h5f:

                    name_list = np.array(h5f['distances'])
                            
                    for dist in name_list:
                        sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k other -1, L,L,2
                        sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)
                        full_spin_list.append(sim_spins)
                        full_defect_list.append(sim_defects)
                        print(sim_spins.shape)
        print(name_list)

        nmb_distances = int(len(name_list))
        max_distances = max(name_list)
        
        print(max_distances)

        defect_nmber = full_defect_list

        if data_type == 'training_data':


            oservable_data_spins = []

            oservable_data_defects = []

            path_name = 'training_data_full_distance_temp_{}_new_{}/'.format(temperature,sim_data_type)

            for idx, name in enumerate(name_list):
                ##transformed_spin = full_spin_list[idx].reshape(-1,lattice_size*lattice_size,2)
                
                
                ##data_norm = torch.sqrt((torch.square(transformed_spin[:,:,1])+torch.square(transformed_spin[:,:,0])))+1e-12
                            ##print(generated_data_norm.size())
                ##transformed_spin_angle = torch.arctan2(transformed_spin[:,:,1]/data_norm,transformed_spin[:,:,0]/data_norm)
                
                ##print(transformed_spin_angle.shape)
                oservable_data_spins.append([full_spin_list[idx].reshape(-1,lattice_size,lattice_size)])
                oservable_data_defects.append([full_defect_list[idx].reshape(-1,lattice_size,lattice_size)])
                
        elif data_type == 'vision_trafo_data':
            
            full_spin_list = []
            full_defect_list = []
            
            with h5py.File(simulation_data_path+'/VIT_check/'+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                ##with h5py.File(simulation_data_path+'full_training_data_gan_defect_distance_select_compare_{}_rotation_{}_later_rot_{}_latticesize_{}_defectnmb_{}_samplesize{}_temperature_{}.h5'.format(int(compare_defect_lattice),int(rotation),int(later_rotation),lattice_size,defect_nmb,training_data_nmb,temperature), 'r') as h5f:

                name_list = np.array(h5f['distances'])
                            
                for dist in name_list:
                    sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size,2)##for constraint_sampling more like 80 k other -1, L,L,2
                    sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)
                    full_spin_list.append(sim_spins)
                    full_defect_list.append(sim_defects)
                    
                    print('check',sim_defects.shape)
            
            oservable_data_spins = []

            oservable_data_defects = []

            path_name = '/VIT_check/training_data_full_distance_temp_{}_new/'.format(temperature)

            for idx, name in enumerate(name_list):
                transformed_spin = full_spin_list[idx].reshape(-1,lattice_size*lattice_size,2)
                
                
                data_norm = torch.sqrt((torch.square(transformed_spin[:,:,1])+torch.square(transformed_spin[:,:,0])))+1e-12
                            ##print(generated_data_norm.size())
                transformed_spin_angle = torch.arctan2(transformed_spin[:,:,1]/data_norm,transformed_spin[:,:,0]/data_norm)
                
                ##print(transformed_spin_angle.shape)
                oservable_data_spins.append([transformed_spin_angle.reshape(-1,lattice_size,lattice_size)])
                oservable_data_defects.append([full_defect_list[idx].reshape(-1,lattice_size,lattice_size)])


        elif data_type == 'generated_data':

            oservable_data_spins = []
            oservable_data_defects = []
            
            if gen_folder is None:

                path_name = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(noise_size,training_data_nmb,*generated_data_atributes,c_depth, *extra_info) 
            else:
                path_name = gen_folder
            
            if res_critic and not res_gen:
                path_name += 'res_critic/'
                
                data_path += 'res_critic/'
            elif res_critic and res_gen:
                path_name += 'residual_blocksres_critic/'
                
                data_path += 'residual_blocksres_critic/'
                
                
            path_name += 'temp_{}/'.format(temperature)
            print(path_name)
            if nmb_label == 0:
                temperature_label = None
            elif nmb_label == 1 or nmb_label == 3:

                if temperature <= 0.3:
                    factor = 0.
                elif temperature <= 0.6:
                    factor = 1.
                else:
                    factor  = 2.0

                temperature_label = (torch.ones(samplesize).view(-1,1)*factor).long()
                
            elif nmb_label == 9:
                temperature_label = (torch.sub(torch.ones(samplesize, device = device).view(-1,1)*temperature*10.,1.)).long()
            else:
                temperature_label = (torch.ones(samplesize, device = device).view(-1,1)*temperature*10.).long()
                
           
            
            for input_defects in full_defect_list:

                
                generated_angles_list_per_epoch = []
                generated_defects_list_per_epoch = []

                for epoch in epochs:


                    generated_data = load_in_generator_network(data_path=data_path,epoch=epoch,temperature_label=temperature_label,input_defects=input_defects,nmb_tlable = nmb_label, upscaling = generator_input_dict['upsample'],downscaling=generator_input_dict['downsample'], activationfunction = generator_input_dict['activation_function'][0],activationfunction_parameter = generator_input_dict['activation_function'][1], max_feat_map = generator_input_dict['max_feature'], extra_depth = generator_input_dict['depth'],noise_factor= noise_size)


                    generated_data_x = generated_data[:,0,:,:].view(-1,lattice_size,lattice_size)
                    generated_data_y = generated_data[:,1,:,:].view(-1,lattice_size,lattice_size)
                    generated_data_norm = torch.sqrt((torch.square(generated_data_x)+torch.square(generated_data_y)))+1e-12
                            ##print(generated_data_norm.size())
                    generated_data_x_normed = torch.div(generated_data_x,generated_data_norm)
                    generated_data_y_normed = torch.div(generated_data_y,generated_data_norm)
                    generated_angles = torch.arctan2(generated_data_y_normed,generated_data_x_normed)
                    if further_simulated == True:

                        generated_angles_add_sim_list = []
                        for add_sim_idx,spin_angles in enumerate(generated_angles):
                            spin_system = XYSystem(temperature = temperature,defect_config=full_defect_list[idx][add_sim_idx].cpu().numpy().reshape(lattice_size*lattice_size),input_analytic_solution=spin_angles.cpu().numpy().reshape(lattice_size*lattice_size),width = lattice_size)
                            spin_system.set_input_configuration()
                            spin_system.multible_sweep_alternative(10)
                            spin_after_add_sim = torch.from_numpy(spin_system.spin_config).view(lattice_size,lattice_size).cuda()
                            generated_angles_add_sim_list.append(spin_after_add_sim)
                        generated_angles_list_per_epoch.append(torch.stack(generated_angles_add_sim_list,dim=0))
                    else:
                        generated_angles_list_per_epoch.append(generated_angles)

                    generated_defects_list_per_epoch.append(input_defects)
    

                oservable_data_spins.append(generated_angles_list_per_epoch)
                oservable_data_defects.append(generated_defects_list_per_epoch)



        elif data_type == 'full_data':

            print(samplesize)

            spin_data_prefix = 'full_cluster_spin_data__'
            defect_data_prefix = 'full_cluster_defect_data_'

            path_name = 'not_fixed_data_full_temp_{}/'.format(temperature)

            spin_configurations = []
            defect_configurations = []
            temperatures = []
            
            for count in range(50):
                training_angles = pd.read_csv(data_path + spin_data_prefix + '{0}_{1}_{2}.csv'.format(temperature,lattice_size, count), delimiter = ',', skipinitialspace = True , header = 0)
                training_defects = pd.read_csv(data_path + defect_data_prefix + '{0}_{1}_{2}.csv'.format(temperature,lattice_size, count), delimiter = ',', skipinitialspace = True , header = 0)

                spin_configuration = [training_angles['Spin_angles{0}'.format(i)] for i in range(1000)]
                defect_configuration = [training_defects['Defects{0}'.format(i)] for i in range(1000)]

                spin_configurations.append(np.array(spin_configuration).reshape((1000,-1)))
                defect_configurations.append(np.array(defect_configuration).reshape((1000,-1)))
                temperatures.append(np.array([temperature for i in range(1000)]).reshape((1000,-1)))


            full_index = [i for i in range(50000)]
            sample_index = sorted(random.sample(full_index,samplesize))
            spin_configurations = np.array(spin_configurations).reshape((50000,-1))
            defect_configurations = np.array(defect_configurations).reshape((50000,-1))
            temperatures = np.array(temperatures).reshape((50000))

            full_data_spins = torch.from_numpy(spin_configurations[sample_index]).to(device).view(-1,lattice_size,lattice_size)
            full_data_defects = torch.from_numpy(defect_configurations[sample_index]).to(device).view(-1,lattice_size,lattice_size)

            #print(full_data_defects[0])

            defect_number_tensor = torch.sum(torch.abs(full_data_defects).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)



            name_list = ['full']

            oservable_data_spins = [[full_data_spins]]
            oservable_data_defects = [[full_data_defects]]



            print('finish load')



        elif data_type == 'random_data':

            path_name = 'random_data_comparison/'

            oservable_data_spins = []

            oservable_data_defects = []

            for idx, name in enumerate(name_list):
                
                oservable_data_spins.append([(torch.rand((samplesize,lattice_size,lattice_size)))*2.*np.pi])

                oservable_data_defects.append([full_defect_list[idx]])

        elif data_type == 'zero_temp':

            path_name = 'zero_temperature_solution_comparison_new/'

            oservable_data_spins = []

            oservable_data_defects = []

            lattice_spacing = 1

            spin_positions = np.array([[[i*lattice_spacing,j*lattice_spacing] for j in range(lattice_size)] for i in range(lattice_size)])

            for input_defects in full_defect_list:

                analys_sol_per_defects = []

                for defects in input_defects:
                    analy_spins = torch.from_numpy(analysitcally_calculation(spin_position=spin_positions,defects=defects.cpu().numpy(),lattice_size=lattice_size)).to(device)
                    analys_sol_per_defects.append(analy_spins)
                oservable_data_spins.append([torch.stack(analys_sol_per_defects,dim=0)])
                oservable_data_defects.append([input_defects])


        output_vorticity_list_full = []
        output_defect_lattice_list_full = []
        output_defect_diff_list_full = []
        output_magentisation_list_full = []
        output_energy_list_full = []
        output_magnet_suscep_list_full = []
        output_magnet_suscep_alternative_list_full = []
        output_specific_heat_list_full = []
        output_helicity_modulus_list_full = []
        output_magnet_suscep_error_list_full = []
        output_magnet_suscep_alternative_error_list_full = []
        output_specific_heat_error_list_full = []
        output_helicity_modulus_error_list_full =[]
        output_vorticity_error_list_full = []
        output_defect_diff_error_list_full = []
        output_magentisation_error_list_full = []
        output_energy_error_list_full = []
        output_corr_mean_list_full = []
        output_corr_var_list_full = []


        output_vorticity_full_list = []
        output_magentisation_full_list = []
        output_energy_full_list = []
        output_defect_diff_full_list = []
        output_correlationfunction_full_list = []

        output_energy_distribution_local_full = []
        
        #name_list = [4.0]


        for idx,types in enumerate(name_list):
            
            ##print('check_defect_nmb',defect_nmb,temperature, torch.all(torch.sum(oservable_data_defects[idx][0].view(-1,lattice_size*lattice_size),dim=1)==0.))

            #print('check',oservable_data_spins[idx][0].size(0))





            ##print(types,oservable_data_spins[idx][0].size(0))

            if int(oservable_data_spins[idx][0].size(0)) == 10000:
                    sample_nmb = 45
                    sample_size = 200
            elif int(oservable_data_spins[idx][0].size(0)) == 1000:
                    sample_nmb = 9
                    sample_size = 100
            elif int(oservable_data_spins[idx][0].size(0)) == 900:
                    sample_nmb = 8
                    sample_size = 100
            elif int(oservable_data_spins[idx][0].size(0)) == 5000:
                    sample_nmb = 35
                    sample_size = 140
            elif int(oservable_data_spins[idx][0].size(0)) == 50000:
                    sample_nmb = 98
                    sample_size = 500
            elif int(oservable_data_spins[idx][0].size(0)) == 20000:
                    sample_nmb = 54
                    sample_size = 350
            elif int(oservable_data_spins[idx][0].size(0)) == 30000:
                    sample_nmb = 64
                    sample_size = 450

            defect_number_tensor = torch.sum(torch.abs(oservable_data_defects[idx][0]).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)
            unique_defect_lattices = torch.unique(oservable_data_defects[idx][0], dim=0)
  
            unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)
            simulation_observables = Observables(lattice_size=lattice_size, temperature = temperature)
            
            print(maximal_defect_nmb)
            simulation_observables.set_full_defect_positions(defect_lattices=oservable_data_defects[idx][0],maximal_defect_nmb=maximal_defect_nmb)
            



            output_vorticity_list = []
            output_defect_lattice_list = []
            output_defect_diff_list = []
            output_magentisation_list = []
            output_energy_list = []
            output_magnet_suscep_list = []
            output_magnet_suscep_alternative_list = []
            output_specific_heat_list = []
            output_helicity_modulus_list = []
            output_magnet_suscep_error_list = []
            output_magnet_suscep_alternative_error_list = []
            output_specific_heat_error_list = []
            output_helicity_modulus_error_list =[]
            output_vorticity_error_list = []
            output_defect_diff_error_list = []
            output_magentisation_error_list = []
            output_energy_error_list = []
            output_corr_mean_list = []
            output_corr_var_list = []

            output_vorticity_full_list_per_name = []
            output_magentisation_full_list_per_name = []
            output_energy_full_list_per_name = []
            output_defect_diff_full_list_per_name = []
            output_correlationfunction_full_list_per_name = []

            output_energy_distribution_local_per_name = []




            for e_idx,epoch in enumerate(epochs):
                print(e_idx)

                defect_lattices, vorticity, vorticity_error, full_vorticity = simulation_observables.faster_vorticity(spin_lattice_tensor=oservable_data_spins[idx][e_idx])

                #print('check v')
                magentisation, magentisation_error, full_magentisation = simulation_observables.faster_mean_magentisation(oservable_data_spins[idx][e_idx])

                #print('check m')
                magentic_suscep = simulation_observables.faster_magnet_suscep()
                magentic_suscep_alternative = magentic_suscep#simulation_observables.check_sucept_corr_function(oservable_data_spins[idx][e_idx])

                energy, energy_error, full_energy = simulation_observables.faster_mean_energy(oservable_data_spins[idx][e_idx])

                energy_distribution_local = simulation_observables.faster_mean_local_energy(oservable_data_spins[idx][e_idx])

                output_energy_distribution_local_per_name.append(energy_distribution_local)

                #print('check e')
                specific_heat = simulation_observables.faster_specfifc_heat()
            
                simulation_observables.set_defects_lattices(sim_defect_lattices=oservable_data_defects[idx][e_idx], gen_defect_lattices=defect_lattices)
                defect_diff, defect_diff_error, full_defect_diff = simulation_observables.defect_distance()
                
                #print(np.nonzero(full_defect_diff > 0.0))
                
                ##if any(defect_diff > 0.0):
                 
                #continue   
            
                helicity_modulus = simulation_observables.faster_get_helicity_modulus(oservable_data_spins[idx][e_idx])

                observables_tensor = [specific_heat,magentic_suscep,magentic_suscep_alternative,helicity_modulus]

                sh_error,ms_error,ms_alternative_error,hm_error = simulation_observables.faster_error_calulcation(full_observable_tensor_list=observables_tensor,full_spin_tensor=oservable_data_spins[idx][e_idx], sample_nmb=sample_nmb,sample_size=sample_size)

                #print(e_idx)

                data_correlation_function = simulation_observables.correlation_function_even_faster_new_try(spin_fields=oservable_data_spins[idx][e_idx])

                print('check corr')

                #print(e_idx)
                
                distance_parameter = [float(key) for key in data_correlation_function.keys()]

                distance_parameter.sort() 

                mean_correlation_data_sorted = [torch.flatten(torch.mean(data_correlation_function['{}'.format(k)])) for k in distance_parameter]

                full_correlation_data_sorted = [data_correlation_function['{}'.format(k)] for k in distance_parameter]

                var_correlation_data_sorted = [torch.flatten(torch.mean(torch.pow(data_correlation_function['{}'.format(k)],2.0))-torch.pow(torch.mean(data_correlation_function['{}'.format(k)]),2.0)) for k in distance_parameter]

                output_vorticity_list.append(vorticity)
                output_defect_lattice_list.append(defect_lattices)
                output_defect_diff_list.append(defect_diff)
                output_magentisation_list.append(magentisation)
                output_energy_list.append(energy)
                output_magnet_suscep_list.append(magentic_suscep)
                output_magnet_suscep_alternative_list.append(magentic_suscep_alternative)
                output_specific_heat_list.append(specific_heat)
                output_helicity_modulus_list.append(helicity_modulus)
                output_magnet_suscep_error_list.append(ms_error)
                output_magnet_suscep_alternative_error_list.append(ms_alternative_error)
                output_specific_heat_error_list.append(sh_error)
                output_helicity_modulus_error_list.append(hm_error)
                output_vorticity_error_list.append(vorticity_error)
                output_defect_diff_error_list.append(defect_diff_error)
                output_magentisation_error_list.append(magentisation_error)
                output_energy_error_list.append(energy_error)
                output_corr_mean_list.append(torch.stack(mean_correlation_data_sorted, dim=0))
                output_corr_var_list.append(torch.stack(var_correlation_data_sorted, dim =0))



                output_vorticity_full_list_per_name.append(full_vorticity)
                output_magentisation_full_list_per_name.append(full_magentisation)
                output_energy_full_list_per_name.append(full_energy)
                output_defect_diff_full_list_per_name.append(full_defect_diff)
                output_correlationfunction_full_list_per_name.append(torch.stack(full_correlation_data_sorted, dim=0))
            #print(output_vorticity_list)
            output_vorticity_list_full.append(torch.stack(output_vorticity_list, dim =0))
            output_defect_lattice_list_full.append(torch.stack(output_defect_lattice_list, dim =0))
            output_defect_diff_list_full.append(torch.stack(output_defect_diff_list, dim =0))
            output_magentisation_list_full.append(torch.stack(output_magentisation_list, dim =0))
            output_energy_list_full.append(torch.stack(output_energy_list, dim =0))
            output_magnet_suscep_list_full.append(torch.stack(output_magnet_suscep_list, dim =0))
            output_magnet_suscep_alternative_list_full.append(torch.stack(output_magnet_suscep_alternative_list, dim =0))
            output_specific_heat_list_full.append(torch.stack(output_specific_heat_list, dim =0))
            output_helicity_modulus_list_full.append(torch.stack(output_helicity_modulus_list, dim =0))
            output_magnet_suscep_error_list_full.append(torch.stack(output_magnet_suscep_error_list, dim =0))
            output_magnet_suscep_alternative_error_list_full.append(torch.stack(output_magnet_suscep_alternative_error_list, dim =0))
            output_specific_heat_error_list_full.append(torch.stack(output_specific_heat_error_list, dim =0))
            output_helicity_modulus_error_list_full.append(torch.stack(output_helicity_modulus_error_list, dim =0))
            output_vorticity_error_list_full.append(torch.stack(output_vorticity_error_list, dim =0))
            output_defect_diff_error_list_full.append(torch.stack(output_defect_diff_error_list, dim =0))
            output_magentisation_error_list_full.append(torch.stack(output_magentisation_error_list, dim =0))
            output_energy_error_list_full.append(torch.stack(output_energy_error_list, dim =0))
            output_corr_mean_list_full.append(torch.stack(output_corr_mean_list, dim =0))
            output_corr_var_list_full.append(torch.stack(output_corr_var_list, dim =0))

            output_vorticity_full_list.append(torch.stack(output_vorticity_full_list_per_name, dim =0))
            output_magentisation_full_list.append(torch.stack(output_magentisation_full_list_per_name, dim =0))
            output_energy_full_list.append(torch.stack(output_energy_full_list_per_name, dim =0))
            output_defect_diff_full_list.append(torch.stack(output_defect_diff_full_list_per_name, dim =0))
            output_correlationfunction_full_list.append(torch.stack(output_correlationfunction_full_list_per_name, dim =0))

            output_energy_distribution_local_full.append(torch.stack(output_energy_distribution_local_per_name,dim=0))




        outname = 'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_batch_gen.csv'.format(data_type,sim_data_type,nmb_distances,max_distances, defect_nmb,samplesize,noise_size,1)

        outname_full = 'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_full_not_mean_compare_{}_batch_gen.h5'.format(data_type,sim_data_type,nmb_distances,max_distances, defect_nmb,samplesize,noise_size,1)

        correspodning_data_outname = 'more_data_basis_data_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_batch_gen.h5'.format(data_type,sim_data_type,nmb_distances,max_distances, defect_nmb,samplesize,noise_size,1)

        os.makedirs(output_data_path+ path_name, exist_ok=True)

        data_observables = {'Epoch':np.array(epochs).flatten()}

        magnetisation_mean = {}

        magnetisation_error = {}

        vorticity_mean = {}

        vorticity_error = {}

        defect_diff_mean = {}

        defect_diff_error = {}

        energy_mean = {}

        energy_error = {}

        ms_mean = {}

        ms_alternative_mean = {}

        ms_error = {}

        ms_alternative_error = {}

        hm_mean = {}

        hm_error = {}

        sh_mean = {}

        sh_error = {}

        correlation_function_mean = {}

        correlation_function_var = {}
        
        print(torch.stack(output_magentisation_list_full, dim=0).shape)

        for i,name in enumerate(name_list):

            magnetisation_mean.update({'Mag_mean_{}'.format(name):torch.flatten(torch.stack(output_magentisation_list_full, dim=0)[i]).cpu().numpy()})
            magnetisation_error.update({'Mag_error_{}'.format(name):torch.flatten(torch.stack(output_magentisation_error_list_full, dim=0)[i]).cpu().numpy()})
            vorticity_mean.update({'Vor_mean_{}'.format(name):torch.flatten(torch.stack(output_vorticity_list_full, dim=0)[i]).cpu().numpy()})
            vorticity_error.update({'Vor_error_{}'.format(name):torch.flatten(torch.stack(output_vorticity_error_list_full, dim=0)[i]).cpu().numpy()})
            defect_diff_mean.update({'Defect_Diff_mean_{}'.format(name):torch.flatten(torch.stack(output_defect_diff_list_full, dim=0)[i]).cpu().numpy()})
            defect_diff_error.update({'Defect_Diff_error_{}'.format(name):torch.flatten(torch.stack(output_defect_diff_error_list_full, dim=0)[i]).cpu().numpy()})
            energy_mean.update({'Energy_mean_{}'.format(name):torch.flatten(torch.stack(output_energy_list_full, dim=0)[i]).cpu().numpy()})
            energy_error.update({'Energy_error_{}'.format(name):torch.flatten(torch.stack(output_energy_error_list_full, dim=0)[i]).cpu().numpy()})
            ms_mean.update({'MS_mean_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_list_full, dim=0)[i]).cpu().numpy()})
            ms_alternative_mean.update({'MS_alternative_mean_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_alternative_list_full, dim=0)[i]).cpu().numpy()})
            ms_error.update({'MS_error_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_error_list_full, dim=0)[i]).cpu().numpy()})
            ms_alternative_error.update({'MS_alternative_error_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_alternative_error_list_full, dim=0)[i]).cpu().numpy()})
            hm_mean.update({'HM_mean_{}'.format(name):torch.flatten(torch.stack(output_helicity_modulus_list_full, dim=0)[i]).cpu().numpy()})
            hm_error.update({'HM_error_{}'.format(name):torch.flatten(torch.stack(output_helicity_modulus_error_list_full, dim=0)[i]).cpu().numpy()})
            sh_mean.update({'SH_mean_{}'.format(name):torch.flatten(torch.stack(output_specific_heat_list_full, dim=0)[i]).cpu().numpy()})
            sh_error.update({'SH_error_{}'.format(name):torch.flatten(torch.stack(output_specific_heat_error_list_full, dim=0)[i]).cpu().numpy()})
            correlation_function_mean.update({'Corr_mean_{}_{}'.format(dist,name):torch.flatten(torch.stack(output_corr_mean_list_full, dim=0)[i,:,k]).cpu().numpy() for k,dist in enumerate(distance_parameter)})
            correlation_function_var.update({'Corr_var_{}_{}'.format(dist,name):torch.flatten(torch.stack(output_corr_var_list_full, dim=0)[i,:,k]).cpu().numpy() for k,dist in enumerate(distance_parameter)})

        data_observables.update(magnetisation_mean)
        data_observables.update(magnetisation_error)
        data_observables.update(vorticity_mean)
        data_observables.update(vorticity_error)
        data_observables.update(defect_diff_mean)
        data_observables.update(defect_diff_error)
        data_observables.update(energy_mean)
        data_observables.update(energy_error)
        data_observables.update(ms_mean)
        data_observables.update(ms_error)
        data_observables.update(ms_alternative_mean)
        data_observables.update(ms_alternative_error)
        data_observables.update(hm_mean)
        data_observables.update(hm_error)
        data_observables.update(sh_mean)
        data_observables.update(sh_error)
        data_observables.update(correlation_function_mean)
        data_observables.update(correlation_function_var)

        data_observables_data_basis = {}


        



        df_type_observables = pd.DataFrame.from_dict(data_observables)
        
        df_type_observables.to_csv(output_data_path+path_name +outname, index= False)

        #print('distance',name_list)

        with h5py.File(output_data_path+path_name +correspodning_data_outname,'w') as h5f:
            if data_type != 'full_data':
                dataset_distances = h5f.create_dataset('distances', shape=(int(len(name_list)),), dtype='float')
                if new_data:
                    dataset_distances[:] = np.array([dist for dist in name_list])#
                else :
                    dataset_distances[:] = np.array([dist for dist in name_list])

            for dist_idx,dist in enumerate(name_list):

                    #print(oservable_data_spins[dist_idx])

                dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(int(len(epochs)),samplesize,lattice_size*lattice_size), dtype='float')
                dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(int(len(epochs)),samplesize,lattice_size*lattice_size), dtype='float')

                for e_dix,epoch in enumerate(epochs):

                        #print(oservable_data_spins[dist_idx][e_dix].size())


                    
                    dataset_spins_compare[e_dix] = oservable_data_spins[dist_idx][e_dix].view(1,-1,lattice_size*lattice_size).cpu().numpy()
                        
                    dataset_defects_compare[e_dix] = oservable_data_defects[dist_idx][e_dix].view(1,-1,lattice_size*lattice_size).cpu().numpy()

        with h5py.File(output_data_path+path_name +outname_full,'w') as h5f:

                    

            for dist_idx,dist in enumerate(name_list):

                print(dist)

                    #print(oservable_data_spins[dist_idx])

                dataset_vorticity_compare = h5f.create_dataset('vorticity_{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_magent_compare = h5f.create_dataset('magentisation{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_energy_compare = h5f.create_dataset('energy_{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_defect_diff_compare = h5f.create_dataset('defect_diff_{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_corr_compare = h5f.create_dataset('corr_funct_{}'.format(dist), shape=(int(len(epochs)),samplesize,int(len(distance_parameter))), dtype='float')

                dataset_energy_local = h5f.create_dataset('local_energy_{}'.format(dist), shape=(int(len(epochs)),samplesize,lattice_size*lattice_size), dtype='float')

                for e_dix,epoch in enumerate(epochs):

                        #print(output_correlationfunction_full_list[dist_idx][e_dix].size())


                    dataset_energy_local[e_dix] = output_energy_distribution_local_full[dist_idx][e_dix].view(1,-1,lattice_size*lattice_size).cpu().numpy()
                    dataset_vorticity_compare[e_dix] = output_vorticity_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                    dataset_magent_compare[e_dix] = output_magentisation_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                    dataset_energy_compare[e_dix] = output_energy_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                    dataset_defect_diff_compare[e_dix] = output_defect_diff_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                        #print('check')
                    dataset_corr_compare[e_dix] = output_correlationfunction_full_list[dist_idx][e_dix].view(1,-1,int(len(distance_parameter))).cpu().numpy()
                        



if __name__ == "__main__":
    pass
