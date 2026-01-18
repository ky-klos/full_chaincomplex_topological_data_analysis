from filtration_function import TopologicalAnalysis
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import random
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerLine2D
import math
from scipy.optimize import curve_fit
from numpy import pi
import colorcet as cc
from matplotlib.cm import get_cmap
from colorsys import hls_to_rgb
import matplotlib.colors

from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline ~ magic function backend for IPython : output of plotting
#commands displayed inline within frontends directly below code, that it produced

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

import h5py

import pickle
import pandas as pd

from matplotlib.collections import LineCollection

import seaborn as sb

import plotly.graph_objects as go

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import random
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerLine2D
import math
from scipy.optimize import curve_fit
from numpy import pi

import time

import os

lattice_spacing = 1
lattice_size = 16
two_pi = 2.0*np.pi

  
def vortex_number(defects,lattice_size:int) -> int:
    defect_array = defects.reshape(lattice_size*lattice_size)
    defect_counter = 0
    vortex_counter = 0
    antivortex_counter = 0

    for vortex in defect_array:
        if vortex == 1 :
            vortex_counter += 1
            defect_counter += 1
        elif vortex == -1:
            antivortex_counter += 1
            defect_counter += 1

    if vortex_counter != antivortex_counter:
        defect_counter = 0

    return defect_counter

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

    return analytic_solution


def load_in_simulated_data(device,start_additional_counter,lattice_Size = 16,temperature=0.1,temp_range=9, number_of_defect_configs_start =0, number_of_defect_configs=50,additional_counter=50, number_of_MC_swips_start= 0,number_of_MC_swips_end=100, defect_data_prefix='after_translation_fixed_my_test_defect_data_compare_', spin_data_prefix ='after_translation_fixed_my_test_spin_data_compare_', data_path = '/localscratch/kyklos/Pytorch_cuda_code/Full_training_data/Fixed_defects/'):
    temp = temperature
    spin_configurations= []
    defect_configurations = []
    temperatures = []


    for t in range(temp_range):
        #print(t)

        temp = np.around(temp, decimals=2)

        for k in range(start_additional_counter,additional_counter):
            for count in range(number_of_defect_configs_start,number_of_defect_configs):
                training_angles = pd.read_csv(data_path + spin_data_prefix + '{0}_{1}_{2}_{3}.csv'.format(temp,lattice_Size, count, k), delimiter = ',', skipinitialspace = True , header = 0)
                training_defects = pd.read_csv(data_path + defect_data_prefix + '{0}_{1}_{2}_{3}.csv'.format(temp,lattice_Size, count, k), delimiter = ',', skipinitialspace = True , header = 0)

                spin_configuration = [training_angles['Spin_angles{0}'.format(i)] for i in range(number_of_MC_swips_start,number_of_MC_swips_end)]
                defect_configuration = [training_defects['Defects{0}'.format(i)] for i in range(number_of_MC_swips_start,number_of_MC_swips_end)]

                spin_configurations.append(spin_configuration)
                defect_configurations.append(defect_configuration)
                temperatures.append([temp for i in range((number_of_MC_swips_end-number_of_MC_swips_start))])
                #if temp <= 0.3 :
                #    temperatures.append([0.1 for i in range((number_of_MC_swips_end-number_of_MC_swips_start))])
                #elif temp >0.3 and temp<=0.6:
                #    temperatures.append([0.2 for i in range((number_of_MC_swips_end-number_of_MC_swips_start))])
                #elif temp > 0.6 and temp <= 0.9:
                #    temperatures.append([0.3 for i in range((number_of_MC_swips_end-number_of_MC_swips_start))])

        temp += 0.1

    spins_array = np.array(spin_configurations)
    defects_array = np.array(defect_configurations)
    temps_array = np.array(temperatures)

    data_number = temp_range*(number_of_MC_swips_end-number_of_MC_swips_start)*number_of_defect_configs*(additional_counter-start_additional_counter)

    spins_real = spins_array.reshape(-1,lattice_Size,lattice_Size)
    defects = defects_array.reshape(-1,lattice_Size,lattice_Size)
    temps = temps_array.reshape(-1,1)


    #spin_config_tensor = torch.from_numpy(spins_real).float().to(device)#traget_image
    #defect_config_tensor = torch.from_numpy(defects).float().to(device)#input_image
    #correspond_temp_labels_tensor= torch.from_numpy(temps).float().to(device)

    return spins_real,defects, temps


spin_configurations = []
defect_configurations = []
temperatures = []
temp_1 = 0.1
temp_2 = 0.1
temp_3 = 0.1
temp_4 = 0.1
spin_configurations = []
defect_configurations = []
temperatures = []
spin_configurations_2 = []
defect_configurations_2 = []
temperatures_2 = []
spin_configurations_3 = []
defect_configurations_3 = []
temperatures_3 = []
spin_configurations_4 = []
defect_configurations_4 = []
temperatures_4 = []



#spin_positions = np.asarray([[[i*lattice_spacing,j*lattice_spacing] for j in range(lattice_size)] for i in range(lattice_size)])
#defect_lattice =  np.zeros([lattice_size,lattice_size])
#defect_lattice[50,47]=1
#defect_lattice[50,53]=-1

#testing_analytic = analysitcally_calculation(spin_position=spin_positions,defects=defect_lattice,lattice_size=lattice_size)
#print(testing_analytic)

#random_spins = np.random.random(lattice_size*lattice_size)*two_pi
##topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size)



state_percolations_analytics = []
bond_percolations_analytics = []
state_percolations_random = []
bond_percolations_random = []
edge_x_matrices = []
edge_y_matrices= []
edge_x_matrices_rand = []
edge_y_matrices_rand = []
plaquette_matrices = []
plaquette_matrices_rand = []

plaquette_matrices_full = []
edge_x_matrices_full = []
edge_y_matrices_full= []

temp =0.1

defects_numer=50
##os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'



def check_defect_type_config_small(simulated_data_path,temperature, lattice_size,device='cpu'):
    with torch.no_grad():

        

        with h5py.File(simulated_data_path+'full_training_data_gan_{}_corrected.h5'.format(temperature), 'r') as h5f:
            sim_spins = torch.from_numpy(h5f['spins'][350000:430000]).to(device).view(-1,lattice_size,lattice_size)
            sim_defects = torch.from_numpy(h5f['defects'][350000:430000]).to(device).view(-1,lattice_size,lattice_size)


        print('check input')

        
        

        defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
        maximal_defect_nmb=torch.max(defect_number_tensor)
        ##print(maximal_defect_nmb)
        unique_defect_lattices = torch.unique(sim_defects, dim=0)
        ##print(unique_defect_lattices.size())
        unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)

        topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


        topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
        defect_distances = topological_analysis_tool.return_full_defect_distances()

        ##make  uniform distributed data of the following type:
        no_distances = defect_distances== 0.0
        small_distances = torch.logical_and(defect_distances > 0.0 ,defect_distances <= 2.0 )
        mid_distances = torch.logical_and(defect_distances > 2.0 ,defect_distances < 5.0 )
        large_distances = defect_distances == 5.0 

        full_sim_data_index = [i for i in range(int(sim_defects.size(0)))]
        no_defect_sim_data_index = [i for i in range(int(sim_defects[no_distances].size(0)))]
        small_defect_sim_data_index = [i for i in range(int(sim_defects[small_distances].size(0)))]
        mid_defect_sim_data_index = [i for i in range(int(sim_defects[mid_distances].size(0)))]
        large_defect_sim_data_index = [i for i in range(int(sim_defects[large_distances].size(0)))]

        no_spins = sim_spins[no_distances]
        small_spins = sim_spins[small_distances]
        mid_spins = sim_spins[mid_distances]
        large_spins = sim_spins[large_distances]

        no_defect_spins = sim_defects[no_distances]
        small_defect_spins = sim_defects[small_distances]
        mid_defect_spins = sim_defects[mid_distances]
        large_defect_spins = sim_defects[large_distances]

        ##print('no', sim_defects[no_distances].size())
        ##print('small',sim_defects[small_distances].size())
        ##print('mid',sim_defects[mid_distances].size())
        ##print('large',sim_defects[large_distances].size())



        ##small_def = sim_defects[small_distances]
        ##unique_small_defect_lattices = torch.unique(small_def, dim=0)
        ##print('small unique',unique_small_defect_lattices.size())

        ##mid_def = sim_defects[mid_distances]
        ##unique_mid_defect_lattices = torch.unique(mid_def, dim=0)
        ##print('mid unique',unique_mid_defect_lattices.size())

        ##large_def = sim_defects[large_distances]
        ##unique_large_defect_lattices = torch.unique(large_def, dim=0)
        ##print('large unique',unique_large_defect_lattices.size())




        ##counting = torch.unique(defect_distances,return_counts= True)
        ##print(defect_distances)
        ##print(counting)

        ##here comes the random sampling :

        full_sampling_idx = random.sample(full_sim_data_index,10000)
        no_defect_sampling_idx = random.sample(no_defect_sim_data_index,1000)
        small_defect_sampling_idx = random.sample(small_defect_sim_data_index,1000)
        mid_defect_sampling_idx = random.sample(mid_defect_sim_data_index,1000)
        large_defect_sampling_idx = random.sample(large_defect_sim_data_index,1000)

        full_sample_spin = sim_spins[full_sampling_idx]
        full_sample_defect = sim_defects[full_sampling_idx]
        no_defects_sample_spin = no_spins[no_defect_sampling_idx]
        no_defects_sample_defect = no_defect_spins[no_defect_sampling_idx]
        small_defects_sample_spin = small_spins[small_defect_sampling_idx]
        small_defects_sample_defect = small_defect_spins[small_defect_sampling_idx]
        mid_defects_sample_spin = mid_spins[mid_defect_sampling_idx]
        mid_defects_sample_defect = mid_defect_spins[mid_defect_sampling_idx]
        large_defects_sample_spin = large_spins[large_defect_sampling_idx]
        large_defects_sample_defect = large_defect_spins[large_defect_sampling_idx]

        print(no_defects_sample_spin.size())

        full_spin_list = [full_sample_spin,no_defects_sample_spin,small_defects_sample_spin,mid_defects_sample_spin,large_defects_sample_spin]

        full_defect_list = [full_sample_defect,no_defects_sample_defect,small_defects_sample_defect,mid_defects_sample_defect,large_defects_sample_defect]

        return full_spin_list, full_defect_list
        #random.sample(,1000)


def check_defect_type_config(simulated_data_path,temperature, lattice_size,device='cpu', samplesize = 1000):
    with torch.no_grad():

        

        with h5py.File(simulated_data_path+'full_training_data_gan_{}_corrected.h5'.format(temperature), 'r') as h5f:
            sim_spins = torch.from_numpy(h5f['spins'][350000:700000]).to(device).view(-1,lattice_size,lattice_size)
            sim_defects = torch.from_numpy(h5f['defects'][350000:700000]).to(device).view(-1,lattice_size,lattice_size)


            print('check input')

        
        

            defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)
            ##print(maximal_defect_nmb)
            ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
            ##print(unique_defect_lattices.size())
            ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)

            topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


            topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
            defect_distances = topological_analysis_tool.return_full_defect_distances()

            print(defect_distances.size())


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

            ##print(sizes)


            return distances_dictionary_spins, distances_dictionary_defects
    

def output_analysis_basis(simulated_data_path,temperature,lattice_size,device='cpu',samplesize=1000):

    spin_dict, defect_dict = check_defect_type_config(simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,device=device,samplesize=samplesize)
    #name_list = ['full', 'no_defect', 'small_defects', 'mid_defects', 'large_defects']


    with h5py.File(simulated_data_path + 'defect_for_analysis_full_distances_{}.h5'.format(temperature),'w') as h5f:

        dataset_distances = h5f.create_dataset('distances', shape=(int(len(spin_dict)),), dtype='float')
        dataset_distances[:] = np.array([dist for dist in spin_dict])
            

        for dist in spin_dict:


            dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            dataset_spins_compare[:] = spin_dict[dist].view(-1,lattice_size*lattice_size).numpy()
            dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            dataset_defects_compare[:] = defect_dict[dist].view(-1,lattice_size*lattice_size).numpy()

    return spin_dict, defect_dict

##check_defect_type_config(simulated_data_path='/localscratch/kklos/XY_GAN_transport_folder/',epoch=50, generated_data_attributes=[64,0.001,0,5,1,1.0,0], temperature=0.1, lattice_size=lattice_size, further_simulated=False,device='cpu')

##same defects for sim and gen for different gens 

def get_edge_state_graphs_sim_gen(res_criritc,extra_info,output_data_path,data_path, simulation_data_path, generated_data_attributes, nmb_label,data_type,temperature,sim_data_type='full_pinned',nmb_defects =2,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= True, further_simulated=False,device='cpu',samplesize = 1000):
    with torch.no_grad():

        
        if new_data:
            spin_dict, defect_dict = output_analysis_basis(simulated_data_path=simulation_data_path,temperature=temperature, lattice_size=lattice_size,device=device,samplesize=samplesize)
            full_spin_list = []
            full_defect_list = []
            name_list = []
            for dist in spin_dict:
                full_spin_list.append(spin_dict[dist])
                full_defect_list.append(defect_dict[dist])
                name_list.append(dist)
        else:
            full_spin_list = []
            full_defect_list = []

            #name_list = return_defect_distances(maximal_distance=maximal_distance)
            if nmb_defects > 2 or lattice_size > 16 or temperature > 0.1:##sim_data_type,
                with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,nmb_defects,samplesize), 'r') as h5f:
                    name_list = np.array(h5f['distances'])
                            
                    for dist in name_list:
                        sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                        full_spin_list.append(sim_spins)
                        full_defect_list.append(sim_defects)
            
            else:
                with h5py.File(simulation_data_path+ 'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,nmb_defects,samplesize), 'r') as h5f:

                    name_list = np.array(h5f['distances'])
                            
                    for dist in name_list:
                        sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                        full_spin_list.append(sim_spins)
                        full_defect_list.append(sim_defects)

        print(name_list)

        nmb_distances = int(len(name_list))
        max_distances = max(name_list)

        if data_type == 'training_data':


            oservable_data_spins = []

            oservable_data_defects = []

            path_name = 'training_data_full_distance_temp_{}_new_{}/'.format(temperature,sim_data_type)

            for idx, name in enumerate(name_list):
                oservable_data_spins.append(full_spin_list[idx])
                oservable_data_defects.append(full_defect_list[idx])


        elif data_type == 'generated_data':

            oservable_data_spins = []
            oservable_data_defects = []

            ##path_name = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(noise_size,training_data_nmb,*generated_data_attributes,0,*extra_info) 
            path_name = 'gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*extra_info)
            if res_criritc:
                path_name += 'res_critic/'
            
            path_name += 'temp_{}/'.format(temperature)

            if new_data:
                if nmb_label == 0:
                    temperature_label = None
                elif nmb_label == 1 or nmb_label == 3:

                    if temperature <= 0.3:
                        factor = 0.
                    elif temperature <= 0.6:
                        factor = 1.
                    else:
                        factor  = 2.0

                    temperature_label = torch.ones(samplesize).view(-1,1)*factor

                else:
                    temperature_label = torch.ones(samplesize).view(-1,1)*temperature*10.

                for input_defects in full_defect_list:

                        
                    gen_model = torch.load(data_path 
                                            + path_name
                                            + 'Generator_check_{}.pth'.format(80),map_location=torch.device('cpu'))
                    generated_data = gen_model(input_defects.float().view(-1,1,lattice_size,lattice_size),label=temperature_label)
                                ##print(generated_data.size())
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
                        oservable_data_spins.append(torch.stack(generated_angles_add_sim_list,dim=0))
                    else:
                        oservable_data_spins.append(generated_angles)

                    oservable_data_defects.append(input_defects)
            else:

                with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_batch_gen.h5'.format(data_type,sim_data_type,nmb_distances,max_distances,nmb_defects,samplesize,noise_size,1), 'r') as h5f:

                        
                    for dist in name_list:
                        spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][-1]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][-1]).to(device).view(-1,lattice_size,lattice_size)

                        oservable_data_spins.append(spins)
                        oservable_data_defects.append(defects)

    



        elif data_type == 'full_data':

            spin_data_prefix = 'full_cluster_spin_data__'
            defect_data_prefix = 'full_cluster_defect_data_'

            path_name = 'not_fixed_data_full_temp_{}/'.format(temperature)

            spin_configurations = []
            defect_configurations = []
            temperatures = []
            
            for count in range(50):
                training_angles = pd.read_csv(data_path + spin_data_prefix + '{0}_{1}_{2}.csv'.format(temp,lattice_size, count), delimiter = ',', skipinitialspace = True , header = 0)
                training_defects = pd.read_csv(data_path + defect_data_prefix + '{0}_{1}_{2}.csv'.format(temp,lattice_size, count), delimiter = ',', skipinitialspace = True , header = 0)

                spin_configuration = [training_angles['Spin_angles{0}'.format(i)] for i in range(1000)]
                defect_configuration = [training_defects['Defects{0}'.format(i)] for i in range(1000)]

                spin_configurations.append(spin_configuration)
                defect_configurations.append(defect_configuration)
                temperatures.append([temp for i in range(1000)])


            full_index = [i for i in range(50000)]
            sample_index = sorted(random.sample(full_index,samplesize))
            spin_configurations = np.array(spin_configurations).reshape((50000,-1))
            defect_configurations = np.array(defect_configurations).reshape((50000,-1))
            temperatures = np.array(temperatures).reshape((50000))*10.

            full_data_spins = torch.from_numpy(spin_configurations[sample_index]).to(device).view(-1,lattice_size,lattice_size)
            full_data_defects = torch.from_numpy(defect_configurations[sample_index]).to(device).view(-1,lattice_size,lattice_size)

            defect_number_tensor = torch.sum(torch.abs(full_data_defects).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)

            print(maximal_defect_nmb)

            topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


            topological_analysis_tool.set_full_defect_positions(defect_lattices=full_data_defects,maximal_defect_nmb=maximal_defect_nmb)
            defect_distances = topological_analysis_tool.return_full_defect_distances()

            print(defect_distances.size())


            ##make  uniform distributed data of the following type:
            distances_dictionary_spins = {}
            distances_dictionary_defects = {}
            sizes = []
            for dist in torch.unique(defect_distances):
                ##print(dist)
                current_defect_distances = full_data_defects[defect_distances==dist]
                current_spin_distances = full_data_spins[defect_distances==dist]
                spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                sample_idx = random.sample(spin_data_index,samplesize)
                distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                sizes.append(int(sim_spins[spin_data_index].size(0)))
            full_spin_list = []
            full_defect_list = []
            name_list = []
            for dist in spin_dict:
                full_spin_list.append(torch.tensor(full_spin_list[dist]))
                full_defect_list.append(torch.tensor(defect_dict[dist]))
                name_list.append(dist)

            oservable_data_spins = full_spin_list
            oservable_data_defects = full_defect_list

        
            with h5py.File(output_data_path + 'defect_for_analysis_full_distances_not_fixed{}.h5'.format(temperature),'w') as h5f:

                dataset_distances = h5f.create_dataset('distances', shape=(int(len(spin_dict)),), dtype='float')
                dataset_distances[:] = np.array([dist for dist in spin_dict])
            

                for dist in spin_dict:


                    dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
                    dataset_spins_compare[:] = spin_dict[dist].view(-1,lattice_size*lattice_size).numpy()
                    dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
                    dataset_defects_compare[:] = defect_dict[dist].view(-1,lattice_size*lattice_size).numpy()



        elif data_type == 'random_data':

            path_name = 'random_data_comparison/'

            oservable_data_spins = []

            oservable_data_defects = []

            if new_data:

                for idx, name in enumerate(name_list):
                    
                    oservable_data_spins.append((torch.rand((samplesize,lattice_size,lattice_size)))*2.*np.pi)

                    oservable_data_defects.append(full_defect_list[idx])

            else:
                with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.h5'.format(data_type,nmb_distances,max_distances,nmb_defects,samplesize,noise_size,1), 'r') as h5f:

                        
                    for dist in name_list:
                        spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                        oservable_data_spins.append(spins)
                        oservable_data_defects.append(defects)

        elif data_type == 'zero_temp':

            path_name = 'zero_temperature_solution_comparison_new/'


            oservable_data_spins = []

            oservable_data_defects = []

            if new_data:

                lattice_spacing = 1

                spin_positions = np.array([[[i*lattice_spacing,j*lattice_spacing] for j in range(lattice_size)] for i in range(lattice_size)])

                for input_defects in full_defect_list:

                    analys_sol_per_defects = []

                    for defects in input_defects:
                        analy_spins = torch.from_numpy(analysitcally_calculation(spin_position=spin_positions,defects=defects.numpy(),lattice_size=lattice_size))
                        analys_sol_per_defects.append(analy_spins)
                    oservable_data_spins.append(torch.stack(analys_sol_per_defects,dim=0))
                    oservable_data_defects.append(input_defects)
            else:
                with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_test.h5'.format(data_type,nmb_distances,max_distances,nmb_defects,samplesize,noise_size,1), 'r') as h5f:

                        
                    for dist in name_list:
                        spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                        oservable_data_spins.append(spins)
                        oservable_data_defects.append(defects)



            
        os.makedirs(output_data_path+ path_name, exist_ok=True)

        for name_idx,name in enumerate(name_list):

            outputname = '{}_{}_Plaquette_edge_graph_rep_before_epsilon_comparison_{}_temp_{}_nmb_defects_{}_latticesize_{}.h5'.format(name,sim_data_type,data_type,temperature,nmb_defects,lattice_size)


            defect_number_tensor = torch.sum(torch.abs(oservable_data_defects[name_idx]).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)
            unique_defect_lattices = torch.unique(oservable_data_defects[name_idx], dim=0)
            #print(unique_defect_lattices.size())
            unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)

            topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


            topological_analysis_tool.set_full_defect_positions(defect_lattices=oservable_data_defects[name_idx],maximal_defect_nmb=maximal_defect_nmb)



            with h5py.File(output_data_path+ path_name+outputname,'w') as h5f:
        
                dataset_plaquette = h5f.create_dataset('Plaquette', shape=(oservable_data_defects[name_idx].size(0),lattice_size*lattice_size), dtype='float')
                dataset_edge_x = h5f.create_dataset('Edge x', shape=(oservable_data_defects[name_idx].size(0),lattice_size*lattice_size), dtype='float')
                dataset_edge_y = h5f.create_dataset('Edge y', shape=(oservable_data_defects[name_idx].size(0),lattice_size*lattice_size), dtype='float')

                dataset_defects = h5f.create_dataset('Defects', shape=(oservable_data_defects[name_idx].size(0),lattice_size*lattice_size), dtype='float')

                dataset_spins = h5f.create_dataset('Spins', shape=(oservable_data_defects[name_idx].size(0),lattice_size*lattice_size), dtype='float')

                dataset_defects[:] = oservable_data_defects[name_idx].view(-1,lattice_size*lattice_size).cpu().numpy()


                dataset_plaquette[:] =topological_analysis_tool.plaquettes_matrix_batch_faster(oservable_data_spins[name_idx]).view(-1,lattice_size*lattice_size).cpu().numpy()


                edge_matrices_x, edge_matrices_y = topological_analysis_tool.edge_matrix_batch_faster(oservable_data_spins[name_idx])
                dataset_edge_x[:]=edge_matrices_x.view(-1,lattice_size*lattice_size).cpu().numpy()

                dataset_edge_y[:]=edge_matrices_y.view(-1,lattice_size*lattice_size).cpu().numpy()

                dataset_spins[:] = oservable_data_spins[name_idx].view(-1,lattice_size*lattice_size).cpu().numpy()


if __name__ == '__main__':
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.01,0,5,1,0], nmb_label=1,data_type='training_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.01,0,5,1,0], nmb_label=1,data_type='random_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.01,0,5,1,0], nmb_label=1,data_type='zero_temp',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    generated_data_atributes_list_v2 = [
            [0.1,0,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'conv_transposed', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 3, 64, 0, 'L2', (10.0 ,0.0)],
        [0.1,4,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (1.0 ,0.0)],
        [0.1,4,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (50.0 ,0.0)]]
    generated_data_atributes_list_v1 = [
            [0.1,10,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (10.0 ,0.0)],
        [0.1,10,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (1.0 ,0.0)],
        [0.1,10,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (50.0 ,0.0)]]
    ##get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/', simulation_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',generated_data_attributes=generated_data_atributes_list_v2[0], nmb_label=0,data_type='zero_temp',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    ##get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/', simulation_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',generated_data_attributes=generated_data_atributes_list_v2[0], nmb_label=0,data_type='random_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=generated_data_atributes_list_v2[1], nmb_label=4,data_type='generated_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=generated_data_atributes_list_v1[0], nmb_label=10,data_type='generated_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=generated_data_atributes_list_v1[1], nmb_label=10,data_type='generated_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
    #get_edge_state_graphs_sim_gen(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', simulation_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=generated_data_atributes_list_v2[2], nmb_label=4,data_type='generated_data',temperature=0.1,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= False, further_simulated=False,device='cpu',samplesize = 1000)
