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

from torch.nn.functional import pad

import torchlens as tl

import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

two_pi = 2.0*np.pi

from collections import Counter

from training_september_25_fixed_sum_fourier_changed_critic import CircularUpscaleConv2d,ResidualBlock, AdaIN, UnetGenerator, CircularPad2d, SumPool2d, InceptionConv2d_mini #_fourier

#other version red
## #B2182B , #D6604D. ## #F4A582
##sb.pointplot(x=distance,y=training_mean_correlation_data_sorted)

##full: '#2166AC', #4393C3','#92C5DE'
##small: '#B2182B', '#D6604D', '#F4A582'
##large: '#762A83', '#9970AB', '#C2A5CF'
##wo: '#1B7837', '#5AAE61', '#ACD39E'
##gen: '#EC7014', '#FB9A29', '#FEC44F'


    ###colors:
     ##pics
    ###4: low: p:'darkorange', e:'darkcyan'; mid: p:r'#ffa756', e:r'#82cbb2'; high: p: r'#fec615', e: r'#24bca8'
    ###46: low: p: r'#7bc8f6', e: r'#ec2d01' ; mid: p: r'#acc2d9', e: r'#db5856'; high: p:r'#107ab0', e: r'#8c000f'
    ### wo: low: p: r'#5ca904', e: r'#e17701'; mid: p:r'#6fc276', e:r'#ffb16d'; high: p:r'#388004', e:r'#efb435'
    ###gen: low: p: r'#c79fef',e:r'#4b57db'; mid: p:r'#dfc5f3' ,e:r'#4e7496' ; high: p:r'#c875c4', e: r'#0b5394'
    ##plots types gets same symbol temp+defect type gets own color
    ###4: low: area:marker_color=r'#F1932D',marker_symbol="cross"), perimeter:marker_color=r'#F1932D',marker_symbol="circle"), euler: marker_color=r'#F1932D',marker_symbol="diamond"); mid:area:,marker_color=r'#F6c141', perimeter:marker_color=r'#F6c141', euler: marker_color=r'#F6c141'; high: area:r'#E8691c', perimeter:r'#E8691c', euler: r'#E8691c'
    ###46: low: area:r"#5289C7"  ; mid: r"#5289C7" ; high:r"#1965b0"
    ### wo: low:"#90C987" ; mid: r"#cae0ab" ; high: r"#4eb265"
    ###gen: low: "#AE76A3"; mid:r"#d1bbd7" ; high: r"#882e72"
    
    ### colortypes_list = [sb.color_palette("BuPu"),sb.color_palette("YlGn"),sb.color_palette("GnBu"),sb.color_palette("OrRd")]
def testing_correlation_function(simualted_data_path='/localscratch/kklos/XY_Cluster_rust_sim_data/',temperature=0.1, lattice_size = 16):
    simul_spin = pd.read_csv(simualted_data_path+'new_cluster_spin_data_observables_{}_{}_0.csv'.format(temperature, lattice_size), delimiter=',', skipinitialspace= True, header = 0)
    

def define_colors():
    colortypes_gen = [r'#77aadd',r'#9398d2', r'#ee99aa',r'#bbcc33',r'#44bb99']
    colortypes_sim = [r'#332288',r'#762a83', r'#882255', r'#997700',r'#225522']
    color_list = ['#9EDAE5','#FFBB78','#BBCCEE']

    colortypes_test =  [r'#762a83',r'#9398d2',r'#ee99aa',r'#44bb99']#[r'#762a83',r'#9398d2']
    
    linestyle = ['solid','dashed', 'dotted', 'dashdot',(0,(1,10)), (0,(5,10)),(0,(3,5,1,5,1,5)),(0,(3,5,3,5,1,5)),(0,(6,5,1,5,1,5))]# loosly dotted, loosley dashed, dashdottdotted, dash dash dotted, long dash dott dott
    
    marker_styles = ['o','v','s','d']
    
    return colortypes_test,linestyle,marker_styles

        
def basis_data_load(new_data,lattice_size,simulation_data_path,temperature,device,defect_nmb,compare_defect_lattice,samplesize,larger):
    if new_data :
        if lattice_size == 32:
            batch = True
            device_check = device
        else:
            batch = False
            device_check = device
        spin_dict, defect_dict = output_analysis_basis(batch=batch,simulated_data_path=simulation_data_path,temperature=temperature, lattice_size=lattice_size,compare_defect_lattice=compare_defect_lattice,device=device_check,samplesize=samplesize,defect_nmb=defect_nmb,larger=larger)
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
            
        if defect_nmb > 2 or lattice_size > 16 :
            #with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
            with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                name_list = np.array(h5f['distances'])
                            
                for dist in name_list:
                    sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                    full_spin_list.append(sim_spins)
                    full_defect_list.append(sim_defects)
                
        else:

            #name_list = return_defect_distances(maximal_distance=maximal_distance)
                #with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_sample_{}.h5'.format(temperature,samplesize), 'r') as h5f:
            with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                ##with h5py.File(simulation_data_path+'full_training_data_gan_defect_distance_select_compare_{}_rotation_{}_later_rot_{}_latticesize_{}_defectnmb_{}_samplesize{}_temperature_{}.h5'.format(int(compare_defect_lattice),int(rotation),int(later_rotation),lattice_size,defect_nmb,training_data_nmb,temperature), 'r') as h5f:

                name_list = np.array(h5f['distances'])
                            
                for dist in name_list:
                    sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                    full_spin_list.append(sim_spins)
                    full_defect_list.append(sim_defects)
                    
    return full_spin_list, full_defect_list, name_list

#def transformation_pos(defects, spins, distances,lattice_size=16,defect_nmb=2):#defect and spins should be lists
    
#    transformed_spins = []
#    transformed_defects = []
    
#    topological_tool = TopologicalAnalysis(lattice_size=lattice_size,device='cpu')
    
#    changed_distances = np.zeros(int(len(distances)))
    

    
    for d_idx, dist in enumerate(distances):
        
        topological_tool.set_full_defect_positions(defect_lattices=torch.from_numpy(defects[d_idx]),maximal_defect_nmb=defect_nmb,individual=True)
        full_defect_distance_x,full_defect_distance_y = topological_tool.get_defect_distance_xy()
        
        spins_per_distance = []
        defects_per_distance = []
        if torch.any(full_defect_distance_x==full_defect_distance_y) or torch.any(full_defect_distance_x==full_defect_distance_y*(-1.)):#fulldiag
            if torch.sum(full_defect_distance_x==full_defect_distance_y) +torch.sum(full_defect_distance_x==full_defect_distance_y*(-1.))>= 500:
                
                changed_distances[d_idx] = 1.
                
                spins_per_distance.append(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x>0.0)])
                defects_per_distance.append(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x>0.0)])
                
                spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x<0.0)],k=2,dims=[1,2]))
                defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x<0.0)],k=2,dims=[1,2]))
                
                spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_x<0.0)],k=1,dims=[1,2]))
                defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_x<0.0)],k=1,dims=[1,2]))
                
                spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_y<0.0)],k=3,dims=[1,2]))
                defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_y<0.0)],k=3,dims=[1,2]))
                #for case: full_defect_distance_x==full_defect_distance_y*(-1)
                
                
            else:
                
                raise ValueError('Not enough data for that')

        if torch.any(full_defect_distance_x==0.) or torch.any(full_defect_distance_y==0.):#horizontal/vertical
            
            changed_distances[d_idx] = 1.
            
            spins_per_distance.append(spins[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x>=0.)])
            defects_per_distance.append(defects[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x>=0.)])
            spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x<0.)],k=2,dims=[1,2]))
            defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x<0.)],k=2,dims=[1,2]))
            spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y>0.)],k=1,dims=[1,2]))
            defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y>0.)],k=1,dims=[1,2]))
            spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y<0.)],k=3,dims=[1,2]))
            defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y<0.)],k=3,dims=[1,2]))
            
        transformed_spins.append(torch.stack(spins_per_distance))
        transformed_defects.append(torch.stack(defects_per_distance))
        
    return transformed_spins,transformed_defects,changed_distances
                        
def loading_spin_defect_data(data_path,epochnmb,simulation_data_path,output_data_path,new_data,lattice_size,defect_nmb,larger,data_type,temperature, compare_defect_lattice,device,further_simulated,generated_data_attributes, critic_depth,training_data_nmb,noise_size,samplesize,sim_data_type='full_pinned'):
    #observable gives either physical_observables, filtration, just_spin_data, persistent_homology
    
    full_spin_list, full_defect_list, name_list = basis_data_load(False,lattice_size,simulation_data_path,temperature,device,defect_nmb,compare_defect_lattice,samplesize,larger)
    #name_list=[3.0]
    
    if epochnmb == -1:
        e_nmb = 120
    else:
        e_nmb = epochnmb
    print('load',name_list)
    nmb_distances = int(len(name_list))
    max_distances = max(name_list)


    observable_data_spins = []

    observable_data_defects = []
    if data_type == 'training_data':
        
        if sim_data_type == 'full_pinned':

            path_name = 'training_data_full_distance_temp_{}_new/'.format(temperature)


                    
            with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.h5'.format(data_type,nmb_distances,max_distances,defect_nmb,samplesize,noise_size,1), 'r') as h5f:
                            
                for dist in name_list:
                    spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][epochnmb]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][epochnmb]).to(device).view(-1,lattice_size,lattice_size)

                    observable_data_spins.append(spins)
                    observable_data_defects.append(defects)
        else:
            path_name = 'training_data_full_distance_temp__{}_{}_new/'.format(sim_data_type,temperature)


                    
            with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.h5'.format(data_type,sim_data_type,nmb_distances,max_distances,defect_nmb,samplesize,noise_size,1), 'r') as h5f:
                            
                for dist in name_list:
                    spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][epochnmb]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][epochnmb]).to(device).view(-1,lattice_size,lattice_size)

                    observable_data_spins.append(spins)
                    observable_data_defects.append(defects)
            
                
        ##observable_data_spins, observable_data_defects,changed_distances = transformation_pos(defects=observable_data_defects, spins=observable_data_spins, distances=name_list,lattice_size=lattice_size,defect_nmb=defect_nmb)
        ##print(changed_distances)

    elif data_type == 'generated_data':


        path_name = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/res_critic/temp_{}/'.format(noise_size,training_data_nmb,*generated_data_attributes,temperature) 
        ##path_name = 'res_critic/temp_{}/'.format(temperature)

        if new_data:
            input_path_name = '{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/res_critic/'.format(*generated_data_attributes) 
            #path_name = '{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_instance_AdaInstart_0_gen_regul_0.5_extranoise_False/test/'.format(temperature,*generated_data_attributes[1:],critic_depth) 
            
            
            nmb_label = generated_data_attributes[1]
            os.makedirs(output_data_path+path_name, exist_ok=True)
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

            else:
                    
                    #print(temperature)
                temperature_label = (torch.mul(torch.ones(samplesize).view(-1,1),temperature*10.)).long() #
                    
                    
                print(torch.unique(temperature_label))
                
            ##observable_data_spins, observable_data_defects,changed_distances = transformation_pos(defects=observable_data_defects, spins=observable_data_spins, distances=name_list,lattice_size=lattice_size,defect_nmb=defect_nmb)

            for input_defects in full_defect_list:
                    

                    
                with torch.no_grad():

                        
                    gen_model = torch.load(data_path 
                                                + input_path_name
                                                 + 'Generator_check_{}.pth'.format(e_nmb),map_location=torch.device('cpu'))
                        #gen_model.set_new_noise_factor(100)
                    generated_data = gen_model(input_defects.float().view(-1,1,lattice_size,lattice_size),labels=temperature_label)
                                    ##print(generated_data.size())
                    for name, params in gen_model.named_parameters():
                        norm = torch.sqrt(torch.sum(params**2))
                        print(name, norm)
                    ##    
                    ##break

                        #print(generated_data[:5])
                    generated_data_x = generated_data[:,0,:,:].view(-1,lattice_size,lattice_size)
                    generated_data_y = generated_data[:,1,:,:].view(-1,lattice_size,lattice_size)
                        
                        #print(generated_data_x.view(-1,lattice_size*lattice_size)[:5])
                        #print(temperature_label[:5])
                    generated_data_norm = torch.sqrt((torch.square(generated_data_x)+torch.square(generated_data_y)))+1e-12
                                    ##print(generated_data_norm.size())
                    generated_data_x_normed = torch.div(generated_data_x,generated_data_norm)
                    generated_data_y_normed = torch.div(generated_data_y,generated_data_norm)
                    generated_angles = torch.arctan2(generated_data_y_normed,generated_data_x_normed)
                        #print(generated_angles[:5])
                        #break
                    if further_simulated == True:

                        generated_angles_add_sim_list = []
                        for add_sim_idx,spin_angles in enumerate(generated_angles):
                            spin_system = XYSystem(temperature = temperature,defect_config=full_defect_list[idx][add_sim_idx].cpu().numpy().reshape(lattice_size*lattice_size),input_analytic_solution=spin_angles.cpu().numpy().reshape(lattice_size*lattice_size),width = lattice_size)
                            spin_system.set_input_configuration()
                            spin_system.multible_sweep_alternative(10)
                            spin_after_add_sim = torch.from_numpy(spin_system.spin_config).view(lattice_size,lattice_size).to(device)
                            generated_angles_add_sim_list.append(spin_after_add_sim)
                        observable_data_spins.append(torch.stack(generated_angles_add_sim_list,dim=0))
                    else:
                        observable_data_spins.append(generated_angles)

                    observable_data_defects.append(input_defects)
        else:

                #with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance_{}_samplesize{}_noisesize_{}.h5'.format(data_type,nmb_distances,max_distances,samplesize,noise_size), 'r') as h5f:
            with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.h5'.format(data_type,nmb_distances,max_distances,defect_nmb,samplesize,noise_size,1), 'r') as h5f:
                        
                for dist in name_list:
                    spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][epochnmb]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][epochnmb]).to(device).view(-1,lattice_size,lattice_size)

                    observable_data_spins.append(spins)
                    observable_data_defects.append(defects)

    elif data_type == 'vision_trafo':
        
        print('TEST')

        path_name = 'VIT_check/'


                
        with h5py.File(output_data_path+path_name+'Test_data_0.1_predictions.h5', 'r') as h5f:
                        
            for dist in name_list:
                spins = torch.from_numpy(np.array(h5f['spin_lattices_{}'.format(dist)])).to(device)##.view(-1,lattice_size,lattice_size,2)##for constraint_sampling more like 80 k 
                defects = torch.from_numpy(np.array(h5f['defects_lattices_{}'.format(dist)])).to(device).view(-1,lattice_size,lattice_size)

                print(dist,spins.shape)
                transformed_spin = spins.reshape(-1,lattice_size*lattice_size,2)
                
                data_norm = torch.sqrt((torch.square(transformed_spin[:,:,1])+torch.square(transformed_spin[:,:,0])))+1e-12
                transformed_spin_angle = torch.arctan2(transformed_spin[:,:,1]/data_norm,transformed_spin[:,:,0]/data_norm)
                
                observable_data_spins.append(transformed_spin_angle.reshape(-1,lattice_size,lattice_size))
                observable_data_defects.append(defects.reshape(-1,lattice_size,lattice_size))

    elif data_type == 'random_data':

        path_name = 'random_data_comparison/'


        if new_data:

            for idx, name in enumerate(name_list):
                    
                observable_data_spins.append((torch.rand((samplesize,lattice_size,lattice_size)))*2.*np.pi)

                observable_data_defects.append(full_defect_list[idx])

        else:
            with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance.h5'.format(data_type,nmb_distances,max_distances), 'r') as h5f:

                        
                for dist in name_list:
                    spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                    observable_data_spins.append(spins)
                    observable_data_defects.append(defects)

    elif data_type == 'zero_temp':

        path_name = 'zero_temperature_solution_comparison_new/'


        if new_data:

            lattice_spacing = 1

            spin_positions = np.array([[[i*lattice_spacing,j*lattice_spacing] for j in range(lattice_size)] for i in range(lattice_size)])

            for input_defects in full_defect_list:

                analys_sol_per_defects = []

                for defects in input_defects:
                    analy_spins = torch.from_numpy(analysitcally_calculation(spin_position=spin_positions,defects=defects.numpy(),lattice_size=lattice_size))
                    analys_sol_per_defects.append(analy_spins)
                observable_data_spins.append(torch.stack(analys_sol_per_defects,dim=0))
                observable_data_defects.append(input_defects)
        else:
            with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.h5'.format(data_type,nmb_distances,max_distances,defect_nmb,samplesize,noise_size,1), 'r') as h5f:

                        
                for dist in name_list:
                    spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                    defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                    observable_data_spins.append(spins)
                    observable_data_defects.append(defects)
                    

    return name_list, observable_data_spins, observable_data_defects,path_name

  
def find_unique_min_distance_tensors(batch_tensors):
    """
    Process a batch of tensors to find indices of tensors where each -1 has a unique minimum-distance +1,
    and all other +1s are at least 1 unit farther away.
    Parameters:
    -----------
    batch_tensors : torch.Tensor
        A batch of tensors of shape (b, L, L) containing only 0, -1, and +1 values
    Returns:
    --------
    torch.Tensor
        A tensor containing batch indices where the condition is met
    dict
        A dictionary with detailed information about minimum distances for each -1 in each tensor
    """
    batch_size = batch_tensors.shape[0]
    L = batch_tensors.shape[1]  # Assuming square tensors (L x L)
    # Store results for each tensor in the batch
    valid_indices = []
    all_min_distances = {}
    # Process each tensor in the batch
    for b in range(batch_size):
        tensor = batch_tensors[b]
        # Find positions of -1s and +1s
        neg_positions = torch.nonzero(tensor == -1, as_tuple=False)  # Shape: (num_neg, 2)
        pos_positions = torch.nonzero(tensor == 1, as_tuple=False)   # Shape: (num_pos, 2)
        # If there are no -1s or +1s, skip this tensor
        if len(neg_positions) == 0 or len(pos_positions) == 0:
            continue
        # Calculate distances from each -1 to each +1
        min_distances = {}
        unique_min_flag = True
        for neg_idx, neg_pos in enumerate(neg_positions):
            # Calculate L1 (Manhattan) distance to all +1 positions
            distances = torch.sqrt(torch.sum(torch.pow(pos_positions - neg_pos,2), dim=1))
            # Find the minimum distance and its index
            min_dist, min_idx = torch.min(distances, dim=0)
            # Find the second minimum distance
            if len(distances) > 1:
                # Create a mask to exclude the minimum value
                mask = torch.ones_like(distances, dtype=torch.bool)
                mask[min_idx] = False
                second_min_dist = torch.min(distances[mask])
                # Check if the second minimum is at least 1 larger than the minimum
                if second_min_dist <= min_dist + 1:
                    unique_min_flag = False
                    break
            # Store the minimum distance for this -1
            min_distances[tuple(neg_pos.tolist())] = {
                'min_dist': min_dist.item(),
                'min_pos': tuple(pos_positions[min_idx].tolist())
            }
        # Check if all -1s have a unique minimum-distance +1
        if unique_min_flag:
            # Check if each +1 is associated with at most one -1
            pos_to_neg = {}
            for neg_pos, info in min_distances.items():
                pos = info['min_pos']
                if pos in pos_to_neg:
                    unique_min_flag = False
                    break
                pos_to_neg[pos] = neg_pos
        # If all conditions are met, add this tensor's index to valid_indices
        if unique_min_flag:
            valid_indices.append(b)
            all_min_distances[b] = min_distances
    return torch.tensor(valid_indices, dtype=torch.long)#, all_min_distances


def check_defect_type_config(simulated_data_path,temperature, lattice_size,device='cpu', samplesize = 1000,compare_defect_lattice=False,reference_temp = 0.1,new=True):
    with torch.no_grad():
        
        if new:
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
            
            #print(torch.any(sim_distances == 5.0))
            for dist in torch.unique(sim_distances):
                        ##print(dist)
                current_defect_distances = sim_defects[sim_distances==dist]
                #print(current_defect_distances.shape)
                current_spin_distances = sim_spins[sim_distances==dist]

                find_unique_min_distance_tensors(current_defect_distances)
                sample_idx = random.sample(spin_data_index,samplesize)
                distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                sizes.append(int(sim_spins[spin_data_index].size(0)))
                    

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


        return distances_dictionary_spins, distances_dictionary_defects
        
    
def defect_distance_distributions(defect_distances,correspon_defects, samplesize):
    
    unique_distances, unique_distance_nmb = torch.unique(defect_distances, return_counts= True)
    
    new_unique_distance_nmb = torch.clone(unique_distance_nmb)
    
    new_distance_with_counts = {}
    
    for d_idx,d in unique_distances:
        if unique_distance_nmb[d_idx] < samplesize:
            if (d_idx-1) > 0:
                lower_uniq_nmb = new_unique_distance_nmb[d_idx-1]
    
        
def check_defect_type_config_larger_lattice(simulated_data_path,temperature, lattice_size,batch = True,device=device, samplesize = 1000, defect_number = 2):
    
    

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
            with h5py.File(simulated_data_path+'lattice_{}_num_pairs{}_temp{}.h5'.format(lattice_size,defect_number//2,temperature), 'r') as h5f:
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
                for dist in torch.unique(defect_distances):
                    #print(dist)
                    
                    current_defect_distances = sim_defects[defect_distances==dist]
                    current_spin_distances = sim_spins[defect_distances==dist]

                    indices_with_correct_distances = find_unique_min_distance_tensors(current_defect_distances)
                    print('before',current_defect_distances.size)
                    print('after',current_defect_distances[indices_with_correct_distances].size)
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    if int(current_defect_distances.size(0)) >= samplesize:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                        sizes.append(int(sim_spins[spin_data_index].size(0)))

                #print('defects',sizes)


        return distances_dictionary_spins, distances_dictionary_defects
    
    

def output_analysis_basis(simulated_data_path,temperature,lattice_size,batch= True,compare_defect_lattice=False,device='cpu',samplesize=1000, larger=True,defect_nmb=2,new=True):

    if larger:
        spin_dict, defect_dict= check_defect_type_config_larger_lattice(batch = batch,simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,device=device,samplesize=samplesize,defect_number=defect_nmb)

    else:

        spin_dict, defect_dict= check_defect_type_config(simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,device=device,samplesize=samplesize,compare_defect_lattice=compare_defect_lattice,new=new)
    #name_list = ['full', 'no_defect', 'small_defects', 'mid_defects', 'large_defects']

    output_spin_dict = {}
    output_defect_dict = {}
    with h5py.File(simulated_data_path + 'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize),'w') as h5f:

        dataset_distances = h5f.create_dataset('distances', shape=(int(len(spin_dict)),), dtype='float')
        if batch:
            dataset_distances[:] = np.array([dist for dist in spin_dict])
        else:
            dataset_distances[:] = np.array([dist.cpu().numpy() for dist in spin_dict])
            

        for dist in spin_dict:
            
            if batch:
                output_spin_dict[dist] = torch.cat(spin_dict[dist], dim = 0)
                output_defect_dict[dist] = torch.cat(defect_dict[dist], dim = 0)
            else:
                output_spin_dict[dist] = spin_dict[dist]
                output_defect_dict[dist] = defect_dict[dist]
                            
            dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            
            if batch:
                
                print(torch.cat(spin_dict[dist], dim = 0).shape)
            
                dataset_spins_compare[:] = torch.cat(spin_dict[dist], dim = 0).view(-1,lattice_size*lattice_size).cpu().numpy()
                
                dataset_defects_compare[:] = torch.cat(defect_dict[dist],dim=0).view(-1,lattice_size*lattice_size).cpu().numpy()
                
            else:
                
                dataset_spins_compare[:] = spin_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()
                
                dataset_defects_compare[:] = defect_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()
                

                
    return output_spin_dict, output_defect_dict


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

def defect_position_simult(defects,number_of_defects, lattice_size):

    position_vector = torch.zeros((defects.shape[0],number_of_defects,3))
    
    #print('ähmm?',position_vector.shape)

    counter = torch.zeros((defects.shape[0],1))


    for i in range(lattice_size):
        for j in range(lattice_size):
            if torch.any(defects[:,i,j] != 0):
                

                
                current_counter = counter[defects[:,i,j] != 0].flatten()
                
                
                current_defects = defects[defects[:,i,j] != 0]
                
                
                non_zero_idx = torch.nonzero(defects[:,i,j] != 0).flatten()
                
                #print(non_zero_idx.shape)
                #print(current_counter.shape)
                #print(torch.clone(current_defects[:,i,j]).shape)
                position_vector[non_zero_idx,current_counter.int(),torch.ones_like(non_zero_idx)*0] += i
                position_vector[non_zero_idx,current_counter.int(),torch.ones_like(non_zero_idx)*1] += j
                position_vector[non_zero_idx,current_counter.int(),torch.ones_like(non_zero_idx)*2] += torch.clone(current_defects[:,i,j])
                counter[defects[:,i,j] != 0] +=1

    #print(position_vector)

    return position_vector

def analysitcally_calculation(spin_position,defects,lattice_size):

    vortex_numb = vortex_number(defects,lattice_size)

    print(vortex_number)

    defect_positions = defect_position(defects = defects,number_of_defects=vortex_numb, lattice_size= lattice_size)

    print(defect_positions)

    analytic_solution_list = []

    print(spin_position.shape)

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



    return analytic_solution


def jensen_shannon(observable_truth, observable_compare,needs_norm = True):

    #KLD = np.where(np.logical_and(observable_truth > 0.,observable_compare > .0),observable_truth*np.log(np.divide(observable_truth,observable_compare)),)

    if np.any(observable_truth > 10.0 ):
        round_nb = 2
    else:
        round_nb = 3

    samplesize = float(len(list(observable_truth.flatten())))


    observables_truth_counter = Counter(list(np.round(observable_truth.flatten(),round_nb)))

    observables_compare_counter = Counter(list(np.round(observable_compare.flatten(),round_nb)))

    #print('truth',observables_truth_counter)

    #print('compare',observables_compare_counter)

    visted_observables = []

    jsd = 0.

    for truth_key in observables_truth_counter.keys():

        if truth_key in observables_compare_counter.keys():

            tru_dist = float(observables_truth_counter[truth_key])/samplesize

            compare_dist = float(observables_compare_counter[truth_key])/samplesize

            jsd+=0.5*(tru_dist*np.log(np.divide(tru_dist,((compare_dist+tru_dist)/2.)))+compare_dist*np.log(np.divide(compare_dist,((compare_dist+tru_dist)/2.))))

        else:

            tru_dist = float(observables_truth_counter[truth_key])/samplesize

            compare_dist = 0.0

            jsd+=0.5*(tru_dist*np.log(np.divide(tru_dist,((compare_dist+tru_dist)/2.))))

        #print(jsd,np.log(np.divide(tru_dist,((compare_dist+tru_dist)/2.))),np.log(np.divide(compare_dist,((compare_dist+tru_dist)/2.))))


        visted_observables.append(truth_key)

    
    for compare_key in observables_compare_counter.keys():

        if compare_key not in visted_observables:

            tru_dist = 0.0

            compare_dist = float(observables_compare_counter[compare_key])/samplesize

            jsd+=0.5*(compare_dist*np.log(np.divide(compare_dist,((compare_dist+tru_dist)/2.))))



    return jsd

def kullback_leibler_alternative(observable_truth, observable_compare):

    #KLD = np.where(np.logical_and(observable_truth > 0.,observable_compare > .0),observable_truth*np.log(np.divide(observable_truth,observable_compare)),)
    
    if np.all(observable_truth > 0.) and np.all(observable_compare > .0):

        kld = observable_truth*np.log(np.divide(observable_truth,observable_compare))

        change_value = 0.0 
        return kld, change_value
    else:
        minmal_value = min([np.min(observable_truth),np.min(observable_compare)])

        change_value = minmal_value + 1e-16

        changed_truth = observable_truth + np.abs(minmal_value) +change_value

        changed_compare = observable_compare + np.abs(minmal_value) + change_value

        kld = changed_truth*np.log(np.divide(changed_truth,changed_compare))

        return kld, change_value


def kullback_leibler(observable_truth, observable_compare,needs_norm = True):

    #KLD = np.where(np.logical_and(observable_truth > 0.,observable_compare > .0),observable_truth*np.log(np.divide(observable_truth,observable_compare)),)

    if np.any(observable_truth > 1.0 ):
        round_nb = 2
        acc = 0.05
    else:
        round_nb = 3
        acc = 0.005

    samplesize = float(len(list(observable_truth.flatten())))


    observables_truth_counter = Counter(list(np.round(observable_truth.flatten(),round_nb)))

    observables_compare_counter = Counter(list(np.round(observable_compare.flatten(),round_nb)))

    #print('truth',observables_truth_counter)

    #print('compare',observables_compare_counter)

    visted_observables = []

    kld = 0.

    for truth_key in observables_truth_counter.keys():

        if truth_key in observables_compare_counter.keys():

            tru_dist = float(observables_truth_counter[truth_key])/samplesize

            compare_dist = float(observables_compare_counter[truth_key])/samplesize

        #elif observables_compare_counter.has_key(truth_key+acc):
        #    tru_dist = observables_truth_counter[truth_key].float()/samplesize
        #    compare_dist = observables_compare_counter[truth_key+acc].float()/samplesize

            

        else:
            tru_dist = float(observables_truth_counter[truth_key])/samplesize

            compare_dist = 1e-10

        kld+=tru_dist*np.log(np.divide(tru_dist,compare_dist))



        visted_observables.append(truth_key)

    for compare_key in observables_compare_counter.keys():

        if compare_key not in visted_observables:

            tru_dist = 1e-10

            compare_dist = float(observables_compare_counter[compare_key])/samplesize


            kld+=tru_dist*np.log(np.divide(tru_dist,compare_dist))


    

    return kld

def magnet_mean_tensor(angle_tensors):

    cosine = torch.mean(torch.cos(angle_tensors), dim = -1)

    sinus =  torch.mean(torch.sin(angle_tensors), dim= -1)

    mean_angle = torch.arctan2(sinus,cosine)

    return mean_angle#torch.mean(angle_tensors, dim=-1)#mean_angle

def angle_diff_tensor(angle_tensor_one, angel_tensor_two):

    difference = angle_tensor_one - angel_tensor_two

    saw_cond_1 = torch.abs(difference - two_pi) < np.abs(difference)

    saw_cond_2 = torch.abs(difference + two_pi) < np.abs(difference)


    difference[saw_cond_1] -= two_pi

    difference[saw_cond_2] += two_pi




    return torch.abs(difference)#torch.mean(angle_tensors, dim=-1)#mean_angle


def new_interpolation(angle_tensor, lattice_size):

    ##frist need to fill in zeros

    ##print(angle_tensor.size())
    ##print(angle_tensor)

    

    enlarged_tensor_1 = torch.repeat_interleave(angle_tensor.view(lattice_size,lattice_size), 2, dim=1)

    #print(enlarged_tensor_1)

    enlarged_tensor_2 = torch.repeat_interleave(enlarged_tensor_1, 2, dim=0)

    #print(enlarged_tensor_2)

    out_put_map = torch.clone(enlarged_tensor_2)




    plaquette_maker = nn.Unfold(kernel_size=(3,3), stride= 2)
    plaq_index = torch.tensor([0,2,3,1,0],device=device)

    spin_lattice_halo = torch.add(F.pad(enlarged_tensor_2.view(-1,1,2*lattice_size,2*lattice_size), (0,1,0,1), 'circular'),1e-8)




    spin_lattice_halo_enlarged = torch.clone(spin_lattice_halo)

    spin_lattice_halo_enlarged_difference = torch.clone(spin_lattice_halo)



          
    spin_lattice_indices = spin_lattice_halo.view(1,-1).nonzero(as_tuple=True)[1]

    #spin_lattice_indices = (spin_lattice_halo==1.0).view(1,-1).nonzero(as_tuple=True)[1]

    spin_lattices_x_indices = (spin_lattice_indices%(2*lattice_size+1)).view(1,1,2*lattice_size+1,2*lattice_size+1)

    spin_lattices_y_indices = (spin_lattice_indices//(2*lattice_size+1)).view(1,1,2*lattice_size+1,2*lattice_size+1)



    spin_lattices_x_indices_plaquette = torch.transpose(plaquette_maker(spin_lattices_x_indices.float()).view(1,9,-1),1,2)

    spin_lattices_y_indices_plaquette = torch.transpose(plaquette_maker(spin_lattices_y_indices.float()).view(1,9,-1),1,2)


    plaquette_temp = plaquette_maker(spin_lattice_halo).view(1,9,-1)

    #print(plaquette_temp.size())
    

    plaquettes = torch.transpose(plaquette_temp, 1, 2)

    plaquettes_difference = torch.clone(plaquettes)



    plaquettes[:,:,1] = magnet_mean_tensor(torch.stack([plaquettes[:,:,2],plaquettes[:,:,0]],dim=2))

    plaquettes[:,:,3] = magnet_mean_tensor(torch.stack([plaquettes[:,:,0],plaquettes[:,:,6]],dim=2))

    plaquettes[:,:,7] = magnet_mean_tensor(torch.stack([plaquettes[:,:,6],plaquettes[:,:,8]],dim=2))

    plaquettes[:,:,5] = magnet_mean_tensor(torch.stack([plaquettes[:,:,8],plaquettes[:,:,2]],dim=2))

    plaquettes[:,:,4] = magnet_mean_tensor(torch.stack([plaquettes[:,:,1],plaquettes[:,:,3],plaquettes[:,:,7],plaquettes[:,:,5]],dim=2))

    plaquettes_difference[:,:,1] = angle_diff_tensor(plaquettes[:,:,2],plaquettes[:,:,0])

    plaquettes_difference[:,:,3] = angle_diff_tensor(plaquettes[:,:,0],plaquettes[:,:,6])

    plaquettes_difference[:,:,7] = angle_diff_tensor(plaquettes[:,:,6],plaquettes[:,:,8])

    plaquettes_difference[:,:,5] = angle_diff_tensor(plaquettes[:,:,8],plaquettes[:,:,2])

    plaquettes_difference[:,:,4] = torch.mean(torch.stack([plaquettes_difference[:,:,1],plaquettes_difference[:,:,3],plaquettes_difference[:,:,7],plaquettes_difference[:,:,5]],dim=2),dim=2)

    
    plaquettes_difference[:,:,0] = torch.mean(torch.stack([plaquettes_difference[:,:,1],plaquettes_difference[:,:,3]],dim=2),dim=2)

    plaquettes_difference[:,:,2] = torch.mean(torch.stack([plaquettes_difference[:,:,1],plaquettes_difference[:,:,5]],dim=2),dim=2)

    plaquettes_difference[:,:,6] = torch.mean(torch.stack([plaquettes_difference[:,:,3],plaquettes_difference[:,:,7]],dim=2),dim=2)

    plaquettes_difference[:,:,8] = torch.mean(torch.stack([plaquettes_difference[:,:,5],plaquettes_difference[:,:,7]],dim=2),dim=2)






    spin_lattice_halo_enlarged[:,:,spin_lattices_y_indices_plaquette[:,:,1].int(),spin_lattices_x_indices_plaquette[:,:,1].int()] = plaquettes[:,:,1]
    spin_lattice_halo_enlarged[:,:,spin_lattices_y_indices_plaquette[:,:,3].int(),spin_lattices_x_indices_plaquette[:,:,3].int()] = plaquettes[:,:,3]
    spin_lattice_halo_enlarged[:,:,spin_lattices_y_indices_plaquette[:,:,7].int(),spin_lattices_x_indices_plaquette[:,:,7].int()] = plaquettes[:,:,7]
    spin_lattice_halo_enlarged[:,:,spin_lattices_y_indices_plaquette[:,:,5].int(),spin_lattices_x_indices_plaquette[:,:,5].int()] = plaquettes[:,:,5]
    spin_lattice_halo_enlarged[:,:,spin_lattices_y_indices_plaquette[:,:,4].int(),spin_lattices_x_indices_plaquette[:,:,4].int()] = plaquettes[:,:,4]


    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,0].int(),spin_lattices_x_indices_plaquette[:,:,0].int()] = plaquettes_difference[:,:,0]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,1].int(),spin_lattices_x_indices_plaquette[:,:,1].int()] = plaquettes_difference[:,:,1]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,2].int(),spin_lattices_x_indices_plaquette[:,:,2].int()] = plaquettes_difference[:,:,2]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,3].int(),spin_lattices_x_indices_plaquette[:,:,3].int()] = plaquettes_difference[:,:,3]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,6].int(),spin_lattices_x_indices_plaquette[:,:,6].int()] = plaquettes_difference[:,:,6]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,7].int(),spin_lattices_x_indices_plaquette[:,:,7].int()] = plaquettes_difference[:,:,7]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,5].int(),spin_lattices_x_indices_plaquette[:,:,5].int()] = plaquettes_difference[:,:,5]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,4].int(),spin_lattices_x_indices_plaquette[:,:,4].int()] = plaquettes_difference[:,:,4]
    spin_lattice_halo_enlarged_difference[:,:,spin_lattices_y_indices_plaquette[:,:,8].int(),spin_lattices_x_indices_plaquette[:,:,8].int()] = plaquettes_difference[:,:,8]


    ##print('after',spin_lattice_halo_enlarged[:,:,:2*lattice_size,:2*lattice_size])

    return_interpolation = F.pad(spin_lattice_halo_enlarged[:,:,:2*lattice_size,:2*lattice_size].view(-1,1,2*lattice_size,2*lattice_size), (1,1,1,1), 'circular')

    return_angel_change = F.pad(spin_lattice_halo_enlarged_difference[:,:,:2*lattice_size,:2*lattice_size].view(-1,1,2*lattice_size,2*lattice_size), (1,1,1,1), 'circular')/np.pi


    return return_interpolation,return_angel_change


def grid(m, n):
    y, x = np.indices((m, n))
    return x, y

def set_edges(x, y, X_pos_right,Y_pos_right,X_pos_down,Y_pos_down):
    if x.shape != y.shape:
        raise ValueError('x and y should have the same shape')

    J,I = X_pos_right,Y_pos_right
    L,K = X_pos_down,Y_pos_down

    x_edges = []
    y_edges = []
    for i, j  in zip(I, J):
        print(i,j)
        x_edges.extend([x[i,j], x[i, j+1], None])
        y_edges.extend([y[i,j], y[i,j], None])

    for k, l in zip(K,L):
        print(k,l)
        x_edges.extend([x[k,l], x[k, l], None])
        y_edges.extend([y[k,l], y[k+1, l], None])    
             
    return x_edges, y_edges


#def make_edges(x,y):



def array_to_lattice_position(pos,lattice_size):
    

    x_pos = pos%lattice_size
    y_pos = int(pos)//lattice_size
    #print(pos)
    #print('x: ', x_pos)
    #print('y: ', y_pos)

    return int(x_pos), int(y_pos)


def array_to_lattice_multipble(positions, lattice_size):
    x_positions, y_positions = [], []

    for pos in positions:
        #print(pos)
        x_pos,y_pos = array_to_lattice_position(pos=pos,lattice_size=lattice_size)
        x_positions.append(x_pos)
        y_positions.append(y_pos)

    return x_positions, y_positions

def local_energy_per_distance(dependence,lattice_size, defect_list_before, dim=2,data_idx=0,number_distances=14):
    
    
    
    defect_list = defect_list_before.reshape(number_distances,2,-1, lattice_size, lattice_size)[:,data_idx,:,:,:]
    
    print('*check again', torch.from_numpy(defect_list).shape)
    if dependence == 'edge_distance':
        
        all_lattice_indices = np.asarray([[[i,j] for j in range(lattice_size)] for i in range(lattice_size)]).reshape(lattice_size*lattice_size,2)

        all_lattice_indices = all_lattice_indices.reshape(-1,2)
        
        top_edge = all_lattice_indices[all_lattice_indices[:,0] == 0].reshape(lattice_size,2)
        

        
        left_edge = all_lattice_indices[all_lattice_indices[:,1] == 0].reshape(lattice_size,2)
        
        bottom_edge = all_lattice_indices[all_lattice_indices[:,0] == lattice_size-1].reshape(lattice_size,2)
        
        right_edge = all_lattice_indices[all_lattice_indices[:,1] == lattice_size-1].reshape(lattice_size,2)
        
        distances = np.zeros((lattice_size*lattice_size,4,lattice_size))
        
        edges = [top_edge,right_edge,bottom_edge,left_edge]
        
        for ed_idx, ed in enumerate(edges):
            for idx in range(lattice_size):
                distances[:,ed_idx,idx] = np.sqrt(
                    (all_lattice_indices[:,0]-ed[idx,0])**2 + (all_lattice_indices[:,1]-ed[idx,1])**2
                )
        distances_min = np.min(np.min(distances, axis= 2), axis= 1)#size l**2
        
    elif dependence == 'min_defect_distance' or dependence == 'max_defect_distance' or dependence == 'indiv_defect_distance':
        
        
        all_lattice_indices = np.asarray([[[i,j] for j in range(lattice_size)] for i in range(lattice_size)]).reshape(lattice_size*lattice_size,2)

        all_lattice_indices = np.repeat(all_lattice_indices.reshape(1,-1,2),defect_list.shape[1], axis = 0).reshape(defect_list.shape[1],-1,2)
        
        
        
        distances_min_per_pair_distance = []
        
        maximal_defect_nmb = torch.max(torch.sum(torch.abs(torch.from_numpy(defect_list.reshape(number_distances,-1, lattice_size*lattice_size).reshape(-1,lattice_size*lattice_size))), dim=1).flatten()).int()
        
        ##print('defect lattice size', defect_list)
        distances = np.zeros((defect_list.shape[0],defect_list.shape[1],lattice_size*lattice_size,maximal_defect_nmb))

        
        for d_idx,defect in enumerate(defect_list):
            
            ##print('ähm',defect.shape)
            
            

        
            defect_positions = defect_position_simult(torch.from_numpy(defect),maximal_defect_nmb, lattice_size).view(-1,maximal_defect_nmb,3).numpy() #output size (l**2, defect_nmb, 3)
        
            print(defect_positions.shape)
        
            for idx in range(maximal_defect_nmb):
                
                if (defect_positions[:,idx,2] == 0).all():
                    ##print('check zero defects')
                    
                    distances[d_idx,:,:,idx] = np.ones((defect_list.shape[1],lattice_size*lattice_size))*lattice_size
                    
                else:
            
                    distances[d_idx,:,:,idx] = np.sqrt(((all_lattice_indices[:,:,0]- defect_positions[:,idx,0].reshape(-1,1))%(lattice_size/2))**2+ ((all_lattice_indices[:,:,1]- defect_positions[:,idx,1].reshape(-1,1))%(lattice_size/2))**2)
            
        if dependence == 'min_defect_distance':
            
            distances_min = np.min(distances, axis=3 )#together before #size batch,l**2
            
        elif dependence == 'max_defect_distance':
            
            distances_min = np.max(distances, axis=3 )
            
        elif dependence == 'indiv_defect_distance':
            
            distances_min = distances #[:,:,0] or 1
            
            
        
    return distances_min


########################
########################
def testing_correlation_function(simulation_data_path = '/localscratch/kklos/XY_Cluster_rust_sim_data/',temperature=0.1, lattice_size=16, nmb_sample = 100):

    

    full_spin_data = pd.read_csv(simulation_data_path+'new_cluster_spin_data_observables_{}_{}_0.csv'.format(temperature,lattice_size), delimiter=',', skipinitialspace= True, header = 0)

    spin_lattices = torch.from_numpy(np.array([full_spin_data['Spin_angles{}'.format(i)] for i in range(nmb_sample)]).reshape(-1,lattice_size*lattice_size))

 
    one_tensor = torch.ones(1,9*9)
    correlation_indices_test = one_tensor.nonzero(as_tuple=True)[1]
    #print('corr idx',correlation_indices_test)
    ##correlation_flipped_right = torch.flip(correlation_indices_test,dims=[0,1])
    #print('corr_flipped once',correlation_flipped_right)

    ##full_corr_indices = torch.cat((correlation_indices_test,correlation_flipped_right[:,1:7*7-1].view(1,-1)),dim=1).flatten()

    #print(full_corr_indices)

    #print(torch.repeat_interleave(spin_lattices[:,0].view(-1,1),lattice_size*lattice_size,dim=1).size())

    ##testing corr change

    corr_distances = np.sqrt(((correlation_indices_test)%9)**2+((correlation_indices_test)//9)**2).view(9, 9)

    #print('corr distances',corr_distances)

    ##distance_flipped_right = torch.flip(corr_distances,dims=[0,1]) for diga down

    distance_flipped_right = torch.fliplr(corr_distances)

    ##print('distance flipped', distance_flipped_right)

    full_corr_indices_up = torch.cat((corr_distances,distance_flipped_right[:,1:9-1]),dim=1)

    ##print('full up corr', full_corr_indices_up)

    distance_flipped_down = torch.fliplr(torch.flip(full_corr_indices_up,dims=[1,0]))

    ##print('full up corr filpped ', distance_flipped_down)

    full_corr = torch.cat((full_corr_indices_up,distance_flipped_down[1:9-1,:]),dim=0)

    #print(full_corr)

    ##full_corr_right = torch.roll(full_corr,1, dims=1)

    ##full_corr_down = torch.roll(full_corr_right,2, dims=0)

    ##print(full_corr_down)

    ##full_corr_down = torch.roll(full_corr_right,3, dims=0)

    ##print(full_corr_down)

    

    ##print(full_corr[1,2])





    simulation_observables = Observables(lattice_size=lattice_size, temperature = temperature)









    sim_data_correlation_function = simulation_observables.correlation_function_even_faster_new_try(spin_fields=spin_lattices)

    #print(sim_data_correlation_function['6.0'])
    #print(sim_data_correlation_function['7.0'])
    #print(sim_data_correlation_function['8.0'])



    ##print(sim_data_correlation_function)



                            
    ##training_correlation_data= {np.sqrt(d_x**2+d_y**2):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
                        #print(training_correlation_data)
    ##training_correlation_data.pop(0.0)


    ##training_correlation_data_key_list = list(training_correlation_data.keys())
    ##training_correlation_data_key_list.sort()
                        #print(training_correlation_data_key_list)
                        #print(training_correlation.keys)
                        #print(training_correlation_data.keys())
    distance_parameter = [float(key) for key in sim_data_correlation_function.keys()]
    distance_parameter.sort() 
    training_mean_correlation_data_sorted = [torch.flatten(torch.mean(sim_data_correlation_function['{}'.format(k)])).cpu().numpy() for k in distance_parameter]
    #print(training_mean_correlation_data_sorted)
    
    #training_mean_correlation_data_sorted.insert(0,np.array([1.0]))
                    #print(training_mean_correlation_data_sorted)
    ##training_var_correlation_data_sorted = [torch.flatten(torch.mean(torch.pow(sim_data_correlation_function['{}'.format(k)],2.0))-torch.pow(torch.mean(sim_data_correlation_function['{}'.format(k)]),2.0)).cpu().numpy() for k in distance_parameter]
    training_var_correlation_data_sorted = [torch.sum(torch.pow(sim_data_correlation_function['{}'.format(k)]-training_mean_correlation_data_sorted[idx],2))/float((nmb_sample-1)) for idx,k in enumerate(distance_parameter)]
    #print(training_var_correlation_data_sorted)
    #training_var_correlation_data_sorted.insert(0,np.array([0.0]))
    ##distance ={np.sqrt(d_x**2+d_y**2):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
    ##                    #distance.pop(0.0)

    ##maximal_distance_correlation = int(simulation_observables.max_corr_distance())
    ##map_tensor_out_put = torch.ones(1,maximal_distance_correlation*maximal_distance_correlation)
    ##correlation_indices_output = map_tensor_out_put.nonzero(as_tuple=True)[1]

    ##distances = np.sqrt(((correlation_indices_output)%maximal_distance_correlation)**2+((correlation_indices_output)//maximal_distance_correlation)**2).view(maximal_distance_correlation*maximal_distance_correlation)
    ##distances.sort()


    #print(training_mean_correlation_data_sorted)
    #print(distance_parameter)
    #print(training_var_correlation_data_sorted)

    

    sb.set_theme(style='whitegrid',palette='gist_earth')

    fig, ax = plt.subplots()


    ax.plot(np.array(distance_parameter).flatten(),np.array(training_mean_correlation_data_sorted).flatten(),marker='o',markersize=3.5,label='T=0.1',linestyle='dashed',linewidth=3,color='#762A83')
    ax.fill_between(np.array(distance_parameter).flatten(),np.array(training_mean_correlation_data_sorted).flatten()-np.array(training_var_correlation_data_sorted).flatten(),np.array(training_mean_correlation_data_sorted).flatten()+np.array(training_var_correlation_data_sorted).flatten(),alpha=0.3,color='#762A83')


    x=np.array(distance_parameter).flatten()

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                        ncol=3, fancybox=True, shadow=True)



    plt.xlabel('Distance r')
    plt.ylabel(r"$\langle \cos(\vartheta_0-\vartheta_r)\rangle $")
    plt.xticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    #ax.set_xlim([0.0, 11.0])
    plt.show()
    #plt.savefig('corr_function_test_{}.png'.format(nmb_sample))

#start_time_1 = time.time()
##testing_correlation_function(nmb_sample=1000)
#end_time_1 = time.time()
#print('full time', end_time_1-start_time_1)
#testing_correlation_function(nmb_sample=1000)
##testing_correlation_function(nmb_sample=10000)


def testing_correlation_function(simulation_data_path = '/localscratch/kklos/XY_Cluster_rust_sim_data/',temperature=0.1, lattice_size=16, nmb_sample = 100):

    

    full_spin_data = pd.read_csv(simulation_data_path+'new_cluster_spin_data_observables_{}_{}_0.csv'.format(temperature,lattice_size), delimiter=',', skipinitialspace= True, header = 0)

    spin_lattices = torch.from_numpy(np.array([full_spin_data['Spin_angles{}'.format(i)] for i in range(nmb_sample)]).reshape(-1,lattice_size*lattice_size))

 
    one_tensor = torch.ones(1,9*9)
    correlation_indices_test = one_tensor.nonzero(as_tuple=True)[1]
    #print('corr idx',correlation_indices_test)
    ##correlation_flipped_right = torch.flip(correlation_indices_test,dims=[0,1])
    #print('corr_flipped once',correlation_flipped_right)

    ##full_corr_indices = torch.cat((correlation_indices_test,correlation_flipped_right[:,1:7*7-1].view(1,-1)),dim=1).flatten()

    #print(full_corr_indices)

    #print(torch.repeat_interleave(spin_lattices[:,0].view(-1,1),lattice_size*lattice_size,dim=1).size())

    ##testing corr change

    corr_distances = np.sqrt(((correlation_indices_test)%9)**2+((correlation_indices_test)//9)**2).view(9, 9)

    #print('corr distances',corr_distances)

    ##distance_flipped_right = torch.flip(corr_distances,dims=[0,1]) for diga down

    distance_flipped_right = torch.fliplr(corr_distances)

    ##print('distance flipped', distance_flipped_right)

    full_corr_indices_up = torch.cat((corr_distances,distance_flipped_right[:,1:9-1]),dim=1)

    ##print('full up corr', full_corr_indices_up)

    distance_flipped_down = torch.fliplr(torch.flip(full_corr_indices_up,dims=[1,0]))

    ##print('full up corr filpped ', distance_flipped_down)

    full_corr = torch.cat((full_corr_indices_up,distance_flipped_down[1:9-1,:]),dim=0)

    #print(full_corr)

    ##full_corr_right = torch.roll(full_corr,1, dims=1)

    ##full_corr_down = torch.roll(full_corr_right,2, dims=0)

    ##print(full_corr_down)

    ##full_corr_down = torch.roll(full_corr_right,3, dims=0)

    ##print(full_corr_down)

    

    ##print(full_corr[1,2])





    simulation_observables = Observables(lattice_size=lattice_size, temperature = temperature)









    sim_data_correlation_function = simulation_observables.correlation_function_even_faster_new_try(spin_fields=spin_lattices)

    #print(sim_data_correlation_function['6.0'])
    #print(sim_data_correlation_function['7.0'])
    #print(sim_data_correlation_function['8.0'])



    ##print(sim_data_correlation_function)



                            
    ##training_correlation_data= {np.sqrt(d_x**2+d_y**2):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
                        #print(training_correlation_data)
    ##training_correlation_data.pop(0.0)


    ##training_correlation_data_key_list = list(training_correlation_data.keys())
    ##training_correlation_data_key_list.sort()
                        #print(training_correlation_data_key_list)
                        #print(training_correlation.keys)
                        #print(training_correlation_data.keys())
    distance_parameter = [float(key) for key in sim_data_correlation_function.keys()]
    distance_parameter.sort() 
    training_mean_correlation_data_sorted = [torch.flatten(torch.mean(sim_data_correlation_function['{}'.format(k)])).cpu().numpy() for k in distance_parameter]
    #print(training_mean_correlation_data_sorted)
    
    #training_mean_correlation_data_sorted.insert(0,np.array([1.0]))
                    #print(training_mean_correlation_data_sorted)
    ##training_var_correlation_data_sorted = [torch.flatten(torch.mean(torch.pow(sim_data_correlation_function['{}'.format(k)],2.0))-torch.pow(torch.mean(sim_data_correlation_function['{}'.format(k)]),2.0)).cpu().numpy() for k in distance_parameter]
    training_var_correlation_data_sorted = [torch.sum(torch.pow(sim_data_correlation_function['{}'.format(k)]-training_mean_correlation_data_sorted[idx],2))/float((nmb_sample-1)) for idx,k in enumerate(distance_parameter)]
    #print(training_var_correlation_data_sorted)
    #training_var_correlation_data_sorted.insert(0,np.array([0.0]))
    ##distance ={np.sqrt(d_x**2+d_y**2):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
    ##                    #distance.pop(0.0)

    ##maximal_distance_correlation = int(simulation_observables.max_corr_distance())
    ##map_tensor_out_put = torch.ones(1,maximal_distance_correlation*maximal_distance_correlation)
    ##correlation_indices_output = map_tensor_out_put.nonzero(as_tuple=True)[1]

    ##distances = np.sqrt(((correlation_indices_output)%maximal_distance_correlation)**2+((correlation_indices_output)//maximal_distance_correlation)**2).view(maximal_distance_correlation*maximal_distance_correlation)
    ##distances.sort()


    #print(training_mean_correlation_data_sorted)
    #print(distance_parameter)
    #print(training_var_correlation_data_sorted)

    

    sb.set_theme(style='whitegrid',palette='gist_earth')

    fig, ax = plt.subplots()


    ax.plot(np.array(distance_parameter).flatten(),np.array(training_mean_correlation_data_sorted).flatten(),marker='o',markersize=3.5,label='T=0.1',linestyle='dashed',linewidth=3,color='#762A83')
    ax.fill_between(np.array(distance_parameter).flatten(),np.array(training_mean_correlation_data_sorted).flatten()-np.array(training_var_correlation_data_sorted).flatten(),np.array(training_mean_correlation_data_sorted).flatten()+np.array(training_var_correlation_data_sorted).flatten(),alpha=0.3,color='#762A83')


    x=np.array(distance_parameter).flatten()

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                        ncol=3, fancybox=True, shadow=True)



    plt.xlabel('Distance r')
    plt.ylabel(r"$\langle \cos(\vartheta_0-\vartheta_r)\rangle $")
    plt.xticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    #ax.set_xlim([0.0, 11.0])
    plt.show()
    #plt.savefig('corr_function_test_{}.png'.format(nmb_sample))


def kullback_leibler(observable_truth, observable_compare):

    #KLD = np.where(np.logical_and(observable_truth > 0.,observable_compare > .0),observable_truth*np.log(np.divide(observable_truth,observable_compare)),)
    
    if np.all(observable_truth > 0.) and np.all(observable_compare > .0):

        kld = observable_truth*np.log(np.divide(observable_truth,observable_compare))

        change_value = 0.0 
        return kld, change_value
    else:
        minmal_value = min([np.min(observable_truth),np.min(observable_compare)])

        change_value = minmal_value + 1e-16

        changed_truth = observable_truth + np.abs(minmal_value) +change_value

        changed_compare = observable_compare + np.abs(minmal_value) + change_value

        kld = changed_truth*np.log(np.divide(changed_truth,changed_compare))

        return kld, change_value