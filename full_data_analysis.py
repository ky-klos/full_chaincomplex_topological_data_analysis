import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import random

from matplotlib.cm import get_cmap
from colorsys import hls_to_rgb

from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import h5py


import pandas as pd




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
    






from training_september_25_fixed_sum_fourier import CircularUpscaleConv2d, ResidualBlock, AdaIN, UnetGenerator, CircularPad2d, SumPool2d, InceptionConv2d_mini


def full_analysis(device,measures,simulation_data_path,outputpath,data_type, sim_type,defect_nmb,temperatures,output_data_path,new_data=False,new_basis_data=False,less_idx_list = [False,True],configuration_attributes_dict=None,gen_folder='./',save_basis_data=False,larger=False,compare_defect_lattice=False,full_sample_size =1000,tda_sample_size=100,noise_size = 1.0,epochs=[-1],lattice_size=16,training_data_nmb=350000,nmb_label=10):
    from filtration_function import TopologicalAnalysis
    from observables_analysis_class_26 import Observables
    import os
    for t_idx,t in enumerate(temperatures):
            if 'statistical_analysis' in measures:
                from observables_calculation_26 import calculation_observables,output_analysis_basis
                calculation_observables(data_type=data_type,output_data_path=output_data_path,save_basis_data=save_basis_data,gen_folder= gen_folder,sim_data_type=sim_type,larger=larger,defect_nmb=defect_nmb,compare_defect_lattice=compare_defect_lattice,noise_size = noise_size, training_data_nmb=training_data_nmb,nmb_label = nmb_label,new_basis_data=new_basis_data,new_data=new_data,data_path='', epochs=epochs, configuration_attributes_dict=configuration_attributes_dict,simulation_data_path  =simulation_data_path, temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=full_sample_size,rotation=False,later_rotation=True)
            if 'graph_analysis' in measures:
                from filtration_function_output_data_faster_july_24 import get_edge_state_graphs_sim_gen
                get_edge_state_graphs_sim_gen(output_data_path=outputpath, simulation_data_path=simulation_data_path, configuration_attributes_dict=configuration_attributes_dict, nmb_label=10,data_type=data_type,temperature=t,gen_folder=gen_folder,sim_data_type=sim_type,nmb_defects =2,noise_size = noise_size, training_data_nmb=training_data_nmb,lattice_size=lattice_size, new_data= new_data, compare_defect_lattice=compare_defect_lattice,further_simulated=False,device=device,samplesize = full_sample_size,larger=larger,epochs=epochs)
            if 'tda_analysis' in measures:
                from topological_analysis_just_measures_26 import get_topological_measures_mean
                if less_idx_list[1]:
                    less_index = fixed_idx(basic_sample_size=full_sample_size, reduced_samplesize= tda_sample_size, temperature = t, new=less_idx_list[0],data_type=sim_type,defect_nmb=defect_nmb)
                else:
                    less_index = None
                get_topological_measures_mean(temperature=t,extra_idx=0, output_data_path=output_data_path,configuration_attributes_dict=configuration_attributes_dict,training_data_nmb=training_data_nmb, simulated_data_path=simulation_data_path, device=device, less_index=less_index,epsilon_size=2000,sim_data_type=sim_type,basis_sample_size=full_sample_size,number_defects=defect_nmb,noise_size=noise_size,samplesize=tda_sample_size,lattice_size=lattice_size,fixed_data=True,data_type_list = [data_type], name_idxs = None)

if __name__ == "__main__":
    import yaml
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.set_default_device(device)
    with open('configuration.yaml', 'r') as file:
        configuration_attributes_dict = yaml.safe_load(file)
        data_type = 'training_data'
        sim_type = 'full_pinned'
        defect_nmb = 2
        full_sample_size = 1000
        tda_sample_size = 100
        epochs = [-1]
        temps = [0.1]
        lattice_size = 16
        measures = ['statistical_analysis','graph_analysis','tda_analysis']
        simulation_data_path = './'
        output_data_path = './'
        gen_folder = './'
        full_analysis(device=device,measures=measures,simulation_data_path=simulation_data_path,data_type=data_type, sim_type=sim_type,defect_nmb=defect_nmb,temperatures=temps,output_data_path=output_data_path,new_data=True,new_basis_data=False,less_idx_list = [False,True],configuration_attributes_dict=configuration_attributes_dict,gen_folder=gen_folder,save_basis_data=True,full_sample_size =full_sample_size,tda_sample_size=tda_sample_size,epochs=epochs,lattice_size=lattice_size)

