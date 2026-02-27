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

from basis_data_production import load_in_data_basis



def get_edge_state_graphs_sim_gen(output_data_path,data_path, simulation_data_path, configuration_attributes_dict, nmb_label,data_type,temperature,gen_folder=None,sim_data_type='full_pinned',nmb_defects =2,noise_size = 1.0, training_data_nmb=100000,lattice_size=16, new_data= True, compare_defect_lattice=False,further_simulated=False,device='cpu',samplesize = 1000,larger=False,epochs=[-1]):
    with torch.no_grad():
            

        oservable_data_spins, oservable_data_defects, path_name, name_list = load_in_data_basis(temperature=temperature, lattice_size=lattice_size, device=device, data_type=data_type, simulation_data_path=simulation_data_path, sim_data_type=sim_data_type, new_data=new_data,new=new_data, compare_defect_lattice=compare_defect_lattice, samplesize=samplesize, defect_nmb=nmb_defects, larger=larger, further_simulated=further_simulated, data_path=data_path, configuration_attributes_dict=configuration_attributes_dict, training_data_nmb=training_data_nmb, noise_size=noise_size, nmb_label=nmb_label, epochs=epochs, output_data_path=output_data_path, gen_folder=gen_folder)

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
    pass