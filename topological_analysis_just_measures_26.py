

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

from numba import jit
from numba.typed import List

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
from matplotlib import colors
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

import plotly.express as px

import matplotlib as mpl

import time

##import torch.multiprocessing as mp
##mp.set_start_method('spawn', force=True)

##from filtration_function_output_data_faster_july_24 import get_edge_state_graphs_sim_gen

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15


##parameter
lattice_size = 16
##-------------
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##torch.set_default_device(device)

##helpier functions--------------

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

#def make_edges(x,y):
@jit(nopython=True)
def minkovski_with_edges_numba(cluster_list):

    #liste mit anzahl 3,2,1 in dim 2 von unterliste
    #now also with 4
    result_before = List()
    result = List()
    for arr in (cluster_list):
        d = 0
        
        
        result_before.append(np.sum(np.logical_and(arr[:,2]>=2,arr[:,2]<4))-np.sum(arr[:,2]==1))
        result.append(np.sum(arr[:,2]>=2)-np.sum(arr[:,2]==1))

    return result_before,result

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



##
##simulated data
def cluster_faster(input_plaquette_tensor,device):

    rows, cols = input_plaquette_tensor.size()

    clusters = []

    visited = torch.zeros_like(input_plaquette_tensor, dtype=bool, device= device)

    def explore_clusters(row, col, cluster,cluster_counter):
            
        if not (0<=row<rows) or not (0<= col < cols) or visited[row,col] or input_plaquette_tensor[row, col] == 0:
            return
        visited[row,col] = True
        cluster.append(torch.tensor([row,col,cluster_counter+1], device = device))

        for i, j in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_row=(row +i) % rows
            new_col = (col+j) % cols
            explore_clusters(new_row,new_col, cluster,cluster_counter)

    cluster_counter = 0

    for i in range(rows):
        for j in range(cols):
            if input_plaquette_tensor[i,j] == 1 and not visited[i,j]:
                current_cluster = []
                explore_clusters(i,j,current_cluster,cluster_counter)
                clusters.append(torch.stack(current_cluster, dim=0))
                cluster_counter += 1


    return clusters

@jit(nopython=True)
def run_bfs_edge_numba(img, x, y, visited):
    height, width = img.shape
    cluster = List()
    cluster.append((x, y, img[x, y]))
    queue = List()
    queue.append((x, y))
    
    while queue:
        x, y = queue.pop(0)
        
        for dx in (-1, 1):
            x_new = (x + dx + height) % height
            
            if img[x_new, y] >= 1 and (x_new, y) not in visited:
                visited.add((x_new, y))
                queue.append((x_new, y))
                cluster.append((x_new, y, img[x_new, y]))
        
        for dy in (-1, 1):
            y_new = (y + dy + width) % width
            
            if img[x, y_new] >= 1 and (x, y_new) not in visited:
                visited.add((x, y_new))
                queue.append((x, y_new))
                cluster.append((x, y_new, img[x, y_new]))
    
    return cluster

@jit(nopython=True)
def cluster_image_bfs_edge_numba(img):
    height, width = img.shape
    visited = set()
    cluster_list = List()
    
    for x_coord in range(height):
        for y_coord in range(width):
            if (x_coord, y_coord) not in visited and img[x_coord, y_coord] >= 1:
                visited.add((x_coord, y_coord))
                cluster = run_bfs_edge_numba(img, x_coord, y_coord, visited)
                cluster_list.append(cluster)
    
    return cluster_list

def cluster_faster_edges(input_edge_tensor_x, input_edge_tensor_y,device):

    rows, cols = input_plaquette_tensor.size()

    clusters = []

    visited = torch.zeros_like(input_plaquette_tensor, dtype=bool, device= device)

    def explore_clusters(row, col, cluster,cluster_counter):
            
        if not (0<=row<rows) or not (0<= col < cols) or visited[row,col] or input_plaquette_tensor[row, col] == 0:
            return
        visited[row,col] = True
        cluster.append(torch.tensor([row,col,cluster_counter+1], device = device))

        for i, j in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_row=(row +i) % rows
            new_col = (col+j) % cols
            explore_clusters(new_row,new_col, cluster,cluster_counter)

    cluster_counter = 0

    for i in range(rows):
        for j in range(cols):
            if input_plaquette_tensor[i,j] == 1 and not visited[i,j]:
                current_cluster = []
                explore_clusters(i,j,current_cluster,cluster_counter)
                clusters.append(torch.stack(current_cluster, dim=0))
                cluster_counter += 1


    return clusters

def run_bfs(img, x, y, visited):
 
    size = img.shape
    cluster = [[x,y]]
    queue = [(x, y)]
 
    while queue:
        x, y = queue.pop(0)
       
        # check neighbors with periodic boundary
        for dx in [-1,1]:
            x_new = (x + dx + size[0]) % size[0]
 
            if img[x_new][y] == 1 and (x_new, y) not in visited:
                visited.add((x_new, y))
                queue.append((x_new, y))
                cluster.append([x_new, y])
       
        for dy in [-1,1]:
            y_new = (y + dy + size[1]) % size[1]
 
            if img[x][y_new] == 1 and (x, y_new) not in visited:
                visited.add((x, y_new))
                queue.append((x, y_new))
                cluster.append([x, y_new])
 
 
       
 
    return cluster
 
def cluster_image_bfs(img):
 
    size = img.shape
 
    visited = set()
    cluster_list = []
 
    for x_coord in range(size[0]):
        for y_coord in range(size[1]):
 
            if img[x_coord][y_coord] == 1:
 
                if (x_coord, y_coord) in visited:
                    continue
                else:
                    visited.add((x_coord, y_coord))
                    cluster = run_bfs(img, x_coord, y_coord, visited)
                    cluster_list.append(cluster)
 
 
    return cluster_list

def run_bfs_edge(img, x, y, visited):
 
    size = img.shape
    cluster = [[x,y,img[x][y]]]
    queue = [(x, y)]
 
    while queue:
        x, y = queue.pop(0)
       
        # check neighbors with periodic boundary
        for dx in [-1,1]:
            x_new = (x + dx + size[0]) % size[0]
 
            if img[x_new][y] >= 1 and (x_new, y) not in visited:
                visited.add((x_new, y))
                queue.append((x_new, y))
                cluster.append([x_new, y,img[x_new][y]])
       
        for dy in [-1,1]:
            y_new = (y + dy + size[1]) % size[1]
 
            if img[x][y_new] >= 1 and (x, y_new) not in visited:
                visited.add((x, y_new))
                queue.append((x, y_new))
                cluster.append([x, y_new,img[x][y_new]])
 
 
       
 
    return cluster
 
def cluster_image_bfs_edge(img):
 
    size = img.shape
 
    visited = set()
    cluster_list = []
    
    ##print(type(img))
    ##print((img == 1.0).any())

 
    for x_coord in range(size[0]):
        for y_coord in range(size[1]):
    
            if img[x_coord][y_coord] >= 1:
    
                if (x_coord, y_coord) in visited:
                    continue
                else:
                    visited.add((x_coord, y_coord))
                    cluster = run_bfs_edge(img, x_coord, y_coord, visited)
                    cluster_list.append(cluster)
 
    
    return cluster_list

def run_edges(edge_x,edge_y, x, y, visited,type):#type 0 for x, 1 for y
 
    size = edge_x.shape
    cluster = [[x,y,type]]
    queue = [(x, y)]
 
    while queue:
        x, y = queue.pop(0)
       
        # check neighbors with periodic boundary
        for dx in [-1,1]:
            x_new = (x + dx + size[0]) % size[0]
 
            if edge_x[x_new][y] == 1 and (x_new, y) not in visited:
                visited.add((x_new, y))
                queue.append((x_new, y))
                cluster.append([x_new, y])
       
        for dy in [-1,1]:
            y_new = (y + dy + size[1]) % size[1]
 
            if img[x][y_new] == 1 and (x, y_new) not in visited:
                visited.add((x, y_new))
                queue.append((x, y_new))
                cluster.append([x, y_new])
 
 
       
 
    return cluster
 
def cluster_image_edges(edge_x, edge_y):
 
    size = edge_x.shape
 
    visited = set()
    cluster_list = []
 
    for x_coord in range(size[0]):
        for y_coord in range(size[1]):
 
            if edge_x[x_coord][y_coord] == 1:
 
                if (x_coord, y_coord,0) in visited:
                    continue
                else:
                    visited.add((x_coord, y_coord,0))#0 means here x
                    cluster = run_edges(edge_x,edge_y, x_coord, y_coord, visited,edge_type=0)
                    cluster_list.append(cluster)

            if edge_y[x_coord][y_coord] == 1:
 
                if (x_coord, y_coord,1) in visited:
                    continue
                else:
                    visited.add((x_coord, y_coord, 1))
                    cluster = run_edges(edge_x,edge_y, x_coord, y_coord, visited,edge_type=1)
                    cluster_list.append(cluster)
 
 
    return cluster_list



def cluster_faster_alternative(input_plaquette_tensor):

    batchsize, rows, cols = input_plaquette_tensor.size()

    print(batchsize)
    print(rows)
    print(cols)

    reshape_tensor = input_plaquette_tensor.view(batchsize,-1)

    ##row_indices = torch.arange(rows).view(-1,1).repeat(1,cols).view(-1)
    ##col_indices = torch.arange(cols).repeat(rows)

    ##indices = torch.arange(reshape_tensor.numel()).view(1,-1)
    indices = reshape_tensor.nonzero(as_tuple=True)[1]


    print(indices)

    row_idx = (indices/cols).long()

    col_idx = (indices%cols).long()

    non_zero_mask = reshape_tensor != 0

    print(non_zero_mask)

    row_idx = (row_idx-1) % rows

    col_idx = (col_idx-1) % cols

    n_row=(row_idx.repeat(4,1)+torch.tensor([-1,1,0,0]).view(-1,1))%rows

    n_col = (col_idx.repeat(4,1)+torch.tensor([0,0,-1,1]).view(-1,1))%cols

    ##neighbor_indices = n_row*cols+n_col

    neighbor_indices = row_idx*cols+col_idx

    print(neighbor_indices)

    ##neighbor_mask = reshape_tensor[:,neighbor_indices]*non_zero_mask.view(batchsize,-1)

    ##print(neighbor_mask.size())

    ##clusters = neighbor_mask.nonzero(as_tuple=True)[1]

    clusters, indices = torch.unique(neighbor_indices, return_inverse= True)

    unique_clusters = [indices[indices==i] for i in range(len(clusters))]

    return unique_clusters

def cluster_faster_other(input_plaquette_tensor, lattice_size=5):

    spins_halo= F.pad(input_plaquette_tensor.float().view(-1,1,lattice_size,lattice_size),(1,1,1,1),'circular')

    nn_maker = nn.Unfold(kernel_size=(3,3),stride=1)

    nn_temp = nn_maker(spins_halo).view(input_plaquette_tensor.size(0),9,lattice_size*lattice_size)

    next_n = torch.transpose(nn_temp, 1,2)

    index = torch.tensor([4,1,3,5,7])

    all_nn = torch.index_select(next_n,dim=2, index=index)

    object_cluster = all_nn!=0

    object_cluster_indices = object_cluster.nonzero(as_tuple=True)

    ##print(object_cluster_indices[:,2])

    #check if zero i


    print('object_cluster_indices',object_cluster_indices)

    return object_cluster_indices





                    
def get_topological_measures_mean(temperature,extra_idx,res_criritc,extra_info, output_data_path,generated_data_attributes,training_data_nmb, simulated_data_path, device, less_index,epsilon_size=2000,sim_data_type='full_pinned',basis_sample_size=1000,number_defects=2,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data','generated_data','random_data','zero_temp'], name_idxs = None):

    import time 
    if fixed_data:
        
        before_sim_load = time.time()
         
         #+'defect_for_analysis_full_distances_{}_sample_{}.h5'.format(temperature,1000)
         
         
        if sim_data_type == 'full_pinned':
            if number_defects > 2 or lattice_size > 16 or temperature > 0.1:
                with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,number_defects,basis_sample_size), 'r') as h5f:
                    name_list = np.array(h5f['distances'])

            else:
                with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,number_defects,basis_sample_size), 'r') as h5f:
                #with h5py.File(simulated_data_path+ 'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_compared.h5'.format(temperature,lattice_size,number_defects), 'r') as h5f:

                    name_list = np.array(h5f['distances'])
        else:
            with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(sim_data_type,temperature,lattice_size,number_defects,basis_sample_size), 'r') as h5f:
                #with h5py.File(simulated_data_path+ 'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_compared.h5'.format(temperature,lattice_size,number_defects), 'r') as h5f:

                name_list = np.array(h5f['distances'])            
                
        after_sim_load = time.time()
        
        print('sim load', after_sim_load-before_sim_load)

    path_name_list = []
    
    if less_index is None:
        less_index = [i for i in range(samplesize)]

    for data in data_type_list:

        if data == 'random_data':
            path_name_list.append('random_data_comparison/')
        elif data == 'zero_temp':
            path_name_list.append('zero_temperature_solution_comparison_new/')
        elif data == 'generated_data':
            path = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(noise_size,training_data_nmb,*generated_data_attributes,0,*extra_info) 
             #path_name_list.append('noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/'.format(noise_size,training_data_nmb,*generated_data_attributes,0)) 
            path = 'gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(*extra_info)
            if res_criritc:
                path += 'res_critic/'
                
            path += 'temp_{}/'.format(temperature)
            print(path)
            path_name_list.append(path)
            
        elif data == 'training_data':
            
            
            path_name_list.append('training_data_full_distance_temp_{}_new_{}/'.format(temperature,sim_data_type))
        elif data == 'full_data':
            path_name_list.append('not_fixed_data_full_temp_{}/'.format(temperature))



    full_list_plaquettes = []
    full_list_edges_x = []
    full_list_edges_y = []
    full_list_defects = []
    full_output_names = []
    full_output_names_long = []

    if name_idxs is None:
        iterable_idxs = [(i,v) for i,v in enumerate(name_list)]
    else:
        iterable_idxs = [(i,name_list[i]) for i in name_idxs]
        
    before_data_load = time.time()
         
    for name_idx, name in iterable_idxs:

        list_plaquettes_per_distance = []
        list_edges_x_per_distance = []
        list_edges_y_per_distance = []
        list_defects_per_distance = []
        list_output_name_per_distance = []
        list_output_name_per_distance_long = []

        for data_idx,data_type in enumerate(data_type_list):

            path_name = path_name_list[data_idx]

            out_put_name = '{}_{}_{}_TDA_Measurements_{}_nmb_defects_{}_latticesize_{}_{}.csv'.format(name,sim_data_type,data_type,samplesize,number_defects,lattice_size,extra_idx)

            out_put_name_long = '{}_{}_{}_TDA_Measurements_{}_nmb_defects_{}_latticesize_{}_{}.h5'.format(name,sim_data_type,data_type,samplesize,number_defects,lattice_size,extra_idx)

            list_output_name_per_distance.append(out_put_name)
            list_output_name_per_distance_long.append(out_put_name_long)


            with h5py.File(output_data_path+ path_name +'{}_{}_Plaquette_edge_graph_rep_before_epsilon_comparison_{}_temp_{}_nmb_defects_{}_latticesize_{}.h5'.format(name,sim_data_type,data_type,temperature,number_defects,lattice_size),'r') as h5f:
                #less_index = random.sample([i for i in range(basis_sample_size)],samplesize)
                plaquettes = torch.from_numpy(np.array(h5f['Plaquette'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
                #print(type(plaquettes))
                edges_x = torch.from_numpy(np.array(h5f['Edge x'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
                edges_y = torch.from_numpy(np.array(h5f['Edge y'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
                defects = torch.from_numpy(np.array(h5f['Defects'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
            print(device)
            list_plaquettes_per_distance.append(plaquettes.to(device))
            list_edges_x_per_distance.append(edges_x.to(device))
            list_edges_y_per_distance.append(edges_y.to(device))
            list_defects_per_distance.append(defects.to(device))
        
        full_list_plaquettes.append(list_plaquettes_per_distance)
        full_list_edges_x.append(list_edges_x_per_distance)
        full_list_edges_y.append(list_edges_y_per_distance)
        full_list_defects.append(list_defects_per_distance)
        full_output_names.append(list_output_name_per_distance)
        full_output_names_long.append(list_output_name_per_distance_long)

    after_data_load = time.time()
    print('data load', after_data_load-before_data_load)
    for n_idx, (name_idx, name) in enumerate(iterable_idxs):

        cluster_edge_dict = {}
        calc_dict = {}
        cluster_calc_dict = {}

        for data_idx,data_type in enumerate(data_type_list):


            topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device,batchsize=samplesize)

            distance_list = []
            plq_percolation_list = []
            edge_percolation_list = []

            plq_area_list = []
            plq_perimeter_list = []
            plq_euler_list = []

            plq_area_mean_per_cluster_list = []
            plq_perimeter_mean_per_cluster_list = []
            plq_euler_mean_per_cluster_list = []

        


            plq_cluster_nmb_list = []
            edge_cluster_nmb_list = []

            edge_epsilon_euler_list= []
            edge_epsilon_euler_list_correct = []



            plq_euler_list_alternative = []
            plq_euler_mean_per_cluster_list_alternative = []


            edge_epsilon_diameter_list= []
            edge_epsilon_radius_list= []

            betti_number_euler_charactistic = []

            #full_euler_list = []

            #start_time = time.time()
            prev_plaq = None
            prev_edge = None

            dictionary_persistence_objects = {}

            dictionary_persistence_holes = {}



            ##epsilon_size = 2000
            angular_filt = (np.pi+0.1)/epsilon_size

            full_cluster_plq_area = {}
            full_cluster_plq_perimeter = {}
            full_cluster_plq_euler = {}
            full_cluster_edge_euler = {}
            full_cluster_edge_diameter = {}
            full_cluster_edge_radius = {}
            for epsilon in range(0,epsilon_size):#3600 0.1 grad steps

                print("EPS:",epsilon)

                
                

                distance_list.append(angular_filt*epsilon)
                    
                plaquette_per_epsilon = full_list_plaquettes[n_idx][data_idx]<=(angular_filt*epsilon)
                plaq_diff = 1
                if prev_plaq is not None:
                    plaq_diff = (torch.sum(prev_plaq!=plaquette_per_epsilon))
                prev_plaq = plaquette_per_epsilon
                #print(plaquette_per_epsilon.shape)

                edges_x_per_epsilon = full_list_edges_x[n_idx][data_idx] <= (angular_filt*epsilon)

                edges_y_per_epsilon = full_list_edges_y[n_idx][data_idx] <= (angular_filt*epsilon)

                full_state_percolation, full_bond_percolation = topological_analysis_tool.percolation_faster(plaquettes= plaquette_per_epsilon.view(-1,lattice_size*lattice_size), edges_x=edges_x_per_epsilon.view(-1,lattice_size*lattice_size), edges_y=edges_y_per_epsilon.view(-1,lattice_size*lattice_size))
                plq_percolation_list.append((full_state_percolation).cpu())
                edge_percolation_list.append((full_bond_percolation).cpu())

                full_edge_matrix_per_epsilon = topological_analysis_tool.make_full_edge_matrix_with_definite_holes(epsilon=0.00175*epsilon, edge_x=full_list_edges_x[n_idx][data_idx].view(-1,lattice_size*lattice_size), edge_y=full_list_edges_y[n_idx][data_idx].view(-1,lattice_size*lattice_size),plaq=full_list_plaquettes[n_idx][data_idx].view(-1,lattice_size*lattice_size))
                #print(full_edge_matrix_per_epsilon[0])
                # check if holes changed at all (4s) or if they stayed the same (persistet) or chluster changed, if vertex is still vertex or know part of cluster, or cluster still same clsuter or part of bigger cluster     
                ##topological_analysis_tool.simplices_ordering(full_edge_matrix_list=full_edge_matrix_per_epsilon,epsilon=epsilon)
                
                full_edge_matrix_per_epsilon_ordering = topological_analysis_tool.make_full_edge_matrix_with_definite_holes(epsilon=0.00175*epsilon, edge_x=full_list_edges_x[n_idx][data_idx].view(-1,lattice_size*lattice_size), edge_y=full_list_edges_y[n_idx][data_idx].view(-1,lattice_size*lattice_size),plaq=full_list_plaquettes[n_idx][data_idx].view(-1,lattice_size*lattice_size),ordering=True)
                topological_analysis_tool.simplices_ordering(full_edge_matrix_list=full_edge_matrix_per_epsilon_ordering,epsilon=epsilon)
                edge_diff = 1
                if prev_edge is not None:
                    edge_diff=(torch.sum(prev_edge!=full_edge_matrix_per_epsilon))
                prev_edge = full_edge_matrix_per_epsilon


 
                full_cluster_list = []

                full_area_plq_list = []
                full_perimter_plq_list = []
                full_euler_plq_list = []

                

                full_cluster_nmb_plq_list = []
                full_cluster_nmb_edge_list = []
                
                full_euler_edge_list = []
                full_euler_edge_list_correct = []

                full_diameter_edge_list = []
                full_radius_edge_list = []
            
                mean_cluster_area_plq_list = []
                mean_cluster_perimter_plq_list = []
                mean_cluster_euler_plq_list = []

                betti_number_euler_charactistic_per_epsilon = []

                mean_cluster_euler_plq_list_alternative = []
                full_euler_plq_list_alternative = []
                per_epsilon_cluster_plq_area = {}
                per_epsilon_cluster_plq_perimeter = {}
                per_epsilon_cluster_plq_euler = {}
                per_epsilon_cluster_edge_euler = {}
                per_epsilon_cluster_edge_diameter = {}
                per_epsilon_cluster_edge_radius = {}


            
                if plaq_diff!=0:
                    for idx, plq in enumerate(plaquette_per_epsilon):

                        hash_val = plq.cpu().numpy().data.tobytes()
                        if hash_val in calc_dict:
                            #print(calc_dict[hash_val])
                            full_euler_plq_list.append(calc_dict[hash_val]['plq_euler'].clone())
                            full_cluster_nmb_plq_list.append(calc_dict[hash_val]['plq_cluster_nmb'].clone())
                            full_area_plq_list.append(calc_dict[hash_val]['plq_area'].clone())
                            full_perimter_plq_list.append(calc_dict[hash_val]['plq_perimeter'].clone())

                            mean_cluster_area_plq_list.append(calc_dict[hash_val]['plq_area_cluster_mean'].clone())
                            mean_cluster_perimter_plq_list.append(calc_dict[hash_val]['plq_perimeter_cluster_mean'].clone())
                            mean_cluster_euler_plq_list.append(calc_dict[hash_val]['plq_euler_cluster_mean'].clone())
                        else:
                            calc_dict[hash_val] = {}

                        
                            cluster_list = cluster_image_bfs(plq.cpu().numpy())##all clusters at the same time



                            ##print(cluster_list)
                            ##print(int(len(cluster_list)))

                            mean_cluster_euler_plq_list_per_epsilon = []
                            mean_cluster_area_plq_list_per_epsilon = []
                            mean_cluster_perimeter_plq_list_per_epsilon = []
                            
                            ##print('pql clusters',int(len(cluster_list)))

                            if int(len(cluster_list)) > 0:


                                cluster_lattices_list = []
                                for cluster in cluster_list :
                                    cluster_tensor = torch.tensor(cluster, device=device)
                                    ##print(cluster_tensor)
                                    ##print(cluster_tensor.size())
                                    cluster_lattices_zero = torch.zeros(lattice_size,lattice_size, device=device)
                                    cluster_lattices_zero[cluster_tensor[:,0],cluster_tensor[:,1]]=1
                                    euler_nmb = topological_analysis_tool.euler_charactistics_faster(cluster_lattices_zero)
                                    mean_cluster_euler_plq_list_per_epsilon.append(torch.tensor([euler_nmb],dtype=float,device=device))
                                    #cluster_lattices_list.append(cluster_lattices_zero)
                                    area_clusters, perimeter_clusters = topological_analysis_tool.minkowski_measure_new_faster(cluster_lattices_zero.view(1,lattice_size,lattice_size))
                                    mean_cluster_area_plq_list_per_epsilon.append(area_clusters)
                                    mean_cluster_perimeter_plq_list_per_epsilon.append(perimeter_clusters)

                                full_euler_plq_list.append(torch.flatten(torch.sum(torch.stack(mean_cluster_euler_plq_list_per_epsilon))))
                                mean_cluster_euler_plq_list.append(torch.flatten(torch.mean(torch.stack(mean_cluster_euler_plq_list_per_epsilon))))
                                full_cluster_nmb_plq_list.append(torch.flatten(torch.tensor([float(len(cluster_list))],device=device)))
                                per_epsilon_cluster_plq_area[idx] = torch.stack(mean_cluster_area_plq_list_per_epsilon)
                                per_epsilon_cluster_plq_perimeter[idx] = torch.stack(mean_cluster_perimeter_plq_list_per_epsilon)
                                per_epsilon_cluster_plq_euler[idx] = torch.stack(mean_cluster_euler_plq_list_per_epsilon)
                                #print(torch.stack(cluster_lattices_list,dim=0).size())

                                
                                #only possible because our cluster do not overlap
                                full_area_plq_list.append(torch.flatten(torch.sum(torch.stack(mean_cluster_area_plq_list_per_epsilon))))
                                full_perimter_plq_list.append(torch.flatten(torch.sum(torch.stack(mean_cluster_perimeter_plq_list_per_epsilon))))

                                mean_cluster_area_plq_list.append(torch.flatten(torch.mean(torch.stack(mean_cluster_area_plq_list_per_epsilon))))
                                mean_cluster_perimter_plq_list.append(torch.flatten(torch.mean(torch.stack(mean_cluster_perimeter_plq_list_per_epsilon))))


                                #save position in dict
                                calc_dict[hash_val]['plq_euler'] = full_euler_plq_list[-1]
                                calc_dict[hash_val]['plq_area'] = full_area_plq_list[-1]
                                calc_dict[hash_val]['plq_perimeter'] = full_perimter_plq_list[-1]

                                calc_dict[hash_val]['plq_euler_cluster_mean'] = mean_cluster_euler_plq_list[-1]
                                calc_dict[hash_val]['plq_area_cluster_mean'] = mean_cluster_area_plq_list[-1]
                                calc_dict[hash_val]['plq_perimeter_cluster_mean'] = mean_cluster_perimter_plq_list[-1]

                                calc_dict[hash_val]['plq_cluster_nmb'] = full_cluster_nmb_plq_list[-1]


                            else:
                                full_euler_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                                full_cluster_nmb_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                                full_area_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                                full_perimter_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                                per_epsilon_cluster_plq_area[idx] = torch.flatten(torch.tensor([0.0],device=device))
                                per_epsilon_cluster_plq_perimeter[idx] = torch.flatten(torch.tensor([0.0],device=device))
                                per_epsilon_cluster_plq_euler[idx] = torch.flatten(torch.tensor([0.0],device=device))

                                mean_cluster_euler_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                                mean_cluster_area_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                                mean_cluster_perimter_plq_list.append(torch.flatten(torch.tensor([0.0],device=device)))

                                calc_dict[hash_val]['plq_euler'] = full_euler_plq_list[-1]
                                calc_dict[hash_val]['plq_area'] = full_area_plq_list[-1]
                                calc_dict[hash_val]['plq_perimeter'] = full_perimter_plq_list[-1]

                                calc_dict[hash_val]['plq_euler_cluster_mean'] = mean_cluster_euler_plq_list[-1]
                                calc_dict[hash_val]['plq_area_cluster_mean'] = mean_cluster_area_plq_list[-1]
                                calc_dict[hash_val]['plq_perimeter_cluster_mean'] = mean_cluster_perimter_plq_list[-1]

                                calc_dict[hash_val]['plq_cluster_nmb'] = full_cluster_nmb_plq_list[-1]




                    plq_area_list.append((torch.stack(full_area_plq_list,dim=0)).cpu())
                    plq_perimeter_list.append((torch.stack(full_perimter_plq_list,dim=0)).cpu())
                    plq_cluster_nmb_list.append((torch.stack(full_cluster_nmb_plq_list, dim=0)).cpu())
                    plq_euler_list.append((torch.stack(full_euler_plq_list, dim =0)).cpu())

                    plq_area_mean_per_cluster_list.append((torch.stack(mean_cluster_area_plq_list,dim=0)).cpu())
                    plq_perimeter_mean_per_cluster_list.append((torch.stack(mean_cluster_perimter_plq_list,dim=0)))
                    plq_euler_mean_per_cluster_list.append((torch.stack(mean_cluster_euler_plq_list, dim =0)))
                    full_cluster_plq_area[epsilon] = per_epsilon_cluster_plq_area
                    full_cluster_plq_perimeter[epsilon] =per_epsilon_cluster_plq_perimeter
                    full_cluster_plq_euler[epsilon] = per_epsilon_cluster_plq_euler

                else:
                    plq_area_list.append(plq_area_list[-1])
                    plq_perimeter_list.append(plq_perimeter_list[-1])
                    plq_cluster_nmb_list.append(plq_cluster_nmb_list[-1])
                    plq_euler_list.append(plq_euler_list[-1])

                    plq_area_mean_per_cluster_list.append(plq_area_mean_per_cluster_list[-1])
                    plq_perimeter_mean_per_cluster_list.append(plq_perimeter_mean_per_cluster_list[-1])
                    plq_euler_mean_per_cluster_list.append(plq_euler_mean_per_cluster_list[-1])
                    full_cluster_plq_area[epsilon] = full_cluster_plq_area[epsilon-1]
                    full_cluster_plq_perimeter[epsilon] =full_cluster_plq_perimeter[epsilon-1]
                    full_cluster_plq_euler[epsilon] = full_cluster_plq_euler[epsilon-1]
                if edge_diff!=0:
                    for idx, edge in enumerate(full_edge_matrix_per_epsilon):

                        
                        full_edge_mat_numpy = full_edge_matrix_per_epsilon[idx].cpu().numpy()
                        hashVal = full_edge_mat_numpy.data.tobytes()

                        if hashVal in cluster_edge_dict:
                            pass
                        else:
                            cluster_edge_dict[hashVal] = cluster_image_bfs_edge(full_edge_mat_numpy)
                    
                        cluster_edge_list = cluster_edge_dict[hashVal]

                        full_cluster_nmb_edge_list.append(torch.tensor([float(len(cluster_edge_list))]))





                        full_euler_plqauette_list_per_epsilon = []

                        #full_euler_list_per_epsilon = []
                        #full_euler_list_per_epsilon_correct = []

                        full_diameter_list_per_epsilon = []
                        full_radius_list_per_epsilon = []

                        hole_counter = []

                        vertex_in_cluster_counter = []

                        
                        ##print('clusters',int(len(cluster_edge_list)))
                        

                        if int(len(cluster_edge_list)) > 0:
                            full_euler_list_per_epsilon,full_euler_list_per_epsilon_correct = minkovski_with_edges_numba([np.array(c) for c in cluster_edge_list])

                            cluster_lattices_list = []
                            for cluster_idx,cluster in enumerate(cluster_edge_list):
                                #print(cluster)
                                clusterHash = str(cluster)
                                if clusterHash not in cluster_calc_dict:
                                    cluster_calc_dict[clusterHash] = topological_analysis_tool.diameter_radius_graph(cluster)
                                diameter,radius = cluster_calc_dict[clusterHash]
                                
                                full_diameter_list_per_epsilon.append(torch.tensor([diameter],dtype=float,device=device))
                                full_radius_list_per_epsilon.append(torch.tensor([radius],dtype=float,device=device))

                                #print(cluster[:][2])

                                cluster_tensor = torch.tensor(cluster)

                                hole_counter.append((cluster_tensor[:,2]==4).sum())
                                vertex_in_cluster_counter.append((cluster_tensor[:,2]==2).sum())

                                #if torch.sum(cluster_tensor[:,2]==3) > 1:
                                #    
                                #    
                                #    if (torch.sum(cluster_tensor[:,2]==4) > 0) and (torch.sum(cluster_tensor[:,2]==3) >= 4):
                                #
                                #        xpositions = cluster_tensor[cluster_tensor[:,2]==4,0]
                                #        ypositions = cluster_tensor[cluster_tensor[:,2]==4,1]
                                #
                                #        cluster_tensor[cluster_tensor[:,2]==3,0]
                                #
                                #    else:
                                #        full_euler_plqauette_list_per_epsilon.append(torch.flatten(torch.tensor([torch.sum(torch.tensor(cluster)[:,2]==3)],device=device)))
                                #
                                #else:
                                #    
                                #    full_euler_plqauette_list_per_epsilon.append(torch.flatten(torch.tensor([0.0],device=device)))

                                #other_euler_nmb = topological_analysis_tool.minkovski_with_edges(cluster)

                                #full_euler_list_per_epsilon.append(torch.tensor([other_euler_nmb],dtype=float,device=device))
                                #cluster_lattices_list.append(cluster_lattices_zero)

                            full_euler_edge_list.append(torch.flatten(torch.mean(torch.tensor(full_euler_list_per_epsilon).float())))
                            full_euler_edge_list_correct.append(torch.flatten(torch.mean(torch.tensor(full_euler_list_per_epsilon_correct).float())))
                            full_diameter_edge_list.append(torch.flatten(torch.mean(torch.stack(full_diameter_list_per_epsilon).float())))
                            full_radius_edge_list.append(torch.flatten(torch.mean(torch.stack(full_radius_list_per_epsilon).float())))
                            
                            objects_nmb = (lattice_size**2-sum(vertex_in_cluster_counter)) + int(len(cluster_edge_list))
                            
                            betti_number_euler_charactistic_per_epsilon.append(torch.flatten(torch.tensor(objects_nmb - sum(hole_counter),device=device)).float())
                            per_epsilon_cluster_edge_euler[idx] =torch.tensor(full_euler_list_per_epsilon_correct).float()
                            per_epsilon_cluster_edge_diameter[idx] = torch.stack(full_diameter_list_per_epsilon).float()
                            per_epsilon_cluster_edge_radius[idx] = torch.stack(full_radius_list_per_epsilon).float()
                            #full_cluster_nmb_plq_list[idx] #number of plq objects 

                            ##mean_cluster_euler_plq_list_alternative.append(torch.flatten(torch.mean(torch.stack(full_euler_plqauette_list_per_epsilon))))
                            ##full_euler_plq_list_alternative.append(torch.flatten(torch.sum(torch.stack(full_euler_plqauette_list_per_epsilon))))



                        else:

                            full_euler_edge_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                            full_diameter_edge_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                            full_radius_edge_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                            full_euler_edge_list_correct.append(torch.flatten(torch.tensor([0.0],device=device)))
                            per_epsilon_cluster_edge_euler[idx] =torch.flatten(torch.tensor([0.0],device=device))
                            per_epsilon_cluster_edge_diameter[idx] = torch.flatten(torch.tensor([0.0],device=device))
                            per_epsilon_cluster_edge_radius[idx] = torch.flatten(torch.tensor([0.0],device=device))

                            betti_number_euler_charactistic_per_epsilon.append(torch.flatten(torch.tensor([lattice_size**2],device=device)).float())

                            ##mean_cluster_euler_plq_list_alternative.append(torch.flatten(torch.tensor([0.0],device=device)))
                            ##full_euler_plq_list_alternative.append(torch.flatten(torch.tensor([0.0],device=device)))




                    #print(full_euler_edge_list)
                    edge_epsilon_euler_list.append((torch.stack(full_euler_edge_list, dim =0)).cpu())
                    edge_epsilon_euler_list_correct.append((torch.stack(full_euler_edge_list_correct, dim =0)).cpu())

                    edge_epsilon_diameter_list.append((torch.stack(full_diameter_edge_list, dim =0)).cpu())
                    edge_epsilon_radius_list.append((torch.stack(full_radius_edge_list, dim =0)).cpu())
                    edge_cluster_nmb_list.append((torch.stack(full_cluster_nmb_edge_list, dim =0)).cpu())

                    full_cluster_edge_euler[epsilon] =per_epsilon_cluster_edge_euler
                    full_cluster_edge_diameter[epsilon] = per_epsilon_cluster_edge_diameter
                    full_cluster_edge_radius[epsilon] = per_epsilon_cluster_edge_radius



                    betti_number_euler_charactistic.append((torch.stack(betti_number_euler_charactistic_per_epsilon, dim =0)).cpu())

                    ##plq_euler_mean_per_cluster_list_alternative.append(torch.mean(torch.stack(mean_cluster_euler_plq_list_alternative, dim =0)).cpu().numpy())

                    ##plq_euler_list_alternative.append(torch.mean(torch.stack(full_euler_plq_list_alternative, dim =0)).cpu().numpy())
                    
                    #old school witzh betti numbers just count objects but everyone, also singular vertices
                    #betti_number_euler_charactistic.
                else:
                    edge_epsilon_euler_list.append(edge_epsilon_euler_list[-1])
                    edge_epsilon_euler_list_correct.append(edge_epsilon_euler_list_correct[-1])
                    edge_epsilon_diameter_list.append(edge_epsilon_diameter_list[-1])
                    edge_epsilon_radius_list.append(edge_epsilon_radius_list[-1])
                    edge_cluster_nmb_list.append(edge_cluster_nmb_list[-1])

                    full_cluster_edge_euler[epsilon] = full_cluster_edge_euler[epsilon-1]
                    full_cluster_edge_diameter[epsilon] = full_cluster_edge_diameter[epsilon-1]
                    full_cluster_edge_radius[epsilon] = full_cluster_edge_radius[epsilon-1]
                    ##plq_euler_mean_per_cluster_list_alternative.append(plq_euler_mean_per_cluster_list_alternative[-1])
                    ##plq_euler_list_alternative.append(plq_euler_list_alternative[-1])

                    betti_number_euler_charactistic.append(betti_number_euler_charactistic[-1])

            ##topological_analysis_tool.boundary_matrices()
            ##reduced_edge_boundary_matrix,reduced_plq_boundary_matrix = topological_analysis_tool.reduced_boundary_matrices()

            ##h_0_info_birth, h0_info_death, h0_info_unpaired = topological_analysis_tool.get_persistent_info_of_boundary_matrix(reduced_edge_boundary_matrix, homology_group=0)
            ##h_1_info_birth, h1_info_death, h1_info_unpaired = topological_analysis_tool.get_persistent_info_of_boundary_matrix(reduced_plq_boundary_matrix, homology_group=1)

            ##out_put_name_ph_barcode = '{}_{}_TDA_Measurements_barcode_{}.csv'.format(name,data_type,samplesize)

            ##tda_barcodes = {'birth_h0': h_0_info_birth, 'death_h0':h0_info_death, 'unpaired_0':h0_info_unpaired,
            #                'birth_h1': h_1_info_birth, 'death_h1':h1_info_death, 'unpaired_1':h1_info_unpaired}

            with h5py.File(output_data_path+path_name_list[data_idx]+full_output_names_long[n_idx][data_idx],'w') as h5f:

                dataset_epsilon = h5f.create_dataset('epsilon', shape=(epsilon_size,), dtype='float')
                dataset_epsilon[:] = torch.tensor(distance_list).cpu().numpy()

                dataset_Plq_area_sum = h5f.create_dataset('Plq_area_sum', shape=(epsilon_size,samplesize), dtype='float')
                dataset_Plq_area_sum[:] = torch.stack(plq_area_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_Plq_area_mean = h5f.create_dataset('Plq_area_mean', shape=(epsilon_size,samplesize), dtype='float')
                dataset_Plq_area_mean[:] = torch.stack(plq_area_mean_per_cluster_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_Plq_perimeter_sum = h5f.create_dataset('Plq_perimeter_sum', shape=(epsilon_size,samplesize), dtype='float')
                dataset_Plq_perimeter_sum[:] = torch.stack(plq_perimeter_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_Plq_perimeter_mean = h5f.create_dataset('Plq_perimeter_mean', shape=(epsilon_size,samplesize), dtype='float')
                dataset_Plq_perimeter_mean[:] = torch.stack(plq_perimeter_mean_per_cluster_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_plq_cluster_nmb = h5f.create_dataset('plq_cluster_nmb', shape=(epsilon_size,samplesize), dtype='float')
                dataset_plq_cluster_nmb[:] = torch.stack(plq_cluster_nmb_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_plq_euler_sum = h5f.create_dataset('plq_euler_sum', shape=(epsilon_size,samplesize), dtype='float')
                dataset_plq_euler_sum[:] = torch.stack(plq_euler_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_plq_euler_mean = h5f.create_dataset('plq_euler_mean', shape=(epsilon_size,samplesize), dtype='float')
                dataset_plq_euler_mean[:] = torch.stack(plq_euler_mean_per_cluster_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_plq_percol = h5f.create_dataset('plq_percol', shape=(epsilon_size,samplesize), dtype='float')
                dataset_plq_percol[:] = torch.stack(plq_percolation_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_edge_percol = h5f.create_dataset('edge_percol', shape=(epsilon_size,samplesize), dtype='float')
                dataset_edge_percol[:] = torch.stack(edge_percolation_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_full_euler_new_wo_faces = h5f.create_dataset('full_euler_new_wo_faces', shape=(epsilon_size,samplesize), dtype='float')#does not
                dataset_full_euler_new_wo_faces[:] = torch.stack(edge_epsilon_euler_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_full_euler_old_w_faces = h5f.create_dataset('full_euler_old_w_faces', shape=(epsilon_size,samplesize), dtype='float')#count also holes as faces
                dataset_full_euler_old_w_faces[:] = torch.stack(edge_epsilon_euler_list_correct,dim=0).view(epsilon_size,samplesize).cpu().numpy()
                dataset_bettinumber_euler = h5f.create_dataset('bettinumber_euler', shape=(epsilon_size,samplesize), dtype='float')
                dataset_bettinumber_euler[:] = torch.stack(betti_number_euler_charactistic,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_full_diameter = h5f.create_dataset('full_diameter', shape=(epsilon_size,samplesize), dtype='float')
                dataset_full_diameter[:] = torch.stack(edge_epsilon_diameter_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_full_radius = h5f.create_dataset('full_radius', shape=(epsilon_size,samplesize), dtype='float')
                dataset_full_radius[:] = torch.stack(edge_epsilon_radius_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

                dataset_full_nmb_edge_clusters = h5f.create_dataset('full_nmb_edge_clusters', shape=(epsilon_size,samplesize), dtype='float')
                dataset_full_nmb_edge_clusters[:] = torch.stack(edge_cluster_nmb_list,dim=0).view(epsilon_size,samplesize).cpu().numpy()

            tda_measurements = {'epsilon':distance_list,'Plq_area_sum':torch.mean(torch.stack(plq_area_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'Plq_area_mean':torch.mean(torch.stack(plq_area_mean_per_cluster_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'Plq_perimeter_sum':torch.mean(torch.stack(plq_perimeter_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'Plq_perimeter_mean':torch.mean(torch.stack(plq_perimeter_mean_per_cluster_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'plq_cluster_nmb':torch.mean(torch.stack(plq_cluster_nmb_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'plq_euler_sum':torch.mean(torch.stack(plq_euler_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'plq_euler_mean':torch.mean(torch.stack(plq_euler_mean_per_cluster_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                ##'plq_euler_sum_alternative_v_edge_cluster':plq_euler_list_alternative,'plq_euler_mean_alternative_v_edge_cluster':plq_euler_mean_per_cluster_list_alternative, 
                                'plq_percol':torch.mean(torch.stack(plq_percolation_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'edge_percol':torch.mean(torch.stack(edge_percolation_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'full_euler_new_wo_faces':torch.mean(torch.stack(edge_epsilon_euler_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'full_euler_old_w_faces':torch.mean(torch.stack(edge_epsilon_euler_list_correct,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'bettinumber_euler':torch.mean(torch.stack(betti_number_euler_charactistic,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'full_diameter':torch.mean(torch.stack(edge_epsilon_diameter_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'full_radius':torch.mean(torch.stack(edge_epsilon_radius_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'full_nmb_edge_clusters':torch.mean(torch.stack(edge_cluster_nmb_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy()}
            
            df_tda_measurements = pd.DataFrame.from_dict(tda_measurements)

            ##df_tda_barcodes = pd.DataFrame.from_dict(dict([(k,pd.Series(v)) for k,v in tda_barcodes.items()]))
            #print(full_output_names[n_idx][data_idx])

            df_tda_measurements.to_csv(output_data_path+path_name_list[data_idx]+full_output_names[n_idx][data_idx],index=False)

            ##df_tda_barcodes.to_csv(output_data_path+path_name_list[data_idx]+out_put_name_ph_barcode,index=False)
            
            topological_analysis_tool.boundary_matrices()
            reduced_edge_boundary_matrix,reduced_plq_boundary_matrix = topological_analysis_tool.reduced_boundary_matrices()
            
            h_0_info_birth, h0_info_death, h0_info_unpaired = topological_analysis_tool.get_persistent_info_of_boundary_matrix(reduced_boundary_matrix_plq=reduced_plq_boundary_matrix, reduced_boundary_matrix_edge=reduced_edge_boundary_matrix, homology_group=0)
            h_1_info_birth, h1_info_death, h1_info_unpaired = topological_analysis_tool.get_persistent_info_of_boundary_matrix(reduced_boundary_matrix_plq = reduced_plq_boundary_matrix,reduced_boundary_matrix_edge=reduced_edge_boundary_matrix, homology_group=1)
            
            
            with h5py.File(output_data_path+path_name_list[data_idx] + '{}_{}_TDA_Measurements_barcode_{}_nmb_defects_{}_latticesize_{}_test_again_{}.h5'.format(name,data_type,samplesize,number_defects,lattice_size,extra_idx),'w') as h5f:

                dataset_birth_h0 = h5f.create_dataset('birth_h0', shape=(samplesize,h_0_info_birth.shape[1],h_0_info_birth.shape[2]), dtype='float')
                dataset_birth_h0[:] = h_0_info_birth.reshape(samplesize,2,-1)
                dataset_birth_h1 = h5f.create_dataset('birth_h1', shape=(samplesize,h_1_info_birth.shape[1],h_1_info_birth.shape[2]), dtype='float')
                dataset_birth_h1[:] = h_1_info_birth.reshape(samplesize,2,-1)
                dataset_death_h0 = h5f.create_dataset('death_h0', shape=(samplesize,h0_info_death.shape[1],h0_info_death.shape[2]), dtype='float')
                dataset_death_h0[:] = h0_info_death.reshape(samplesize,2,-1)
                dataset_death_h1 = h5f.create_dataset('death_h1', shape=(samplesize,h1_info_death.shape[1],h1_info_death.shape[2]), dtype='float')
                dataset_death_h1[:] = h1_info_death.reshape(samplesize,2,-1)
                dataset_unpaired_h0 = h5f.create_dataset('unpaired_h0', shape=(samplesize,h0_info_unpaired.shape[1],h0_info_unpaired.shape[2]), dtype='float')
                dataset_unpaired_h0[:] = h0_info_unpaired.reshape(samplesize,2,-1)
                dataset_unpaired_h1 = h5f.create_dataset('unpaired_h1', shape=(samplesize,h1_info_unpaired.shape[1],h1_info_unpaired.shape[2]), dtype='float')
                dataset_unpaired_h1[:] = h1_info_unpaired.reshape(samplesize,2,-1)
                dataset_sample_idx =  h5f.create_dataset('sample_idx', shape=(samplesize,), dtype='float')
                dataset_sample_idx[:] = np.array(less_index)


generated_data_atributes_list_v2 = [
        [0.1,0,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'conv_transposed', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 3, 64, 0, 'L2', (10.0 ,0.0)],
    [0.1,4,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (1.0 ,0.0)],
    [0.1,10,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (10.0 ,0.0)],
     [0.1,10,'adam',(0.9, 0.999), 40, 1.0, 5, 'avg', 'bilinear_interpol', 0, 'leaky_Relu',0.01,0,0.0,0.0,'leaky_Relu',0.01,0.0001,1e-05,0.0001,0.0001, (0.0, 0.0), 2, 64, 0, 'L2', (1.0 ,0.0)]]


#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['random_data'], name_idxs = None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['zero_temp'], name_idxs = None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.2,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.3,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.4,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.5,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.6,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.7,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.8,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
#get_topological_measures_mean(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', device='cpu', temperature=0.9,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs =None)
##get_topological_measures_mean(output_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',generated_data_attributes=generated_data_atributes_list_v2[0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/', device='cuda', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['random_data'], name_idxs = None)
##get_topological_measures_mean(output_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/',generated_data_attributes=generated_data_atributes_list_v2[0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/Pytorch_cuda_code/XY_Model/XY_model_WGAN/Full_training_wgan/observables_output/', device='cuda', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['zero_temp'], name_idxs = None)
#get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=generated_data_atributes_list_v2[1],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs = None)
#get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=generated_data_atributes_list_v2[2],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs = None)
#get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=generated_data_atributes_list_v2[3],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs = None)
###generated_data_atributes_list=[
####         ['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'avg', 'conv_transposed', 0, 'leaky_Relu',0.01,None,0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0)]]
#['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'avg', 'conv_transposed', 0, 'leaky_Relu',0.01,'instance',0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0)]]
###temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
###extra_infos = [[None, 0, 0.5, False,'{}_avg'.format(5)]]#
###import argparse
###if __name__ == '__main__':
###    
###    def fixed_idx(basic_sample_size, reduced_samplesize, temperature, new=True):
###        if new :
###            less_index = random.sample([i for i in range(basic_sample_size)],reduced_samplesize)
###            with open('topological_idx_{}_{}.txt'.format(reduced_samplesize,temperature), 'w') as f:
###                for i in less_index:
###                    f.write(f"{i}\n")
###            return less_index
###        else:
###            less_idx = []
###            file =open('topological_idx_{}_{}.txt'.format(reduced_samplesize,temperature))
###            for i in range(reduced_samplesize):
###                less_idx.append(int(file.readline()[:-1]))
###                
###            print(less_idx)
###            return less_idx
     #parse name_idxs
###    parser = argparse.ArgumentParser(description='Process some integers.')

# ##    #can be None
###    parser.add_argument('--name_idxs', type=int, nargs='+', help='an integer for name idxs to calculate')
###    args = parser.parse_args()
    
###    for att in generated_data_atributes_list:
###    
###        for temp in temps:
###            less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = temp, new=False)
###            get_topological_measures_mean(extra_idx = 0,less_index =less_index,res_criritc= True,extra_info = extra_infos[0],output_data_path='./observables_output/new/',generated_data_attributes=att,basis_sample_size=1000,number_defects=2,noise_size = 1.0, training_data_nmb=350000,simulated_data_path='./observables_output/new/', device='cuda', temperature=temp,samplesize=100,lattice_size=lattice_size,fixed_data=True,data_type_list = ['generated_data'], name_idxs = args.name_idxs)#calculation_observables(extra_info =['instance', 0, 0.5, True],larger=False,defect_nmb=d,compare_defect_lattice=False,c_depth=0, generator_input_dict=generator_input_dict[0],noise_size=1.0,data_type='generated_data',output_data_path='./observables_output/',nmb_label = generated_data_atributes_list[0][1],maximal_distance= 5,new_data=False,data_path='./AdaIn/{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}/res_critic/'.format(*generated_data_atributes_list[0],0, 'instance', 0, 0.5, True), epochs=[120], generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./observables_output/new/', temperature = t, lattice_size= lattice_size,further_simulated=False,samplesize=1000)

            ##get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs = args.name_idxs)
#    get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['random_data'], name_idxs = args.name_idxs)
#    get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['zero_temp'], name_idxs = args.name_idxs)
#     get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=generated_data_atributes_list_v2[0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs = args.name_idxs)
#     get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=generated_data_atributes_list_v2[1],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs = args.name_idxs)
#     get_topological_measures_mean(output_data_path='/localscratch/kyklos/topo_scribt/',generated_data_attributes=generated_data_atributes_list_v2[2],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/localscratch/kyklos/topo_scribt/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['generated_data'], name_idxs = args.name_idxs)

#     #get_topological_measures_mean(output_data_path='/home/jadissel/Programming/KyraCode/NEWNEWNEW/output/',generated_data_attributes=[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0],noise_size = 1.0, training_data_nmb=100000,simulated_data_path='/home/jadissel/Programming/KyraCode/NEWNEWNEW/', device='cpu', temperature=0.1,samplesize=1000,lattice_size=16,fixed_data=True,data_type_list = ['training_data'], name_idxs = args.name_idxs)
    





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

##get_topological_measures_mean(attribute_list=[[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0]],type_list=[ 'no_defect', 'small_defects', 'mid_defects', 'large_defects'],length_list =[1000,1000,1000,1000] ,generated_data_path='/localscratch/kyklos/DPG_24/homeoffice/filtration_function_june_24/', simulated_data_path='/localscratch/kklos/topological_analysis_idea/', device='cpu', temperature=0.1,simulated=True, lattice_size=16)
##get_topological_measures_mean(attribute_list=[[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0]],type_list=[ 'no_defect', 'small_defects', 'mid_defects', 'large_defects'],length_list =[1000,1000,1000,1000] ,generated_data_path='/localscratch/kyklos/DPG_24/homeoffice/filtration_function_june_24/', simulated_data_path='/localscratch/kklos/topological_analysis_idea/', device='cpu', temperature=0.1,simulated=True, lattice_size=16)
##get_topological_measures_mean(attribute_list=[[64,0.0001,2,5,1,10.0,0, 1e-05, 40,2,1,0,1,0,'bilinear', 0.01, 1,0,0,0]],type_list=[ 'no_defect', 'small_defects', 'mid_defects', 'large_defects'],length_list =[1000,1000,1000,1000] ,generated_data_path='/localscratch/kyklos/DPG_24/homeoffice/filtration_function_june_24/', simulated_data_path='/localscratch/kklos/topological_analysis_idea/', device='cpu', temperature=0.1,simulated=False, lattice_size=16)


#compare = 'sim', 'zero'
def plot_topological_measurements(output_data_path,data_type_list, simulated_data_path, generated_data_attributes, device,temperature,sample_size=100,training_data_nmb=100000,noise_size=1.0,compare='sim',full_distances= False,delta_data=False,KLD= False,fixed_data=True):

    if fixed_data:

        with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_sample_1000.h5'.format(temperature), 'r') as h5f:

            name_list = np.array(h5f['distances'])

    path_name_list = []

    for data in data_type_list:

        if data == 'random_data':
            path_name_list.append('random_data_comparison/')
        elif data == 'zero_temp':
            path_name_list.append('zero_temperature_solution_comparison/')
        elif data == 'generated_data':
            path_name_list.append('noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/'.format(noise_size,training_data_nmb,*generated_data_attributes)) 
        elif data == 'training_data':
            path_name_list.append('training_data_full_distance_temp_{}/'.format(temperature))
        elif data == 'full_data':
            path_name_list.append('not_fixed_data_full_temp_{}/'.format(temperature))

    save_path = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/'.format(noise_size,training_data_nmb,*generated_data_attributes)

    plot_area_plq = []
    plot_perimeter_plq = []
    plot_nmb_clusters_plq = []
    plot_euler_characteristics_plq = []
    plot_edge_percolation = []
    plot_plq_percolation = []
    plot_euler_characteristics_full = []
    plot_radius = []
    plot_diameter = []
    plot_nmb_clusters_edge = []

    for data_idx,data_type in enumerate(data_type_list):

        plot_per_datatype_area_plq = []
        plot_per_datatype_perimeter_plq = []
        plot_per_datatype_nmb_clusters_plq = []
        plot_per_datatype_euler_characteristics_plq = []
        plot_per_datatype_edge_percolation = []
        plot_per_datatype_plq_percolation = []
        plot_per_datatype_euler_characteristics_full = []
        plot_per_datatype_radius = []
        plot_per_datatype_diameter = []
        plot_per_datatype_nmb_clusters_edge = []
        for name in name_list[:6]:

            tda_observables = pd.read_csv(output_data_path+path_name_list[data_idx]+'{}_{}_TDA_Measurements_{}.csv'.format(name,data_type,sample_size), delimiter=',', skipinitialspace= True, header = 0)
            dataset_distances = np.array(tda_observables['epsilon'])[:2500]
            dataset_area_plq= np.array(tda_observables['Plq_area'])[:2500]
            dataset_nmb_clusters_plq = np.array(tda_observables['plq_cluster_nmb'])[:2500]
            dataset_perimeter_plq = np.array(tda_observables['Plq_perimeter'])[:2500]
            dataset_euler_characteristics_plq = np.array(tda_observables['plq_euler'])[:2500]
            dataset_edge_percolation = np.array(tda_observables['edge_percol'])[:2500]
            dataset_plq_percolation = np.array(tda_observables['plq_percol'])[:2500]
            dataset_euler_characteristics_full = np.array(tda_observables['full_euler'])[:2500]
            dataset_radius = np.array(tda_observables['full_radius'])[:2500]
            dataset_diameter = np.array(tda_observables['full_diameter'])[:2500]
            dataset_nmb_clusters_edge = np.array(tda_observables['full nmb clusters'])[:2500]

            plot_per_datatype_area_plq.append(dataset_area_plq)
            plot_per_datatype_perimeter_plq.append(dataset_perimeter_plq)
            plot_per_datatype_nmb_clusters_plq.append(dataset_nmb_clusters_plq)
            plot_per_datatype_euler_characteristics_plq.append(dataset_euler_characteristics_plq)
            plot_per_datatype_edge_percolation.append(dataset_edge_percolation)
            plot_per_datatype_plq_percolation.append(dataset_plq_percolation)
            plot_per_datatype_euler_characteristics_full.append(dataset_euler_characteristics_full)
            plot_per_datatype_radius.append(dataset_radius)
            plot_per_datatype_diameter.append(dataset_diameter)
            plot_per_datatype_nmb_clusters_edge.append(dataset_nmb_clusters_edge)


        plot_area_plq.append(plot_per_datatype_area_plq)
        plot_perimeter_plq.append(plot_per_datatype_perimeter_plq)
        plot_nmb_clusters_plq.append(plot_per_datatype_nmb_clusters_plq)
        plot_euler_characteristics_plq.append(plot_per_datatype_euler_characteristics_plq)
        plot_edge_percolation.append(plot_per_datatype_edge_percolation)
        plot_plq_percolation.append(plot_per_datatype_plq_percolation)
        plot_euler_characteristics_full.append(plot_per_datatype_euler_characteristics_full)
        plot_radius.append(plot_per_datatype_radius)
        plot_diameter.append(plot_per_datatype_diameter)
        plot_nmb_clusters_edge.append(plot_per_datatype_nmb_clusters_edge)

    plot_full_measures = [
            plot_area_plq,
            plot_perimeter_plq,
            plot_nmb_clusters_plq,
            plot_euler_characteristics_plq,
            plot_edge_percolation,
            plot_plq_percolation,
            plot_euler_characteristics_full,
            plot_radius,
            plot_diameter,
            plot_nmb_clusters_edge
        ]

    plot_names = ['area_plq','perimeter_plq', 'nmb_cluster_plq', 'euler_plq', 'bond_Percolation', 'sate_percolation', 'euler_full', 'radius','diameter', 'nmb_cluster_edge']
    plot_names_axes = [r'$A(\Theta)$',r'$P(\Theta)$', r'$N_{C_P}(\Theta)$', r'$\chi_{P}(\Theta)$', r'$P^B(\Theta)$', r'$P^S(\Theta)$', r'$\chi_{full}(\Theta)$',  r'$R(\Theta)$', r'$PD(\Theta)$', r'$N_{C_E}(\Theta)$']

    colortypes_sim = [r'#762a83', r'#882255', r'#997700',r'#225522']
    colortypes_sim_edge = [r'#33bbee',r'#f67e4b',r'#ddaa33',r'#4eb265']
    colortypes_gen = [r'#9398d2', r'#ee99aa',r'#bbcc33',r'#44bb99']
    colortypes_gen_edge = [r'#222255',r'#a50026',r'#999933',r'#125a56']
    specs_plot = [[{}],[{}],[{}],[{}],[{}],[{}],[{}],[{}],[{}],[{}]]


    
    for name_idx, name in enumerate(name_list[:1]):

        print(int(len(plot_full_measures)))
        sns.set_theme(style="whitegrid")
        fig = make_subplots(rows=int(len(plot_full_measures)),cols=1, shared_xaxes= True)
        fig.update_layout(width=2000, height=3500, template='seaborn', xaxis_visible=True, yaxis_visible=True,font=dict(
        family="Courier New, monospace",
        size=27,  # Set the font size here
        color="black"),xaxis_range=[0.,2.])#legend_title = plot_names[plot_idx] 
        
        for plot_idx,plotmeasure in enumerate(plot_full_measures):


            if full_distances:

                #sns.set_theme(style="whitegrid")
                #fig = go.Figure()
                #fig.update_layout(width=2000, height=1000, template='none', xaxis_visible=True, yaxis_visible=True,font=dict(
                #family="Courier New, monospace",
                #size=27,  # Set the font size here
                #color="black"), legend_title = plot_names[plot_idx],xaxis_range=[0.,4.2])
                #fig.update_xaxes(ticklabelposition="inside top", title= dict(text= '$\Theta$'),title_font=dict(size=35))#, size= 25))
                #fig.update_yaxes(ticklabelposition="inside top", title= dict(text= plot_names_axes[plot_idx]),title_font=dict(size=35))

            
                for data_idx,data_type in data_type_list:

                    if delta_data:
                        if compare == 'sim':

                            if data_type != 'training_data':
                                output_plot = np.abs(plotmeasure[0][name_idx]-plotmeasure[data_idx][name_idx])


                        elif compare == 'zero':
                            if data_type != 'zero_temp':
                                output_plot = np.abs(plotmeasure[1][name_idx]-plotmeasure[data_idx][name_idx])

                    elif KLD:
                        if compare == 'sim':

                            if data_type != 'training_data':
                                output_plot = kullback_leibler(plotmeasure[0][name_idx],plotmeasure[data_idx][name_idx])


                        elif compare == 'zero':
                            if data_type != 'zero_temp':
                                output_plot = kullback_leibler(plotmeasure[1][name_idx],plotmeasure[data_idx][name_idx])

                    else:
                        output_plot = plotmeasure[data_idx][name_idx]

                    fig.add_trace(go.Scatter(x=dataset_distances, y=output_plot,
                                    mode='markers', name=data_type,marker_color=colortypes_sim[data_idx],marker_symbol="cross"))#"#88CCEE"  ##
                        
                #fig.show()

            #fig.write_image(output_data_path+save_path+'TDA_measures_{}_defect_full_distance{}_delta{}_kld{}_compare_{}_noise_{}_samplesize_{}.png'.format(plot_names[plot_idx],delta_data,KLD,compare,noise_size, sample_size))


            else:
                            

                #sns.set_theme(style="whitegrid")
                #fig = go.Figure()
                #fig.update_layout(width=2000, height=1000, template='none', xaxis_visible=True, yaxis_visible=True,font=dict(
                #family="Courier New, monospace",
                #size=27,  # Set the font size here
                #color="black"), legend_title = plot_names[plot_idx] ,xaxis_range=[0.,4.2])
                #fig.update_xaxes(ticklabelposition="inside top", title= dict(text= '$\Theta$'),title_font=dict(size=35))#, size= 25))
                #fig.update_yaxes(ticklabelposition="inside top", title= dict(text= plot_names_axes[plot_idx]),title_font=dict(size=35))

                for data_idx,data_type in enumerate(data_type_list):

                        #zero should be sim, one should be zero temp

                    if delta_data:
                        if compare == 'sim':

                            if data_type != 'training_data':
                                output_plot = np.abs(plotmeasure[0][name_idx]-plotmeasure[data_idx][name_idx])


                        elif compare == 'zero':
                            if data_type != 'zero_temp':
                                output_plot = np.abs(plotmeasure[1][name_idx]-plotmeasure[data_idx][name_idx])

                    elif KLD:
                        if compare == 'sim':

                            if data_type != 'training_data':
                                output_plot = kullback_leibler(plotmeasure[0][name_idx],plotmeasure[data_idx][name_idx])


                        elif compare == 'zero':
                            if data_type != 'zero_temp':
                                output_plot = kullback_leibler(plotmeasure[1][name_idx],plotmeasure[data_idx][name_idx])

                    else:
                        output_plot = plotmeasure[data_idx][name_idx]


                    if plot_idx > 0:

                        fig.add_trace(go.Scatter(x=dataset_distances, y=output_plot,
                                            mode='lines + markers', marker_color=colortypes_sim[data_idx],marker_symbol="cross",showlegend=False), row=plot_idx+1,col=1)#"#88CCEE"  ##
                    else:

                        fig.add_trace(go.Scatter(x=dataset_distances, y=output_plot,
                                    mode='lines + markers', name=data_type,marker_color=colortypes_sim[data_idx],marker_symbol="cross"), row=plot_idx+1,col=1)#"#88CCEE"  ##

                    fig.update_yaxes(ticklabelposition="outside top", title= dict(text= plot_names_axes[plot_idx]),title_font=dict(size=35), row=plot_idx+1,col=1)


        fig.update_xaxes(ticklabelposition="inside top", title= dict(text= '$\Theta$'),title_font=dict(size=35))#, size= 25))
        fig.show()

        #fig.write_image(output_data_path+save_path+'TDA_measures_{}_defect_distance{}_delta{}_kld{}_compare_{}_noise_{}_samplesize_{}.png'.format('full',name,delta_data,KLD,compare,noise_size,sample_size))


#plot_topological_measurements(output_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/',data_type_list=['training_data','generated_data'], simulated_data_path='/localscratch/kklos/WGAN_XY_Model_Analysis/July_24/', generated_data_attributes=generated_data_atributes_list_v2[0], device='cpu',temperature=0.1,training_data_nmb=100000,noise_size=1.0,compare='sim',full_distances= False,delta_data=False,KLD= False,fixed_data=True,sample_size=100)
#plot_topological_measurements(h5d=False,generated_data_path = '/localscratch/kklos/topological_analysis_idea/', simulated_data_path = '/localscratch/kklos/DPG_24/homeoffice/', generator_attributes = [[80,64,0.0001,0,5,1,1.0,0,1e-05,2]],types = ['mid_defects', 'large_defects'],percolation=True,device='cpu',temperatures=[0.7],only_sim=True)
#plot_topological_measurements(h5d=False,generated_data_path = '/localscratch/kyklos/DPG_24/homeoffice/filtration_function_june_24/', simulated_data_path = '/localscratch/kklos/DPG_24/homeoffice/', generator_attributes = [[80,64,0.0001,0,5,1,1.0,0,1e-05,2]],types = ['no_defect','small_defects','mid_defects', 'large_defects'],percolation=True,device='cpu',temperatures=[0.1],only_sim=False)
#plot_topological_measurements(h5d=False,generated_data_path = '/localscratch/kklos/topological_analysis_idea/', simulated_data_path = '/localscratch/kklos/DPG_24/homeoffice/', generator_attributes = [[80,64,0.001,0,5,1,1.0,0,1e-05,2]],types =[ 'no_defect',  'mid_defects', 'large_defects'],temperature=0.1,percolation=False,device='cpu')


    #fig.update_layout(width=1000, height=1000, template='none', xaxis_visible=True, yaxis_visible=True)




    #sns.despine(f, left=True, bottom=True)
    #clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    #sns.scatterplot(x="distances", y="areas_analytics",
    #                hue="clarity", size="depth",
    #                palette="ch:r=-.2,d=.3_r",
    #                hue_order=clarity_ranking,
    #                sizes=(1, 8), linewidth=0,
    #                data=diamonds, ax=ax)

    #all_data_points = areas_analytics+ perimeter_analytics+eulercharcteristics_analytics+areas_random+perimeter_random

    #sns.pointplot(x=distances, y=areas_analytics
    #fig = px.scatter( x=distances, y=areas_analytics, 
    #                title="Automatic Labels Based on Data Frame Column Names")
    #fig.show()



######################
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
        










