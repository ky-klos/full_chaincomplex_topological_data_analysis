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
from skimage.measure import euler_number, label
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

import time

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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torch.autograd import Function

import networkx as nx

import sys

from collections import OrderedDict


import copy
two_pi = 2*np.pi
#lattice_size = 100
lattice_spacing = 1
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###this is the working class !
#torch.set_default_device(device)

np.set_printoptions(threshold=sys.maxsize)

class TopologicalAnalysis():
    def __init__(self,device,lattice_size=5, epsilon= 0.0,batchsize=100):

        

        self.device = device
        torch.set_default_device(self.device)
        self.lattice_spacing = 1
        self.lattice_size = lattice_size
        self.num_spins = lattice_size**2
        self.saved_L = 1
        self.saved_N = 1
        L,N = self.lattice_size,self.num_spins
        self.nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                    (i // L) * L + (i - 1) % L, (i - L) % N) \
                                            for i in list(range(N))}#right,down,left,up
        self.pqts = {i : (i  ,((i+L)%N),(i+1-((i%L)//(L-1))*L+L)%N ,i+1-((i%L)//(L-1))*L,i)  \
                                                for i in list(range(N))}
        self.pqts_nbr = {i : ((i - L) % N ,(i // L) * L + (i - 1) % L,
        ((i // L) * L + (i - 1) % L +L)%N,
        (i + 2*L) % N,((i+1-((i%L)//(L-1))*L+L)%N + L) % N,
        ((i // L) * L + (i + 2) % L+L)%N,
        (i // L) * L + (i + 2) % L,
        ((i+1-((i%L)//(L-1))*L)- L) % N  ) \
                                                for i in list(range(N))}#up left down-left


        #upp-left, left,pos, up, upper-left; normal plaquette;
        self.upleft_pqts = {i : ((((i // L) * L + (i - 1) % L) - L) % N,(i // L) * L + (i - 1) % L,i,(i - L) % N, (((i // L) * L + (i - 1) % L) - L) % N) \
                                            for i in list(range(N))}

        #left, down-left, down, pos, left

        self.downleft_pqts = {i : ((i // L) * L + (i - 1) % L,((i // L) * L + (i - 1) % L +L)%N, (i + L) % N,i,(i // L) * L + (i - 1) % L) \
                                            for i in list(range(N))}
        #up,pos, right, up-right, up
        self.upright_pqts = {i : ( (i - L) % N,i,(i // L) * L + (i + 1) % L,((i+1-((i%L)//(L-1))*L)- L) % N ,(i - L) % N) \
                                            for i in list(range(N))}

        self.defect_plaquette = {i : ((((i // L) * L + (i - 1) % L) - L) % N, (i // L) * L + (i - 1) % L ,i ,(i - L) % N)  \
                                                for i in list(range(N))}# up-left, left, main, up

        self.nn_x_direction = {i : ((i // L) * L + (i + 1) % L)\
                                            for i in list(range(N))}#right
        self.nn_y_direction = {i : ((i + L) % N)\
                                            for i in list(range(N))}#down, up: ,(i - L) % N

        self.nn_cluster = {i : ((i // L) * L + (i - 1) % L,(i - L) % N)\
                                            for i in list(range(N))}  #left nn, up
        
        self.nn_cluster_edge_h = {i : ((i // L) * L + (i - 1) % L,(i - L) % N)\
                                            for i in list(range(N))}  #left (edge x) up (edge y)
        
        self.nn_cluster_edge_v = {i : ((i // L) * L + (i - 1) % L,(i - L) % N)\
                                            for i in list(range(N))}  #same (edge x) up (edge y) left (edge x) diag up left (edge x)
        
        self.nn_cluster_boundary = {i : ((i // L) * L + (i + 1) % L,(i + L) % N)\
                                            for i in list(range(N))}  #right, down nn

        self.epsilon = epsilon

        self.clusters_per_epsilon = {}

        self.two_pi = 2*np.pi

        # these folowwing two dictionary will have indice in 2 dim lattice*2 (because of edge matrix) as key and array of birth depenent on epsilon each epsilon with depth of batch size

        #where are possible edges
        self.edge_ordering = {batch_idx: OrderedDict() for batch_idx in range(batchsize)}
        # where are possible plaquettes
        self.plaquette_ordering = {batch_idx: OrderedDict() for batch_idx in range(batchsize)}




    def saw(self, angle_one, angle_two):
        difference = angle_one - angle_two

        if np.abs(difference - two_pi) < np.abs(difference):
            difference -= two_pi
        elif np.abs(difference + two_pi) < np.abs(difference):
            difference += two_pi

        return difference
    
    def saw_batch(self, angle_one, angle_two):
        
        #print(angle_one.shape)
        difference = angle_one - angle_two
        return_diff = np.copy(difference)
        #print(difference.shape)

        change_condition_1 = np.abs(difference - two_pi) < np.abs(difference)
        #print(change_condition_1)
        change_condition_index_1 = change_condition_1.nonzero()
        #print(difference)
        change_condition_2 = np.abs(difference + two_pi) < np.abs(difference)
        change_condition_index_2 = change_condition_2.nonzero()



        return_diff[change_condition_index_1] = difference[change_condition_index_1] - two_pi
        return_diff[change_condition_index_2] = difference[change_condition_index_2] + two_pi
        #print(return_diff)

        return return_diff

    def setAngleInterval(self,angle):

        if angle >= two_pi:
            return_angle = angle%two_pi
        elif angle < 0.0 and angle >= -two_pi:
            return_angle = two_pi + angle
        elif angle < 0.0 and angle < -two_pi:
            return_angle = angle%two_pi
        else:
            return_angle = angle

        return return_angle
    
    def setAngleInterval_batch(self,angle_batch):

        return_angle_batch = np.copy(angle_batch)



        indizes_larger = angle_batch >= two_pi
        indizes_less_zero = angle_batch < 0.0 

        indizes_small = angle_batch < -two_pi
        indizes_middle =  np.logical_and(indizes_less_zero,indizes_small)
        #print(indizes_less_zero)
        #print(indizes_small)
        #print(indizes_middle)
        

        indizes_larger_1 = indizes_larger.nonzero()
        indizes_middle_1 = indizes_middle.nonzero()
        indizes_small_1 = indizes_small.nonzero()

        #print(indizes_middle)

        return_angle_batch[indizes_larger_1] = angle_batch[indizes_larger_1]%two_pi
        return_angle_batch[indizes_middle_1] = angle_batch[indizes_middle_1]+two_pi
        return_angle_batch[indizes_small_1] = angle_batch[indizes_small_1]%two_pi

        return return_angle_batch
    
    def defect_position_tensor(self,maximal_defect_number,input_defect_lattice= None):


        counter = 0

        if input_defect_lattice == None:
            defect_lattice = self.defect_config
        else:
            defect_lattice = input_defect_lattice

        #number_of_defects = torch.sum(torch.abs(defect_lattice))

        defect_position_vector = torch.zeros(int(maximal_defect_number),3,device = self.device)


        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if defect_lattice[i,j] != 0:
                    defect_position_vector[counter,0] += i
                    defect_position_vector[counter,1] += j
                    defect_position_vector[counter,2] += torch.clone(defect_lattice[i,j])
                    counter +=1

        if counter == 0:
            defect_distance = torch.tensor(0)
            #print(defect_distance)
        elif counter == 2:
            defect_distance = torch.sqrt(torch.pow((defect_position_vector[0,0]-defect_position_vector[1,0]),2) +torch.pow((defect_position_vector[0,1]-defect_position_vector[1,1]),2))
            #print(defect_distance)
        else:
            #print(counter,defect_position_vector)
            vortices = defect_position_vector[defect_position_vector[:,2]== 1]
            anti_vortices = defect_position_vector[defect_position_vector[:,2]== -1]

            defect_distances_list = torch.zeros(int(counter/2))

            #print(counter)

            for c in range(int(counter/2)):


                defect_distances_list[c] = torch.min(torch.sqrt(torch.pow((vortices[:,0]-anti_vortices[c,0]),2) +torch.pow((vortices[:,1]-anti_vortices[c,1]),2)))

            defect_distance = torch.mean(defect_distances_list)
        

        #print('caluclated_defects',position_vector)

        return defect_position_vector, defect_distance
    
    def defect_position_tensor_simult(self,maximal_defect_number,input_defect_lattices= None):


        

        if input_defect_lattices == None:
            defect_lattices = self.defect_config
        else:
            defect_lattices = input_defect_lattices

        counter = torch.zeros(defect_lattices.size(0))

        #number_of_defects = torch.sum(torch.abs(defect_lattice))

        defect_position_vector = torch.zeros(defect_lattices.size(0),int(maximal_defect_number),3,device = self.device)

        defect_distances_vector = torch.zeros(defect_lattices.size(0))


        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if torch.any(defect_lattices[:,i,j] != 0):

                    current_counter = counter[defect_lattices[:,i,j] != 0]
                    current_defects = defect_lattices[defect_lattices[:,i,j] != 0]

                    non_zero_index = torch.nonzero(defect_lattices[:,i,j] != 0)

                    

                    defect_position_vector[non_zero_index,current_counter,torch.ones(non_zero_index.shape())*0] += i
                    defect_position_vector[non_zero_index,current_counter,torch.ones(non_zero_index.shape())*1] += j
                    defect_position_vector[non_zero_index,current_counter,torch.ones(non_zero_index.shape())*2] += torch.clone(current_defects[:,i,j])
                    counter[defect_lattices[:,i,j] != 0] += 1

        print('works?',torch.sum(counter))

        if torch.any(counter == 2):
            defect_distances_vector[counter == 2] = torch.sqrt(torch.pow((defect_position_vector[counter == 2,0,0]-defect_position_vector[counter == 2,1,0]),2) +torch.pow((defect_position_vector[counter == 2,0,1]-defect_position_vector[counter == 2,1,1]),2))
        if torch.any(counter == 0):
            defect_distances_vector[counter == 2] = torch.zeros_like(counter[counter == 2])
        if torch.any(counter > 2):

            vortices = defect_position_vector[defect_position_vector[counter > 2,:,2]== 1]
            anti_vortices = defect_position_vector[defect_position_vector[counter > 2,:,2]== -1]

            defect_distances_vector[counter > 2]
            defect_distances_list = torch.zeros(int(counter/2))

            #print(counter)

            for c in range(int(counter/2)):


                defect_distances_list[c] = torch.min(torch.sqrt(torch.pow((vortices[:,0]-anti_vortices[c,0]),2) +torch.pow((vortices[:,1]-anti_vortices[c,1]),2)))

            defect_distance = torch.mean(defect_distances_list)
        

        #print('caluclated_defects',position_vector)

        return defect_position_vector, defect_distance
    
    def set_full_defect_positions(self,defect_lattices,maximal_defect_nmb=2,individual=False):
        full_defect_position_list = []
        defect_distance_list = []
        defect_distance_list_x = []
        defect_distance_list_y = []
        #print('size',defect_lattices.size())
        

        for defect_lattice in defect_lattices:
            defect_position,defect_distance = self.defect_position_tensor(maximal_defect_number=maximal_defect_nmb,input_defect_lattice = defect_lattice)
            if individual:
                vortices = defect_position[defect_position[:,2]== 1]
                anti_vortices = defect_position[defect_position[:,2]== -1]

                defect_distances_list_x = torch.zeros(int(maximal_defect_nmb/2))
                defect_distances_list_y = torch.zeros(int(maximal_defect_nmb/2))
                
                #print(int(vortices.size(0)))

            #print(counter)
                

                for c in range(int(maximal_defect_nmb/2)):
                    if int(vortices.size(0))>0:


                        defect_distances_list_y[c] = torch.min((vortices[:,0]-anti_vortices[c,0]))
                        defect_distances_list_x[c] = torch.min((vortices[:,1]-anti_vortices[c,1]))
                        
                    else:
                        defect_distances_list_y[c] = torch.tensor([0.])
                        defect_distances_list_x[c] = torch.tensor([0.])
                    

                defect_distance_list_x.append(torch.mean(defect_distances_list_x))
                defect_distance_list_y.append(torch.mean(defect_distances_list_y))

            full_defect_position_list.append(defect_position)
            #print(defect_distance.size())
            defect_distance_list.append(defect_distance)


        self.full_defect_positions = torch.stack(full_defect_position_list,dim=0)
        self.full_defect_distance = torch.stack(defect_distance_list, dim =0)
        
        if individual:
            self.full_defect_distance_x = torch.stack(defect_distance_list_x, dim =0)
            self.full_defect_distance_y = torch.stack(defect_distance_list_y, dim =0)
        else:
            self.full_defect_distance_x = None
            self.full_defect_distance_y = None
    
    def get_defect_distance_xy(self):
        return self.full_defect_distance_x,self.full_defect_distance_y

    def set_full_defect_positions_simultanously(self,defect_lattices,maximal_defect_nmb=2):

        defect_position,defect_distance = self.defect_position_tensor_simult(maximal_defect_number=maximal_defect_nmb,input_defect_lattices = defect_lattices)


        self.full_defect_positions = defect_position
        self.full_defect_distance = defect_distance

    def return_full_defect_positions(self):
        return self.full_defect_positions
    
    def return_full_defect_distances(self):
        return self.full_defect_distance
    
    def get_full_defect_distance(self, defect_nmb):
       
       ##only for two defects

       anti_vortices = self.full_defect_positions[:,:, 2]== -1
       vortices = self.full_defect_positions[:,:, 2]== 1
       no_defects = self.full_defect_positions[:,:, 2]== 0

    ##    np.array([np.sqrt(((defects[0]-current_cluster_com[0])**2+(defects[1]-current_cluster_com[1])**2)/2.0)
    
    def set_analytic_spin_config_tensor(self,input_defect_positions=None, input_vortex_numb = 2):

        self.spin_position = torch.from_numpy(np.asarray([[[i*self.lattice_spacing,j*self.lattice_spacing] for j in range(self.lattice_size)] for i in range(self.lattice_size)]))
        ##print(self.spin_position.size())
            

        if input_vortex_numb == None:
            vortex_numb = self.vortex_number()
        else:
            vortex_numb = input_vortex_numb

        if input_defect_positions == None:
            defect_positions = self.defect_position()
        else:
            defect_positions = input_defect_positions

        analytic_solution_list = []

        if input_vortex_numb == 0:
            analytic_solution = torch.zeros(self.lattice_size,self.lattice_size)

        else:

            for i in range(self.lattice_size):#should be x direction
                for j in range(self.lattice_size):#should be y driection
                    analystic_angles = torch.stack([(defect_positions[v,2])*torch.arctan2((torch.tensor(self.spin_position[i,j,1])-(defect_positions[v,1]+0.5)),(torch.tensor(self.spin_position[i,j,0])-(defect_positions[v,0]+0.5))) for v in range(vortex_numb)])

                    analytic_angle = (torch.sum(analystic_angles))%(2*np.pi)

                            #we want it in area [0,2pi]

                    if analytic_angle < 0.0:
                        analytic_angle += (2*np.pi)


                    analytic_solution_list.append(analytic_angle)



            analytic_solution = torch.stack(analytic_solution_list).view(self.lattice_size,self.lattice_size)


            self.current_analyse_spin = analytic_solution

            return analytic_solution
    
    def set_analytic_spin_lattices(self, unique_defect_lattices, defect_nb_list):
   

        full_analytic_spin_lattices_list = []

        max_defect_nmb = max(defect_nb_list)

        

        for idx,defect_lattice in enumerate(unique_defect_lattices):

            defect_pos = self.defect_position_tensor(maximal_defect_number=max_defect_nmb,input_defect_lattice=defect_lattice)

            
            full_analytic_spin_lattices_list.append(self.set_analytic_spin_config_tensor(input_defect_positions=defect_pos, input_vortex_numb = defect_nb_list[idx]))

        full_analytic_spin_lattices = torch.stack(full_analytic_spin_lattices_list,dim=0)
       

        self.analytic_spin_lattices = full_analytic_spin_lattices


        return full_analytic_spin_lattices
    
    


    def epsilon_change(self,change=0.0175):
        self.epsilon += change # 1 angle grad in rad
        #print(self.epsilon)

    def epsilon_change_multible(self,times,change=0.0175):
        self.epsilon = times*change # 1 angle grad in rad
        #print(self.epsilon)

    def vertex(self):
        vertex = np.ones((self.lattice_size)*(self.lattice_size))
        self.vertex = vertex
        return vertex

    def set_edge_matrix(self, edge_matrix_x, edge_matrix_y):
        self.edges_x = edge_matrix_x
        self.edges_y = edge_matrix_y

    def set_plaquette_matrix(self, plaquette_matrix):
        self.plaquettes = plaquette_matrix

    def set_batch_edge_matrix(self, edge_batch_matrix_x, edge_batch_matrix_y):
        self.edges_x_batch = edge_batch_matrix_x
        self.edges_y_batch = edge_batch_matrix_y

    def set_batch_plaquette_matrix(self, plaquette_batch_matrix):
        self.plaquettes_batch = plaquette_batch_matrix



    def edge_matrix_batch(self, spin_lattice,batch):
        spin_lattice_array = spin_lattice.reshape(-1,self.lattice_size*self.lattice_size)
        edges_x = np.zeros((batch,self.lattice_size*self.lattice_size))
        edges_y = np.zeros((batch,self.lattice_size*self.lattice_size))
        spin_lattice_array_two_pi = self.setAngleInterval_batch(spin_lattice_array)
        for idx in range(self.lattice_size*self.lattice_size):
            nn_angle_distance_x = abs(self.saw_batch(spin_lattice_array_two_pi[:,idx],spin_lattice_array_two_pi[:,self.nn_x_direction[idx]]))
            nn_angle_distance_y = abs(self.saw_batch(spin_lattice_array_two_pi[:,idx],spin_lattice_array_two_pi[:,self.nn_y_direction[idx]]))

            #print('x',nn_angle_distance_x)
            #print('y',nn_angle_distance_y)



            connection_constraint_x = nn_angle_distance_x<= self.epsilon
            connection_constraint_y = nn_angle_distance_y<= self.epsilon
            if sum(connection_constraint_x) > 0:
                connection_constraint_x_idx = connection_constraint_x.nonzero()
                edges_x[connection_constraint_x_idx,idx] = 1
            if sum(connection_constraint_y) > 0:
                connection_constraint_y_idx = connection_constraint_y.nonzero()
                edges_y[connection_constraint_y_idx,idx] = 1


        self.edges_x_batch = edges_x
        self.edges_y_batch = edges_y

        return edges_x, edges_y
    
    def make_full_edge_matrix(self, epsilon, edge_x, edge_y, plaq):

        full_edges = torch.zeros((edge_x.size(0),self.lattice_size*2,self.lattice_size*2))
        connection_constraint_x = edge_x<= epsilon
        connection_constraint_y = edge_y<= epsilon
        plaq_constraint = plaq <= epsilon

        ##vertex_count_lattice = torch.zeros((edge_x.size(0),self.lattice_size*2,self.lattice_size*2))

        #np.set_printoptions(threshold = np.inf)

        #print('edge_x',connection_constraint_x[0].view(self.lattice_size,self.lattice_size).int())
        #print('edge_y',connection_constraint_y[0].view(self.lattice_size,self.lattice_size).int())
        #print('plaq',plaq_constraint[0].view(self.lattice_size,self.lattice_size).int())
        ##count_edge = 0
        for idx in range(self.lattice_size*self.lattice_size):
            
            if connection_constraint_x[:,idx].sum() > 0:
                connection_constraint_x_idx = connection_constraint_x[:,idx].nonzero()
                ##count_edge += 1
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = count_edge

                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 1

                ##count_edge += 1
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 2
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = count_edge

            if connection_constraint_y[:,idx].sum() > 0:
                connection_constraint_y_idx = connection_constraint_y[:,idx].nonzero()
                ##count_edge += 1

                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = count_edge

                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2+1)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 1

                ##count_edge += 1

                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2+2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2+2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = count_edge

                #full_edges[connection_constraint_y_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 1

            if plaq_constraint[:,idx].sum() > 0:
                plaq_constraint_idx = plaq_constraint[:,idx].nonzero()
                full_edges[plaq_constraint_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 3

        #print('full',full_edges[0].numpy())


        return full_edges
    
    def make_full_edge_matrix_with_definite_holes(self, epsilon, edge_x, edge_y, plaq,ordering=False):

        # 2 vertex, 1 edge, 4 hole , 3 filled hole

        full_edges = torch.zeros((edge_x.size(0),self.lattice_size*2,self.lattice_size*2))
        connection_constraint_x = edge_x<= epsilon
        connection_constraint_y = edge_y<= epsilon
        plaq_constraint = plaq <= epsilon
        
        #print(plaq_constraint)

        #print('x edges', connection_constraint_x)
        #print('y edges', connection_constraint_y)

        ##vertex_count_lattice = torch.zeros((edge_x.size(0),self.lattice_size*2,self.lattice_size*2))

        #np.set_printoptions(threshold = np.inf)

        #print('edge_x',connection_constraint_x[0].view(self.lattice_size,self.lattice_size).int())
        #print('edge_y',connection_constraint_y[0].view(self.lattice_size,self.lattice_size).int())
        #print('plaq',plaq_constraint[0].view(self.lattice_size,self.lattice_size).int())
        ##count_edge = 0




        for idx in range(self.lattice_size*self.lattice_size):
            
            if ordering:
                
                full_edges[:,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[:,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 2
                full_edges[:,((idx//self.lattice_size)*2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[:,((idx//self.lattice_size)*2+2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2

            
            if connection_constraint_x[:,idx].sum() > 0:
                connection_constraint_x_idx = connection_constraint_x[:,idx].nonzero()
                ##count_edge += 1
                
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = count_edge

                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 1
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 2

                ##count_edge += 1
                
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = count_edge

            if connection_constraint_y[:,idx].sum() > 0:
                connection_constraint_y_idx = connection_constraint_y[:,idx].nonzero()
                ##count_edge += 1

                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2+2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = count_edge

                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2+1)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 1

                ##count_edge += 1

                
                ##vertex_count_lattice[connection_constraint_x_idx,(idx//self.lattice_size)*2+2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = count_edge

                #full_edges[connection_constraint_y_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 1

            if plaq_constraint[:,idx].sum() > 0:
                plaq_constraint_idx = plaq_constraint[:,idx].nonzero()
                full_edges[plaq_constraint_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 3


            #should fill holes of plaquette graph/cluster with 4s

            #this is not working:


            if  torch.logical_and((connection_constraint_x[:,idx]>0),(connection_constraint_y[:,idx]>0)).sum() > 0:
                
                #print('check')
                
                ##connection_constraint_x[:,idx]##wtf ???
                
                
                possible_hole = torch.logical_and(torch.logical_and((connection_constraint_x[:,idx]>0),(connection_constraint_y[:,idx]>0)),torch.logical_and((connection_constraint_x[:,(idx+self.lattice_size)%(self.lattice_size**2)] >0),(connection_constraint_y[:,(idx // self.lattice_size) * self.lattice_size + (idx + 1) % self.lattice_size]> 0)))
                
                #print(possible_hole.nonzero())
                if possible_hole.sum() > 0:
                    
                    ##plaq_hole_idx = torch.logical_and(torch.logical_not(plaq_constraint[:,idx]),possible_hole).nonzero()
                    plaq_hole_idx = torch.logical_and(plaq_constraint[:,idx] == 0,possible_hole).nonzero()
                    
                    #print(full_edges[plaq_hole_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)])
                    
                    #print(plaq_constraint[:,idx])
                    

                
                #print('check',connection_constraint_x[:,idx],connection_constraint_y[:,idx])

                    

                    full_edges[plaq_hole_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 4

        #if torch.any(full_edges==4.):
        #    print('full_edges',full_edges[0])
        

        return full_edges

        #print('full',full_edges[0].numpy())

    def simplices_ordering(self,full_edge_matrix_list,epsilon):
        #vertices come at the same time at epsilon = 0
        #edges

        self.full_edge_matrix_list = full_edge_matrix_list

        #print(self.full_edge_matrix_list.shape)


        for idx in range(int(full_edge_matrix_list.size(0))):

            #these should give back indices between 0 and (L*2)^2
            new_lattice_size = self.lattice_size*2



            edge_indices = torch.nonzero((full_edge_matrix_list[idx].flatten()==1))
            plq_indices = torch.nonzero((full_edge_matrix_list[idx].flatten()==3))#because other 3 is filled and  empty and here it is important when ist 3 born
            vertex_indices = torch.nonzero((full_edge_matrix_list[idx].flatten()==2))

            #print('edge_indices',edge_indices)
            #print('vertex_indices',vertex_indices)


            for edg_idx in edge_indices:

                
                if edg_idx.item() not in self.edge_ordering[idx].keys():
                    #print('new key',edg_idx)

                    ##check more and explizit
                    if torch.sum(((edg_idx//new_lattice_size)*new_lattice_size + (edg_idx+1)%(new_lattice_size))==vertex_indices) == 1:
                        self.edge_ordering[idx][edg_idx.item()] = torch.tensor([epsilon,(edg_idx//new_lattice_size)*new_lattice_size + (edg_idx+1)%(new_lattice_size),(edg_idx//new_lattice_size)*new_lattice_size + (edg_idx-1)%(new_lattice_size)])#birth epsilon and boundary: here two vertices
                    else: #needs to be boundary of y edge
                        self.edge_ordering[idx][edg_idx.item()] = torch.tensor([epsilon,(edg_idx+new_lattice_size)%(new_lattice_size**2),(edg_idx-new_lattice_size)%(new_lattice_size**2)])#modulu should not be nessescary since new lattice was made with implicit bc
                else:
                    #print(edg_idx)
                    continue

            for plq_idx in plq_indices:
                if plq_idx.item() not in self.plaquette_ordering[idx].keys():

                    self.plaquette_ordering[idx][plq_idx.item()]= torch.tensor([epsilon,(plq_idx-1)%(new_lattice_size**2),(plq_idx+new_lattice_size)%(new_lattice_size**2),(plq_idx+1 - new_lattice_size*(((plq_idx+1)%(2*new_lattice_size))==0))%(new_lattice_size**2),(plq_idx-new_lattice_size)%(new_lattice_size**2)])#birth epsilon and boundary: here four edges


            self.vertex_indices = vertex_indices.flatten()
            #self.edge_ordering[epsilon][idx].append((full_edge_matrix_list[idx]==1).nonzero())
            #self.plaquette_ordering[epsilon][idx].append((full_edge_matrix_list[idx]==4).nonzero())

    def return_implices_ordering(self):
        return self.edge_ordering,self.plaquette_ordering

    def boundary_matrices(self):
        #need ordering of edges and plaquettes, vertices are anyways 'born' at the same time

        #check again number of vertices!!



        edge_boundary_matrix = torch.zeros((int(len(list(self.edge_ordering.keys()))),(int(torch.numel(self.vertex_indices))),int(len(list(self.edge_ordering[0].keys())))))#last dimenion shoul be maximum of edges ()

        plaq_boundary_matrix = torch.zeros((int(len(list(self.plaquette_ordering.keys()))),int(len(list(self.edge_ordering[0].keys()))),int(len(list(self.plaquette_ordering[0].keys())))))

        #print('check vertex',torch.numel(self.vertex_indices),self.vertex_indices)

        #print('size', int(len(list(self.edge_ordering.keys()))))

        #print('check vertex infices full', self.vertex_indices)

        for batch_idx in self.edge_ordering.keys():
            edge_indices = list(self.edge_ordering[batch_idx].keys())#these are the edge positions 
            plq_indices = list(self.plaquette_ordering[batch_idx].keys())

            #print('edge indices', edge_indices)
            for e_idx,edge_matrix_indices in enumerate(edge_indices):

                #print('edge check',e_idx, edge_matrix_indices)

                

                #print('check vertex infices full', self.vertex_indices)

                boundary_indices = self.edge_ordering[batch_idx][edge_matrix_indices][1:]

                #print('epsilon', self.edge_ordering[batch_idx][edge_matrix_indices][0])

                #print('boudnary_check', boundary_indices)

                #pos in real matrix to get pos in new 

                coresponding_matrix_indice = (torch.logical_or(self.vertex_indices == boundary_indices[0],self.vertex_indices == boundary_indices[1])).nonzero()

                #if torch.any(self.vertex_indices == boundary_indices[0]) == False:
                #    print('boudnary indice', boundary_indices)
                #    print('edge indices full', edge_indices)
                #    print('edge indices',coresponding_matrix_indice)
                #    print('batch indices',batch_idx)
                #    print('edge_indices count',e_idx)

                edge_boundary_matrix[batch_idx,coresponding_matrix_indice,e_idx] = 1

            for p_idx,plaquette_matrix_indices in enumerate(plq_indices):
                boundary_indices = self.plaquette_ordering[batch_idx][plaquette_matrix_indices][1:]

                #print('plq boundary',boundary_indices)
                #print('check boudnary', p_idx)
                #print('edge',edge_indices)

                for b_idx in boundary_indices:

                    sub_b_idx = (torch.tensor(edge_indices)==b_idx).nonzero(as_tuple=True)

                    #print('check boudnary', sub_b_idx)
                    #print('check boudnary', p_idx)

                    plaq_boundary_matrix[batch_idx,sub_b_idx,p_idx] = 1

        self.edge_boundary_matrix = edge_boundary_matrix
        self.plaq_boundary_matrix = plaq_boundary_matrix

        torch.set_printoptions(threshold=sys.maxsize)

        #print(torch.sum(edge_boundary_matrix, dim=1))
        #print(torch.sum(plaq_boundary_matrix, dim=1))

        ##check here if boudnary matrix makes sense 

        #check_zeros = torch.sum(edge_boundary_matrix.view(self.lattice_size*self.lattice_size, 2*self.lattice_size*self.lattice_size).T,dim=1) == 0

        #print('edge boundary',torch.sum(edge_boundary_matrix.view(self.lattice_size*self.lattice_size, 2*self.lattice_size*self.lattice_size).T,dim=1))

        #print(self.vertex_indices)

        #print('check e b', torch.tensor(edge_indices).flatten()[check_zeros])

        #print('check edge bound', [self.edge_ordering[0][k.item()] for k in torch.tensor(edge_indices).flatten()[check_zeros]])



        



        #print('plq boudnary',torch.nonzero(plaq_boundary_matrix))

        #return edge_boundary_matrix,plaq_boundary_matrix

        #this should give two boundary matrices one for H_0 one for H_1

    def low_idx(self, transposed_boundary_matrix, boundary, column=False):

        if boundary == 2 :
            low_indices = transposed_boundary_matrix.nonzero().view(-1,boundary,2)[:,-1,1]
            if column:
                low_cloumn = transposed_boundary_matrix.nonzero().view(-1,boundary,2)[:,-1,0]
        elif boundary == 4:

            nonzeros = transposed_boundary_matrix.nonzero()
            low_indices = [0 for i in range(transposed_boundary_matrix.size(0))]#this includes all 
            
            for (x,y) in nonzeros:
                low_indices[x]= max(low_indices[x],y)#wrong?

        else:
            raise ValueError('Boundary unkown')
        
        if column:
            return low_indices, low_cloumn
        else:

            return low_indices
            
        

    def lowest_one(self, boundary_matrix, boundary, output= False,reduction=True):

        #either max vaule of one or if everythin is zero 

        transposed_boundary_matrix = torch.t(boundary_matrix)

        if output:

                transposed_boundary_matrix[torch.all(torch.t(boundary_matrix)==0, dim =1),:2] = -1

                #low_indices = transposed_boundary_matrix.nonzero().view(-1,boundary,2)[:,-1,1]

                low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)


        else:

            if reduction:
                #zero rows need to be handled as well, here we charactises them as low on position zero

                #print('check if low_change works', torch.all(torch.t(boundary_matrix)==0, dim =1))


                if torch.any(torch.all(torch.t(boundary_matrix)==0, dim =1)):

                    transposed_boundary_matrix[torch.all(torch.t(boundary_matrix)==0, dim =1),:2] = -1

                    low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)




                else:

                    low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)

            else:

                transposed_boundary_matrix[transposed_boundary_matrix==-1] = 0#should filter out zero rows
                
                low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)


    
        return torch.tensor(low_indices)




    def reduced_boundary_matrices(self):
    #     #input boundary matrix 
    #     #output_reduced_boudnary matrix

        temp_edge_boudnary_matrix = torch.clone(self.edge_boundary_matrix)
        temp_plaq_boundary_matrix  = torch.clone(self.plaq_boundary_matrix)

        


        for batch in range(int(temp_edge_boudnary_matrix.size(0))):
            #break
            temp_edge_low_indices = self.lowest_one(temp_edge_boudnary_matrix[batch],boundary=2)
            temp_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4)

            #print('low',temp_plq_low_indices)
            #print('size',int(temp_edge_boudnary_matrix.size(2)))
            plq_check_unique, plq_check_counts = torch.unique(temp_plq_low_indices,return_counts=True)

            #print(torch.sum(temp_plaq_boundary_matrix[batch], dim=0))


            while torch.any(plq_check_counts>1):

                #print('plq count',plq_check_counts)

                for j in range(int(temp_plaq_boundary_matrix.size(2))):#should be columns

                    i=0

                    #print(j)
                    #print(temp_plaq_boundary_matrix[batch])
                    while i < j:
                    #print(i,j,temp_edge_low_indices[i] == temp_edge_low_indices[j])
                        
                        if (temp_plq_low_indices[i] == temp_plq_low_indices[j]) and torch.logical_and(torch.all(temp_plaq_boundary_matrix[batch,:,j] != -1),torch.all(temp_plaq_boundary_matrix[batch,:,i] != -1)):
                            #print(i,j)
                            #print((temp_plaq_boundary_matrix[batch,:,j]+temp_plaq_boundary_matrix[batch,:,i])==1)
                            #print(temp_plaq_boundary_matrix[batch])
                            temp_plaq_boundary_matrix[batch,:,j]=((temp_plaq_boundary_matrix[batch,:,j]+temp_plaq_boundary_matrix[batch,:,i])==1).long()
                            #print(torch.sum(temp_plaq_boundary_matrix[batch], dim=0))
                            temp_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4)
                    #else:
                    #    continue
                    
                        i += 1

                #print('check')


                check_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4,reduction=False)
                
                plq_check_unique, plq_check_counts = torch.unique(check_plq_low_indices,return_counts=True)


            edge_check_unique, edge_check_counts = torch.unique(temp_edge_low_indices,return_counts=True)

            #print(plq_check_counts>1)

            while torch.any(edge_check_counts>1):

                #print(edge_check_counts)

                for j in range(int(temp_edge_boudnary_matrix.size(2))):#should be columns

                    i=0

                    #print(j)

                    while i < j:
                    #print(i,j,temp_edge_low_indices[i] == temp_edge_low_indices[j])
                        if (temp_edge_low_indices[i] == temp_edge_low_indices[j]) and torch.logical_and(torch.all(temp_edge_boudnary_matrix[batch,:,j] != -1),torch.all(temp_edge_boudnary_matrix[batch,:,i] != -1)):
                            #print(i,j)
                            #print((temp_edge_boudnary_matrix[batch,:,j]+temp_edge_boudnary_matrix[batch,:,i])==1)
                            temp_edge_boudnary_matrix[batch,:,j]=((temp_edge_boudnary_matrix[batch,:,j]+temp_edge_boudnary_matrix[batch,:,i])==1).long()
                            temp_edge_low_indices = self.lowest_one(temp_edge_boudnary_matrix[batch],boundary=2)
                    #else:
                    #    continue
                    
                        i += 1

                    #print('check',j, temp_edge_boudnary_matrix)
                    #print('low', j , temp_edge_low_indices)

                check_edge_low_indices = self.lowest_one(temp_edge_boudnary_matrix[batch],boundary=2,reduction=False)
                
                edge_check_unique, edge_check_counts = torch.unique(check_edge_low_indices,return_counts=True)

                #print('edge_counts', edge_check_counts)

            #print('out of edge')
                

            #for j in range(int(temp_plaq_boundary_matrix.size(2))):#should be columns
            #    
            #    i=0
            #
            #    while i < j:
            #        if (temp_plq_low_indices[i] == temp_plq_low_indices[j]):
            #            temp_plaq_boundary_matrix[batch,:,j]=int((temp_plaq_boundary_matrix[batch,:,j]+temp_plaq_boundary_matrix[batch,:,i])==1)
            #            temp_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4)
            #        else:
            #            continue
            #        
            #        i += 1

        return temp_edge_boudnary_matrix,temp_plaq_boundary_matrix
    
    def get_persistent_info_of_boundary_matrix(self, reduced_boundary_matrix_edge,reduced_boundary_matrix_plq, homology_group = 0):

        birth_position_and_epsilon = []

        death_position_and_epsilon = []

        unpaired_just_brith_pos_and_epslion = []

        

        copy_vertex_ordering = torch.clone(self.vertex_indices)

        #copy_edge_ordering = torch.clone(self.edge_ordering)
        
        



        for batch in range(int(reduced_boundary_matrix_edge.size(0))):

            if homology_group == 0:


                # if 

                lowest_indices = self.lowest_one(reduced_boundary_matrix_edge[batch],boundary=2,reduction=False)

                #print('low',lowest_indices)

                


                birth = self.vertex_indices[lowest_indices] #should give back vertex indices epsilon is zero in our case

                #print('before', copy_vertex_ordering)

                #print('to remove', birth)

                

                #print('keys',self.edge_ordering[batch].keys())

                #print('death', torch.logical_not(torch.all(torch.t(reduced_boundary_matrix[batch])==0, dim =1)))

                unpaired = torch.from_numpy(np.delete(copy_vertex_ordering.flatten().cpu().numpy(), lowest_indices.flatten().cpu().numpy()))# does this work ?

                

                unpaired_epsilon = torch.zeros_like(unpaired)# torch.tensor([unpaired[batch][u.item()][0] for u in unpaired])

                death = torch.tensor(list(self.edge_ordering[batch].keys()))[torch.logical_not(torch.all(torch.t(reduced_boundary_matrix_edge[batch])==0, dim =1))]
                #print('birth!', birth)
                #print('death!', death)

                #print('death?',torch.tensor(list(self.edge_ordering[batch].keys())))

                death_epsilon = torch.tensor([self.edge_ordering[batch][d.item()][0] for d in death])

                birth_position_and_epsilon.append(torch.cat((birth,torch.zeros_like(birth)),dim=0).view(2,-1).cpu().numpy())
                ##birth_position_and_epsilon.append(torch.zeros_like(birth))

                death_position_and_epsilon.append(torch.cat((death,death_epsilon), dim=0).view(2,-1).cpu().numpy())

                ##death_position_and_epsilon.append(death_epsilon.cpu().numpy())



                #unpaired_just_brith_pos_and_epslion.append(unpaired)
                unpaired_just_brith_pos_and_epslion.append(torch.cat((unpaired,unpaired_epsilon), dim=0).view(2,-1).cpu().numpy())

                #print(torch.stack([death,death_epsilon], dim= 0))
                #print(unpaired)

                #olny relevant if epsilons are not the same


            elif homology_group == 1:

                copy_edge_ordering = copy.deepcopy(self.edge_ordering[batch])

                lowest_indices = self.lowest_one(reduced_boundary_matrix_plq[batch],boundary=4,reduction=False)[torch.logical_not(torch.all(torch.t(reduced_boundary_matrix_plq[batch])==0, dim =1))]#need to exclude zero columns
                
                print(self.lowest_one(reduced_boundary_matrix_plq[batch],boundary=4,reduction=False)[torch.all(torch.t(reduced_boundary_matrix_plq[batch])==0, dim =1)])
                #edge_check_unique, edge_check_counts = torch.unique(lowest_indices,return_counts=True)
                
                #print('low count', edge_check_counts)
                #print(lowest_indices.size())

                birth =  torch.tensor(list(self.edge_ordering[batch].keys()))[lowest_indices]
                
                ##unpaired differently:
                
                unpaired_birth = torch.tensor(list(self.edge_ordering[batch].keys()))[torch.all(torch.t(reduced_boundary_matrix_edge[batch])==0, dim =1)] #since h_2 is not tested should be 
                
                fully_unpaired = []
                #print('before',list(copy_edge_ordering.keys()))
                
                for u in unpaired_birth: 
                    ##copy_edge_ordering.pop(b.item()) 
                    if (torch.any(u == birth)):
                        continue
                    else:
                        fully_unpaired.append(u)
                
                #unpaired = torch.from_numpy(copy_edge_ordering.keys())
                
                print('after',list(copy_edge_ordering.keys()))

                unpaired_epsilon = torch.tensor([[self.edge_ordering[batch][u.item()][0],u.item()] for u in fully_unpaired]).cpu()

                #unpaired = np.delete(copy_edge_ordering.cpu().numpy(),lowest_indices.cpu().numpy())

                brith_epsilon = torch.tensor([self.edge_ordering[batch][b.item()][0] for b in birth])


                death = torch.tensor(list(self.plaquette_ordering[batch].keys()))[torch.logical_not(torch.all(torch.t(reduced_boundary_matrix_plq[batch])==0, dim =1))]

                death_epsilon = torch.tensor([self.plaquette_ordering[batch][d.item()][0] for d in death])

                birth_position_and_epsilon.append(torch.stack([birth,brith_epsilon], dim= 0).cpu().numpy())

                #birth_position_and_epsilon.append(brith_epsilon.cpu().numpy())

                death_position_and_epsilon.append(torch.stack([death,death_epsilon],dim=0).cpu().numpy())

                #death_position_and_epsilon.append(death_epsilon.cpu().numpy())

                #unpaired_just_brith_pos_and_epslion.append(unpaired)
                
                unpaired_just_brith_pos_and_epslion.append(torch.t(unpaired_epsilon).cpu().numpy())#torch.stack([unpaired,unpaired_epsilon],dim=0).numpy())



            else:
                raise ValueError('Working currently only with two dimensions')
            
        return np.stack(birth_position_and_epsilon,axis=0), np.stack(death_position_and_epsilon,axis=0),np.stack(unpaired_just_brith_pos_and_epslion, axis=0)
        

        

        


        






        





    
    def minkovski_with_edges(self, full_cluster_list):#only one cluster but multible 
        #check if 

        cluster_tensor = torch.tensor(full_cluster_list)

        cluster_lattice = torch.zeros(self.lattice_size*2,self.lattice_size*2)
        cluster_lattice[cluster_tensor[:,0].int(),cluster_tensor[:,1].int()]=cluster_tensor[:,2]


        padded_cluster_lattice = F.pad(cluster_lattice, (3,3,3,3), mode='constant')

        #euler characteristic

        count_vertex = torch.sum(padded_cluster_lattice==2)
        count_edges = torch.sum(padded_cluster_lattice==1)
        count_faces = torch.sum(padded_cluster_lattice==3)

        euler_characteristic = count_vertex-count_edges+count_faces

        return euler_characteristic
    
    def diameter_radius_graph(self, full_cluster_list):

        #make cluster list to graph with connections

        L = (2*self.lattice_size)
        N = L*L

        if self.saved_L != L or self.saved_N != N:
            self.saved_L = L
            self.saved_N = N

            self.nn_x_direction_larger = {i : ((i // L) * L + (i + 1) % L)\
                                                for i in list(range(N))}#right
            self.nn_y_direction_larger = {i : ((i + L) % N)\
                                                for i in list(range(N))}#down, up: ,(i - L) % N
            
            self.nn_x_direction_larger_second = {i : ((i // L) * L + (i + 2) % L)\
                                                for i in list(range(N))}#right
            self.nn_y_direction_larger_second = {i : ((i + 2*L) % N)\
                                                for i in list(range(N))}#down, up: ,(i - L) % N


        #print(full_cluster_list)

        cluster_tensor = torch.tensor(full_cluster_list).view(-1,3)

        cluster_lattice = torch.zeros(self.lattice_size*2,self.lattice_size*2)
        cluster_lattice[cluster_tensor[:,0].int(),cluster_tensor[:,1].int()]=cluster_tensor[:,2]

        ##np.set_printoptions(threshold=sys.maxsize)

        #print(cluster_lattice.numpy())

        cluster_lattice = cluster_lattice.flatten()

        

        vertex_indices = torch.nonzero(cluster_lattice==2)

        #print([idx.item() for idx in vertex_indices])
        ##test = [cluster_lattice[self.nn_x_direction_larger[idx.item()]]==1. for idx in vertex_indices]
        #print(test)

        graph_list_right = [[idx.item(),self.nn_x_direction_larger_second[idx.item()]] for idx in vertex_indices if cluster_lattice[self.nn_x_direction_larger[idx.item()]]==1.]
        #print('right',graph_list_right)
        graph_list_down = [[idx.item(),self.nn_y_direction_larger_second[idx.item()]] for idx in vertex_indices if cluster_lattice[self.nn_y_direction_larger[idx.item()]]==1]

        
        #print('down',graph_list_down)

        


        graph_list = graph_list_right+graph_list_down

        graph = nx.Graph(graph_list)

        #print(graph)

        #print(nx.diameter(graph))

        return nx.diameter(graph),nx.radius(graph)



        











        #get maximal distant vertex min path : diamter

        #min nmb until min one  point reaches every point: radius

        #make cluster_lattice into graph




    

    
    
    def saw_function_tensor(self, spin_lattice_tensor):

        first_saw = torch.where(torch.abs(spin_lattice_tensor-self.two_pi)< torch.abs(spin_lattice_tensor),spin_lattice_tensor-self.two_pi, spin_lattice_tensor)

        second_saw = torch.where(torch.abs(spin_lattice_tensor+self.two_pi)< torch.abs(spin_lattice_tensor), spin_lattice_tensor+self.two_pi, first_saw)

        return second_saw
    
    def edge_matrix_batch_faster(self, spin_lattice_tensor):

        sample_nmb =int(spin_lattice_tensor.size(0))

        plaquette_maker = nn.Unfold(kernel_size=(2,2), stride= 1)
        plaq_index = torch.tensor([0,1,2],device=self.device)

        ##edge_x = torch.zeros(spin_lattice_tensor.size())
        ##edge_y = torch.zeros(spin_lattice_tensor.size())

        ##print(spin_lattice_tensor.size())

        spin_lattice_halo = F.pad(spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        plaquette_temp = plaquette_maker(spin_lattice_halo).view(spin_lattice_tensor.size(0),4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquette_temp, 1, 2)

        all_plaquettes =  torch.index_select(plaquettes,dim=2, index=plaq_index)

        ##print(all_plaquettes)

        ##changed_plaquette_angles = torch.sub(all_plaquettes[:,:,:],all_plaquettes[:,:,0].view(-1,self.lattice_size*self.lattice_size,1))

        ##angle in interval
        ##print(changed_plaquette_angles)

        angel_intervaled_plaquette_angles_1 = torch.where(torch.logical_and(all_plaquettes < 0.0,all_plaquettes > -self.two_pi), all_plaquettes + self.two_pi,all_plaquettes)

        angel_intervaled_plaquette_angles = torch.where(torch.abs(all_plaquettes) >= self.two_pi ,angel_intervaled_plaquette_angles_1%self.two_pi,angel_intervaled_plaquette_angles_1)

        ## angle diff

        nn_angle_distance_x=torch.abs(self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,0] - angel_intervaled_plaquette_angles[:,:,1])))
        nn_angle_distance_y=torch.abs(self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,0] - angel_intervaled_plaquette_angles[:,:,2])))

        edges_x = (nn_angle_distance_x<= self.epsilon).view(-1,self.lattice_size,self.lattice_size).float()
        edges_y = (nn_angle_distance_y<= self.epsilon).view(-1,self.lattice_size,self.lattice_size).float()

        self.edges_x_batch = edges_x
        self.edges_y_batch = edges_y

        return nn_angle_distance_x.view(-1,self.lattice_size,self.lattice_size),nn_angle_distance_y.view(-1,self.lattice_size,self.lattice_size)## careful here return before compared to epsilon##edges_x, edges_y

    def plaquettes_matrix_batch(self, spin_lattice,batch):
        spin_lattice_array = spin_lattice.reshape(-1,self.lattice_size*self.lattice_size)
        plaquett_matrix = np.zeros((batch,self.lattice_size*self.lattice_size))
        spin_lattice_array_two_pi = self.setAngleInterval_batch(spin_lattice_array)
        for idx in range(self.lattice_size*self.lattice_size):

            plaquetes_angles_two_pi = np.array([spin_lattice_array_two_pi[:,plq_idx] for plq_idx in self.pqts[idx]])
            #print(plaquetes_angles_two_pi.shape)
            plaquetes_angles_normed = np.array([(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)])
            #print(plaquetes_angles_normed.shape)
            plaquetes_angles_normed_two_pi = self.setAngleInterval_batch(plaquetes_angles_normed)


            plaquette_distances = np.abs(np.array([self.saw_batch(plaquetes_angles_normed_two_pi[i,:],plaquetes_angles_normed_two_pi[i+1,:]) for i in range(4)]))
            #print(plaquette_distances.shape)
            #print(np.transpose(plaquette_distances))

            maximal_plaqt_distance = np.max(np.transpose(plaquette_distances), axis=1)

            #print(maximal_plaqt_distance.shape)


            fill_constraint = maximal_plaqt_distance<= self.epsilon
    
            if sum(fill_constraint) > 0:
                fill_constraint_idx = fill_constraint.nonzero()
                #print(plaquett_matrix[fill_constraint_idx,idx])
                plaquett_matrix[fill_constraint_idx,idx] = 1
            #print(sum(plaquett_matrix))



        self.plaquettes = plaquett_matrix

        return plaquett_matrix
    
    def plaquettes_matrix_batch_faster(self, spin_lattice_tensor):

        plaquette_maker = nn.Unfold(kernel_size=(2,2), stride= 1)
        plaq_index = torch.tensor([0,2,3,1,0],device=self.device)

        #print(spin_lattice_tensor.size())

        spin_lattice_halo = F.pad(spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        plaquette_temp = plaquette_maker(spin_lattice_halo).view(spin_lattice_tensor.size(0),4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquette_temp, 1, 2)

        all_plaquettes =  torch.index_select(plaquettes,dim=2, index=plaq_index)

        ##print(all_plaquettes)

        changed_plaquette_angles = torch.sub(all_plaquettes[:,:,:],all_plaquettes[:,:,0].view(-1,self.lattice_size*self.lattice_size,1))

        ##angle in interval
        ##print(changed_plaquette_angles)

        angel_intervaled_plaquette_angles_1 = torch.where(torch.logical_and(changed_plaquette_angles < 0.0,changed_plaquette_angles > -self.two_pi), changed_plaquette_angles + self.two_pi,changed_plaquette_angles)

        angel_intervaled_plaquette_angles = torch.where(torch.abs(changed_plaquette_angles) >= self.two_pi ,angel_intervaled_plaquette_angles_1%self.two_pi,angel_intervaled_plaquette_angles_1)

        ## angle diff

        full_diff_list =[self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,1] - angel_intervaled_plaquette_angles[:,:,0])) 
        ,self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,2] - angel_intervaled_plaquette_angles[:,:,1])) 
        ,self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,3] - angel_intervaled_plaquette_angles[:,:,2])) 
        ,self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,4] - angel_intervaled_plaquette_angles[:,:,3]))]

        full_diff_tensor = torch.stack(full_diff_list,dim=2)

        #print(full_diff_tensor.size())

        max_diff_tensor = torch.max(torch.abs(full_diff_tensor),dim=2)[0]

        #print(max_diff_tensor.size())

        plaquette_matrices= max_diff_tensor.view(-1,self.lattice_size,self.lattice_size) <= self.epsilon

        self.plaquettes = plaquette_matrices

        return max_diff_tensor.view(-1,self.lattice_size,self.lattice_size)## careful here return before compared to epsilon plaquette_matrices


    
    def plaquettes_matrix(self, spin_lattice):
        spin_lattice_array = spin_lattice.reshape(self.lattice_size*self.lattice_size)
        plaquett_matrix = np.zeros((self.lattice_size)*(self.lattice_size))
        spin_lattice_array_two_pi = [self.setAngleInterval(spin) for spin in spin_lattice_array]
        for idx in range(self.lattice_size*self.lattice_size):

            plaquetes_angles_two_pi = [spin_lattice_array_two_pi[plq_idx] for plq_idx in self.pqts[idx]]
            plaquetes_angles_normed = [(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)]
            plaquetes_angles_normed_two_pi = [self.setAngleInterval(spin) for spin in plaquetes_angles_normed]


            plaquette_distances = [abs(self.saw(plaquetes_angles_normed_two_pi[i],plaquetes_angles_normed_two_pi[i+1])) for i in range(4)]

            maximal_plaqt_distance = max(plaquette_distances)


            fill_constraint = (maximal_plaqt_distance)


            if fill_constraint <= self.epsilon:
                plaquett_matrix[idx] = 1


        self.plaquettes = plaquett_matrix

        return plaquett_matrix
    
    
    
    def plaquettes_matrix_tensor(self, spin_lattice,batch):
        spin_lattice_array = spin_lattice.view(batch,self.lattice_size*self.lattice_size)
        plaquett_matrix = np.zeros((batch,self.lattice_size*self.lattice_size))
        spin_lattice_array_two_pi = self.setAngleInterval_batch(spin_lattice_array)

        spin_lattice_halo = F.pad(spin_lattice_array, (0,1,0,1), 'circular')
        plaquette_maker = nn.Unfolad(kernel_size=(2,2),stride=1)

        plaquettes_temp = plaquette_maker(spin_lattice_halo).view(batch,1,4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquettes_temp,2,3)

        index = torch.tensor([0,2,3,1,0])

        index_full = index.repeat(batch,1,self.lattice_size*self.lattice_size,1)

        all_plaquettes = torch.gather(plaquettes, index= index_full, dim=-1)

        angled_distances = [torch.sub(all_plaquettes[:,0,:,1],all_plaquettes[:,0,:,0]),torch.sub(all_plaquettes[:,0,:,2],all_plaquettes[:,0,:,1]),torch.sub(all_plaquettes[:,0,:,3],all_plaquettes[:,0,:,2]),torch.sub(all_plaquettes[:,0,:,4],all_plaquettes[:,0,:,3])]

        changed_angle_diff = []
        for distances in angled_distances:
                changed_distances = torch.where(torch.abs(distances - two_pi)>= torch.abs(distances), distances, distances-two_pi)
                changed_distances_2 = torch.where(changed_distances == distances and torch.abs(changed_distances + two_pi)>= torch.abs(changed_distances), changed_distances, changed_distances+two_pi)
                changed_angle_diff.append(changed_distances_2)

        maximal_diff = n


        #torch.where(abs())







        
        for idx in range(self.lattice_size*self.lattice_size):

            plaquetes_angles_two_pi = np.array([spin_lattice_array_two_pi[:,plq_idx] for plq_idx in self.pqts[idx]])
            #print(plaquetes_angles_two_pi.shape)
            plaquetes_angles_normed = np.array([(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)])
            #print(plaquetes_angles_normed.shape)
            plaquetes_angles_normed_two_pi = self.setAngleInterval_batch(plaquetes_angles_normed)


            plaquette_distances = np.array([self.saw_batch(plaquetes_angles_normed_two_pi[i,:],plaquetes_angles_normed_two_pi[i+1,:]) for i in range(4)])

            maximal_plaqt_distance = np.max(np.transpose(plaquette_distances), axis=1)

            #print(maximal_plaqt_distance.shape)


            fill_constraint = maximal_plaqt_distance<= self.epsilon
            fill_constraint_idx = fill_constraint.nonzero()
            plaquett_matrix[fill_constraint_idx,idx] = 1



        self.plaquettes = plaquett_matrix

        return plaquett_matrix



    def edge_matrix(self, spin_lattice):
        spin_lattice_array = spin_lattice.reshape(self.lattice_size*self.lattice_size)
        edges_x = np.zeros((self.lattice_size)*(self.lattice_size))
        edges_y = np.zeros((self.lattice_size)*(self.lattice_size))
        spin_lattice_array_two_pi = [self.setAngleInterval(spin) for spin in spin_lattice_array]
        for idx in range(self.lattice_size*self.lattice_size):
            nn_angle_distance_x = abs(self.saw(spin_lattice_array_two_pi[idx],spin_lattice_array_two_pi[self.nn_x_direction[idx]]))
            nn_angle_distance_y = abs(self.saw(spin_lattice_array_two_pi[idx],spin_lattice_array_two_pi[self.nn_y_direction[idx]]))

            #print('x',nn_angle_distance_x)
            #print('y',nn_angle_distance_y)

            connection_constraint_x = (nn_angle_distance_x)
            connection_constraint_y = (nn_angle_distance_y)

            if connection_constraint_x <= self.epsilon:
                edges_x[idx] = 1

            if connection_constraint_y <= self.epsilon:
                edges_y[idx] = 1

        self.edges_x = edges_x
        self.edges_y = edges_y

        return edges_x, edges_y


    
    def get_key_from_value(self,value, dic):
        return [k for k,v in dic.items() if np.any(value in v)]
    
    def defect_position(self,defects,number_of_defects):
        position_vector = np.zeros((number_of_defects,3))
        counter = 0
        
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if defects[i,j] != 0:
                    position_vector[counter,0] += i ##y
                    position_vector[counter,1] += j ##x
                    position_vector[counter,2] += np.copy(defects[i,j])
                    counter +=1

        return position_vector

    def cluster_plq_scipy(self, plq_graph =None,size_factor=1):
        if np.all(plq_graph == None):
            states = self.plaquettes.reshape(size_factor*self.lattice_size,size_factor*self.lattice_size) ##should be array of array 
        else:
            states = plq_graph.reshape(size_factor*self.lattice_size,size_factor*self.lattice_size)
            
        cluster_plaqu_array, num_cluster_plq_full = sp.ndimage.label(states)##returns 2d array with numbers of cluster
        if np.any(cluster_plaqu_array[0, 0] > 0) and np.any(cluster_plaqu_array[self.lattice_size-1, self.lattice_size-1] > 0):
            cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[self.lattice_size-1, self.lattice_size-1]] = cluster_plaqu_array[0, 0]
            
        if np.any(cluster_plaqu_array[0, self.lattice_size-1] > 0) and np.any(cluster_plaqu_array[self.lattice_size-1, 0] > 0):
            cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[self.lattice_size-1, 0]] = cluster_plaqu_array[0, self.lattice_size-1]


        for l in range(self.lattice_size):
            if np.any(cluster_plaqu_array[l, 0] > 0) and np.any(cluster_plaqu_array[l, -1] > 0):
                cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[l, -1]] = cluster_plaqu_array[l, 0]
            if np.any(cluster_plaqu_array[0, l] > 0) and np.any(cluster_plaqu_array[-1, l] > 0):
                cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[-1, l]] = cluster_plaqu_array[0, l]


        ##if plq_graph ==None:
        ##    self.number_cluster_plq_full = num_cluster_plq_full

        self.scipy_cluster_pql = cluster_plaqu_array

        return cluster_plaqu_array
    
    def cluster_plq_tensor(self, plq_graph):
        #first change only onmes to one, two ,three etc..
        #afterwards it should take min kernel to go over everyhing
        plq_graph_copy = np.copy(plq_graph)
        number_cluster_elements_full = np.sum(plq_graph.reshape(-1,self.lattice_size*self.lattice_size), axis=1)
        plq_graph_tensor = torch.from_numpy(plq_graph)
        plq_graph_halo = F.pad(plq_graph_tensor, (1,1,1,1), 'circular')
        plaquette_maker = nn.Unfold(kernel_size=(3,3),stride=1)


        for counter in range(number_cluster_elements_full):

            plaquettes_temp = plaquette_maker(plq_graph_halo).view(plq_graph.size(0),1,9,self.lattice_size*self.lattice_size)

            plaquettes = torch.transpose(plaquettes_temp,2,3)

            index = torch.tensor([1,3,4,5,7])

            index_full = index.repeat(plq_graph.size(0),1,self.lattice_size*self.lattice_size,1)
            print(torch.gather(plaquettes, index= index_full, dim=-1).size())

            all_plaquettes = torch.min(torch.gather(plaquettes, index= index_full, dim=-1), dim=-1)




    
    def cluster_plq_scipy_batch(self,batchsize, plq_graph =None,size_factor=1):
        if plq_graph ==None:
            states = self.plaquettes_batch.reshape(batchsize,size_factor*self.lattice_size,size_factor*self.lattice_size) ##should be array of array 
        else:
            states = plq_graph
            
        cluster_plaqu_array, num_cluster_plq_full = sp.ndimage.label(states)##returns 2d array with numbers of cluster
        print(cluster_plaqu_array)
        

        if np.any(np.logical_and(cluster_plaqu_array[:,0, 0] > 0, cluster_plaqu_array[:,self.lattice_size-1, self.lattice_size-1] > 0)):
            left_upper_corner = cluster_plaqu_array[:,0, 0] > 0
            right_lower_corner = cluster_plaqu_array[:,self.lattice_size-1, self.lattice_size-1] > 0
            indexing_corner = left_upper_corner & right_lower_corner
            ##print(cluster_plaqu_array == cluster_plaqu_array[indexing_corner,self.lattice_size-1, self.lattice_size-1])
            cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[indexing_corner,self.lattice_size-1, self.lattice_size-1]] = cluster_plaqu_array[indexing_corner,0, 0]
            
        if np.any(np.logical_and(cluster_plaqu_array[:,0, self.lattice_size-1] > 0, cluster_plaqu_array[:,self.lattice_size-1, 0] > 0)):
            left_upper_corner_2 = cluster_plaqu_array[:,0, self.lattice_size-1] > 0
            right_lower_corner_2 = cluster_plaqu_array[:,self.lattice_size-1, 0] > 0
            indexing_corner_2 = left_upper_corner_2 & right_lower_corner_2            
            cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[indexing_corner_2,self.lattice_size-1, 0]] = cluster_plaqu_array[indexing_corner_2,0, self.lattice_size-1]


        for l in range(self.lattice_size):
            if np.any(np.logical_and(cluster_plaqu_array[:,l, 0] > 0,cluster_plaqu_array[:,l, self.lattice_size-1] > 0)):
                left_side = cluster_plaqu_array[:,l, 0] > 0
                right_side = cluster_plaqu_array[:,l, self.lattice_size-1] > 0
                indexing_sides = np.logical_and(left_side,right_side).flatten()
                lattice_cluster_plaqu_array = cluster_plaqu_array[indexing_sides]
                print(lattice_cluster_plaqu_array)
                print(lattice_cluster_plaqu_array.shape)
                print((lattice_cluster_plaqu_array == cluster_plaqu_array[indexing_sides,l, self.lattice_size-1]).shape)

                full_indexing = np.where([lattice_cluster_plaqu_array == cluster_plaqu_array[indexing_sides,l, self.lattice_size-1]],lattice_cluster_plaqu_array)

                print(full_indexing)
                print(full_indexing.shape)


                print(cluster_plaqu_array[indexing_sides,l, self.lattice_size-1].shape)

                full_cluster_plaqu_array = np.repeat(cluster_plaqu_array[indexing_sides,l, self.lattice_size-1].reshape(-1,1), self.lattice_size*self.lattice_size, axis=1)
                print(full_cluster_plaqu_array.shape)               
                cluster_indx = (cluster_plaqu_array[indexing_sides]-full_cluster_plaqu_array.reshape(-1,self.lattice_size,self.lattice_size))==0
                cluster_indx_full = indexing_sides
                
                print(cluster_plaqu_array[indexing_sides].shape)

                print(cluster_indx)

                cluster_plaqu_array[indexing_sides,cluster_indx] = cluster_plaqu_array[indexing_sides,l, 0]
            if np.any(np.logical_and(cluster_plaqu_array[:,0, l] > 0, cluster_plaqu_array[:,self.lattice_size-1, l] > 0)):
                up_side = cluster_plaqu_array[:,0, l] > 0
                down_side = cluster_plaqu_array[:,self.lattice_size-1, l] > 0
                indexing_upsides = up_side & down_side

                cluster_plaqu_array[cluster_plaqu_array == cluster_plaqu_array[indexing_upsides,self.lattice_size-1, l]] = cluster_plaqu_array[indexing_upsides,0, l]


        ##if plq_graph ==None:
        ##    self.number_cluster_plq_full = num_cluster_plq_full

        self.scipy_cluster_pql_batch = cluster_plaqu_array

        return cluster_plaqu_array
    
    
    def com_cluster_pql_scipy(self, cluster_position):
        cluster_matrix = self.scipy_cluster_pql

        com = sp.ndimage.measurements.center_of_mass(cluster_matrix)

        cluster_objects = cluster_matrix != 0

        nrb_cluster_objects = np.sum(cluster_objects, axis= 1)#should only sum in one lattice

        cluster_positions = self.defect_position(defects=cluster_matrix,number_of_defects=nrb_cluster_objects)

    def cluster_faster(self,input_plaquette_tensor):

        rows, cols = input_plaquette_tensor.size(1,2)

        clusters = []

        visited = torch.zeros_like(input_plaquette_tensor, dtype=bool)

        def explore_clusters(row, col, cluster):
            
            if not (0<=row<rows) or not (0<= col < cols) or visited[row,col] or input_plaquette_tensor[row, col] == 0:
                return
            visited[row,col] = True
            cluster.append((row,col))

            for i, j in [(-1,0),(1,0),(0,-1),(0,1)]:
                new_row=(row +i) % rows
                new_col = (col+j) % cols
                explore_clusters(new_row,new_col, cluster)

        for i in range(rows):
            for j in range(cols):
                if input_plaquette_tensor[i,j] == 1 and not visited[i,j]:

                    current_cluster = []
                    explore_clusters(i,j,current_cluster)
                    clusters.append(current_cluster)

        return clusters
    
    def cluster_faster_other(self,input_plaquette_tensor):

        spins_halo= F.pad(input_plaquette_tensor.view(-1,1,self.lattice_size,self.lattice_size),(1,1,1,1),'circular')

        nn_maker = nn.Unfold(kernel_size=(3,3),stride=1)

        nn_temp = nn_maker(spins_halo).view(input_plaquette_tensor.size(0),9,self.lattice_size*self.lattice_size)

        next_n = torch.transpose(nn_temp, 1,2)

        index = torch.tensor([4,1,3,5,7],device=self.device)

        all_nn = torch.index_select(next_n,dim=2, index=index)

        object_cluster = all_nn!=0

        object_cluster_indices = object_cluster.nonzero()


        print(object_cluster_indices)

        return object_cluster_indices

 




    
    def cluster_edge_scipy(self):
        ##should transform the cluster graphs of x and y direction into something in one object 
        ## so that x direction neibour, but y not 
        bonds_x = self.edges_x.reshape((self.lattice_size,self.lattice_size)) ##should be np array with x positions of conncetion
        bonds_y = self.edges_y.reshape((self.lattice_size,self.lattice_size))

        edge_cluster_preprocessed = np.zeros((2*self.lattice_size+1,2*self.lattice_size+1))

        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if bonds_x[i,j] == 1:
                    edge_cluster_preprocessed[2*i,2*j+1] = 1 ##middle part in x to the right


                    if edge_cluster_preprocessed[2*i,2*j] != 1:
                        edge_cluster_preprocessed[2*i,2*j] = 1
                    if edge_cluster_preprocessed[2*i,2*j+2] != 1:
                        edge_cluster_preprocessed[2*i,2*j+2] = 1
                if bonds_y[i,j] == 1:
                    edge_cluster_preprocessed[2*i+1,2*j] = 1 ##middle part in x to the right


                    if edge_cluster_preprocessed[2*i,2*j] != 1:
                        edge_cluster_preprocessed[2*i,2*j] = 1
                    if edge_cluster_preprocessed[2*i+2,2*j] != 1:
                        edge_cluster_preprocessed[2*i+2,2*j] = 1


        ##last line should be first 
        edge_cluster_preprocessed


    def cluster_plq_scipy_dep_distance(self, defect_lattice, number_defects):
        ##defect_lattice = defect_lattice.reshape((self.lattice_size,self.lattice_size))
        defect_position = self.defect_position(defects=defect_lattice, number_of_defects=number_defects)

        full_cluster_array = self.cluster_plq_scipy(plq_graph =None)

        cluster_objects = full_cluster_array != 0

        nrb_cluster_objects = np.sum(cluster_objects, axis= 1)#should only sum in one lattice

        cluster_object_positions = self.defect_position(defects=full_cluster_array, number_of_defects=nrb_cluster_objects)

        clusters_per_distance = {}




    
    def cluster(self,index):#only for state percolation
        cluster_nmb = 0
        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = []
        states = self.plaquettes_batch[index]
        #bonds_x = self.edges_x
        #bonds_y = self.edges_y
        clusters_position_states = {}
        #clusters_position_states['{0}'.format(cluster_nmb)] = []
        for idx in range(int(self.lattice_size*self.lattice_size)):
                
            if states[idx] == 1:
                #print(idx)
                #print(clusters_position_states)
                #if not already in cluster but neigbours are add to that cluster,
                #if npt in cluster and neigbours aren't in cluster make new cluster
                #insert(init array, arr.shape[0], [row added],axis=0)axis=1 directly in list)
                cluster_values_array = sorted({x for v in clusters_position_states.values() for x in v})
                #print(cluster_values_array)
                if idx not in cluster_values_array:
                    #before_neigbour = [states[cluster] for cluster in self.nn_cluster[idx]]# left nn, upper nn diagonal left up
                    #print(before_neigbour)

                    #check if over boundary 

                    
                    before_neigbour_indices_boundary = [cluster_index for cluster_index in self.nn_cluster_boundary[idx]]#right,down
                   #print('idx, right,down', idx, before_neigbour_indices_boundary)

                    

                    
                    before_neigbour_indices = [cluster_index for cluster_index in self.nn_cluster[idx]]
                    #print(before_neigbour_indices)
                    #print(before_neigbour_indices[1] in clusters_position_states.values())
                    if before_neigbour_indices[0] in cluster_values_array or before_neigbour_indices[1] in cluster_values_array:
                        #should be put in corresponding cluster
                        #print(idx)
                        if before_neigbour_indices[0] in cluster_values_array and before_neigbour_indices[1] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                #connect both clusters 
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                cluster_nmb -= 1
                            else:
                                clusters_position_states['{0}'.format(cluster_number_one[0])].append(idx)
                        elif before_neigbour_indices[0] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                        elif before_neigbour_indices[1] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                    else:
                        cluster_nmb += 1
                        clusters_position_states['{0}'.format(cluster_nmb)] = []
                        clusters_position_states['{0}'.format(cluster_nmb)].append(idx)

                    cluster_values_array_2 = sorted({x for v in clusters_position_states.values() for x in v})

                    #now the edge case need to behandled (clusters going over the edge)

                    if idx % self.lattice_size == (self.lattice_size-1) or int(idx/self.lattice_size) == (self.lattice_size-1):
                        #print(idx)
                        #print(idx % self.lattice_size)
                        #the very lower edge case
                        if idx == (self.lattice_size*self.lattice_size)-1 and before_neigbour_indices_boundary[0] in cluster_values_array_2 and before_neigbour_indices_boundary[1] in cluster_values_array_2:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)
                            cluster_number_three = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            #print('?')
                            #print(cluster_number_one)
                            #print(cluster_number_two)
                            #print(cluster_number_three)

                            if cluster_number_one != cluster_number_two and cluster_number_one != cluster_number_three and cluster_number_two!= cluster_number_three:
                                #print(cluster_number_one)
                                #print(cluster_number_two)
                                #print(cluster_number_three)
                                cluster_number_list = cluster_number_one+cluster_number_two+cluster_number_three
                                #print(cluster_number_list)
                                earlier_cluster = min(cluster_number_one,cluster_number_two,cluster_number_three)
                                #print(earlier_cluster[0])
                                later_cluster = max(cluster_number_one,cluster_number_two,cluster_number_three)
                                middle_cluster = [cluster_number for cluster_number in cluster_number_list if cluster_number != earlier_cluster[0] and cluster_number != later_cluster[0]]
                                #print(middle_cluster)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]+clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                clusters_position_states['{0}'.format(middle_cluster[0])] = []
                            elif cluster_number_one != cluster_number_two and cluster_number_one == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one == cluster_number_two and cluster_number_one != cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one != cluster_number_two and cluster_number_two == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            else:
                                continue

                        #the right edge full

                        elif idx % self.lattice_size == (self.lattice_size-1) and before_neigbour_indices_boundary[0] in cluster_values_array_2:
                            #print(idx)
                            #print(before_neigbour_indices_boundary[0])
                            #print(idx % self.lattice_size)
                            #print(cluster_values_array)
                            #print(clusters_position_states)

                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)

                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        #lowest line (edge) of lattice 
                        elif int(idx/self.lattice_size) == (self.lattice_size-1) and before_neigbour_indices_boundary[1] in cluster_values_array_2:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        else:
                            continue
                else:
                    continue


        self.elements_per_clusters = clusters_position_states

        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = clusters_position_states

        deep_copy_cluster_position_states = clusters_position_states.copy()

        for cluster_key in deep_copy_cluster_position_states.keys():
            if clusters_position_states.get('{0}'.format(cluster_key[0])) == []:
                empty_cluster = clusters_position_states.pop('{0}'.format(cluster_key[0]))




        return clusters_position_states
    
    def cluster_batch(self):#only for state percolation
        cluster_nmb = 0
        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = []
        states = self.plaquettes
        #bonds_x = self.edges_x
        #bonds_y = self.edges_y
        clusters_position_states = {}
        #clusters_position_states['{0}'.format(cluster_nmb)] = []
        for idx in range(int(self.lattice_size*self.lattice_size)):
                
            if states[:,idx] == 1:
                #print(idx)
                #print(clusters_position_states)
                #if not already in cluster but neigbours are add to that cluster,
                #if npt in cluster and neigbours aren't in cluster make new cluster
                #insert(init array, arr.shape[0], [row added],axis=0)axis=1 directly in list)
                cluster_values_array = sorted({x for v in clusters_position_states.values() for x in v})
                #print(cluster_values_array)
                if idx not in cluster_values_array:
                    #before_neigbour = [states[cluster] for cluster in self.nn_cluster[idx]]# left nn, upper nn diagonal left up
                    #print(before_neigbour)

                    #check if over boundary 

                    
                    before_neigbour_indices_boundary = [cluster_index for cluster_index in self.nn_cluster_boundary[idx]]#right,down
                   #print('idx, right,down', idx, before_neigbour_indices_boundary)

                    

                    
                    before_neigbour_indices = [cluster_index for cluster_index in self.nn_cluster[idx]]
                    #print(before_neigbour_indices)
                    #print(before_neigbour_indices[1] in clusters_position_states.values())
                    if before_neigbour_indices[0] in cluster_values_array or before_neigbour_indices[1] in cluster_values_array:
                        #should be put in corresponding cluster
                        #print(idx)
                        if before_neigbour_indices[0] in cluster_values_array and before_neigbour_indices[1] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                #connect both clusters 
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                cluster_nmb -= 1
                            else:
                                clusters_position_states['{0}'.format(cluster_number_one[0])].append(idx)
                        elif before_neigbour_indices[0] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                        elif before_neigbour_indices[1] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                    else:
                        cluster_nmb += 1
                        clusters_position_states['{0}'.format(cluster_nmb)] = []
                        clusters_position_states['{0}'.format(cluster_nmb)].append(idx)

                    cluster_values_array_2 = sorted({x for v in clusters_position_states.values() for x in v})

                    #now the edge case need to behandled (clusters going over the edge)

                    if idx % self.lattice_size == (self.lattice_size-1) or int(idx/self.lattice_size) == (self.lattice_size-1):
                        #print(idx)
                        #print(idx % self.lattice_size)
                        #the very lower edge case
                        if idx == (self.lattice_size*self.lattice_size)-1 and before_neigbour_indices_boundary[0] in cluster_values_array_2 and before_neigbour_indices_boundary[1] in cluster_values_array_2:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)
                            cluster_number_three = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            #print('?')
                            #print(cluster_number_one)
                            #print(cluster_number_two)
                            #print(cluster_number_three)

                            if cluster_number_one != cluster_number_two and cluster_number_one != cluster_number_three and cluster_number_two!= cluster_number_three:
                                #print(cluster_number_one)
                                #print(cluster_number_two)
                                #print(cluster_number_three)
                                cluster_number_list = cluster_number_one+cluster_number_two+cluster_number_three
                                #print(cluster_number_list)
                                earlier_cluster = min(cluster_number_one,cluster_number_two,cluster_number_three)
                                #print(earlier_cluster[0])
                                later_cluster = max(cluster_number_one,cluster_number_two,cluster_number_three)
                                middle_cluster = [cluster_number for cluster_number in cluster_number_list if cluster_number != earlier_cluster[0] and cluster_number != later_cluster[0]]
                                #print(middle_cluster)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]+clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                clusters_position_states['{0}'.format(middle_cluster[0])] = []
                            elif cluster_number_one != cluster_number_two and cluster_number_one == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one == cluster_number_two and cluster_number_one != cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one != cluster_number_two and cluster_number_two == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            else:
                                continue

                        #the right edge full

                        elif idx % self.lattice_size == (self.lattice_size-1) and before_neigbour_indices_boundary[0] in cluster_values_array_2:
                            #print(idx)
                            #print(before_neigbour_indices_boundary[0])
                            #print(idx % self.lattice_size)
                            #print(cluster_values_array)
                            #print(clusters_position_states)

                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)

                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        #lowest line (edge) of lattice 
                        elif int(idx/self.lattice_size) == (self.lattice_size-1) and before_neigbour_indices_boundary[1] in cluster_values_array_2:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        else:
                            continue
                else:
                    continue


        self.elements_per_clusters = clusters_position_states

        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = clusters_position_states

        deep_copy_cluster_position_states = clusters_position_states.copy()

        for cluster_key in deep_copy_cluster_position_states.keys():
            if clusters_position_states.get('{0}'.format(cluster_key[0])) == []:
                empty_cluster = clusters_position_states.pop('{0}'.format(cluster_key[0]))




        return clusters_position_states
    


    def cluster_edge(self):#only for state percolation
        cluster_nmb = 0
        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = []
        edges_x,edges_y = self.edge_matrix
        #bonds_x = self.edges_x
        #bonds_y = self.edges_y
        clusters_position_states = {}
        #clusters_position_states['{0}'.format(cluster_nmb)] = []
        for idx in range(int(self.lattice_size*self.lattice_size)):
                
            if states[idx] == 1:
                #print(idx)
                #print(clusters_position_states)
                #if not already in cluster but neigbours are add to that cluster,
                #if npt in cluster and neigbours aren't in cluster make new cluster
                #insert(init array, arr.shape[0], [row added],axis=0)axis=1 directly in list)
                cluster_values_array = sorted({x for v in clusters_position_states.values() for x in v})
                if idx not in cluster_values_array:
                    #before_neigbour = [states[cluster] for cluster in self.nn_cluster[idx]]# left nn, upper nn diagonal left up
                    #print(before_neigbour)

                    #check if over boundary 

                    
                    before_neigbour_indices_boundary = [cluster_index for cluster_index in self.nn_cluster_boundary[idx]]#right,down
                   #print('idx, right,down', idx, before_neigbour_indices_boundary)

                    

                    
                    before_neigbour_indices = [cluster_index for cluster_index in self.nn_cluster[idx]]
                    #print(before_neigbour_indices)
                    #print(before_neigbour_indices[1] in clusters_position_states.values())
                    if before_neigbour_indices[0] in cluster_values_array or before_neigbour_indices[1] in cluster_values_array:
                        #should be put in corresponding cluster
                        if before_neigbour_indices[0] in cluster_values_array and before_neigbour_indices[1] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                #connect both clusters 
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                cluster_nmb -= 1
                            else:
                                clusters_position_states['{0}'.format(cluster_number_one[0])].append(idx)
                        elif before_neigbour_indices[0] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                        elif before_neigbour_indices[1] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                    else:
                        cluster_nmb += 1
                        clusters_position_states['{0}'.format(cluster_nmb)] = []
                        clusters_position_states['{0}'.format(cluster_nmb)].append(idx)

                    if idx % self.lattice_size == (self.lattice_size-1) or int(idx/self.lattice_size) == (self.lattice_size-1):
                        if idx == (self.lattice_size*self.lattice_size)-1 and before_neigbour_indices_boundary[0] in cluster_values_array and before_neigbour_indices_boundary[1] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)
                            cluster_number_three = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            #print('?')

                            if cluster_number_one != cluster_number_two and cluster_number_one != cluster_number_three:
                                #print(cluster_number_one)
                                #print(cluster_number_two)
                                #print(cluster_number_three)
                                cluster_number_list = cluster_number_one+cluster_number_two+cluster_number_three
                                #print(cluster_number_list)
                                earlier_cluster = min(cluster_number_one,cluster_number_two,cluster_number_three)
                                #print(earlier_cluster[0])
                                later_cluster = max(cluster_number_one,cluster_number_two,cluster_number_three)
                                middle_cluster = [cluster_number[0] for cluster_number in cluster_number_list if cluster_number[0] != earlier_cluster[0] and cluster_number[0] != later_cluster[0]]
                                #print(middle_cluster)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]+clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                clusters_position_states['{0}'.format(middle_cluster[0])] = []
                            elif cluster_number_one != cluster_number_two and cluster_number_one == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one == cluster_number_two and cluster_number_one != cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            else:
                                continue

                        elif idx % self.lattice_size == (self.lattice_size-1) and before_neigbour_indices_boundary[0] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        elif int(idx/self.lattice_size) == (self.lattice_size-1) and before_neigbour_indices_boundary[1] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                #clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        else:
                            continue
                else:
                    continue


 

        deep_copy_cluster_position_states = clusters_position_states.copy()

        for cluster_key in deep_copy_cluster_position_states.keys():
            if clusters_position_states.get('{0}'.format(cluster_key[0])) == []:
                empty_cluster = clusters_position_states.pop('{0}'.format(cluster_key[0]))

        self.elements_per_clusters = clusters_position_states

        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = clusters_position_states



        return clusters_position_states
    
    def number_of_clusters(self):
        clusters = self.elements_per_clusters #need to call self.cluster() before self.cluster()#return dic of all clusters idx
        number_of_clusters = int(len(list(clusters.keys())))
        return number_of_clusters
    



    def percolation(self):
        states = self.plaquettes
        #print(np.where(states == 1.0))
        bonds_x = self.edges_x
        bonds_y = self.edges_y
        filled_states = np.sum(states == 1.0).astype(float)
        #print(filled_states)
        empty_states = np.sum(states == 0.0).astype(float)
        #print(empty_states)
        filled_bonds_x = np.sum(bonds_x == 1.0).astype(float)
        empty_bonds_x = np.sum(bonds_x == 0.0).astype(float)
        filled_bonds_y = np.sum(bonds_y == 1.0).astype(float)
        empty_bonds_y = np.sum(bonds_y == 0.0).astype(float)

        filled_bonds = filled_bonds_x+filled_bonds_y
        empty_bonds = empty_bonds_x +empty_bonds_y

        ##print('percolation?')

        state_percolation = filled_states/(empty_states+filled_states)

        bond_percolation = filled_bonds/(empty_bonds+filled_bonds)

        ##print('percolation!')

        self.state_percolation = state_percolation

        self.bond_percolation = bond_percolation

        return state_percolation,bond_percolation
    
    def percolation(self):
        states = self.plaquettes
        #print(np.where(states == 1.0))
        bonds_x = self.edges_x
        bonds_y = self.edges_y
        filled_states = np.sum(states == 1.0).astype(float)
        #print(filled_states)
        empty_states = np.sum(states == 0.0).astype(float)
        #print(empty_states)
        filled_bonds_x = np.sum(bonds_x == 1.0).astype(float)
        empty_bonds_x = np.sum(bonds_x == 0.0).astype(float)
        filled_bonds_y = np.sum(bonds_y == 1.0).astype(float)
        empty_bonds_y = np.sum(bonds_y == 0.0).astype(float)

        filled_bonds = filled_bonds_x+filled_bonds_y
        empty_bonds = empty_bonds_x +empty_bonds_y

        ##print('percolation?')

        state_percolation = filled_states/(empty_states+filled_states)

        bond_percolation = filled_bonds/(empty_bonds+filled_bonds)

        ##print('percolation!')

        self.state_percolation = state_percolation

        self.bond_percolation = bond_percolation

        return state_percolation,bond_percolation
    
    def percolation_faster(self,plaquettes, edges_x, edges_y):
            #all size batch, ls* ls
        filled_states = torch.sum(plaquettes == 1.0, dim = 1).float()
            #print(filled_states)
        empty_states = torch.sum(plaquettes == 0.0, dim = 1).float()
            #print(empty_states)
        filled_bonds_x = torch.sum(edges_x == 1.0, dim = 1).float()
        empty_bonds_x = torch.sum(edges_x == 0.0, dim = 1).float()
        filled_bonds_y = torch.sum(edges_y == 1.0, dim = 1).float()
        empty_bonds_y = torch.sum(edges_y == 0.0, dim = 1).float()

        filled_bonds = filled_bonds_x+filled_bonds_y
        empty_bonds = empty_bonds_x +empty_bonds_y

            ##print('percolation?')

        state_percolation = filled_states/(empty_states+filled_states)

        bond_percolation = filled_bonds/(empty_bonds+filled_bonds)

            ##print('percolation!')

            ##self.state_percolation = state_percolation

            ##self.bond_percolation = bond_percolation

        return state_percolation,bond_percolation

    



    def full_rotation(self, dictionary, rot_direction='right'):#'down'

        deep_cops_dicitionary = {}

        if rot_direction == 'right':
            rotation_index = self.nn_x_direction
        
        else:#down
            rotation_index = self.nn_y_direction

        dictionary_keys = dictionary.keys()

        for key in dictionary_keys:
           deep_cops_dicitionary['{0}'.format(key[0])] = [self.nn_x_direction[cluster_idx] for cluster_idx in dictionary[key]]

        return deep_cops_dicitionary
        


    

    def translation_of_pb_overlapping_cluster(self, pre_rot_cluster,inputclustername, x_rot_nmb=0, y_rot_nmb=0,end=False):
        #if cluster goes over boundary, it needs to be translated as much as needed
        
        #define boundary positions
        #check if boundary on opisite filled as well

        #print(pre_rot_cluster)
        #print(inputclustername)

        padded_lattice_size =self.lattice_size

        cluster_name = inputclustername#pre_rot_cluster.key()

        pre_rot_cluster_indices = pre_rot_cluster['{0}'.format(cluster_name)]

        post_rot_cluster = {}

        post_rot_cluster = pre_rot_cluster.copy()

        post_cluster_indices = post_rot_cluster['{0}'.format(cluster_name)]

        x_rotation_number = x_rot_nmb#would be more smart if rotation dependent on how much is filled on either side
        y_rotation_number = y_rot_nmb

        #print('x:',x_rotation_number)
        #print('y:',y_rotation_number)

        counter = 0

        while counter <  int(len(pre_rot_cluster_indices)):
            counter = 0
            
            for idx,cluster_index in enumerate(pre_rot_cluster_indices):

                post_cluster_indices = post_rot_cluster['{0}'.format(cluster_name)]
                
                current_cluster_idx = post_cluster_indices[idx]

                if current_cluster_idx%self.lattice_size == 0 and x_rotation_number < self.lattice_size and current_cluster_idx+(self.lattice_size-1) in post_cluster_indices:
                    #print(current_cluster_idx)
                    #left boundary
                    #check if oppiste on right boundary is filled as well

                    #print('True one')
                    ##print(cluster_index+(self.lattice_size-1))
                    #rotate to right x_boundary_overlap
                    #print(pre_rot_cluster_indices)
                    new_cluster_indices = [self.nn_x_direction[cluster_idx] for cluster_idx in post_cluster_indices]
                    #print(new_cluster_indices)

                    x_rotation_number += 1

                    post_rot_cluster['{0}'.format(cluster_name[0])] =  new_cluster_indices

                    #print(post_rot_cluster)
 

                elif current_cluster_idx%self.lattice_size == self.lattice_size-1  and x_rotation_number < self.lattice_size and current_cluster_idx-(self.lattice_size-1) in post_cluster_indices:
                    

                    #print('True two')
                    
                    #print(cluster_index+(self.lattice_size-1))
                    #rotate to right x_boundary_overlap
                    new_cluster_indices = [self.nn_x_direction[cluster_idx] for cluster_idx in post_cluster_indices]

                    post_rot_cluster['{0}'.format(cluster_name[0])] =  new_cluster_indices

                    x_rotation_number += 1

                    #print(post_rot_cluster)



                elif current_cluster_idx < self.lattice_size and y_rotation_number < self.lattice_size and current_cluster_idx+(self.lattice_size*(self.lattice_size-1)) in post_cluster_indices:
                    #up boundary
                    #check if oppiste on right boundary is filled as well

                    #print('True three')
                    #rotate to down y_boundary_overlap
                    new_cluster_indices = [self.nn_y_direction[cluster_idx] for cluster_idx in post_cluster_indices]

                    y_rotation_number +=  1

                    post_rot_cluster['{0}'.format(cluster_name[0])] =  new_cluster_indices

                    #print(post_rot_cluster)


                    
                elif current_cluster_idx > self.lattice_size*(self.lattice_size-1)  and y_rotation_number < self.lattice_size and current_cluster_idx-(self.lattice_size*(self.lattice_size-1)) in post_cluster_indices:
                    

                    #print('True four')
                    new_cluster_indices = [self.nn_y_direction[cluster_idx] for cluster_idx in post_cluster_indices]
                        
                    post_rot_cluster['{0}'.format(cluster_name[0])] =  new_cluster_indices
                        
                    y_rotation_number +=1 

                    #print(post_rot_cluster)

                else:
                    counter +=1



        return post_rot_cluster
    

    def translation_of_pb_overlapping_cluster_tensor(self, cluster_lattice):
        #if cluster goes over boundary, it needs to be translated as much as needed
        
        #define boundary positions
        #check if boundary on opisite filled as well

        #print(pre_rot_cluster)
        #print(inputclustername)
        #post_rot_tensor = torch.clone(cluster_lattice)


        rot_nmb_d = 0
        rot_nmb_r = 0
        while torch.sum(torch.logical_and(cluster_lattice[0,:],cluster_lattice[self.lattice_size-1,:])) > 0 and rot_nmb_d< self.lattice_size:
            #print(rot_nmb_d)
            cluster_lattice = torch.roll(cluster_lattice, 1 , 0)
            rot_nmb_d+=1


        while torch.sum(torch.logical_and(cluster_lattice[:,0],cluster_lattice[:,self.lattice_size-1])) > 0 and rot_nmb_r< self.lattice_size:
            cluster_lattice = torch.roll(cluster_lattice, 1 ,1)
            rot_nmb_r+=1

        return cluster_lattice

    




    def euler_charactistics(self):#basically need cluster in cluster bettinumber 0: vertices; betti1: holes
        states = self.plaquettes
        clusters = self.elements_per_clusters #need to call self.cluster() before self.cluster()#return dic of all clusters idx
        number_of_clusters = list(clusters.keys())
        ##print(clusters)
        #print('full_cluster_list',number_of_clusters)
        holes_to_clusters = {}
        euler_to_clusters = {}
        #holes_to_clusters['{0}'.format(0)] = []
        holes_to_clusters_list = []

        for idx,cluster_name in enumerate(list(clusters.keys())):
            cluster_array = np.zeros(self.lattice_size*self.lattice_size)
            #clustervalues only the areas of cluster with position
            # one is objects zero is nothing/hole
            #cluster_name = cluster.key()

            #print(cluster_name)

            temp_dict_cluster = {cluster_name:clusters[cluster_name]}

            #print('current_cluster',temp_dict_cluster)

            transformed_cluster = self.translation_of_pb_overlapping_cluster(pre_rot_cluster=temp_dict_cluster,inputclustername=cluster_name)
            #print(transformed_cluster)

            for cluster_pos in list(transformed_cluster.values()):
                cluster_array[cluster_pos] += 1#add one on position

            
            cluster_lattice = cluster_array.reshape(self.lattice_size,self.lattice_size)

            padded_cluster_lattice = np.pad(cluster_lattice, 3, mode='constant')

            #print(clusters[cluster_name])



            e = euler_number(padded_cluster_lattice,connectivity=1) #full euelr charc.
            
            object = label(padded_cluster_lattice,connectivity=1).max()#returns cluster objects 
            number_of_holes = object - e#betti number one
            holes_to_clusters['{0}'.format(cluster_name)] = [e,object,number_of_holes]
            euler_to_clusters[cluster_name] = e
            holes_to_clusters_list.append([e])#,object,number_of_holes])

        if len(holes_to_clusters_list) == 0:
            holes_to_clusters_list = [0]#[0,0,0]]

        self.euler_per_cluster_dict = euler_to_clusters


        return holes_to_clusters,holes_to_clusters_list#
    

    def euler_charactistics_faster(self, cluster_lattices):#takes directly a lattice with zeros and ones for each cluster

        #print('euler')



        #start_time = time.time()


        transformed_cluster = self.translation_of_pb_overlapping_cluster_tensor(cluster_lattices).cpu().numpy()

        #end_time = time.time()

        #print('time',end_time-start_time)

        padded_cluster_lattice = np.pad(transformed_cluster, 3, mode='constant')

        e = euler_number(padded_cluster_lattice,connectivity=1) #full euelr charc.


            
        ##object = label(padded_cluster_lattice,connectivity=1).max()#returns cluster objects 
       #number_of_holes = object - e#betti number one
        #holes_to_clusters = [e,object,number_of_holes]
        #euler_to_clusters = e
        

        #if len(holes_to_clusters_list) == 0:
        #    holes_to_clusters_list = [0]#[0,0,0]]

        ###
        #alternative

        plaquette_positions = torch.nonzero(cluster_lattices)





        return e#








        #holes_in_clusters = {'{0}'.format(i) :[] for i in range(number_of_clusters+1)}
        #nmb_of_holes_in_clusters = {'{0}'.format(i) :[] for i in range(number_of_clusters)}
        #
        #for idx in range(int(self.lattice_size*self.lattice_size)):
        #    if states[idx] == 0:
        #        states_nn = [state_nn for state_nn in self.nbr[idx]]
        #        nn_state_in_clusters = [self.get_key_from_value(state_nn, clusters) for state_nn in states_nn]
        #        if int(len(nn_state_in_clusters)) > 1:#nn are in multible clusters
        #            holes_in_clusters['{0}'.format(0)].append(idx)#part of outside 'hole' so everything which is not a hole, if nn are part of more then one cluster
        #        elif int(len(nn_state_in_clusters)) == 1:#use same key as cluster for corresponding hole
        #            cluster_nmb = int(nn_state_in_clusters.item())
        #            holes_in_clusters['{0}'.format(cluster_nmb)].append(idx)
        #        elif int(len(nn_state_in_clusters)) == 0 and np.any(states_nn in holes_in_clusters.values()):#add to hole if part of larger hole
        #            #get hole cluster key
        #            hole_nmb =5
        #        else:
        #            holes_in_clusters['{0}'.format(0)].append(idx)
        #
        #return holes_in_clusters,nmb_of_holes_in_clusters






    def minkowski_measure(self):

        states = self.plaquettes

        clusters = self.elements_per_clusters #need to call self.cluster() #this is a dictionary for all clusters and the positions of their parts

        areas_of_clusters = []
        perimeters_of_clusters = []
        areas_of_clusters_dict = {}
        perimeters_of_clusters_dict = {}


        for idx,cluster_name in enumerate(list(clusters.keys())):
            cluster_pos_list = clusters[cluster_name]
            area_per_cluster = len(cluster_pos_list)
            areas_of_clusters_dict[cluster_name] = area_per_cluster
            areas_of_clusters.append(area_per_cluster)
            perimeter_cluster = 0
            #print(cluster_positons)
            for cluster_idx in cluster_pos_list:
                #print(cluster_idx)
                nn_cluster_indices = [nn_cluster_idx for nn_cluster_idx in self.nbr[cluster_idx]]
                nn_cluster_indices.sort()
                #print(nn_cluster_indices)
                nn_cluster = [nn_cluster_i in list(cluster_pos_list) for nn_cluster_i in nn_cluster_indices ]
                #print(nn_cluster)
                perimeter_per_cluster_element = len(nn_cluster)-np.sum(nn_cluster)
                #print(perimeter_per_cluster_element)
                perimeter_cluster += perimeter_per_cluster_element
            perimeters_of_clusters.append(perimeter_cluster)
            perimeters_of_clusters_dict[cluster_name] = perimeter_cluster

        self.cluster_areas = areas_of_clusters
        self.cluster_perimeter=perimeters_of_clusters
        self.cluster_areas_dict = areas_of_clusters_dict
        self.cluster_perimeters_dict = perimeters_of_clusters_dict

        return areas_of_clusters,perimeters_of_clusters
    
    def minkowski_measure_new_faster(self,clusters_zero_one_tensors):# should take a tensor of all the clusters for one exmaple and one epsilon

        #print('mink')


        areas_of_clusters = []
        perimeters_of_clusters = []
        areas_of_clusters_dict = {}
        perimeters_of_clusters_dict = {}
       

        areas_of_clusters = torch.sum(clusters_zero_one_tensors.view(-1,self.lattice_size*self.lattice_size), dim = 1 )

        n_maker = nn.Unfold(kernel_size=(3,3), stride=1)

        #print('minko',clusters_zero_one_tensors.size())

        ##circ_pad = nn.CircularPad2d((1,1,1,1))
        
        ##cluster_halo = circ_pad(clusters_zero_one_tensors)

        cluster_halo = F.pad(clusters_zero_one_tensors.view(-1,1,self.lattice_size,self.lattice_size),pad = (1,1,1,1),mode = 'circular')

        nn_cluster_temp = n_maker(cluster_halo).view(clusters_zero_one_tensors.size(0),9,self.lattice_size*self.lattice_size)

        nn_cluster_transposed = torch.transpose(nn_cluster_temp,1,2)

        indices_cluster = torch.tensor([4,1,3,5,7], device=self.device)

        nn_cluster_indices = torch.index_select(nn_cluster_transposed, dim=2, index=indices_cluster)


        nn_check = nn_cluster_indices[:,:,0]==1


        
        perimeter_cluster_start = nn_cluster_indices[nn_cluster_indices[:,:,0] == 1].view(clusters_zero_one_tensors.size(0),-1,5)

        #print(perimeter_cluster_start)

        #print(perimeter_cluster_start.size())

        perimeters_of_clusters = torch.sum(torch.sum(perimeter_cluster_start[:,:,0:]==0,dim=2),dim=1).float()


        return torch.flatten(areas_of_clusters),torch.flatten(perimeters_of_clusters)
    
    
    def minkowski_measure_faster(self):

        states = self.plaquettes

        clusters = self.elements_per_clusters #need to call self.cluster() #this is a dictionary for all clusters and the positions of their parts

        areas_of_clusters = []
        perimeters_of_clusters = []
        areas_of_clusters_dict = {}
        perimeters_of_clusters_dict = {}


        for idx,cluster_name in enumerate(list(clusters.keys())):
            cluster_pos_list = clusters[cluster_name]
            area_per_cluster = len(cluster_pos_list)
            areas_of_clusters_dict[cluster_name] = area_per_cluster
            areas_of_clusters.append(area_per_cluster)
            perimeter_cluster = 0
            #print(cluster_positons)
            for cluster_idx in cluster_pos_list:
                #print(cluster_idx)
                nn_cluster_indices = [nn_cluster_idx for nn_cluster_idx in self.nbr[cluster_idx]]
                nn_cluster_indices.sort()
                #print(nn_cluster_indices)
                nn_cluster = [nn_cluster_i in list(cluster_pos_list) for nn_cluster_i in nn_cluster_indices ]
                #print(nn_cluster)
                perimeter_per_cluster_element = len(nn_cluster)-np.sum(nn_cluster)
                #print(perimeter_per_cluster_element)
                perimeter_cluster += perimeter_per_cluster_element
            perimeters_of_clusters.append(perimeter_cluster)
            perimeters_of_clusters_dict[cluster_name] = perimeter_cluster

        self.cluster_areas = areas_of_clusters
        self.cluster_perimeter=perimeters_of_clusters
        self.cluster_areas_dict = areas_of_clusters_dict
        self.cluster_perimeters_dict = perimeters_of_clusters_dict

        return areas_of_clusters,perimeters_of_clusters
    

    def cluster_max_size(self):
        clusters = self.elements_per_clusters #need to call self.cluster() 

        cluster_max_size = {}

        #need maximal distance in x and y direction
        #first find min and maximal x and y position

        for idx,cluster_name in enumerate(clusters.keys()):
            cluster_max_size['{0}'.format(idx)] = []
            #make the indeces into an x and y positions
            x_pos_cluster_index = clusters[cluster_name]%self.lattice_size
            y_pos_cluster_index = int(clusters[cluster_name]/self.lattice_size)

            temp_min_pos_x = x_pos_cluster_index.min(x_pos_cluster_index)
            temp_min_pos_y = y_pos_cluster_index.min(y_pos_cluster_index)
            temp_max_pos_x = x_pos_cluster_index.max(x_pos_cluster_index)
            temp_max_pos_y = y_pos_cluster_index.max(y_pos_cluster_index)



    def cluster_com(self):
        clusters = self.elements_per_clusters #need to call self.cluster() 


        clusters_com = {}

        clusters_com_list = []

        full_mass_list = self.cluster_areas

        for idx,cluster_name in enumerate(list(clusters.keys())):
            cluster_mass= full_mass_list[idx]
            cluster_positions = clusters['{0}'.format(cluster_name)]
            #print(cluster_positions)
            cluster_x_positions = np.array(cluster_positions)%self.lattice_size
            cluster_y_positions = np.array(cluster_positions)//self.lattice_size
            #print('x', cluster_x_positions)
            #print('y', cluster_y_positions)
            cluster_com_x = float(np.sum(cluster_x_positions))/float(cluster_mass)
            cluster_com_y = float(np.sum(cluster_y_positions))/float(cluster_mass)
            clusters_com['{0}'.format(cluster_name[0])] = [cluster_com_x,cluster_com_y]
            clusters_com_list.append([cluster_com_x,cluster_com_y])

        return clusters_com, clusters_com_list
    
    def cluster_com_edges(self):
        clusters = self.cluster()


        clusters_com = {}

        clusters_com_list = []

        full_mass_list = self.cluster_areas

        for idx,cluster_name in enumerate(clusters.keys()):
            cluster_mass= full_mass_list[idx]
            cluster_positions = clusters['{0}'.format(cluster_name)]
            cluster_x_positions = np.array(cluster_positions%self.lattice_size)
            cluster_y_positions = np.array(int(cluster_positions/self.lattice_size))
            cluster_com_x = float(np.sum(cluster_x_positions))/float(cluster_mass)
            cluster_com_y = float(np.sum(cluster_y_positions))/float(cluster_mass)
            clusters_com['{0}'.format(cluster_name[0])] = [cluster_com_x,cluster_com_y]
            clusters_com_list.append([cluster_com_x,cluster_com_y])

        return clusters_com, clusters_com_list
    
    def cluster_size_distance(self):
        clusters_com_dict,clusters_com_list = self.cluster_com()

        cluster_distances = {}
        cluster_distances_list = []
        ##print(len(cluster_distances_list))


        for idx, key in enumerate(clusters_com_dict):
            #print(key)
            #current_cluster_com = clusters_com_dict[key]
            
            cluster_distance = [np.sqrt((np.power(clusters_com_list[idx][0]-clusters_com_list[other_com][0],2)+np.power(clusters_com_list[idx][1]-clusters_com_list[other_com][1],2))) for other_com in range(idx+1,int(len(clusters_com_list)))]
            ##cluster_distance_full = [np.sqrt((np.power(clusters_com_list[idx,0]-clusters_com_list[other_com,0],2)+np.power(clusters_com_list[idx,1]-clusters_com_list[other_com,1],2))) for other_com in range(0,int(len(clusters_com_list)))]
            cluster_distances_list.append(cluster_distance)
            if int(len(cluster_distance)) > 0:
                cluster_distances['{0}'.format(key[0])] = cluster_distance
            else:
                cluster_distances['{0}'.format(key[0])] = [0]
            ##can just claculate eyverytime less distances
        self.cluster_distances = cluster_distances

        #permutation of cluster coms should give option to get all distances



        if int(len(cluster_distances_list)) > 1:
            cluster_distances_sum = 0.0
            leng_cluster_list = 0
            for cl_list in cluster_distances_list:
                if int(len(cl_list)) > 0:
                    leng_cluster_list += int(len(cl_list))
                    cluster_distances_sum += sum(cl_list)
            self.mean_cluster_distance = float(cluster_distances_sum)/float(leng_cluster_list)
        else:
            self.mean_cluster_distance = 0.0

    def defect_position(self,defects,number_of_defects):
        position_vector = np.zeros((number_of_defects,3))
        counter = 0
        
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if defects[i,j] != 0:
                    position_vector[counter,0] += i
                    position_vector[counter,1] += j
                    position_vector[counter,2] += np.copy(defects[i,j])
                    counter +=1

        return position_vector



    def cluster_distance_defect(self, defect_lattice, graph_typ, number_defects=2 ):
        if graph_typ=='Plaquette':
            clusters_com_dict,clusters_com_list = self.cluster_com()
        elif graph_typ== 'Edges':
            clusters_com_dict,clusters_com_list = self.cluster_com_edges()

        defects_positions = self.defect_position(defects= defect_lattice,number_of_defects=number_defects)#should have form of (defectnumber,x,y)

        distance_cluster_com_nearest_defects = {}

        mean_defect_distance_cluster = {}

        cluster_nearest_defect_distances_list = []

        clusters_sort_by_defect_distance = {}

        #min_distances_defect_list = []

        for idx, key in enumerate(clusters_com_dict):
            #print(key)
            current_cluster_com = clusters_com_dict[key]#(x,y)
            distances_defects_per_cluster = np.array([np.sqrt(((defects[0]-current_cluster_com[0])**2+(defects[1]-current_cluster_com[1])**2)/2.0) for defects in defects_positions])
            min_distance_defect_idx = np.argmin(distances_defects_per_cluster)
            min_distance_defect = distances_defects_per_cluster[min_distance_defect_idx]
            #min_distances_defect_list.append(min_distance_defect)
            means_distance = np.mean(distances_defects_per_cluster)
            mean_defect_distance_cluster['{0}'.format(key)] = means_distance
            distance_cluster_com_nearest_defects['{0}'.format(key)] = (min_distance_defect_idx,min_distance_defect)

            if '{0}'.format(min_distance_defect) in clusters_sort_by_defect_distance:
                clusters_sort_by_defect_distance['{0}'.format(min_distance_defect)].append(key)
            else:
                clusters_sort_by_defect_distance['{0}'.format(min_distance_defect)] = []
                clusters_sort_by_defect_distance['{0}'.format(min_distance_defect)].append(key)



        ##how get clusters for different distances

        self.clusters_idx_by_defect_distance = clusters_sort_by_defect_distance

        return distance_cluster_com_nearest_defects
    

    def means_dep_distance(self):


        cluster_nb_per_dist = {}
        cluster_areas_per_dist = {}
        cluster_perimeters_per_dist = {}
        cluster_euler_per_dist = {}
        cluster_dist_per_dist = {}

        for key in self.clusters_idx_by_defect_distance:
            cluster_list = self.clusters_idx_by_defect_distance[key]
            cluster_nb_per_dist[key] = int(len(cluster_list))
            list_area_per_distance = [ self.cluster_areas_dict['{0}'.format(cluster)] for cluster in cluster_list]
            list_perimeters_per_distance = [ self.cluster_perimeters_dict ['{0}'.format(cluster)] for cluster in cluster_list]
            list_euler_per_distance = [ self.euler_per_cluster_dict['{0}'.format(cluster)] for cluster in cluster_list]
            cluster_areas_per_dist[key] = np.mean(np.array(list_area_per_distance))
            cluster_perimeters_per_dist[key] = np.mean(np.array(list_perimeters_per_distance))
            cluster_euler_per_dist[key] = np.mean(np.array(list_euler_per_distance))

            list_cluster_distances = [self.cluster_distances['{0}'.format(cluster)] for cluster in cluster_list]

            cluster_distances_sum = 0.0
            leng_cluster_list = 0
            if int(len(list_cluster_distances)) > 1:
                for cl_list in list_cluster_distances:
                    if int(len(cl_list)) > 0:
                        leng_cluster_list += int(len(cl_list))
                        cluster_distances_sum += sum(cl_list)
                cluster_dist_per_dist[key] = np.array(cluster_distances_sum/float(leng_cluster_list))
            else:
                cluster_dist_per_dist[key] = 0.0


        return cluster_nb_per_dist, cluster_areas_per_dist,cluster_perimeters_per_dist,cluster_euler_per_dist,cluster_dist_per_dist







        #size just check for each distance len()

            #first calcu nearest defect for each cluster then see how many clusters there are dependent on distance
            ##here comes this
            #also calculate mean distance from cluster and nearest defect

        

        ###how does the cluster cde changes if now intsead of plaquettes edges x,y are used?
        ###cluster positions of x and y toget
