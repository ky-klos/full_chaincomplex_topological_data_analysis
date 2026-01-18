import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import matplotlib.image as mpimg
#from matplotlib.legend_handler import HandlerLine2D
import math
#from scipy.optimize import curve_fit
from numpy import pi
import colorcet as cc
#from matplotlib.cm import get_cmap
#from colorsys import hls_to_rgb
import matplotlib.colors
import time

#from skimage.io import imread
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
import random

import h5py

import pickle
import pandas as pd

from collections import Counter

#from matplotlib.collections import LineCollection

#import seaborn as sb

#import plotly.graph_objects as go

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)

#-----header end --------------------------------------------------------------

#here comes the class observables ,which has objects depending on the model


class Observables():
    def __init__(self, lattice_size, temperature,spin_config_simulation = None, spin_config_generation = None, defect_configuration = None, spin_config_analytic = None, model_typ = None, lattice_spacing =1):
#defualt model typ is at the current moment 2 dimensional xy_model

        self.numb_spins = lattice_size**2

        self.lattice_size = lattice_size

        self.interaction_const = 1.0

        self.two_pi = np.pi*2.0

        L = lattice_size

        self.lattice_spacing = lattice_spacing

        self.temperature = temperature

        self.nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % (L*L),
                    (i // L) * L + (i - 1) % L, (i - L) % (L*L)) \
                                            for i in list(range(L*L))}#right,down,left,up
        self.pqts = {i : (i  ,((i+L)%(L*L)),(i+1-((i%L)//(L-1))*L+L)%(L*L) ,i+1-((i%L)//(L-1))*L,i)  \
                                                for i in list(range((L*L)))}
        self.pqts_nbr = {i : ((i - L) % (L*L) ,(i // L) * L + (i - 1) % L,
        ((i // L) * L + (i - 1) % L +L)%(L*L),
        (i + 2*L) % (L*L),((i+1-((i%L)//(L-1))*L+L)%(L*L) + L) % (L*L),
        ((i // L) * L + (i + 2) % L+L)%(L*L),
        (i // L) * L + (i + 2) % L,
        ((i+1-((i%L)//(L-1))*L)- L) % (L*L)  ) \
                                                for i in list(range((L*L)))}

        self.nn_x_direction = {i : ((i // L) * L + (i + 1) % L)\
                                            for i in list(range(L*L))}#right
        self.nn_y_direction = {i : ((i + L) % (L*L),(i - L) % (L*L))\
                                            for i in list(range((L*L)))}#down, up
        


        self.analyse_spin = spin_config_analytic

        self.simulation_spin = spin_config_simulation #should be 2 dim

        self.generation_spin = spin_config_generation #should be 2 dim

        self.defect_config = defect_configuration #should be 2 dim

        self.vortex_number = self.set_vortex_number()

        self.spin_position = np.asarray([[[i*self.lattice_spacing,j*self.lattice_spacing] for j in range(self.lattice_size)] for i in range(self.lattice_size)])

        self.right = {i : ((i // L) * L + (i + 1) % L)\
                                            for i in list(range(L*L))}#right
        
        self.left = {i : ((i // L) * L + (i - 1) % L)\
                                            for i in list(range(L*L))}#n times left
        
        self.down = {i : ((i + L) % (L*L))\
                                            for i in list(range(L*L))}
        
        self.up = {i : ((i - L) % (L*L))\
                                            for i in list(range(L*L))}
        
        self.upleft = {i : ((((i // L) * L + (i - 1) % L) - L) % (L*L))\
                                            for i in list(range(L*L))}
        
        self.downleft = {i : (((i // L) * L + (i - 1) % L +L)%(L*L))\
                                            for i in list(range(L*L))}
        
        self.upright = {i : (((i+1-((i%L)//(L-1))*L)- L) % (L*L))\
                                            for i in list(range(L*L))}
        
        self.downright = {i : (((i // L) * L + (i - 1) % L +L)%(L*L))\
                                            for i in list(range(L*L))}

    
    def update_defect_config(self, new_defect_config):
        self.defect_config =  new_defect_config

    def set_current_spin_field(self, spin_field):
        self.current_spin_field = spin_field

    def set_current_temperature(self, temp):
        self.temperature = temp

    def set_vortex_number(self):
        if self.defect_config == None:
            return 0
        else:
            vortex_counter = 0
            defect_array = self.defect_config.reshape(self.lattice_size*self.lattice_size)
            for vortex in defect_array:
                if vortex == 1 or vortex == -1:
                    vortex_counter += 1

            self.vortex_number = vortex_counter

            return vortex_counter

    def defect_position(self,input_defect_lattice= None):


        counter = 0

        if input_defect_lattice == None:
            defect_lattice = self.defect_config
        else:
            defect_lattice = input_defect_lattice

        number_of_defects = np.sum(np.abs(defect_lattice))

        defect_position_vector = np.zeros((number_of_defects,3))


        for i in range(self.lattice_siz):
            for j in range(self.lattice_siz):
                if defect_lattice[i,j] != 0:
                    defect_position_vector[counter,0] += i
                    defect_position_vector[counter,1] += j
                    defect_position_vector[counter,2] += np.copy(defect_lattice[i,j])
                    counter +=1

        #print('caluclated_defects',position_vector)

        return defect_position_vector
    
    def defect_position_tensor(self,maximal_defect_number,input_defect_lattice= None):


        counter = 0

        if input_defect_lattice == None:
            defect_lattice = self.defect_config
        else:
            defect_lattice = input_defect_lattice

        #number_of_defects = torch.sum(torch.abs(defect_lattice))

        defect_position_vector = torch.zeros(int(maximal_defect_number),3,device = device)
        
        defect_distance_vector = torch.zeros(int(maximal_defect_number/2),1,device = device)


        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if defect_lattice[i,j] != 0:
                    defect_position_vector[counter,0] += i
                    defect_position_vector[counter,1] += j
                    defect_position_vector[counter,2] += torch.clone(defect_lattice[i,j])
                    counter +=1

        #print('caluclated_defects',position_vector)
        #print(defect_distance_vector) 
        
        
        if counter == 0:
            
            defect_distance = torch.tensor(0).to(device)
            defect_distance_vector[:counter] = defect_distance.view(-1,1)
            #print(counter, defect_distance)
        elif counter == 2:
            defect_distance = torch.sqrt(torch.pow((defect_position_vector[0,0]-defect_position_vector[1,0]),2) +torch.pow((defect_position_vector[0,1]-defect_position_vector[1,1]),2)).view(-1,1)
            defect_distance_vector[:counter] = defect_distance
            #print(counter, defect_distance)
        else:
            vortices = defect_position_vector[defect_position_vector[:,2]== 1]
            anti_vortices = defect_position_vector[defect_position_vector[:,2]== -1]
            defect_distance = torch.sqrt(torch.pow((vortices[:,0]-anti_vortices[:,0]),2) +torch.pow((vortices[:,1]-anti_vortices[:,1]),2)).view(-1,1)
            ##print(counter,defect_distance_vector.shape, defect_distance.shape)
            counter_idx = (counter//2)
            ##print(counter_idx)
            defect_distance_vector[:counter_idx] = defect_distance
            
                    
        

        return defect_position_vector, defect_distance_vector

    def set_analytic_spin_config(self,input_defect_positions=None, input_vortex_numb = None):
        if self.analyse_spin == None:

            if input_vortex_numb == None:
                vortex_numb = self.vortex_number()
            else:
                vortex_numb = input_vortex_numb

            if input_defect_positions == None:
                defect_positions = self.defect_position()
            else:
                defect_positions = input_defect_positions

            analytic_solution_list = []

            for i in range(self.lattice_size):#should be x direction
                    for j in range(self.lattice_size):#should be y driection
                        analystic_angles = [(defect_positions[v,2])*np.arctan2(np.asarray(self.spin_position[i,j,1]-(defect_positions[v,1]+0.5)),np.asarray(self.spin_position[i,j,0]-(defect_positions[v,0]+0.5))) for v in range(vortex_numb)]

                        analytic_angle = (np.sum(analystic_angles))%(2*np.pi)

                        #we want it in area [0,2pi]

                        if analytic_angle < 0.0:
                            analytic_angle += (2*np.pi)


                        analytic_solution_list.append(analytic_angle)



            analytic_solution = np.asarray(analytic_solution_list).reshape(self.lattice_size,self.lattice_size)





            self.current_analyse_spin = analytic_solution

            return analytic_solution
        
    def set_analytic_spin_config_tensor(self,input_defect_positions=None, input_vortex_numb = 2):
        if self.analyse_spin == None:

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


    def get_random_angle_config(self,lattice_size):#L >= 20, better 28,30 for small finite size effects
        random_angle_config = np.random.random(lattice_size*lattice_size)*2*np.pi
        random_angle_lattice = random_angle_config.reshape(lattice_size,lattice_size)
        return random_angle_config,random_angle_lattice

    def angle_difference(self, angle_lattice = 'analytic', comparison_angle_lattice = 'sim'):

        if angle_lattice == 'analytic':
            angle_lattice = self.analyse_spin
        elif angle_lattice == 'simulation':
            angle_lattice = self.simulation_spin

        else:
            print('Wrong string')


        if comparison_angle_lattice == 'simulation':
            angle_lattice = self.simulation_spin
        elif angle_lattice == 'generation':
            angle_lattice = self.generation_spin

        else:
            print('Wrong string')

        angle_diff_list = []

        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                temp_angle_diff = angle_lattice[i,j] - comparison_angle_lattice[i,j]

                if temp_angle_diff >= 0.0:
                    if temp_angle_diff >= np.pi :
                        angle_diff = -2*np.pi + temp_angle_diff
                    else:
                        angle_diff = temp_angle_diff
                else:
                    if temp_angle_diff <= -np.pi :
                        angle_diff = 2*np.pi + temp_angle_diff
                    else:
                        angle_diff = temp_angle_diff

                angle_diff_list.append(angle_diff)

        self.angle_diff_lattice = np.asarray(angle_diff_list).reshape(self.lattice_size,self.lattice_size)

        return np.asarray(angle_diff_list)

    def far_field(self, spin_field=None):#for sepcific_defect position

        #need far field which is minimal > lattice_size/2 of any defect

        if spin_field == None:
            spin_field = self.current_spin_field
        far_field_magnetisation_angle = self.far_field(spin_field)

        vortex_numb = self.get_vortex_number()

        defect_positions = self.defect_position()

        all_lattice_indices = np.asarray([[[i,j] for j in range(self.lattice_size)] for i in range(self.lattice_size)])

        all_lattice_indices = all_lattice_indices.reshape(-1,2)

        far_field_index_list = []

        for idx in all_lattice_indices:

            #print(idx)

            if np.all(np.asarray([np.sqrt((idx[0]-defect_positions[counter,0])**2+(idx[1]-defect_positions[counter,1])**2) for counter in range(vortex_numb)]) > self.lattice_size/2.0):
                far_field_index_list.append(idx)


        far_field_index = np.asarray(far_field_index_list)

        test_index_1 = far_field_index_list[0][0]
        test_index_2 = far_field_index_list[0][1]



        #print(angle_lattice[test_index_1,test_index_2])


        far_field_spins = [ spin_field[index[0], index[1]] for index in  far_field_index]

        mean_angle_far_field = np.mean(np.asarray(far_field_spins))

        return mean_angle_far_field

    def rotation_comparison_angle(self,spin_field= None): #should rorated far field to spin in x direction
        
        if spin_field == None:
            spin_field = self.current_spin_field
        far_field_magnetisation_angle = self.far_field(spin_field)
        #print('far field',far_field_magnetisation_angle)

        if far_field_magnetisation_angle > np.pi:
            rotation_angle = (2*np.pi - far_field_magnetisation_angle)
        else:
            rotation_angle = -far_field_magnetisation_angle

        return rotation_angle

    def magentisation(self, spinfield):

        maget_x = np.sum(np.cos(spinfield))
        maget_y = np.sum(np.sin(spinfield))

        return maget_x,maget_y

    def mean_magentisation(self,spin_lattice_array):

        magnetisation_abs_series = []
        for spin_lattice in spin_lattice_array:
            maget_x,maget_y = self.magentisation(spin_lattice)
            magnetisation_abs_series.append(np.sqrt(maget_x**2+maget_y**2)/float(self.numb_spins))

        magnetisation_abs_array = np.asarray(magnetisation_abs_series)

        magnetic_mean = np.average(magnetisation_abs_array)

        magnetic_mean_2 = np.average(magnetisation_abs_array**2)

        return magnetic_mean, magnetic_mean_2

    def get_magnet_suscep(self,spin_lattice_array):
        magnetic_mean, magnetic_mean_2 = self.mean_magentisation(spin_lattice_array)
        magnet_suscep = (magnetic_mean_2-magnetic_mean**2)/self.temperature

        return magnet_suscep

    def energy(self,spinfield):

        spinfield_array = spinfield.reshape(self.lattice_size*self.lattice_size)
        energy = np.sum(-self.interaction_const*np.sum(np.cos(spin-spinfield[n]) for n in self.nbr[idx]) for idx,spin in enumerate(spinfield_array))/2.0

        return energy
    
    def make_energy_plaquettes(self,spinangle_tensor):

        spins_halo= F.pad(spinangle_tensor,(1,1,1,1),'circular')

        nn_maker = nn.Unfold(kernel_size=(3,3),stride=1)

        nn_temp = nn_maker(spins_halo).view(spinangle_tensor.size(0),9,self.lattice_size*self.lattice_size)

        next_n = torch.transpose(nn_temp, 1,2)

        index = torch.tensor([4,1,3,5,7],device=device)

        all_nn = torch.index_select(next_n,dim=2, index=index)

        return all_nn
    
    def faster_mean_energy(self,spin_lattice_tensor):
        ##shape input array : (batchsize,lattice_size,lattice_size)
        spin_lattice_tensor = spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size)
        sample_nmb =int(spin_lattice_tensor.size(0))
        nn_tensor = self.make_energy_plaquettes(spinangle_tensor=spin_lattice_tensor)##should have size: batchsize, latticesize**2, 5
        ##print('nntensor',nn_tensor.size())

        energy_sum_tensor = -(self.interaction_const/2.0)*torch.sum((torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,1])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,2])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,3])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,4])),dim=1)

        energy_tensor_mean = torch.mean(energy_sum_tensor)
        energy_tensor_mean_2 = torch.mean(torch.pow(energy_sum_tensor,2))


        self.energy_tensor_mean = energy_tensor_mean
        self.energy_tensor_mean_2 = energy_tensor_mean_2

        ##print('e sum tensor',energy_sum_tensor.size())

        energy_tensor_variance = torch.sum(torch.pow((energy_sum_tensor-energy_tensor_mean),2))/(sample_nmb-1)



        return energy_tensor_mean,energy_tensor_variance,energy_sum_tensor

    def faster_mean_local_energy(self,spin_lattice_tensor):

        spin_lattice_tensor = spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size)
        sample_nmb =int(spin_lattice_tensor.size(0))
        nn_tensor = self.make_energy_plaquettes(spinangle_tensor=spin_lattice_tensor)##should have size: batchsize, latticesize**2, 

        energy_distribution_tensor = -(self.interaction_const)*(torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,1])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,2])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,3])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,4]))

        return energy_distribution_tensor.view(-1,self.lattice_size*self.lattice_size)
    
    def faster_mean_local_energy_connections(self,spin_lattice_tensor):

        spin_lattice_tensor = spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size)
        sample_nmb =int(spin_lattice_tensor.size(0))
        nn_tensor = self.make_energy_plaquettes(spinangle_tensor=spin_lattice_tensor)##should have size: batchsize, latticesize**2, 

        energy_conncetions_tensor = torch.zeros((spin_lattice_tensor.size(0),self.lattice_size*self.lattice_size))

        energy_conncetions_tensor[:,]

        energy_distribution_tensor = -(self.interaction_const)*(torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,1])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,2])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,3])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,4]))

        return energy_distribution_tensor.view(-1,self.lattice_size*self.lattice_size)
    
    def faster_specfifc_heat(self):

        ##need faster mean energy before hand
            
        specfifc_heat = (self.energy_tensor_mean_2 -torch.pow(self.energy_tensor_mean,2))/(self.temperature**2)

        return specfifc_heat
    
    def faster_mean_magentisation(self,spin_lattice_tensor):

        spin_lattice_tensor = spin_lattice_tensor.view(-1,self.lattice_size*self.lattice_size)
        sample_nmb =int(spin_lattice_tensor.size(0))
        

        spin_lattice_tensor_sin = torch.sum(torch.sin(spin_lattice_tensor),dim=1)
        spin_lattice_tensor_cos = torch.sum(torch.cos(spin_lattice_tensor),dim=1)

        magnetisation_tensor = torch.sqrt((torch.pow(spin_lattice_tensor_sin,2)+torch.pow(spin_lattice_tensor_cos,2)))/(self.lattice_size*self.lattice_size)

        magnetisation_tensor_mean = torch.mean(magnetisation_tensor)
        magnetisation_tensor_mean_2 = torch.mean(torch.pow(magnetisation_tensor,2))

        self.magnetisation_tensor_mean=magnetisation_tensor_mean
        self.magnetisation_tensor_mean_2 = magnetisation_tensor_mean_2

        self.magent_tensor_mean_x = torch.mean(spin_lattice_tensor_cos)
        self.magent_tensor_mean_y = torch.mean(spin_lattice_tensor_sin)
        self.magent_tensor_mean_x_2 = torch.mean(torch.pow(spin_lattice_tensor_cos,2))
        self.magent_tensor_mean_y_2 = torch.mean(torch.pow(spin_lattice_tensor_sin,2))

        self.magent_tensor_mean_full_2 = torch.mean(torch.pow(spin_lattice_tensor_cos,2)+torch.pow(spin_lattice_tensor_sin,2))

        magnetisation_tensor_variance = torch.sum(torch.pow((magnetisation_tensor-magnetisation_tensor_mean),2))/(sample_nmb-1)

        return magnetisation_tensor_mean,magnetisation_tensor_variance, magnetisation_tensor
    
    def faster_magnet_suscep(self):

        #need fastermagnetisation beforehand
        ##this is for absolute magent but should be magentic vector
        ##magnet_suscep = (self.magnetisation_tensor_mean_2-torch.pow(self.magnetisation_tensor_mean,2))/self.temperature
        #print('compare norm',self.magent_tensor_mean_full_2)
        magnet_suscep = (self.magent_tensor_mean_full_2-torch.pow(self.magent_tensor_mean_x,2)-torch.pow(self.magent_tensor_mean_y,2))/(self.temperature)

        return magnet_suscep

    def check_sucept_corr_function(self, spin_lattices_angle):

        spin_lattices_angle = spin_lattices_angle.view(-1,self.lattice_size*self.lattice_size)

        spin_lattices_x = torch.mean(torch.sum(torch.cos(spin_lattices_angle), dim = 1))

        spin_lattices_y = torch.mean(torch.sum(torch.sin(spin_lattices_angle), dim = 1))



        list_full_corr = []

        list_extra_part = []

        spin_test = torch.clone(spin_lattices_angle)
        #print('check ')

        ##for idx in range(self.lattice_size*self.lattice_size):
            #print(torch.stack([torch.cos(spin_lattices_angle[:,idx] - spin_lattices_angle[:,idx_2]) for idx_2 in range(self.lattice_size*self.lattice_size)], dim = 0).size())
        ##    list_full_corr.append(torch.sum(torch.stack([torch.cos(spin_lattices_angle[:,idx] - spin_lattices_angle[:,idx_2]) for idx_2 in range(self.lattice_size*self.lattice_size)], dim = 0),dim=0))
            #list_extra_part.append(torch.stack([(spin_lattices_x[idx]*spin_lattices_x[idx_2])+(spin_lattices_y[idx]*spin_lattices_y[idx_2]) for idx_2 in range(self.lattice_size*self.lattice_size)]))
        #print('compare alternative',torch.mean(torch.sum(torch.stack(list_full_corr, dim = 0), dim = 0)))
        ##sum_corr = torch.mean(torch.sum(torch.stack(list_full_corr, dim = 0), dim = 0)) - torch.pow(spin_lattices_x,2)-torch.pow(spin_lattices_y,2)#torch.sum(torch.stack(list_extra_part)) ## version 2 without extra part 


        #This does the same as this:

        res = spin_test[:,:,None] - spin_test[:,None,:]
        res = torch.cos(res).sum(dim=[1,2])
        sum_corr = torch.mean(res) - torch.pow(spin_lattices_x,2)-torch.pow(spin_lattices_y,2)
        alternative_suscept = sum_corr/(self.temperature)

        return alternative_suscept





    
    def faster_energy_current_per_link(self,spinangle_tensor,direction = 0):

        nn_maker = nn.Unfold(kernel_size=(2,2),stride=1)

        spins_halo= F.pad(spinangle_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        nn_temp = nn_maker(spins_halo).view(spinangle_tensor.size(0),4,self.lattice_size*self.lattice_size)

        next_n = torch.transpose(nn_temp, 1,2)

        if direction == 0:
            nneb = 1

        else:
            nneb = 2

        index = torch.tensor([0,nneb],device=device)

        all_xn = torch.index_select(next_n,dim=2, index=index)


        energy_per_link = torch.cos(all_xn[:,:,0]-all_xn[:,:,1]) 

        current_per_link = torch.sin(all_xn[:,:,0]-all_xn[:,:,1]) 

        sum_energy_per_link = torch.sum(energy_per_link,dim=1)

        #self.energy_per_link = sum_energy_per_link

        sum_current_per_link = torch.pow(torch.sum(current_per_link,dim=1),2)

        #self.current_per_link = sum_current_per_link

        return sum_energy_per_link, sum_current_per_link

    def faster_get_helicity_modulus(self,spin_lattice_tensor):

        sum_energy_per_link, sum_current_per_link = self.faster_energy_current_per_link(spinangle_tensor=spin_lattice_tensor,direction = 0)

        mean_energy_per_link = torch.mean(sum_energy_per_link)
        mean_current_per_link = torch.mean(sum_current_per_link)

        output_helicity_modulus = (mean_energy_per_link-(mean_current_per_link/ self.temperature))/self.numb_spins

        self.helicity_modulus = output_helicity_modulus

        return output_helicity_modulus
    
    def mean_squared_error(self, mean, var_tensor, sample_nmb):

        sigma_squared = torch.sum(torch.pow((var_tensor-mean),2))/(sample_nmb-1.0)
        mean_squared_error = torch.sqrt(sigma_squared)/np.sqrt(sample_nmb)
        return mean_squared_error
    
    def faster_error_calulcation(self,full_observable_tensor_list,full_spin_tensor, sample_nmb,sample_size): #error for [SH, MS, HM]
        #sample_size = full_observable_tensor
        #sample_nmb *sample_size = 0.8 of int(full_spin_tensor.size(0))

        #full_observable_tensor should be tensor for HM, MS,SH

        error_index = [i for i in range(0,int(full_spin_tensor.size(0)))]

        sh_error_list = []
        ms_error_list = []
        ms_alternative_error_list = []
        hm_error_list = []

        for sample in range(sample_nmb):
            observables_indices = random.sample(error_index,sample_size)
            current_spins = full_spin_tensor[observables_indices]
            current_energy, current_energy_var, energy_full = self.faster_mean_energy(spin_lattice_tensor=current_spins)
            current_sh = self.faster_specfifc_heat()
            sh_error_list.append(current_sh)
            current_magnet, current_magnet_var, magnet_full = self.faster_mean_magentisation(spin_lattice_tensor=current_spins)
            current_ms = self.faster_magnet_suscep()
            current_ms_alternative = current_ms#self.check_sucept_corr_function(current_spins)
            ms_error_list.append(current_ms)
            ms_alternative_error_list.append(current_ms_alternative)
            current_hm = self.faster_get_helicity_modulus(spin_lattice_tensor=current_spins)
            hm_error_list.append(current_hm)

        sh_error = self.mean_squared_error(mean=full_observable_tensor_list[0], var_tensor=torch.stack(sh_error_list), sample_nmb=sample_nmb)
        ms_error = self.mean_squared_error(mean=full_observable_tensor_list[1], var_tensor=torch.stack(ms_error_list), sample_nmb=sample_nmb)
        hm_error = self.mean_squared_error(mean=full_observable_tensor_list[3], var_tensor=torch.stack(hm_error_list), sample_nmb=sample_nmb)
        ms_alternative_error = self.mean_squared_error(mean=full_observable_tensor_list[2], var_tensor=torch.stack(ms_alternative_error_list), sample_nmb=sample_nmb)
       


        return sh_error,ms_error,ms_alternative_error,hm_error
    

    def saw_function_tensor(self, spin_lattice_tensor):

        first_saw = torch.where(torch.abs(spin_lattice_tensor-self.two_pi)< torch.abs(spin_lattice_tensor),spin_lattice_tensor-self.two_pi, spin_lattice_tensor)

        second_saw = torch.where(torch.abs(spin_lattice_tensor+self.two_pi)< torch.abs(spin_lattice_tensor), spin_lattice_tensor+self.two_pi, first_saw)

        return second_saw

    

    
    def faster_vorticity(self, spin_lattice_tensor):

        ##for vorticity and defect lattice
        sample_nmb =int(spin_lattice_tensor.size(0))

        self.plaquette_maker = nn.Unfold(kernel_size=(2,2), stride= 1)
        self.plaq_index = torch.tensor([0,2,3,1,0],device=device)

        #print(spin_lattice_tensor.size())

        spin_lattice_halo = F.pad(spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        plaquette_temp = self.plaquette_maker(spin_lattice_halo).view(spin_lattice_tensor.size(0),4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquette_temp, 1, 2)

        all_plaquettes =  torch.index_select(plaquettes,dim=2, index=self.plaq_index)

        ##print(all_plaquettes)

        changed_plaquette_angles = torch.sub(all_plaquettes[:,:,:],all_plaquettes[:,:,0].view(-1,self.lattice_size*self.lattice_size,1))

        ##angle in interval
        ##print(changed_plaquette_angles)

        angel_intervaled_plaquette_angles_1 = torch.where(torch.logical_and(changed_plaquette_angles < 0.0,changed_plaquette_angles > -self.two_pi), changed_plaquette_angles + self.two_pi,changed_plaquette_angles)

        angel_intervaled_plaquette_angles = torch.where(torch.abs(changed_plaquette_angles) >= self.two_pi ,angel_intervaled_plaquette_angles_1%self.two_pi,angel_intervaled_plaquette_angles_1)

        ## angle diff

        full_diff = ((self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,1] - angel_intervaled_plaquette_angles[:,:,0])) 
        +self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,2] - angel_intervaled_plaquette_angles[:,:,1])) 
        +self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,3] - angel_intervaled_plaquette_angles[:,:,2])) 
        +self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,4] - angel_intervaled_plaquette_angles[:,:,3])))/self.two_pi)

        self.defect_lattices = torch.round(full_diff).int()###changed !

        full_vorticity = torch.abs(full_diff).float()

        vorticity = torch.mean(full_vorticity,dim=1)
        vorticity_mean = torch.mean(vorticity)

        
        ##print(full_diff)
        ##print(vorticity)

        vorticity_tensor_variance = torch.sum(torch.pow((vorticity-vorticity_mean),2))/(sample_nmb-1)

        return full_diff,  vorticity_mean, vorticity_tensor_variance, vorticity






    def set_defects_lattices(self, sim_defect_lattices, gen_defect_lattices):

        self.gen_defect_lattices = gen_defect_lattices.view(-1,self.lattice_size,self.lattice_size)
        self.sim_defect_lattices = sim_defect_lattices.view(-1,self.lattice_size,self.lattice_size)

    
    def defect_distance(self):

        sample_nmb =int(self.sim_defect_lattices.size(0))

        defect_difference = torch.mean(torch.abs((self.gen_defect_lattices-self.sim_defect_lattices)).view(-1,self.lattice_size*self.lattice_size).float(),dim=1)
        defect_difference_mean = torch.mean(defect_difference)

        defect_diff_variance = torch.sum(torch.pow((defect_difference-defect_difference_mean),2))/(sample_nmb-1)

        return defect_difference_mean,defect_diff_variance,defect_difference
    

    ##def faster_angle_diff(self,zero_temp_spin_lattice, other_spinlattices):

    def set_gen_spins(self, spin_lattices):

        self.gen_lattices = spin_lattices

    def set_sim_spins(self, spin_lattices):

        self.sim_lattices = spin_lattices

    def set_comparison_defect_positions(self, defect_lattices, vortex_number =2):
        self.comparison_defect_pos_lattice = defect_lattices


    def set_full_defect_positions(self,defect_lattices,maximal_defect_nmb=2):
        full_defect_position_list = []
        defect_distance_list = []
        #print(defect_lattices.size())
        

        for defect_lattice in defect_lattices:
            defect_position,defect_distance = self.defect_position_tensor(maximal_defect_number=maximal_defect_nmb,input_defect_lattice = defect_lattice)
            full_defect_position_list.append(defect_position)
            defect_distance_list.append(defect_distance)

        self.full_defect_positions = torch.stack(full_defect_position_list,dim=0)
        
        #print(maximal_defect_nmb)
        
        #print(defect_distance_list)
        self.full_defect_distance = torch.stack(defect_distance_list, dim =0)

    def return_full_defect_positions(self):
        return self.full_defect_positions
    
    def return_full_defect_distances(self):
        return self.full_defect_distance   

 
    def set_analytic_spin_lattices(self, unique_defect_lattices, defect_nb_list):
   

        full_analytic_spin_lattices_list = []
        

        for idx,defect_lattice in enumerate(unique_defect_lattices):

            
            full_analytic_spin_lattices_list.append(self.set_analytic_spin_config_tensor(input_defect_positions=defect_lattice, input_vortex_numb = defect_nb_list[idx]))

        full_analytic_spin_lattices = torch.stack(full_analytic_spin_lattices_list,dim=0)
       

        self.analytic_spin_lattices = full_analytic_spin_lattices


        return full_analytic_spin_lattices

    def far_field_faster(self, spin_lattices,defect_number_list,maximal_defect_nmb,minimal_distance = 0):#for sepcific_defect position

        #need far field which is minimal > lattice_size/2 of any defect plus the full boundary

        #set_analytic_spin_lattices need to be mode beforehand

        #self.full_defect_positions should have size [batchsize, counter, 3] 0:x,1:y,2:type

        ##
        if minimal_distance == 0:
            distance_barrier = self.lattice_size/4.0
        else:
            distance_barrier = minimal_distance
        all_lattice_indices = torch.tensor([[[i,j] for j in range(self.lattice_size)] for i in range(self.lattice_size)]).view(1,self.lattice_size,self.lattice_size,2)


        all_lattice_indices_batches = torch.repeat_interleave(all_lattice_indices,int(spin_lattices.size(0)),dim=0)

        ##print(all_lattice_indices_batches.size())

        ##in case of just two defects:

        distance_defect_list = []

        for max_defect in range(maximal_defect_nmb):
            distance_defect_spin_side = torch.sqrt((torch.pow((all_lattice_indices_batches[:,:,:,0] - self.full_defect_positions[:,max_defect,0].view(-1,1,1)),2)+torch.pow((all_lattice_indices_batches[:,:,:,1] - self.full_defect_positions[:,max_defect,1].view(-1,1,1)),2)))> distance_barrier

        distance_defect_spin_side = torch.logical_and(distance_defect_list).view(-1,self.lattice_size*self.lattice_size)##mask for the spins with minimal distance to the defects

        nmb_farfield_spins = torch.sum(distance_defect_spin_side,dim=1).float()
        
        spin_lattice_tensor = spin_lattices.view(-1,self.lattice_size*self.lattice_size)
        

        far_spin_lattice_tensor_sin = torch.sum(torch.where(distance_defect_spin_side,torch.sin(spin_lattice_tensor),0.0),dim=1)
        far_spin_lattice_tensor_cos = torch.sum(torch.where(distance_defect_spin_side,torch.cos(spin_lattice_tensor),0.0),dim=1)

        self.farfield_magnet = torch.sqrt((torch.pow(far_spin_lattice_tensor_sin,2)+torch.pow(far_spin_lattice_tensor_cos,2)))/nmb_farfield_spins

        farfield_angle = torch.arctan2((far_spin_lattice_tensor_sin/nmb_farfield_spins,far_spin_lattice_tensor_cos/nmb_farfield_spins))##angle of mean magnetisation


        return farfield_angle

    def rotation_comparison_angle_faster(self,spin_fields,defect_number_list,maximal_defect_nmb,minimal_distance=0): #should rorated far field to spin in x direction
        

        far_field_magnetisation_angles = self.far_field_faster(spin_fields,minimal_distance=minimal_distance,maximal_defect_nmb=maximal_defect_nmb,defect_number_list=defect_number_list)
        #print('far field',far_field_magnetisation_angle)

        rotation_angles = torch.where(far_field_magnetisation_angles> np.pi,(2*np.pi - far_field_magnetisation_angles),-far_field_magnetisation_angles)

        return rotation_angles
    



    def faster_angle_difference(self, spin_lattices, analytic_angle_lattices, defect_positions,maximal_defect_nmb,defect_number_list):

        ##both magentic field need to be in the same direction

        rotation_angles = self.rotation_comparison_angle_faster(spin_fields=spin_lattices,defect_number_list=defect_number_list, maximal_defect_nmb=maximal_defect_nmb,minimal_distance = 0)

        rotated_spin_lattices = spin_lattices+rotation_angles

        angel_intervaled_rotated_spin_angles_1 = torch.where(torch.logical_and( rotated_spin_lattices< 0.0,rotated_spin_lattices > -self.two_pi), rotated_spin_lattices + self.two_pi,rotated_spin_lattices)

        angel_intervaled_rotated_spin_angles = torch.where(torch.abs(rotated_spin_lattices) >= self.two_pi ,angel_intervaled_rotated_spin_angles_1%self.two_pi,angel_intervaled_rotated_spin_angles_1)


        


        angle_diff_list = []

                ##angle in interval
        
        angle_difference  = self.saw_function_tensor((angel_intervaled_rotated_spin_angles - analytic_angle_lattices)) ##should be minimal angle

        self.angle_diff_full = angle_difference

        mean_angle_diff = torch.mean(torch.abs(angle_difference),dim=0)

        

        angel_intervaled_diff_angles_1 = torch.where(angle_difference < 0.0 and  angle_difference > -self.two_pi , angle_difference + self.two_pi,angle_difference)

        angel_intervaled_diff_angles = torch.where(torch.abs(angle_difference) >= self.two_pi ,angel_intervaled_diff_angles_1%self.two_pi,angel_intervaled_diff_angles_1)

        self.check_spinwaves = angel_intervaled_diff_angles

        return angle_difference, mean_angle_diff








    



    def mean_energy(self,spin_lattice_array):
        energy_series = []
        for spin_lattice in spin_lattice_array:
            energy_series.append(self.energy(spinfield=spin_lattice))
        energy_array = np.asarray(energy_series)

        energy_mean = np.average(energy_array)
        energy_mean_2 = np.average(energy_array**2)

        return energy_mean,energy_mean_2



    def get_specfifc_heat(self,spin_lattice_array):
            energy_mean, energy_mean_2 = self.mean_energy(spin_lattice_array)
            specfifc_heat = (energy_mean_2-energy_mean**2)/(self.temperature**2)

            return specfifc_heat


    def energy_current_per_link(self,spinfield,direction = 0):

        energy_per_link = [np.cos(spin_lattice-spinfield[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(spinfield)]

        current_per_link = [np.sin(spin_lattice-spinfield[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(spinfield)]

        sum_energy_per_link = np.sum(energy_per_link)

        #self.energy_per_link = sum_energy_per_link

        sum_current_per_link = ((np.sum(current_per_link))**2)

        #self.current_per_link = sum_current_per_link

        return sum_energy_per_link, sum_current_per_link

    def get_helicity_modulus(self,spin_lattice_array):

        energies_per_link = []
        currents_per_link = []



        for spin_lattice in spin_lattice_array:
            energy_per_link, current_per_link = self.energy_current_per_link(direction = 0,spinfield=spin_lattice)
            energies_per_link.append(energy_per_link)
            currents_per_link.append(current_per_link)

        mean_energy_per_link = np.average(np.asarray(energies_per_link))
        mean_current_per_link = np.average(np.asarray(currents_per_link))

        output_helicity_modulus = (mean_energy_per_link-mean_current_per_link/ self.temperature)/self.numb_spins

        self.helicity_modulus = output_helicity_modulus

        return output_helicity_modulus



    def pair_distance(self):

        number_vortex = self.get_vortex_number()
        defect_positions = self.defect_position()#returns [counter,(i,j,vortex number)]

        defect_pair_nmb = int(number_vortex/2.0)

        pair_distances_x_list = []

        vortex_positions = np.where(defect_positions[:,2]==1,defect_positions)

        anti_vortex_positions = np.where(defect_positions[:,2]==-1,defect_positions)

        for counter in range(defect_pair_nmb):
            vortex_position_x = vortex_positions[counter,0]
            vortex_position_y = vortex_positions[counter,1]

            pair_distances_x = np.abs(anti_vortex_positions[:,0]-vortex_position_x)
            pair_distances_y = np.abs(anti_vortex_positions[:,1]-vortex_position_y)
    


    def max_corr_distance(self, max_corr = 0):

        if max_corr == 0:

            if int(self.lattice_size) % 2 == 0:
                maximum_distance_x_y = ((self.lattice_size)/2)-1
            else:
                maximum_distance_x_y = int((self.lattice_size)/2)#maximal needs tp be left,right and up,down 


        else:
            maximum_distance_x_y = max_corr

        self.maximum_distance_corr = maximum_distance_x_y

        return maximum_distance_x_y

        
    

    def correlation_function(self,spin_field= None):#mean different ways

        correlation_function_full = [] 

        maximum_distance_x_y = self.max_corr_distance()


        possible_spin_distances = np.arange(0,maximum_distance_x_y,1)
        correlation_function_dict = {}
        correlation_function_per_distance = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
        #print(correlation_function_per_distance)
        correlation_function_per_distance.pop('{0}'.format(0.0))
        for idx,spin in enumerate(spin_field):
            correlation_function_per_distance_per_spin = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
            
            #correlation_function_dict['{0}'.format(idx)] = []
            for distances_x in possible_spin_distances:
                
                for distances_y in possible_spin_distances:

                    full_distance = np.sqrt((distances_x+1)**2+(distances_y+1)**2)

                
                    correlation_function_per_distance_per_pos = []

                    #print(self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)])


                    corr_per_distance = [self.left[(idx-distances_x)%self.lattice_size],
                                         self.right[(idx+distances_x)%self.lattice_size], 
                                         self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)], 
                                         self.down[(idx+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]]#left,right,up,down
                    corr_per_distance_diag = [self.up[((self.left[(idx-distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.left[(idx-distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.right[(idx+distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.up[((self.right[(idx+distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]] # upleft, downleft, downright, upright
                    

                    correlation_function_per_distance_per_pos = [np.cos(spin_field[idx]-spin_field[distance]) for distance in corr_per_distance]
                    correlation_function_per_distance_per_pos = [np.cos(spin_field[idx]-spin_field[distance]) for distance in corr_per_distance_diag]

                    correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[0])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[1])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[2])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[3])

                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[0])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[1])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[2])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[3])


            correlation_function_per_distance_per_spin.pop('{0}'.format(0.0))


            for distance_key in list(correlation_function_per_distance_per_spin.keys()):
                correlation_function_per_distance['{0}'.format(distance_key)].append(np.mean(correlation_function_per_distance_per_spin['{0}'.format(distance_key)]))

     #   correlation_function_dict_mean_per_distance = {}

    #    for distance_key in list(correlation_function_per_distance.keys()):
    #        correlation_function_per_distance_list = list(correlation_function_per_distance[distance_key])
    #        #x = map(lambda x: x[0], correlation_function_per_distance_list)
    #        #y = map(lambda x: x[1], correlation_function_per_distance_list)
    #        #x = list(x)
    #        #y = list(y) 
    #        correlation_function_per_distance_list_flatten = correlation_function_per_distance_list
    #        correlation_function_per_distance_array = np.mean(np.asarray(correlation_function_per_distance_list_flatten))
    #        correlation_function_dict_mean_per_distance[distance_key] = correlation_function_per_distance_array


            


                                 

        correlation_function_dict_mean_per_distance = { distance_key: np.mean(np.array(correlation_function_per_distance[distance_key]).flatten()) for distance_key in list(correlation_function_per_distance.keys())}

        

        return correlation_function_dict_mean_per_distance


    def correlation_function_mean(self,spin_array):

        maximal_distance_correlation = self.max_corr_distance()

        correlation_per_temp= {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
        correlation_per_temp.pop('{0}'.format(0.0))
        for spin_config in spin_array:
            correlation_dict_per_spin_temp = self.correlation_function(spin_field= spin_config)
            for distance_key in list(correlation_per_temp.keys()):
                correlation_per_temp['{0}'.format(distance_key)].append(correlation_dict_per_spin_temp['{0}'.format(distance_key)])

        correlation_mean_per_temp = { 'distance'+ distance_key: [np.mean(correlation_per_temp[distance_key])] for distance_key in list(correlation_per_temp.keys())}

        #print(correlation_per_temp.keys())

        return correlation_mean_per_temp
    
    def correlation_function_faster(self,spin_field= None):#mean different ways

        correlation_function_full = [] 

        maximum_distance_x_y = self.max_corr_distance()


        possible_spin_distances = np.arange(0,maximum_distance_x_y,1)
        correlation_function_dict = {}
        correlation_function_per_distance = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
        #print(correlation_function_per_distance)
        correlation_function_per_distance.pop('{0}'.format(0.0))
        for idx,spin in enumerate(spin_field):
            correlation_function_per_distance_per_spin = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
            
            #correlation_function_dict['{0}'.format(idx)] = []
            for distances_x in possible_spin_distances:
                
                for distances_y in possible_spin_distances:

                    full_distance = np.sqrt((distances_x+1)**2+(distances_y+1)**2)

                
                    correlation_function_per_distance_per_pos = []

                    #print(self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)])


                    corr_per_distance = [self.left[(idx-distances_x)%self.lattice_size],
                                         self.right[(idx+distances_x)%self.lattice_size], 
                                         self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)], 
                                         self.down[(idx+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]]#left,right,up,down
                    corr_per_distance_diag = [self.up[((self.left[(idx-distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.left[(idx-distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.right[(idx+distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.up[((self.right[(idx+distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]] # upleft, downleft, downright, upright
                    

                    correlation_function_per_distance_per_pos = [torch.cos(spin_field[idx]-spin_field[distance]) for distance in corr_per_distance]
                    correlation_function_per_distance_per_pos = [torch.cos(spin_field[idx]-spin_field[distance]) for distance in corr_per_distance_diag]

                    correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[0])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[1])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[2])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[3])

                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[0])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[1])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[2])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[3])


            correlation_function_per_distance_per_spin.pop('{0}'.format(0.0))


            for distance_key in list(correlation_function_per_distance_per_spin.keys()):
                correlation_function_per_distance['{0}'.format(distance_key)].append(torch.mean(torch.stack(correlation_function_per_distance_per_spin['{0}'.format(distance_key)])))

     #   correlation_function_dict_mean_per_distance = {}

    #    for distance_key in list(correlation_function_per_distance.keys()):
    #        correlation_function_per_distance_list = list(correlation_function_per_distance[distance_key])
    #        #x = map(lambda x: x[0], correlation_function_per_distance_list)
    #        #y = map(lambda x: x[1], correlation_function_per_distance_list)
    #        #x = list(x)
    #        #y = list(y) 
    #        correlation_function_per_distance_list_flatten = correlation_function_per_distance_list
    #        correlation_function_per_distance_array = np.mean(np.asarray(correlation_function_per_distance_list_flatten))
    #        correlation_function_dict_mean_per_distance[distance_key] = correlation_function_per_distance_array


            


                                 

        correlation_function_dict_mean_per_distance = { distance_key: torch.mean(torch.flatten(torch.stack(correlation_function_per_distance[distance_key],dim=0))) for distance_key in list(correlation_function_per_distance.keys())}

        

        return correlation_function_dict_mean_per_distance

    def correlation_function_even_faster(self,spin_fields):#mean different ways

        correlation_function_full = [] 
        spin_fields = spin_fields.view(-1,self.lattice_size*self.lattice_size)

        maximum_distance_x_y = self.max_corr_distance()


        possible_spin_distances = np.arange(0,maximum_distance_x_y,1)
        correlation_function_dict = {}
        correlation_function_per_distance = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
        #print(correlation_function_per_distance)
        correlation_function_per_distance.pop('{0}'.format(0.0))

        for idx in range(self.lattice_size*self.lattice_size):

            correlation_function_per_distance_per_spin = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
            correlation_function_per_distance_per_spin.pop('{0}'.format(0.0))
            
            
            #correlation_function_dict['{0}'.format(idx)] = []
            for distances_x in possible_spin_distances:
                
                for distances_y in possible_spin_distances:

                    full_distance = np.sqrt((distances_x+1)**2+(distances_y+1)**2)

                
                    correlation_function_per_distance_per_pos = []

                    #print(self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)])


                    corr_per_distance = [self.left[(idx-distances_x)%self.lattice_size],
                                         self.right[(idx+distances_x)%self.lattice_size], 
                                         self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)], 
                                         self.down[(idx+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]]#left,right,up,down
                    corr_per_distance_diag = [self.up[((self.left[(idx-distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.left[(idx-distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.right[(idx+distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.up[((self.right[(idx+distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]] # upleft, downleft, downright, upright
                    

                    correlation_function_per_distance_per_pos = [torch.cos(spin_fields[:,idx]-spin_fields[:,distance]) for distance in corr_per_distance]
                    correlation_function_per_distance_per_pos = [torch.cos(spin_fields[:,idx]-spin_fields[:,distance]) for distance in corr_per_distance_diag]

                    correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[0])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[1])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[2])
                    correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[3])

                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[0])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[1])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[2])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[3])


            for distance_key in list(correlation_function_per_distance_per_spin.keys()):
                correlation_function_per_distance['{0}'.format(distance_key)].append(torch.mean(torch.stack(correlation_function_per_distance_per_spin['{0}'.format(distance_key)],dim=1),dim=1))
        
        correlation_function_dict_mean_per_distance = { distance_key: torch.flatten(torch.mean(torch.stack(correlation_function_per_distance[distance_key],dim=1),dim=1)) for distance_key in list(correlation_function_per_distance.keys())}
     #   correlation_function_dict_mean_per_distance = {}

    #    for distance_key in list(correlation_function_per_distance.keys()):
    #        correlation_function_per_distance_list = list(correlation_function_per_distance[distance_key])
    #        #x = map(lambda x: x[0], correlation_function_per_distance_list)
    #        #y = map(lambda x: x[1], correlation_function_per_distance_list)
    #        #x = list(x)
    #        #y = list(y) 
    #        correlation_function_per_distance_list_flatten = correlation_function_per_distance_list
    #        correlation_function_per_distance_array = np.mean(np.asarray(correlation_function_per_distance_list_flatten))
    #        correlation_function_dict_mean_per_distance[distance_key] = correlation_function_per_distance_array
        

        #print(correlation_function_dict_mean_per_distance.keys())


            


                                 



        return correlation_function_dict_mean_per_distance
    

    def correlation_function_even_faster_corrected(self,spin_fields):#mean different ways

        correlation_function_full = [] 
        spin_fields = spin_fields.view(-1,self.lattice_size*self.lattice_size)

        maximum_distance_x_y = self.max_corr_distance()


        possible_spin_distances = np.arange(0,maximum_distance_x_y,1)
        correlation_function_dict = {}
        correlation_function_per_distance = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
        #print(correlation_function_per_distance)
        correlation_function_per_distance.pop('{0}'.format(0.0))

        for idx in range(self.lattice_size*self.lattice_size):
            ##correlation_lattices = torch.cos(spin_fields-spin_fields[:,idx])

            ##print(correlation_lattices.size())

            ##print(torch.mean(correlation_lattices,dim=0).view(self.lattice_size,self.lattice_size))

            correlation_function_per_distance_per_spin = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
            correlation_function_per_distance_per_spin.pop('{0}'.format(0.0))
            
            
            #correlation_function_dict['{0}'.format(idx)] = []
            for distances_x in possible_spin_distances:

                corr_per_distance = [self.left[(idx-distances_x)%self.lattice_size],
                                            self.right[(idx+distances_x)%self.lattice_size], 
                                            self.up[(idx-self.lattice_size*distances_x)%(self.lattice_size*self.lattice_size)], 
                                            self.down[(idx+self.lattice_size*distances_x)%(self.lattice_size*self.lattice_size)]]#left,right,up,down
                        
                correlation_function_per_distance_per_pos = [torch.cos(spin_fields[:,idx]-spin_fields[:,distance]) for distance in corr_per_distance]

                correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[0])
                correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[1])
                correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[2])
                correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[3])
                
                for distances_y in possible_spin_distances:

                    full_distance = np.sqrt((distances_x+1)**2+(distances_y+1)**2)

                
                    correlation_function_per_distance_per_pos = []

                    #print(self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)])


                    corr_per_distance_diag = [self.up[((self.left[(idx-distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.left[(idx-distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.down[((self.right[(idx+distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                              self.up[((self.right[(idx+distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]] # upleft, downleft, downright, upright
                    

                    
                    correlation_function_per_distance_per_pos_diag = [torch.cos(spin_fields[:,idx]-spin_fields[:,distance]) for distance in corr_per_distance_diag]



                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos_diag[0])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos_diag[1])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos_diag[2])
                    correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos_diag[3])




            for distance_key in list(correlation_function_per_distance_per_spin.keys()):
                #print('counter')
                #print('distance:',distance_key)
                #print('count',len(correlation_function_per_distance_per_spin['{0}'.format(distance_key)]))
                correlation_function_per_distance['{0}'.format(distance_key)].append(torch.mean(torch.stack(correlation_function_per_distance_per_spin['{0}'.format(distance_key)],dim=1),dim=1))
        
        correlation_function_dict_mean_per_distance = { distance_key: torch.flatten(torch.mean(torch.stack(correlation_function_per_distance[distance_key],dim=1),dim=1)) for distance_key in list(correlation_function_per_distance.keys())}
     #   correlation_function_dict_mean_per_distance = {}

    #    for distance_key in list(correlation_function_per_distance.keys()):
    #        correlation_function_per_distance_list = list(correlation_function_per_distance[distance_key])
    #        #x = map(lambda x: x[0], correlation_function_per_distance_list)
    #        #y = map(lambda x: x[1], correlation_function_per_distance_list)
    #        #x = list(x)
    #        #y = list(y) 
    #        correlation_function_per_distance_list_flatten = correlation_function_per_distance_list
    #        correlation_function_per_distance_array = np.mean(np.asarray(correlation_function_per_distance_list_flatten))
    #        correlation_function_dict_mean_per_distance[distance_key] = correlation_function_per_distance_array
        

        #print(correlation_function_dict_mean_per_distance.keys())


            




        return correlation_function_dict_mean_per_distance
    

    def correlation_function_even_faster_new_try(self,spin_fields):#mean different ways

        correlation_function_full = [] 
        spin_fields = spin_fields.view(-1,self.lattice_size,self.lattice_size)

        maximum_distance_x_y = self.max_corr_distance(max_corr= int(self.lattice_size//2. +1))


        correlation_function_per_distance = {'{0}'.format(torch.sqrt(d_x**2+d_y**2)):[] for d_x in torch.arange(0,maximum_distance_x_y,1) for d_y in torch.arange(0,maximum_distance_x_y,1)}

        


        map_tensor = torch.ones(1,maximum_distance_x_y*maximum_distance_x_y, device=device)
        correlation_indices_test = map_tensor.nonzero(as_tuple=True)[1]
        corr_distances = torch.sqrt(((correlation_indices_test)%maximum_distance_x_y)**2+((correlation_indices_test)//maximum_distance_x_y)**2).view(maximum_distance_x_y, maximum_distance_x_y)
        distance_flipped_right = torch.fliplr(corr_distances)
        full_corr_indices_up = torch.cat((corr_distances,distance_flipped_right[:,1:maximum_distance_x_y-1]),dim=1)
        distance_flipped_down = torch.fliplr(torch.flip(full_corr_indices_up,dims=[1,0]))
        full_corr_distances = torch.cat((full_corr_indices_up,distance_flipped_down[1:maximum_distance_x_y-1,:]),dim=0)

     

        #my version:

        for x_idx in range(self.lattice_size):#self.lattice_size*self.lattice_size

            full_corr_distances_xrot = torch.roll(full_corr_distances,x_idx,dims=1)


            for y_idx in range(self.lattice_size):



                correlation_lattices = torch.cos((torch.repeat_interleave(spin_fields[:,y_idx,x_idx].view(-1,1),self.lattice_size*self.lattice_size,dim=1).view(-1,self.lattice_size,self.lattice_size)-spin_fields[:,:,:])).view(-1,self.lattice_size,self.lattice_size)

                full_corr_distances_fullrot = torch.roll(full_corr_distances_xrot,y_idx,dims=0)



                for distance in torch.unique(corr_distances.flatten()):
                    
                    #print(correlation_lattices.device)
                    #print(full_corr_distances_fullrot.device)
                    #print(distance.device)



                    distance_dep_corr = torch.masked_select(correlation_lattices,(full_corr_distances_fullrot == distance).view(-1,self.lattice_size,self.lattice_size)).view(correlation_lattices.size(0),-1)


                    correlation_function_per_distance['{0}'.format(distance)].append(torch.mean(distance_dep_corr,dim=1))

        # Jans faster version:

        print('check in corr')
        ##correlation_lattices_big = torch.zeros((self.lattice_size*self.lattice_size,spin_fields.shape[0],self.lattice_size,self.lattice_size))
        #full_corr_distances_big = torch.zeros((self.lattice_size*self.lattice_size,*full_corr_distances.shape))
        ##full_corr_distances_big = full_corr_distances.unsqueeze(0)
        ##for x_idx in range(self.lattice_size):#self.lattice_size*self.lattice_size
        ##    #full_corr_distances_xrot = torch.roll(full_corr_distances,x_idx,dims=1)
        ##    for y_idx in range(self.lattice_size):
        ##        correlation_lattices = torch.cos((torch.repeat_interleave(spin_fields[:,y_idx,x_idx].view(-1,1),self.lattice_size*self.lattice_size,dim=1).view(-1,self.lattice_size,self.lattice_size)-spin_fields[:,:,:])).view(-1,self.lattice_size,self.lattice_size)
        ##        correlation_lattices_big[x_idx*self.lattice_size+y_idx] = torch.roll(torch.roll(correlation_lattices,-x_idx,dims=1),-y_idx,dims=0)
        ##        #full_corr_distances_big[x_idx*self.lattice_size+y_idx] = torch.roll(full_corr_distances_xrot,y_idx,dims=0)
        
        
        ##for distance in torch.unique(corr_distances.flatten()):
        ##    mask = (full_corr_distances_big == distance).view(1,-1,self.lattice_size,self.lattice_size)
        ##    distance_dep_corr = (correlation_lattices_big*mask).view(self.lattice_size*self.lattice_size,correlation_lattices.size(0),-1)
        ##    res = torch.mean(distance_dep_corr,dim=2)
        ##    correlation_function_per_distance['{0}'.format(distance)] = torch.mean(res,dim=0)
        
        ##correlation_function_dict_mean_per_distance = { distance_key: torch.flatten(correlation_function_per_distance[distance_key]) for distance_key in list(correlation_function_per_distance.keys())}

        correlation_function_dict_mean_per_distance = { distance_key: torch.flatten(torch.mean(torch.stack(correlation_function_per_distance[distance_key],dim=1),dim=1)) for distance_key in list(correlation_function_per_distance.keys())}


        return correlation_function_dict_mean_per_distance

    def correlation_function_mean_faster(self,spin_array):

        maximal_distance_correlation = self.max_corr_distance()

        correlation_per_temp= {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
        correlation_per_temp.pop('{0}'.format(0.0))
        count = 0
        for spin_config in spin_array:
            #print(count)
            correlation_dict_per_spin_temp = self.correlation_function_faster(spin_field= spin_config.view(self.lattice_size*self.lattice_size))
            for distance_key in list(correlation_per_temp.keys()):
                correlation_per_temp['{0}'.format(distance_key)].append(correlation_dict_per_spin_temp['{0}'.format(distance_key)])

            count += 1

        correlation_mean_per_temp = { 'distance'+ distance_key: [torch.mean(correlation_per_temp[distance_key])] for distance_key in list(correlation_per_temp.keys())}

        #print(correlation_per_temp.keys())

        return correlation_mean_per_temp
    

    def error_calulcation(self,observable_list): #error for HM, MS, SH
        norm = float(len(observable_list))
        observables_array = np.asarray(observable_list)
        observables_mean = np.mean(observables_array)
        sigma_squared = 0.0

        for observable in observables_array:
            
            sigma_squared+= (observable-observables_mean)**2/(norm-1.0)

        observable_error = np.sqrt(sigma_squared)/(np.sqrt(norm))

        return observable_error
    
    def corrlation_function_var(self, correlation_function_list_dict): 
       
        #should be a list [batch, corrleationfunction dep on r]
        
        maximal_distance_correlation = self.max_corr_distance()
 
        var_correlation_per_temp= {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximal_distance_correlation+1,1) for d_y in np.arange(0,maximal_distance_correlation+1,1)}
        var_correlation_per_temp.pop('{0}'.format(0.0))
        counter = 0.0
        for corr_dict in correlation_function_list_dict:
            counter += 1.0
            correlation_dict_per_spin_temp = corr_dict
            for distance_key in list(var_correlation_per_temp.keys()):
                var_correlation_per_temp['{0}'.format(distance_key)].append(correlation_dict_per_spin_temp['{0}'.format(distance_key)])

        correlation_var_per_temp = { 'distance'+ distance_key: np.sqrt((np.mean(np.power(np.asarray(var_correlation_per_temp[distance_key]),2))-np.power(np.mean(np.asarray(var_correlation_per_temp[distance_key])),2)))/(np.sqrt(counter)) for distance_key in list(var_correlation_per_temp.keys())}

        return correlation_var_per_temp

        





    def correlation_function_just_one(self,spin_field= None):#mean different ways

            correlation_function_full = [] 

            maximum_distance_x_y = self.max_corr_distance()


            possible_spin_distances = np.arange(0,maximum_distance_x_y,1)
            correlation_function_dict = {}
            correlation_function_per_distance = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
            #print(correlation_function_per_distance)
            correlation_function_per_distance.pop('{0}'.format(0.0))
            for idx in range(0,1):
                correlation_function_per_distance_per_spin = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
                
                #correlation_function_dict['{0}'.format(idx)] = []
                for distances_x in possible_spin_distances:
                    
                    for distances_y in possible_spin_distances:

                        full_distance = np.sqrt((distances_x+1)**2+(distances_y+1)**2)

                    
                        correlation_function_per_distance_per_pos = []

                        #print(self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)])


                        corr_per_distance = [self.left[(idx-distances_x)%self.lattice_size],
                                            self.right[(idx+distances_x)%self.lattice_size], 
                                            self.up[(idx-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)], 
                                            self.down[(idx+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]]#left,right,up,down
                        corr_per_distance_diag = [self.up[((self.left[(idx-distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                                self.down[((self.left[(idx-distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                                self.down[((self.right[(idx+distances_x)%self.lattice_size])+self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)],
                                                self.up[((self.right[(idx+distances_x)%self.lattice_size])-self.lattice_size*distances_y)%(self.lattice_size*self.lattice_size)]] # upleft, downleft, downright, upright
                        

                        correlation_function_per_distance_per_pos = [np.cos(spin_field[idx]-spin_field[distance]) for distance in corr_per_distance]
                        correlation_function_per_distance_per_pos = [np.cos(spin_field[idx]-spin_field[distance]) for distance in corr_per_distance_diag]

                        correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[0])
                        correlation_function_per_distance_per_spin['{0}'.format(distances_x+1)].append(correlation_function_per_distance_per_pos[1])
                        correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[2])
                        correlation_function_per_distance_per_spin['{0}'.format(distances_y+1)].append(correlation_function_per_distance_per_pos[3])

                        correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[0])
                        correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[1])
                        correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[2])
                        correlation_function_per_distance_per_spin['{0}'.format(full_distance)].append(correlation_function_per_distance_per_pos[3])


                correlation_function_per_distance_per_spin.pop('{0}'.format(0.0))


                for distance_key in list(correlation_function_per_distance_per_spin.keys()):
                    correlation_function_per_distance['{0}'.format(distance_key)].append(np.mean(correlation_function_per_distance_per_spin['{0}'.format(distance_key)]))

        #   correlation_function_dict_mean_per_distance = {}

        #    for distance_key in list(correlation_function_per_distance.keys()):
        #        correlation_function_per_distance_list = list(correlation_function_per_distance[distance_key])
        #        #x = map(lambda x: x[0], correlation_function_per_distance_list)
        #        #y = map(lambda x: x[1], correlation_function_per_distance_list)
        #        #x = list(x)
        #        #y = list(y) 
        #        correlation_function_per_distance_list_flatten = correlation_function_per_distance_list
        #        correlation_function_per_distance_array = np.mean(np.asarray(correlation_function_per_distance_list_flatten))
        #        correlation_function_dict_mean_per_distance[distance_key] = correlation_function_per_distance_array


                


                                    

            correlation_function_dict_mean_per_distance = { distance_key: np.mean(np.array(correlation_function_per_distance[distance_key]).flatten()) for distance_key in list(correlation_function_per_distance.keys())}

            

            return correlation_function_dict_mean_per_distance
        
