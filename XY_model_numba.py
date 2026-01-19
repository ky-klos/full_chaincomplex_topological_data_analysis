import numpy as np
from math import cos

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
import random
import numba as nb
import cProfile
from math import exp as math_exp
import line_profiler

import re

import time

## applying Metropolis algorithm
# input: T/temperature
#        S/spins configuration(in 1d list)
#        H/ecternal field.default value=0
two_pi = 2*np.pi
one_pi = np.pi

N=16*16
L=16

GLOBAL_pqts = tuple([(i  ,((i+L)%N),(i+1-((i%L)//(L-1))*L+L)%N ,i+1-((i%L)//(L-1))*L,i)  \
                                                for i in list(range(N))])

GLOBAL_pqts_nbr = tuple([((i - L) % N ,(i // L) * L + (i - 1) % L,
        ((i // L) * L + (i - 1) % L +L)%N,
        (i + 2*L) % N,((i+1-((i%L)//(L-1))*L+L)%N + L) % N,
        ((i // L) * L + (i + 2) % L+L)%N,
        (i // L) * L + (i + 2) % L,
        ((i+1-((i%L)//(L-1))*L)- L) % N  ) \
                                                for i in list(range(N))])#up left down-left


#upp-left, left,pos, up, upper-left; normal plaquette;
GLOBAL_upleft_pqts = tuple([((((i // L) * L + (i - 1) % L) - L) % N,(i // L) * L + (i - 1) % L,i,(i - L) % N, (((i // L) * L + (i - 1) % L) - L) % N) \
                                    for i in list(range(N))])

#left, down-left, down, pos, left

GLOBAL_downleft_pqts = tuple([((i // L) * L + (i - 1) % L,((i // L) * L + (i - 1) % L +L)%N, (i + L) % N,i,(i // L) * L + (i - 1) % L) \
                                    for i in list(range(N))])
#up,pos, right, up-right, up
GLOBAL_upright_pqts = tuple([( (i - L) % N,i,(i // L) * L + (i + 1) % L,((i+1-((i%L)//(L-1))*L)- L) % N ,(i - L) % N) \
                                    for i in list(range(N))])

GLOBAL_defect_plaquette = tuple([ ((((i // L) * L + (i - 1) % L) - L) % N, (i // L) * L + (i - 1) % L ,i ,(i - L) % N)  \
                                                for i in list(range(N))])


def fix_global_vars(lattice_size):
    global L,N 
    L = lattice_size
    N = L*L

    global GLOBAL_pqts,GLOBAL_pqts_nbr,GLOBAL_upleft_pqts,GLOBAL_downleft_pqts,GLOBAL_upright_pqts,GLOBAL_defect_plaquette


        
    GLOBAL_pqts = tuple([(i  ,((i+L)%N),(i+1-((i%L)//(L-1))*L+L)%N ,i+1-((i%L)//(L-1))*L,i)  \
                                                    for i in list(range(N))])

    GLOBAL_pqts_nbr = tuple([((i - L) % N ,(i // L) * L + (i - 1) % L,
            ((i // L) * L + (i - 1) % L +L)%N,
            (i + 2*L) % N,((i+1-((i%L)//(L-1))*L+L)%N + L) % N,
            ((i // L) * L + (i + 2) % L+L)%N,
            (i // L) * L + (i + 2) % L,
            ((i+1-((i%L)//(L-1))*L)- L) % N  ) \
                                                    for i in list(range(N))])#up left down-left


    #upp-left, left,pos, up, upper-left; normal plaquette;
    GLOBAL_upleft_pqts = tuple([((((i // L) * L + (i - 1) % L) - L) % N,(i // L) * L + (i - 1) % L,i,(i - L) % N, (((i // L) * L + (i - 1) % L) - L) % N) \
                                        for i in list(range(N))])

    #left, down-left, down, pos, left

    GLOBAL_downleft_pqts = tuple([((i // L) * L + (i - 1) % L,((i // L) * L + (i - 1) % L +L)%N, (i + L) % N,i,(i // L) * L + (i - 1) % L) \
                                        for i in list(range(N))])
    #up,pos, right, up-right, up
    GLOBAL_upright_pqts = tuple([( (i - L) % N,i,(i // L) * L + (i + 1) % L,((i+1-((i%L)//(L-1))*L)- L) % N ,(i - L) % N) \
                                        for i in list(range(N))])

    GLOBAL_defect_plaquette = tuple([ ((((i // L) * L + (i - 1) % L) - L) % N, (i // L) * L + (i - 1) % L ,i ,(i - L) % N)  \
                                                    for i in list(range(N))])

@nb.jit(nopython=True)
def setAngleInterval(angle):

    if angle >= two_pi:
        return_angle = angle%two_pi
    elif angle < 0.0 and angle >= -two_pi:
        return_angle = two_pi + angle
    elif angle < 0.0 and angle < -two_pi:
        return_angle = angle%two_pi
    else:
        return_angle = angle

    return return_angle

@nb.jit(nopython=True)
def calc_energy(config,idx,nbrs, dtheta):
    energy_i = -sum([math.cos(config[idx]-config[n]) for n in nbrs])#why not full energy here ?
    energy_f = -sum([math.cos(config[idx]+dtheta-config[n]) for n in nbrs])#why not full energy here ?
    
    return energy_f-energy_i
    
            
@nb.jit(nopython=True)
def get_defect_config_direction_only_angles(plaquetes_angles):

    
    plaquetes_angles_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles]
    #plaquetes_angles_normed_other = [setAngleInterval(spin-plaquetes_angles[0]) for spin in plaquetes_angles]
    #print(plaquetes_angles_normed_other)
    #print(plaquetes_angles_two_pi)
    plaquetes_angles_normed = [(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)]
    #print(plaquetes_angles_normed)
    plaquetes_angles_normed_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles_normed]
    #print(plaquetes_angles_normed_two_pi)
    #print(plaquetes_angles_normed_other==plaquetes_angles_normed_two_pi)
    #after_spin_stuff = time.time()
    #print("The time of execution of above program is :",(after_spin_stuff-before_spin_stuff) * 10**3, "ms")

    #print(len(plaquetes_angles_normed_two_pi))


    #all_diff = sum(self.saw(plaquetes_angles_normed_two_pi[n],plaquetes_angles_normed_two_pi[n+1]) for n in range(4))
    #all_diff = 0
    old_all_diff = 0
    new_all_diff = 0
    #differences = []
    fulldifference = 0.0
    #differences = []
    #print(plaquetes_angles_normed_two_pi)
    
    for n in range(4):
        difference = plaquetes_angles_normed_two_pi[n+1] -plaquetes_angles_normed_two_pi[n]

        if abs(difference - two_pi) < abs(difference):
            difference -= two_pi
        elif abs(difference + two_pi) < abs(difference):
            difference += two_pi

        fulldifference += difference
        #differences.append(difference)
    #differences = [(plaquetes_angles_normed_two_pi[n+1] - plaquetes_angles_normed_two_pi[n] + one_pi)%two_pi - one_pi for n in range(4)]
    #fulldifference = sum(differences)


    #print(all_diff)

    winding_number = int(fulldifference/(2*pi))

    
    #print(winding_number)

    #winding_number = int(winding_number)
    defect_config = winding_number
    #print('winding',defect_config)

    return defect_config

@nb.jit(nopython=True)
def get_defect_config_direction_numba(spins,idx,direction):#directions: 0: up-left; 1: down-left; 2: down-right; 3: up-right
        #print(spins)
        #print(idx)
        #print(direction)

        if direction == 0:
            plaquetes_angles = ([spins[n] for n in GLOBAL_upleft_pqts[idx]])
        elif direction == 1:
            plaquetes_angles = ([spins[n] for n in GLOBAL_downleft_pqts[idx]])
        elif direction == 2:
            plaquetes_angles = ([spins[n] for n in GLOBAL_pqts[idx]])
        elif direction == 3:
            plaquetes_angles = ([spins[n] for n in GLOBAL_upright_pqts[idx]])

        #print(plaquetes_angles)
        return get_defect_config_direction_only_angles(plaquetes_angles)

@nb.jit(nopython=True)
def sweep_alternative_numba_variant(repetitions,temp,num_spins,spin_config,nbr):#for fixed defect configuration

    for r in range(repetitions):
        beta = 1.0 / temp
        spin_idx = np.array(list(range(num_spins)))
        np.random.shuffle(spin_idx)

        #initital_energy = [-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) for idx in spin_idx]
        

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                spin_config[idx] += dtheta
    return spin_config
    
@nb.jit(nopython=True)
def sweep_alternative_old_numba_variant(repetitions,temp,num_spins,spin_config,nbr,defect_config):#for fixed defect configuration

    for r in range(repetitions):
        beta = 1.0 / temp
        spin_idx = np.array(list(range(num_spins)))
        np.random.shuffle(spin_idx)

        #initital_energy = [-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) for idx in spin_idx]
        

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                directions = [0,1,2,3]
                
                
                spin_lattice_temp = np.copy(spin_config)
                spin_lattice_temp[idx] += dtheta
                

                #four_plaquettes_direction = [self.get_defect_config_direction(spins=spin_lattice_temp,idx=idx,direction=direct) for direct in directions]

                
                #defect_plaq = [self.defect_config[n] for n in self.defect_plaquette[idx]]
                all_same=True
                for i in range(4):
                    four_plaquettes_direction_sinlge = get_defect_config_direction_numba(spin_lattice_temp,idx=idx,direction=directions[i])
                    defect_plaq_single = defect_config[GLOBAL_defect_plaquette[idx][i]]
                    if four_plaquettes_direction_sinlge!=defect_plaq_single:
                        all_same=False
                        break
                
                if all_same:
                    spin_config[idx] += dtheta

                
    return spin_config
    
@nb.jit(nopython=True)
def sweep_alternative_old_numba_variant_not_full_fixed(repetitions,temp,num_spins,spin_config,nbr,defect_config):#for fixed defect configuration

    for r in range(repetitions):
        beta = 1.0 / temp
        spin_idx = np.array(list(range(num_spins)))
        np.random.shuffle(spin_idx)

        #initital_energy = [-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) for idx in spin_idx]
        

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                directions = [0,1,2,3]
                
                
                spin_lattice_temp = np.copy(spin_config)
                spin_lattice_temp[idx] += dtheta
                

                #four_plaquettes_direction = [self.get_defect_config_direction(spins=spin_lattice_temp,idx=idx,direction=direct) for direct in directions]

                
                #defect_plaq = [self.defect_config[n] for n in self.defect_plaquette[idx]]
                all_same=True
                for i in range(4):
                    defect_plaq_single = defect_config[GLOBAL_defect_plaquette[idx][i]]
                    if int(np.abs(defect_plaq_single)) == 1:
                        four_plaquettes_direction_sinlge = get_defect_config_direction_numba(spin_lattice_temp,idx=idx,direction=directions[i])
                    
                        ##print('origin',defect_plaq_single)
                        ##print(four_plaquettes_direction_sinlge)
                        ##print(four_plaquettes_direction_sinlge==defect_plaq_single)
                        if four_plaquettes_direction_sinlge!=int(defect_plaq_single) and int(np.abs(defect_plaq_single)) == 1:
                            all_same=False
                            break
                    
                if all_same:
                    spin_config[idx] += dtheta

                
    return spin_config
    



@nb.jit(nopython=True)
def sweep_alternative_original(repetitions,temp,num_spins,spin_config,nbr):#for fixed defect configuration

    for r in range(repetitions):
        beta = 1.0 / temp
        spin_idx = np.array(list(range(num_spins)))
        np.random.shuffle(spin_idx)

        #initital_energy = [-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) for idx in spin_idx]
        

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                spin_config[idx] += dtheta


                
    return spin_config

                


class XYSystem():
    def __init__(self,defect_config=None,input_analytic_solution=None,temperature = 3,width=10):
        self.width = width
        self.num_spins = width**2
        L,N = self.width,self.num_spins
        self.nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                    (i // L) * L + (i - 1) % L, (i - L) % N) \
                                            for i in list(range(N))}#right,down,left,up
        self.nbr_list = [self.nbr[i] for i in range(N)]
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
        self.nn_y_direction = {i : ((i + L) % N,(i - L) % N)\
                                            for i in list(range(N))}#down, up

        self.nn_left_direction = {i : ((i // L) * L + (i - 1) % L)\
                                            for i in list(range(N))}#left

        self.nn_down_direction = {i : ((i + L) % N)\
                                            for i in list(range(N))}#down

        self.nn_up_direction = {i : ((i - L) % N)\
                                            for i in list(range(N))}#up

        self.upleft = {i : ((((i // L) * L + (i - 1) % L) - L) % (L*L))\
                                            for i in list(range(L*L))}

        self.downleft = {i : (((i // L) * L + (i - 1) % L +L)%(L*L))\
                                            for i in list(range(L*L))}

        self.upright = {i : (((i+1-((i%L)//(L-1))*L)- L) % (L*L))\
                                            for i in list(range(L*L))}

        self.downright = {i : ((i+1-((i%L)//(L-1))*L+L)%N)\
                                            for i in list(range(L*L))}



        self.temperature = temperature

        if self.temperature < 0.7:#cold start
            self.spin_config =  np.zeros(self.num_spins)#np.random.random(self.num_spins)*2*np.pi#
        else:# hot start
            self.spin_config = np.random.random(self.num_spins)*2*np.pi

        if input_analytic_solution is not None:
            self.start_spin_config = input_analytic_solution

        #self.start_spin_config = input_analytic_solution



        if defect_config is not None:
            self.defect_config = defect_config
            #self.energy_vortex = np.sum(self.get_energy_vortex(self.defect_config))/self.num_spins
            #self.energy_fix_vortex = np.sum(self.get_energy_fix_vortex())/self.num_spins

        self.energy = np.sum(self.get_energy())/self.num_spins

        self.M = []
        self.Cv = []
        self.HM = []
        self.energy_per_link = []
        self.current_per_link = []
        self.zdim = np.zeros(self.num_spins)
        self.magenetisation = []
        self.susceptibility = []
        self.mean_magnetisation = []

    def set_input_configuration(self):
        self.spin_config = self.start_spin_config

    def set_defects_into_place(self,defect_positions):
        #need defect configuration should change only other spins
        for defect_pos in defect_positions:
            #plaquette_spin = [self.start_spin_config[idx] for idx in self.pqts[defect_pos]]
            for spin_idx in self.pqts[defect_pos]:
                plaquette_spin = self.start_spin_config[spin_idx]
                self.spin_config[spin_idx] = plaquette_spin


    def set_fix_vortex_energy(self):
        self.energy_fix_vortex = np.sum(self.get_energy_fix_vortex())/self.num_spins

    def get_vortex_nmb(self,defects):
        defect_array = defects
        vortex_counter = 0
        for vortex in defect_array:
            if vortex == 1 or vortex == -1:
                vortex_counter += 1

        return vortex_counter


    def check_indz(self,testvector):
        for i in range(self.num_spins):
            test_plaquette_angles = [testvector[n] for n in self.defect_plaquette[i]]
            test_plaquet_upright = [testvector[n] for n in self.upright_pqts[i]]
            test_plaquet_downleft = [testvector[n] for n in self.downleft_pqts[i]]
            test_plaquet_main = [testvector[n] for n in self.pqts[i]]
            test_plaquet_upleft = [testvector[n] for n in self.upleft_pqts[i]]

            print('Test plaquette :',test_plaquette_angles)
            print('upleft',test_plaquet_upleft)
            print('downleft',test_plaquet_downleft)
            print('main',test_plaquet_main)
            print('upright',test_plaquet_upright)

    def set_temperature(self,temperature):
        self.temperature = temperature

    def get_helicity_modulus(self):
        return self.HM

    def get_energy_per_link(self):
        return self.energy_per_link

    def get_current_per_link(self):
        return self.current_per_link

    def get_origin_defect_config(self):
        return self.defect_config

    def sweep_vortex_3dim(self,input_defect_config,K_1=5,K_2=10,K_3=5):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        #random.shuffle(spin_idx)
        #print(input_defect_config.shape)
        #print(self.spin_config.shape)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
                    #k = np.random.randint(0, N - 1)#randomly choose a spin

            #energy_i = (-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) -100*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2)/101
            #energy_i = 10*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2
            energy_i = (-1*sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx])
            +K_1*(self.zdim[idx]**2)+(K_2/2)*(self.get_defect_config(self.spin_config,idx)
            -input_defect_config[idx])**2
            +K_3*sum((1-self.zdim[pn]**2)for pn in self.pqts_nbr[idx]))
            #print(energy_i)
            dtheta = np.random.uniform(-np.pi/8,np.pi/8)
            dz = np.random.uniform(0,0.2)
            spin_temp = self.spin_config[idx] + dtheta
            zdim_temp = self.zdim[idx] + dz

            zdim_lattice_temp = self.zdim
            zdim_lattice_temp[idx] += dz
            spin_lattice_temp = self.spin_config
            spin_lattice_temp[idx] +=dtheta

            #def_temp = self.get_defect_config(spin_lattice_temp,idx)
            #energy_f = (-sum(np.cos(spin_temp-self.spin_config[n]) for n in self.nbr[idx])+100*(def_temp-input_defect_config[idx])**2)/101
            energy_f = (-1*sum(np.cos(spin_lattice_temp[idx]-self.spin_config[n]) for n in self.nbr[idx])
            +K_1*(zdim_lattice_temp[idx]**2)+(K_2/2)*(self.get_defect_config(self.spin_config,idx)
            -input_defect_config[idx])**2
            +K_3*sum((1-zdim_lattice_temp[pn]**2)for pn in self.pqts_nbr[idx]))
            #print(energy_f)
            delta_E = energy_f - energy_i
            #print('deltaE', delta_E)
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta
                self.zdim[idx] += dz

    def sweep(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            energy_i = -sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx])
            dtheta = np.random.uniform(-np.pi,np.pi)#better have gaussian?
            changed_spin = self.spin_config[idx] + dtheta
            #get a new spinangle in gaussian distribution around old angle, with +/- pi is ~max of change
            #changed_spin = np.random.normal(self.spin_config[idx],np.pi/3.0,1)

            energy_f = -sum(np.cos(changed_spin-self.spin_config[n]) for n in self.nbr[idx])
            delta_E = energy_f - energy_i
            #print('deltaE', delta_E)
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta

    def get_defect_config_direction(self,spins,idx,direction):#directions: 0: up-left; 1: down-left; 2: down-right; 3: up-right
        #print(spins)
        #print(idx)
        #print(direction)

        if direction == 0:
            plaquetes_angles = tuple([spins[n] for n in self.upleft_pqts[idx]])
        elif direction == 1:
            plaquetes_angles = tuple([spins[n] for n in self.downleft_pqts[idx]])
        elif direction == 2:
            plaquetes_angles = tuple([spins[n] for n in self.pqts[idx]])
        elif direction == 3:
            plaquetes_angles = tuple([spins[n] for n in self.upright_pqts[idx]])

        #print(plaquetes_angles)
        return get_defect_config_direction_only_angles(plaquetes_angles)
        """
        #print(self.pqts[idx][2])
        #print(spins)

        #print(plaquetes_angles)


        #print('indx',idx)
        #print('spins',np.array(spins).reshape(16,16))
        #print('plaquettes',plaquetes_angles)
        #if idx == 255:
        #    print(no)

        #ANgel differences:


        #print(plaquetes_angles)
        #before_spin_stuff = time.time()
        plaquetes_angles_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles]
        #plaquetes_angles_normed_other = [setAngleInterval(spin-plaquetes_angles[0]) for spin in plaquetes_angles]
        #print(plaquetes_angles_normed_other)
        #print(plaquetes_angles_two_pi)
        plaquetes_angles_normed = [(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)]
        #print(plaquetes_angles_normed)
        plaquetes_angles_normed_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles_normed]
        #print(plaquetes_angles_normed_two_pi)
        #print(plaquetes_angles_normed_other==plaquetes_angles_normed_two_pi)
        #after_spin_stuff = time.time()
        #print("The time of execution of above program is :",(after_spin_stuff-before_spin_stuff) * 10**3, "ms")

        #print(len(plaquetes_angles_normed_two_pi))


        #all_diff = sum(self.saw(plaquetes_angles_normed_two_pi[n],plaquetes_angles_normed_two_pi[n+1]) for n in range(4))
        #all_diff = 0
        old_all_diff = 0
        new_all_diff = 0
        differences = []
        fulldifference = 0.0
        differences = []
        #print(plaquetes_angles_normed_two_pi)
        
        
        differences = [(plaquetes_angles_normed_two_pi[n+1] - plaquetes_angles_normed_two_pi[n] + one_pi)%two_pi - one_pi for n in range(4)]
        fulldifference = sum(differences)
        """
        for n in range(4):
            difference = plaquetes_angles_normed_two_pi[n+1] -plaquetes_angles_normed_two_pi[n]

            if abs(difference - two_pi) < abs(difference):
                difference -= two_pi
            elif np.abs(difference + two_pi) < abs(difference):
                difference += two_pi

            fulldifference += difference
            differences.append(difference)
        """


        #if np.all(np.asarray(differences) > 0.0) or np.all(np.asarray(differences) < 0.0):
        #    winding_number = int(fulldifference/(2*np.pi))
        #else:
    #        winding_number = 0



        #if np.any(np.abs(differences) > (np.pi-(np.pi/6.0))):
                #stronger constarint on defect to have more stable defects
        #        fulldifference = 0.0
        #else:
        #    fulldifference = np.sum(differences)

            #new_all_diff = old_all_diff + self.saw(plaquetes_angles_normed_two_pi[n],plaquetes_angles_normed_two_pi[n+1])
            #print('new', new_all_diff)
            #print('old', old_all_diff)
            #non_physics_diff += (plaquetes_angles_normed_two_pi[n+1]-plaquetes_angles_normed_two_pi[n])
            #if (old_all_diff > 0 and new_all_diff < 0) or (old_all_diff < 0 and new_all_diff > 0) or (np.abs(old_all_diff)>np.abs(new_all_diff)):
            #    all_diff = 0
                #print('stop')
            #    break
            #else:
            #all_diff = new_all_diff
            #old_all_diff = new_all_diff



        #print(all_diff)

        winding_number = int(fulldifference/(2*np.pi))
        #print(winding_number)

        #winding_number = int(winding_number)
        defect_config = winding_number
        #print('winding',defect_config)

        return defect_config
        """

    
    #@line_profiler.profile
    def sweep_alternative_new(self):#for fixed defect configuration
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)

        #initital_energy = [-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) for idx in spin_idx]
        

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(self.spin_config,idx,self.nbr[idx],dtheta)


            if (random.random() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                self.spin_config[idx] += dtheta
    @line_profiler.profile
    def sweep_alternative(self):#for fixed defect configuration
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)

        #initital_energy = [-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) for idx in spin_idx]
        

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            #energy_i = -sum(math.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx])#why not full energy here ?
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(self.spin_config,idx,self.nbr[idx],dtheta)
            #changed_spin = np.random.normal(self.spin_config[idx],np.pi/2.0,1)
            #print('before',self.spin_config)
            
            #print('after',self.spin_config)
            #print('change', dtheta)
            #spin_lattice_temp[idx] = changed_spin
            #spin_temp = self.spin_config[idx] + dtheta

            #energy_f = -sum(math.cos(spin_lattice_temp[idx]-self.spin_config[n]) for n in self.nbr[idx])#spin_temp
            #delta_E = energy_f - energy_i
            #print('deltaE', delta_E)
            #vortex number part

            #need only the four plaquettes surronding the current spin

            if (random.random() < math.exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                directions = [0,1,2,3]
                
                
                spin_lattice_temp = np.copy(self.spin_config)
                spin_lattice_temp[idx] += dtheta
                

                #four_plaquettes_direction = [self.get_defect_config_direction(spins=spin_lattice_temp,idx=idx,direction=direct) for direct in directions]

                
                #defect_plaq = [self.defect_config[n] for n in self.defect_plaquette[idx]]
                all_same=True
                for i in range(len(directions)):
                    four_plaquettes_direction_sinlge = self.get_defect_config_direction(spin_lattice_temp,idx=idx,direction=directions[i])
                    defect_plaq_single = self.defect_config[self.defect_plaquette[idx][i]]
                    if four_plaquettes_direction_sinlge!=defect_plaq_single:
                        all_same=False
                        break
                
                if all_same:
                    self.spin_config[idx] += dtheta
                #print(self.defect_config)
                #print('origin',defect_plaq)
                #if idx == 135 or idx == 140:
                #    print(four_plaquettes_direction)
                #    print(defect_plaq)
                #    print(idx)
                #print('check')
                


            #print(four_plaquettes_direction)
            #print(defect_plaq)



            #print(self.defect_config)

            #if idx == 135 or idx == 140:
            #    print(idx)
            #    print(four_plaquettes_direction)

            #print(idx)
            #print('plaquet',four_plaquettes_direction)

            #print('origin',defect_plaq)

            #print(np.all(four_plaquettes_direction == defect_plaq))




            #defect_configuration_temp = self.get_defect_config_full(spin_temp)

            #vortex_nmb = self.vortex_number_external(defect_configuration_temp)

            #if ((np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E)) and np.all(four_plaquettes_direction == defect_plaq)):
            #    #print(self.defect_config)
            #    #print('origin',defect_plaq)
            #    #if idx == 135 or idx == 140:
            #    #    print(four_plaquettes_direction)
            #    #    print(defect_plaq)
            #    #    print(idx)
            #    #print('check')
            #    self.spin_config[idx] += dtheta
            #    #self.spin_config[idx] = changed_spin
            #    #break
            #break
                #self.spin_config[idx] = changed_spin

    def multible_sweep_alternative(self, times, temperature=None):
        if temperature is None:
            temperature= self.temperature
        else:
            self.temperature = temperature

        self.spin_config =sweep_alternative_old_numba_variant(times,temperature,self.num_spins,self.spin_config,np.array(self.nbr_list),self.defect_config)
        
        #t = 0
        #while t < times:
            
        #    self.sweep_alternative()

        #    t += 1
        
    def multible_sweep_alternative_not_ful_fixed(self, times, temperature=None):
        if temperature is None:
            temperature= self.temperature
        else:
            self.temperature = temperature
        
        self.spin_config = sweep_alternative_old_numba_variant_not_full_fixed(times,temperature,self.num_spins,self.spin_config,np.array(self.nbr_list),self.defect_config)

    def multible_sweep_original(self, times, temperature=None):
        if temperature is None:
            temperature= self.temperature
        else:
            self.temperature = temperature
        
        self.spin_config = sweep_alternative_original(times,temperature,self.num_spins,self.spin_config,np.array(self.nbr_list))

    def sweep_fix_vortex(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            empty_spins =np.zeros(self.start_spin_config.shape)
            empty_defects = np.zeros(self.defect_config.shape)
            energy_i = -sum(np.cos(self.start_spin_config[idx]-self.start_spin_config[n]) for n in self.nbr[idx])

            L=16
            N=16*16

            defect_plaquettes = [self.defect_config[idx],self.defect_config[(idx//L)*L+(idx+1)%L],self.defect_config[(((idx // L) * L + (idx - 1) % L)-L)%N],self.defect_config[(idx-L)%N]]
            no_defect = [0.0,0.0,0.0,0.0]




            dtheta = np.random.uniform(-np.pi,np.pi)

            if np.any(defect_plaquettes != no_defect):
                dtheta = dtheta/2

            #print('before',self.start_spin_config)
            spin_temp = self.start_spin_config[idx] + dtheta
            empty_spins[idx] += dtheta
            spin_lattice_temp = self.start_spin_config + empty_spins
            #print('after',spin_lattice_temp-self.start_spin_config)

            #def_temp = self.get_defect_config(spin_lattice_temp,idx)
            energy_f = -sum(np.cos(spin_temp-self.start_spin_config[n]) for n in self.nbr[idx])
            #print(spin_lattice_temp.shape)
            #defect_config_full = self.get_defect_config_full(spin_lattice_temp)
            #break
            #print(defect_config_i)
            #empty_defects +=defect_config_full
            #check if defects work:
            #print(self.defect_config)
            #print(self.get_defect_config_full(self.start_spin_config))
            #break

            #plaquettes of defects:

            #print(self.start_spin_config.reshape(16,16))
            #print([[self.start_spin_config[i],self.start_spin_config[(((i // L) * L + (i - 1) % L)-L)%N]] for i in list(range(N))])
            #print(((0-1)-((0%L)//(L-1))*L+L)%N )
            #print(((0-L)%N-1)%L)
            #für hoch muss nur (-L)%N hinzugefügt werden

            #defect_plaquettes = [self.defect_config[idx],self.defect_config[(idx//L)*L+(idx+1)%L],self.defect_config[(((idx // L) * L + (idx - 1) % L)-L)%N],self.defect_config[(idx-L)%N]]
            #no_defect = [0.0,0.0,0.0,0.0]

            #print(self.defect_config[idx])
            #print(self.defect_config[(idx//L)*L+(idx+1)%L])
            #print(self.defect_config[(((idx // L) * L + (idx - 1) % L)-L)%N])
            #print(self.defect_config[(idx-L)%N])

            #i

            #print(np.all(defect_plaquettes == no_defect))

            #print(np.all(no_defect == no_defect))

            #(i  ,((i+L)%N),(i+1-((i%L)//(L-1))*L+L)%N ,i+1-((i%L)//(L-1))*L,i)
            # This vector indicates the spin and the spins of which it is part of its plaquette
            #plaquette_influence_vector = (i, )





            defect_config_temp =  self.get_defect_config_full(spin_lattice_temp)

            #defect_config_temp[idx] = defect_config_i
            #print(self.defect_config)
            #print(defect_config_temp)
            #self.show_fix_vortex_change(spin_lattice_temp,colored=False)
            #print(defect_config_i.shape)
            #print('origin',self.defect_config)
            #print('changed',defect_config_i)
            delta_E = energy_f - energy_i
                #print('deltaE', delta_E)

            #old version
            #if np.all(defect_config_temp == self.defect_config):# and np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                #print('true')
                #for n in range(16*16):
                #    print(self.get_defect_config(spin_lattice_temp,n))
            #    if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
            #        self.start_spin_config[idx] += dtheta

            #new version

            if np.all(defect_config_temp == self.defect_config) and np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.start_spin_config[idx] += dtheta

    def sweep_fix_vortex_second(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            empty_spins =np.zeros(self.start_spin_config.shape)
            empty_defects = np.zeros(self.defect_config.shape)
            energy_i = -sum(np.cos(self.start_spin_config[idx]-self.start_spin_config[n]) for n in self.nbr[idx])

            dtheta = np.random.uniform(-np.pi/4,np.pi/4)
            #print('before',self.start_spin_config)
            spin_temp = self.start_spin_config[idx] + dtheta
            empty_spins[idx] += dtheta
            spin_lattice_temp = self.start_spin_config + empty_spins
            #print('after',spin_lattice_temp-self.start_spin_config)

            #def_temp = self.get_defect_config(spin_lattice_temp,idx)
            energy_f = -sum(np.cos(spin_temp-self.start_spin_config[n]) for n in self.nbr[idx])
            #print(spin_lattice_temp.shape)
            #defect_config_full = self.get_defect_config_full(spin_lattice_temp)
            #break
            #print(defect_config_i)
            #empty_defects +=defect_config_full
            #check if defects work:
            #print(self.defect_config)
            #print(self.get_defect_config_full(self.start_spin_config))
            #break

            #plaquettes of defects:
            L=16
            N=16*16
            #print(self.start_spin_config.reshape(16,16))
            #print([[self.start_spin_config[i],self.start_spin_config[(((i // L) * L + (i - 1) % L)-L)%N]] for i in list(range(N))])
            #print(((0-1)-((0%L)//(L-1))*L+L)%N )
            #print(((0-L)%N-1)%L)
            #für hoch muss nur (-L)%N hinzugefügt werden

            #defect_plaquettes = [self.defect_config[idx],self.defect_config[(idx//L)*L+(idx+1)%L],self.defect_config[(((idx // L) * L + (idx - 1) % L)-L)%N],self.defect_config[(idx-L)%N]]
            #no_defect = [0.0,0.0,0.0,0.0]
            #print(self.defect_config[idx])
            #print(self.defect_config[(idx//L)*L+(idx+1)%L])
            #print(self.defect_config[(((idx // L) * L + (idx - 1) % L)-L)%N])
            #print(self.defect_config[(idx-L)%N])

            #i

            #print(np.all(defect_plaquettes == no_defect))

            #print(np.all(no_defect == no_defect))

            #(i  ,((i+L)%N),(i+1-((i%L)//(L-1))*L+L)%N ,i+1-((i%L)//(L-1))*L,i)
            # This vector indicates the spin and the spins of which it is part of its plaquette
            #plaquette_influence_vector = (i, )





            defect_config_temp =  self.get_defect_config_full(spin_lattice_temp)

            #defect_config_temp[idx] = defect_config_i
            #print(self.defect_config)
            #print(defect_config_temp)
            #self.show_fix_vortex_change(spin_lattice_temp,colored=False)
            #print(defect_config_i.shape)
            #print('origin',self.defect_config)
            #print('changed',defect_config_i)
            delta_E = energy_f - energy_i
                #print('deltaE', delta_E)

            #old version
            #if np.all(defect_config_temp == self.defect_config):# and np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                #print('true')
                #for n in range(16*16):
                #    print(self.get_defect_config(spin_lattice_temp,n))
            #    if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
            #        self.start_spin_config[idx] += dtheta

            #new version

            if np.all(defect_config_temp == self.defect_config):

                if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                    self.start_spin_config[idx] += dtheta


            #else:
            #    print('flase')


    def sweep_vortex(self,input_defect_config):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        #random.shuffle(spin_idx)
        #print(input_defect_config.shape)
        #print(self.spin_config.shape)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
                    #k = np.random.randint(0, N - 1)#randomly choose a spin

            #energy_i = (-sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) -100*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2)/101
            energy_i = 10*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2
            #print(energy_i)
            dtheta = np.random.uniform(-np.pi/16,np.pi/16)
            spin_temp = self.spin_config[idx] + dtheta
            spin_lattice_temp = self.spin_config
            spin_lattice_temp[idx] +=dtheta
            def_temp = self.get_defect_config(spin_lattice_temp,idx)
            #energy_f = (-sum(np.cos(spin_temp-self.spin_config[n]) for n in self.nbr[idx])+100*(def_temp-input_defect_config[idx])**2)/101
            energy_f = 10*(def_temp-input_defect_config[idx])**2
            #print(energy_f)
            delta_E = energy_f - energy_i
            #print('deltaE', delta_E)
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta


    

    def saw(self,spin_angle_i,spin_angle_j):
        spin_angle_diff = (spin_angle_j-spin_angle_i)#%2*np.pi
        #temp_angle_diff = spin_angle_diff
        if np.abs(spin_angle_diff - two_pi) < np.abs(spin_angle_diff):
            spin_angle_diff -= two_pi
            #print(spin_angle_diff - two_pi)
        elif np.abs(spin_angle_diff + two_pi) < np.abs(spin_angle_diff):
            spin_angle_diff += two_pi
            #print(spin_angle_diff + two_pi)

        return spin_angle_diff


    def get_defect_config(self,spins,idx):

        #print(self.pqts[idx][2])
        #print(spins)
        plaquetes_angles = [spins[n] for n in self.pqts[idx]]
        #print(plaquetes_angles)


        #print('indx',idx)
        #print('spins',np.array(spins).reshape(16,16))
        #print('plaquettes',plaquetes_angles)
        #if idx == 255:
        #    print(no)

        #ANgel differences:


        #print(plaquetes_angles)
        plaquetes_angles_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles]
        #print(plaquetes_angles_two_pi)
        plaquetes_angles_normed = [(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)]
        #print(plaquetes_angles_normed)
        plaquetes_angles_normed_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles_normed]
        #print(plaquetes_angles_normed_two_pi)

        #print(len(plaquetes_angles_normed_two_pi))


        #all_diff = sum(self.saw(plaquetes_angles_normed_two_pi[n],plaquetes_angles_normed_two_pi[n+1]) for n in range(4))
        #all_diff = 0
        old_all_diff = 0
        new_all_diff = 0
        #print(plaquetes_angles_normed_two_pi)
        for n in range(4):
            new_all_diff = old_all_diff + self.saw(plaquetes_angles_normed_two_pi[n],plaquetes_angles_normed_two_pi[n+1])
            #print('new', new_all_diff)
            #print('old', old_all_diff)
            #non_physics_diff += (plaquetes_angles_normed_two_pi[n+1]-plaquetes_angles_normed_two_pi[n])
            #if (old_all_diff > 0 and new_all_diff < 0) or (old_all_diff < 0 and new_all_diff > 0) or (np.abs(old_all_diff)>np.abs(new_all_diff)):
            #    all_diff = 0
                #print('stop')
            #    break
            #else:
            all_diff = new_all_diff
            old_all_diff = new_all_diff



        #print(all_diff)

        winding_number = int(all_diff/(2*np.pi))
        #print(winding_number)

        #winding_number = int(winding_number)
        defect_config = winding_number
        #print('winding',defect_config)

        return defect_config



    def get_defect_config_full(self,spins):

        #print(spins)

        number_spins = list(range(self.num_spins))
        full_defect_config = np.zeros(spins.shape)
        #print(spins)

        for idx in number_spins:

            full_defect_config[idx] = self.get_defect_config(spins,idx)



        return full_defect_config


    ## calculate the energy of a given configuration
    #  input: S/spin configuration in list
    #         H/external field, defult 0
    def get_energy(self):
        energy_=np.zeros(np.shape(self.spin_config))
        idx = 0
        for spin in self.spin_config: #calculate energy per spin
            energy_[idx] = -1*sum(np.cos(spin-self.spin_config[n]) for n in self.nbr[idx])#nearst neighbor of kth spin
            idx +=1
        return energy_

    def get_energy_fix_vortex(self):
        energy_=np.zeros(np.shape(self.start_spin_config))
        #idx = 0
        for idx,spin in enumerate(self.start_spin_config): #calculate energy per spin
            energy_[idx] = -1*sum(np.cos(spin-self.start_spin_config[n]) for n in self.nbr[idx])#nearst neighbor of kth spin
            #idx +=1
        return energy_



    def get_energy_vortex(self,input_defect_config):
        energy_=np.zeros(np.shape(self.spin_config))
        idx = 0
        for spin in self.spin_config: #calculate energy per spin
            energy_[idx] = (-1*sum(np.cos(spin-self.spin_config[n]) for n in self.nbr[idx])-100*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2)/101
            #energy_[idx] = 10*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2
            idx +=1
        return energy_

    def get_energy_vortex_3dim(self,input_defect_config,K_1=5,K_2=10,K_3=5):
        energy_=np.zeros(np.shape(self.spin_config))
        idx = 0
        for spin in self.spin_config: #calculate energy per spin
            energy_[idx] = (-1*sum(np.cos(spin-self.spin_config[n]) for n in self.nbr[idx])+K_1*(self.zdim[idx]**2)+(K_2/2)*(self.get_defect_config(self.spin_config,idx)
            -input_defect_config[idx])**2
            +K_3*sum((1-self.zdim[pn]**2)for pn in self.pqts_nbr[idx]))
            #energy_[idx] = 10*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2
            idx +=1
        return energy_

    def set_helicity_modulus(self, direction=0, interaction_constant = 1):

        #print(self.spin_config.reshape(self.width,self.width))

        #print([(spin_lattice,self.spin_config[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(self.spin_config)])


        energy_per_link = [np.cos(spin_lattice-self.spin_config[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(self.spin_config)]

        current_per_link = [np.sin(spin_lattice-self.spin_config[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(self.spin_config)]

        sum_energy_per_link = np.sum(energy_per_link)

        self.energy_per_link = sum_energy_per_link

        sum_current_per_link = ((np.sum(current_per_link))**2)

        self.current_per_link = sum_current_per_link

        return sum_energy_per_link, sum_current_per_link


    def set_magentisation(self):
        mag_x = np.power(np.sum(np.cos(self.spin_config)),2)

        mag_y = np.power(np.sum(np.sin(self.spin_config)),2)

        magnetisation = (mag_x+mag_y)/self.num_spins#should be sqrtroot

        self.magenetisation = magnetisation

    def get_magentisation(self):

        return self.magenetisation

    def get_susceptibility(self):
        return self.susceptibility

    def get_mean_magnetisation(self):

        return self.mean_magnetisation




    def equilibrate_vortex_3dim(self,input_defect_config,max_nsweeps=int(1e6),temperature=None,H=None,show = False):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep_vortex_3dim(input_defect_config)
            #list_M.append(np.abs(np.sum(S)/N))
            energy = np.sum(self.get_energy_vortex_3dim(input_defect_config))/self.num_spins
            #print('energy',energy)
            dic_thermal_t['energy'] += [energy]
            #print( abs(energy-energy_temp)/abs(energy))
            if show  & (k%1e3 ==0):
                #print('#sweeps=%i'% (k+1))
                #print('energy=%.2f'%energy)
                self.show()
            #if ((abs(energy-energy_temp)/abs(energy)<1e-4) & (k>700)) or k == max_nsweeps-1:
            if ((self.get_energy_vortex_3dim(input_defect_config)<1e-4).all() & (k>700)) or k == max_nsweeps-1:
                #print('\nequilibrium state is reached at T=%.1f'%self.temperature)
                #print('#sweep=%i'%k)
                print('k',k)
                break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy=np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy_vortex = energy
        energy2=np.average(np.power(dic_thermal_t['energy'][int(nstates/2):],2))
        self.Cv=(energy2-energy**2)*beta**2

    def equilibrate_vortex(self,input_defect_config,max_nsweeps=int(1e6),temperature=None,H=None,show = False):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep_vortex(input_defect_config)
            #list_M.append(np.abs(np.sum(S)/N))
            energy = np.sum(self.get_energy_vortex(input_defect_config))/self.num_spins
            print('energy',energy)
            dic_thermal_t['energy'] += [energy]
            #print( abs(energy-energy_temp)/abs(energy))
            if show  & (k%1e3 ==0):
                #print('#sweeps=%i'% (k+1))
                #print('energy=%.2f'%energy)
                self.show()
            #if ((abs(energy-energy_temp)/abs(energy)<1e-4) & (k>700)) or k == max_nsweeps-1:
            if ((self.get_energy_vortex(input_defect_config)<1e-4).all() & (k>700)) or k == max_nsweeps-1:
                #print('\nequilibrium state is reached at T=%.1f'%self.temperature)
                #print('#sweep=%i'%k)
                print('k',k)
                break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy=np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy_vortex = energy
        energy2=np.average(np.power(dic_thermal_t['energy'][int(nstates/2):],2))
        self.Cv=(energy2-energy**2)*beta**2



    ## Let the system evolve to equilibrium state
    def equilibrate(self,max_nsweeps=int(1e6),temperature=None,show = False):
        if temperature != None:
            self.temperature = temperature
        #dic_thermal_t = {}
        #dic_thermal_t['energy']=[]
        #dic_thermal_t['energy_per_link'] = []
        #dic_thermal_t['current_per_link'] = []
        #dic_thermal_t['number_vortex'] = []
        #dic_thermal_t['magentisation'] = []
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(300)):

            self.sweep_alternative()
        c = 0
        while c < 3000:
            self.sweep_alternative()
            #print(c)
            c += 1


        #for k in list(range(max_nsweeps)):
        #    #break
        #    
        #    #list_M.append(np.abs(np.sum(S)/N))
        #    #energy = np.sum(self.get_energy())/self.num_spins/2.0
        #    #print(energy)
        #    #print('energy',energy)
        #    #dic_thermal_t['energy'] += [energy]
        #    #defect_config = self.get_defect_config_full(self.spin_config)
        # 
        #    #dic_thermal_t['number_vortex'] += [self.get_vortex_nmb(defect_config)]
        #    #energy_per_link_t, current_per_link_t = self.set_helicity_modulus()
        #    #dic_thermal_t['energy_per_link'] += [energy_per_link_t]
        #    #dic_thermal_t['current_per_link'] += [current_per_link_t]
        #    #self.set_magentisation()
        #    #print(self.get_magentisation())
        #    #dic_thermal_t['magentisation'] += [self.get_magentisation()]
        #
        # 
        #
        #    #break
        #
        #    #print( abs(energy-energy_temp)/abs(energy))
        #
        #    #energy_mean = np.average(dic_thermal_t['energy'][k:k+300])
        #    
        #
        #
        #    #if show  & (k%1e4 ==0):
        #    #    #print('#sweeps=%i'% (k+1))
        #    #    #print('energy=%.2f'%energy)
        #    #    self.show()
        #    if ((k>3000)):#(abs(energy_mean-energy)/abs(energy_mean)<1e-4) &
        #        #self.show()
        #        #print('\nequilibrium state is reached at T=%.1f'%self.temperature)
        #        #print('#sweep=%i'%k)
        #        #print('energy=%.2f'%energy)
        #        break
        #    energy_temp = energy
        #nstates = len(dic_thermal_t['energy'])
        #print(dic_thermal_t['energy'][:])
        #sweeps = np.arange(0,nstates, 1)

        #plt.plot(sweeps,dic_thermal_t['energy']/(dic_thermal_t['energy'][0]),'.')
        #plt.plot(sweeps,dic_thermal_t['number_vortex'][:],'.')
        #plt.ylabel(r'$ E $')
        #plt.xlabel('sweep')
        #plt.show()
        #energy=np.average(dic_thermal_t['energy'][1500::])
        #print('mean energy',energy)

        #self.energy = energy
        #energy2=np.average(np.power(dic_thermal_t['energy'][1500:],2))
        #self.Cv=(energy2-energy**2)*beta**2
        #mean_magnetisation = np.average(dic_thermal_t['magentisation'][1500:])
        #self.mean_magnetisation = mean_magnetisation
        #mean_magnetisation2 = np.average(np.power(dic_thermal_t['magentisation'][1500:],2))

        #self.susceptibility = (mean_magnetisation2-np.power(mean_magnetisation,2))/self.temperature


        #self.HM = (np.average(dic_thermal_t['energy_per_link'][5000:])-(np.average(dic_thermal_t['current_per_link'][5000:]))/(self.temperature))/(self.num_spins)

    def equilibrate_fix_vortex(self,max_nsweeps=int(2e5),temperature=None,H=None,show = True):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep_fix_vortex()
            #list_M.append(np.abs(np.sum(S)/N))
            energy = np.sum(self.get_energy_fix_vortex())/self.num_spins/2
            #print('energy',energy)
            dic_thermal_t['energy'] += [energy]
            #print( abs(energy-energy_temp)/abs(energy))
            if show & (k%1e3 ==0) & k != 0:
                #print('#sweeps=%i'% (k+1))
                print('energy=%.2f'%energy)
                break
                #self.show_fix_vortex()
            #if ((abs(energy-energy_temp)/abs(energy)<1e-7) & (k>700)) or k == max_nsweeps-1:
            #    #print('\nequilibrium state is reached at T=%.1f'%self.temperature)
            #    #print('#sweep=%i'%k)
            #    #print('energy=%.2f'%energy)
            #    break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy=np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy_fix_vortex = energy
        energy2=np.average(np.power(dic_thermal_t['energy'][int(nstates/2):],2))
        self.Cv=(energy2-energy**2)*beta**2

    def equilibrate_fix_vortex_second(self,max_nsweeps=int(1e5),temperature=None,H=None,show = True):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep_fix_vortex_second()
            #list_M.append(np.abs(np.sum(S)/N))
            energy = np.sum(self.get_energy_fix_vortex())/self.num_spins/2
            #print('energy',energy)
            dic_thermal_t['energy'] += [energy]
            #print( abs(energy-energy_temp)/abs(energy))
            if show  & (k%2e3 ==0) & k != 0:
                #print('#sweeps=%i'% (k+1))
                #print('energy=%.2f'%energy)
                #self.show_fix_vortex()
                break
            #if ((abs(energy-energy_temp)/abs(energy)<1e-7) & (k>700)) or k == max_nsweeps-1:
                #print('\nequilibrium state is reached at T=%.1f'%self.temperature)
                #print('#sweep=%i'%k)
                #print('energy=%.2f'%energy)
            #    break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy=np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy_fix_vortex = energy
        energy2=np.average(np.power(dic_thermal_t['energy'][int(nstates/2):],2))
        self.Cv=(energy2-energy**2)*beta**2

    ## To see thermoquantities evolve as we cooling the systems down
    # input: T_inital: initial tempreature
    #        T_final: final temperature
    #        sample/'log' or 'lin',mean linear sampled T or log sampled( centered at critical point)
    def annealing(self,T_init=0.5,T_final=0.1,nsteps = 20,show_equi=False):
        # initialize spins. Orientations are taken from 0 - 2pi randomly.
        #initialize spin configuration
        dic_thermal = {}
        dic_thermal['temperature']=list(np.linspace(T_init,T_final,nsteps))
        dic_thermal['energy']=[]
        dic_thermal['Cv']=[]
        dic_thermal['HM'] = []
        for T in dic_thermal['temperature']:
            self.equilibrate(temperature=T)
            if show_equi:
                self.show()
            dic_thermal['energy'] += [self.energy]
            dic_thermal['Cv'] += [self.Cv]
            dic_thermal['HM'] += [self.HM]
        plt.plot(dic_thermal['temperature'],dic_thermal['Cv'],'.')
        plt.ylabel(r'$C_v$')
        plt.xlabel('T')
        plt.show()
        plt.plot(dic_thermal['temperature'],dic_thermal['energy'],'.')
        plt.ylabel(r'$\langle E \rangle$')
        plt.xlabel('T')
        plt.show()
        plt.plot(dic_thermal['temperature'],dic_thermal['HM'],'.')
        plt.ylabel(r'$\langle \gamma \rangle$')
        plt.xlabel('T')
        plt.show()
        return dic_thermal

    def annealing_vortex(self,input_defect_config,T_init=2.5,T_final=0.1,nsteps = 20,show_equi=False):
        # initialize spins. Orientations are taken from 0 - 2pi randomly.
        #initialize spin configuration
        dic_thermal = {}
        dic_thermal['temperature']=list(np.linspace(T_init,T_final,nsteps))
        dic_thermal['energy']=[]
        dic_thermal['Cv']=[]
        for T in dic_thermal['temperature']:
            self.equilibrate_vortex(input_defect_config,temperature=T)
            if show_equi:
                self.show()
            dic_thermal['energy'] += [self.energy_vortex]
            dic_thermal['Cv'] += [self.Cv]

        self.show()
        #plt.plot(dic_thermal['temperature'],dic_thermal['Cv'],'.')
        #plt.ylabel(r'$C_v$')
        #plt.xlabel('T')
        #plt.show()
        #plt.plot(dic_thermal['temperature'],dic_thermal['energy'],'.')
        #plt.ylabel(r'$\langle E \rangle$')
        #plt.xlabel('T')
        #plt.show()
        return dic_thermal




    @staticmethod
    ## convert configuration inz list to matrix form
    def list2matrix(S):
        N=int(np.size(S))
        print(N)
        L = int(np.sqrt(N))
        S=S.reshape((L,L))
        return S

    ## visulize a configurtion
    #  input：S/ spin configuration in list form
    def show(self,colored=False):
        config_matrix = self.list2matrix(self.spin_config)

        X, Y = np.meshgrid(np.arange(0,self.width ),np.arange(0, self.width))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)
        plt.figure(figsize=(4,4), dpi=100)
        #plt.title('Arrows scale with plot width, not view')
        Q = plt.quiver(X, Y, U, V, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()

    def show_fix_vortex(self,colored=False):
        config_matrix = self.list2matrix(self.start_spin_config)
        #print(config_matrix)
        #print(config_matrix.shape)
        X, Y = np.meshgrid(np.arange(0,self.width ),np.arange(self.width ,0,-1))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)

        plt.figure(figsize=(4,4), dpi=100)
        #plt.title('Arrows scale with plot width, not view')
        Q = plt.quiver(X, Y, U, V,config_matrix, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()
    def show_fix_vortex_change(self,spins,colored=False):
        config_matrix = self.list2matrix(spins)
        #print(config_matrix)
        #print(config_matrix.shape)
        X, Y = np.meshgrid(np.arange(0,self.width ),np.arange(self.width ,0,-1))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)

        plt.figure(figsize=(4,4), dpi=100)
        #plt.title('Arrows scale with plot width, not view')
        Q = plt.quiver(X, Y, U, V,config_matrix, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()
