import numpy as np

import matplotlib.pyplot as plt

import random

import math

from numpy import pi

import numba as nb

from math import exp as math_exp



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
    plaquetes_angles_normed = [(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)]
    plaquetes_angles_normed_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles_normed]
    
    old_all_diff = 0
    new_all_diff = 0
    
    fulldifference = 0.0
    
    
    for n in range(4):
        difference = plaquetes_angles_normed_two_pi[n+1] -plaquetes_angles_normed_two_pi[n]

        if abs(difference - two_pi) < abs(difference):
            difference -= two_pi
        elif abs(difference + two_pi) < abs(difference):
            difference += two_pi

        fulldifference += difference
       


    winding_number = int(fulldifference/(2*pi))

    defect_config = winding_number

    return defect_config

@nb.jit(nopython=True)
def get_defect_config_direction_numba(spins,idx,direction):#directions: 0: up-left; 1: down-left; 2: down-right; 3: up-right

        if direction == 0:
            plaquetes_angles = ([spins[n] for n in GLOBAL_upleft_pqts[idx]])
        elif direction == 1:
            plaquetes_angles = ([spins[n] for n in GLOBAL_downleft_pqts[idx]])
        elif direction == 2:
            plaquetes_angles = ([spins[n] for n in GLOBAL_pqts[idx]])
        elif direction == 3:
            plaquetes_angles = ([spins[n] for n in GLOBAL_upright_pqts[idx]])

        return get_defect_config_direction_only_angles(plaquetes_angles)

@nb.jit(nopython=True)
def sweep_alternative_numba_variant(repetitions,temp,num_spins,spin_config,nbr):#for fixed defect configuration

    for r in range(repetitions):
        beta = 1.0 / temp
        spin_idx = np.array(list(range(num_spins)))
        np.random.shuffle(spin_idx)

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): 

                spin_config[idx] += dtheta
    return spin_config
    
@nb.jit(nopython=True)
def sweep_alternative_old_numba_variant(repetitions,temp,num_spins,spin_config,nbr,defect_config):#for fixed defect configuration

    for r in range(repetitions):
        beta = 1.0 / temp
        spin_idx = np.array(list(range(num_spins)))
        np.random.shuffle(spin_idx)

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): 

                directions = [0,1,2,3]
                
                
                spin_lattice_temp = np.copy(spin_config)
                spin_lattice_temp[idx] += dtheta
                
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

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                directions = [0,1,2,3]
                
                
                spin_lattice_temp = np.copy(spin_config)
                spin_lattice_temp[idx] += dtheta
                

                all_same=True
                for i in range(4):
                    defect_plaq_single = defect_config[GLOBAL_defect_plaquette[idx][i]]
                    if int(np.abs(defect_plaq_single)) == 1:
                        four_plaquettes_direction_sinlge = get_defect_config_direction_numba(spin_lattice_temp,idx=idx,direction=directions[i])
                    
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

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(spin_config,idx,nbr[idx],dtheta)


            if (np.random.rand() < math_exp(-beta * delta_E)): 

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

        if defect_config is not None:
            self.defect_config = defect_config

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
        for idx in spin_idx:#one sweep in defined as N attempts of flip

            energy_i = (-1*sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx])
            +K_1*(self.zdim[idx]**2)+(K_2/2)*(self.get_defect_config(self.spin_config,idx)
            -input_defect_config[idx])**2
            +K_3*sum((1-self.zdim[pn]**2)for pn in self.pqts_nbr[idx]))
           
            dtheta = np.random.uniform(-np.pi/8,np.pi/8)
            dz = np.random.uniform(0,0.2)
            spin_temp = self.spin_config[idx] + dtheta
            zdim_temp = self.zdim[idx] + dz

            zdim_lattice_temp = self.zdim
            zdim_lattice_temp[idx] += dz
            spin_lattice_temp = self.spin_config
            spin_lattice_temp[idx] +=dtheta

            energy_f = (-1*sum(np.cos(spin_lattice_temp[idx]-self.spin_config[n]) for n in self.nbr[idx])
            +K_1*(zdim_lattice_temp[idx]**2)+(K_2/2)*(self.get_defect_config(self.spin_config,idx)
            -input_defect_config[idx])**2
            +K_3*sum((1-zdim_lattice_temp[pn]**2)for pn in self.pqts_nbr[idx]))
            
            delta_E = energy_f - energy_i
            
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta
                self.zdim[idx] += dz

    def sweep(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            
            energy_i = -sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx])
            dtheta = np.random.uniform(-np.pi,np.pi)#better have gaussian?
            changed_spin = self.spin_config[idx] + dtheta
            
            

            energy_f = -sum(np.cos(changed_spin-self.spin_config[n]) for n in self.nbr[idx])
            delta_E = energy_f - energy_i
            
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta

    def get_defect_config_direction(self,spins,idx,direction):#directions: 0: up-left; 1: down-left; 2: down-right; 3: up-right


        if direction == 0:
            plaquetes_angles = tuple([spins[n] for n in self.upleft_pqts[idx]])
        elif direction == 1:
            plaquetes_angles = tuple([spins[n] for n in self.downleft_pqts[idx]])
        elif direction == 2:
            plaquetes_angles = tuple([spins[n] for n in self.pqts[idx]])
        elif direction == 3:
            plaquetes_angles = tuple([spins[n] for n in self.upright_pqts[idx]])

        
        return get_defect_config_direction_only_angles(plaquetes_angles)
        
        

    

    def sweep_alternative_new(self):#for fixed defect configuration
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)

       
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(self.spin_config,idx,self.nbr[idx],dtheta)


            if (random.random() < math_exp(-beta * delta_E)): #(random.random() = Uniform(0,1))

                self.spin_config[idx] += dtheta
    
    def sweep_alternative(self):#for fixed defect configuration
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)

        for idx in spin_idx:#one sweep in defined as N attempts of flip
            
            dtheta = random.uniform(-np.pi/2.0,np.pi/2.0)
            
            delta_E = calc_energy(self.spin_config,idx,self.nbr[idx],dtheta)
            
            #need only the four plaquettes surronding the current spin

            if (random.random() < math.exp(-beta * delta_E)): 

                directions = [0,1,2,3]
                
                
                spin_lattice_temp = np.copy(self.spin_config)
                spin_lattice_temp[idx] += dtheta
                
                all_same=True
                for i in range(len(directions)):
                    four_plaquettes_direction_sinlge = self.get_defect_config_direction(spin_lattice_temp,idx=idx,direction=directions[i])
                    defect_plaq_single = self.defect_config[self.defect_plaquette[idx][i]]
                    if four_plaquettes_direction_sinlge!=defect_plaq_single:
                        all_same=False
                        break
                
                if all_same:
                    self.spin_config[idx] += dtheta
             

    def multiple_sweep_alternative(self, times, temperature=None):
        if temperature is None:
            temperature= self.temperature
        else:
            self.temperature = temperature

        self.spin_config =sweep_alternative_old_numba_variant(times,temperature,self.num_spins,self.spin_config,np.array(self.nbr_list),self.defect_config)
        
    
    def multiple_sweep_alternative_not_ful_fixed(self, times, temperature=None):
        if temperature is None:
            temperature= self.temperature
        else:
            self.temperature = temperature
        
        self.spin_config = sweep_alternative_old_numba_variant_not_full_fixed(times,temperature,self.num_spins,self.spin_config,np.array(self.nbr_list),self.defect_config)

    def multiple_sweep_original(self, times, temperature=None):
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

            
            spin_temp = self.start_spin_config[idx] + dtheta
            empty_spins[idx] += dtheta
            spin_lattice_temp = self.start_spin_config + empty_spins
            
            energy_f = -sum(np.cos(spin_temp-self.start_spin_config[n]) for n in self.nbr[idx])
          

            defect_config_temp =  self.get_defect_config_full(spin_lattice_temp)

          
            delta_E = energy_f - energy_i
          

            if np.all(defect_config_temp == self.defect_config) and np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.start_spin_config[idx] += dtheta

    def sweep_fix_vortex_second(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
           
            empty_spins =np.zeros(self.start_spin_config.shape)
            empty_defects = np.zeros(self.defect_config.shape)
            energy_i = -sum(np.cos(self.start_spin_config[idx]-self.start_spin_config[n]) for n in self.nbr[idx])

            dtheta = np.random.uniform(-np.pi/4,np.pi/4)
            
            spin_temp = self.start_spin_config[idx] + dtheta
            empty_spins[idx] += dtheta
            spin_lattice_temp = self.start_spin_config + empty_spins
            
            energy_f = -sum(np.cos(spin_temp-self.start_spin_config[n]) for n in self.nbr[idx])
            
            L=16
            N=16*16
          
            defect_config_temp =  self.get_defect_config_full(spin_lattice_temp)
           
            delta_E = energy_f - energy_i
           
            if np.all(defect_config_temp == self.defect_config):

                if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                    self.start_spin_config[idx] += dtheta



    def sweep_vortex(self,input_defect_config):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        
        for idx in spin_idx:#one sweep in defined as N attempts of flip

            energy_i = 10*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2
            dtheta = np.random.uniform(-np.pi/16,np.pi/16)
            spin_temp = self.spin_config[idx] + dtheta
            spin_lattice_temp = self.spin_config
            spin_lattice_temp[idx] +=dtheta
            def_temp = self.get_defect_config(spin_lattice_temp,idx)
            energy_f = 10*(def_temp-input_defect_config[idx])**2
            delta_E = energy_f - energy_i
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta


    

    def saw(self,spin_angle_i,spin_angle_j):
        spin_angle_diff = (spin_angle_j-spin_angle_i)#%2*np.pi
        if np.abs(spin_angle_diff - two_pi) < np.abs(spin_angle_diff):
            spin_angle_diff -= two_pi
            
        elif np.abs(spin_angle_diff + two_pi) < np.abs(spin_angle_diff):
            spin_angle_diff += two_pi
            

        return spin_angle_diff


    def get_defect_config(self,spins,idx):

        plaquetes_angles = [spins[n] for n in self.pqts[idx]]
        

        plaquetes_angles_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles]

        plaquetes_angles_normed = [(plaquetes_angles_two_pi[i]-plaquetes_angles_two_pi[0]) for i in range(5)]

        plaquetes_angles_normed_two_pi = [setAngleInterval(spin) for spin in plaquetes_angles_normed]
    
        old_all_diff = 0
        new_all_diff = 0
        
        for n in range(4):
            new_all_diff = old_all_diff + self.saw(plaquetes_angles_normed_two_pi[n],plaquetes_angles_normed_two_pi[n+1])
            
            all_diff = new_all_diff
            old_all_diff = new_all_diff


        winding_number = int(all_diff/(2*np.pi))

        defect_config = winding_number

        return defect_config



    def get_defect_config_full(self,spins):

        number_spins = list(range(self.num_spins))
        full_defect_config = np.zeros(spins.shape)

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
        for idx,spin in enumerate(self.start_spin_config): #calculate energy per spin
            energy_[idx] = -1*sum(np.cos(spin-self.start_spin_config[n]) for n in self.nbr[idx])#nearst neighbor of kth spin

        return energy_



    def get_energy_vortex(self,input_defect_config):
        energy_=np.zeros(np.shape(self.spin_config))
        idx = 0
        for spin in self.spin_config: #calculate energy per spin
            energy_[idx] = (-1*sum(np.cos(spin-self.spin_config[n]) for n in self.nbr[idx])-100*(self.get_defect_config(self.spin_config,idx)-input_defect_config[idx])**2)/101
            idx +=1
        return energy_

    def get_energy_vortex_3dim(self,input_defect_config,K_1=5,K_2=10,K_3=5):
        energy_=np.zeros(np.shape(self.spin_config))
        idx = 0
        for spin in self.spin_config: #calculate energy per spin
            energy_[idx] = (-1*sum(np.cos(spin-self.spin_config[n]) for n in self.nbr[idx])+K_1*(self.zdim[idx]**2)+(K_2/2)*(self.get_defect_config(self.spin_config,idx)
            -input_defect_config[idx])**2
            +K_3*sum((1-self.zdim[pn]**2)for pn in self.pqts_nbr[idx]))
            idx +=1
        return energy_

    def set_helicity_modulus(self, direction=0, interaction_constant = 1):


        energy_per_link = [np.cos(spin_lattice-self.spin_config[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(self.spin_config)]

        current_per_link = [np.sin(spin_lattice-self.spin_config[self.nn_x_direction[idx]]) for idx, spin_lattice in enumerate(self.spin_config)]

        sum_energy_per_link = np.sum(energy_per_link)

        self.energy_per_link = sum_energy_per_link

        sum_current_per_link = ((np.sum(current_per_link))**2)

        self.current_per_link = sum_current_per_link

        return sum_energy_per_link, sum_current_per_link


    def set_magnetisation(self):
        mag_x = np.power(np.sum(np.cos(self.spin_config)),2)

        mag_y = np.power(np.sum(np.sin(self.spin_config)),2)

        magnetisation = (mag_x+mag_y)/self.num_spins#should be sqrtroot

        self.magenetisation = magnetisation

    def get_magnetisation(self):

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
            energy = np.sum(self.get_energy_vortex_3dim(input_defect_config))/self.num_spins
            
            dic_thermal_t['energy'] += [energy]
            
            if show  & (k%1e3 ==0):
                self.show()
            
            if ((self.get_energy_vortex_3dim(input_defect_config)<1e-4).all() & (k>700)) or k == max_nsweeps-1:
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
            
            energy = np.sum(self.get_energy_vortex(input_defect_config))/self.num_spins
            print('energy',energy)
            dic_thermal_t['energy'] += [energy]
            
            if show  & (k%1e3 ==0):
                
                self.show()
            
            if ((self.get_energy_vortex(input_defect_config)<1e-4).all() & (k>700)) or k == max_nsweeps-1:
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
       
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(300)):
            self.sweep_alternative()
        c = 0
        while c < 3000:
            self.sweep_alternative()
            c += 1


      
    def equilibrate_fix_vortex(self,max_nsweeps=int(2e5),temperature=None,H=None,show = True):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep_fix_vortex()
            
            energy = np.sum(self.get_energy_fix_vortex())/self.num_spins/2
            
            dic_thermal_t['energy'] += [energy]
            
            if show & (k%1e3 ==0) & k != 0:
                
                print('energy=%.2f'%energy)
                break
                
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
            
            energy = np.sum(self.get_energy_fix_vortex())/self.num_spins/2
           
            dic_thermal_t['energy'] += [energy]
            
            if show  & (k%2e3 ==0) & k != 0:
                break
            
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
        Q = plt.quiver(X, Y, U, V, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()

    def show_fix_vortex(self,colored=False):
        config_matrix = self.list2matrix(self.start_spin_config)
        X, Y = np.meshgrid(np.arange(0,self.width ),np.arange(self.width ,0,-1))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)

        plt.figure(figsize=(4,4), dpi=100)
        Q = plt.quiver(X, Y, U, V,config_matrix, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()

    def show_fix_vortex_change(self,spins,colored=False):
        config_matrix = self.list2matrix(spins)
        X, Y = np.meshgrid(np.arange(0,self.width ),np.arange(self.width ,0,-1))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)

        plt.figure(figsize=(4,4), dpi=100)
        Q = plt.quiver(X, Y, U, V,config_matrix, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()
