import numpy as np
import torch
import torch.nn.functional as F
import random
from torch import nn

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)


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
 
    def defect_position_tensor(self,maximal_defect_number,input_defect_lattice= None):


        counter = 0

        if input_defect_lattice == None:
            defect_lattice = self.defect_config
        else:
            defect_lattice = input_defect_lattice

        defect_position_vector = torch.zeros(int(maximal_defect_number),3,device = device)
        
        defect_distance_vector = torch.zeros(int(maximal_defect_number/2),1,device = device)


        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if defect_lattice[i,j] != 0:
                    defect_position_vector[counter,0] += i
                    defect_position_vector[counter,1] += j
                    defect_position_vector[counter,2] += torch.clone(defect_lattice[i,j])
                    counter +=1
        
        if counter == 0:
            
            defect_distance = torch.tensor(0).to(device)
            defect_distance_vector[:counter] = defect_distance.view(-1,1)
            
        elif counter == 2:
            defect_distance = torch.sqrt(torch.pow((defect_position_vector[0,0]-defect_position_vector[1,0]),2) +torch.pow((defect_position_vector[0,1]-defect_position_vector[1,1]),2)).view(-1,1)
            defect_distance_vector[:counter] = defect_distance
            
        else:
            vortices = defect_position_vector[defect_position_vector[:,2]== 1]
            anti_vortices = defect_position_vector[defect_position_vector[:,2]== -1]
            defect_distance = torch.sqrt(torch.pow((vortices[:,0]-anti_vortices[:,0]),2) +torch.pow((vortices[:,1]-anti_vortices[:,1]),2)).view(-1,1)
            
            counter_idx = (counter//2)
            
            defect_distance_vector[:counter_idx] = defect_distance
            
                    
        

        return defect_position_vector, defect_distance_vector

    def make_energy_plaquettes(self,spinangle_tensor):

        spins_halo= F.pad(spinangle_tensor,(1,1,1,1),'circular')

        nn_maker = nn.Unfold(kernel_size=(3,3),stride=1)

        nn_temp = nn_maker(spins_halo).view(spinangle_tensor.size(0),9,self.lattice_size*self.lattice_size)

        next_n = torch.transpose(nn_temp, 1,2)

        index = torch.tensor([4,1,3,5,7],device=device)

        all_nn = torch.index_select(next_n,dim=2, index=index)

        return all_nn
    
    def faster_mean_energy(self,spin_lattice_tensor):
        
        spin_lattice_tensor = spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size)
        sample_nmb =int(spin_lattice_tensor.size(0))
        nn_tensor = self.make_energy_plaquettes(spinangle_tensor=spin_lattice_tensor)##should have size: batchsize, latticesize**2, 5
        

        energy_sum_tensor = -(self.interaction_const/2.0)*torch.sum((torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,1])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,2])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,3])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,4])),dim=1)

        energy_tensor_mean = torch.mean(energy_sum_tensor)
        energy_tensor_mean_2 = torch.mean(torch.pow(energy_sum_tensor,2))


        self.energy_tensor_mean = energy_tensor_mean
        self.energy_tensor_mean_2 = energy_tensor_mean_2

        

        energy_tensor_variance = torch.sum(torch.pow((energy_sum_tensor-energy_tensor_mean),2))/(sample_nmb-1)



        return energy_tensor_mean,energy_tensor_variance,energy_sum_tensor

    def faster_mean_local_energy(self,spin_lattice_tensor):

        spin_lattice_tensor = spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size)
        
        nn_tensor = self.make_energy_plaquettes(spinangle_tensor=spin_lattice_tensor)##should have size: batchsize, latticesize**2, 

        energy_distribution_tensor = -(self.interaction_const)*(torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,1])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,2])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,3])+torch.cos(nn_tensor[:,:,0]-nn_tensor[:,:,4]))

        return energy_distribution_tensor.view(-1,self.lattice_size*self.lattice_size)
    
    def faster_specfifc_heat(self):
            
        specfifc_heat = (self.energy_tensor_mean_2 -torch.pow(self.energy_tensor_mean,2))/(self.temperature**2)

        return specfifc_heat
    
    def faster_mean_magnetisation(self,spin_lattice_tensor):

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
        self.magent_tensor_mean_full = torch.mean(torch.sqrt(torch.pow(spin_lattice_tensor_cos,2)+torch.pow(spin_lattice_tensor_sin,2)))

        magnetisation_tensor_variance = torch.sum(torch.pow((magnetisation_tensor-magnetisation_tensor_mean),2))/(sample_nmb-1)

        return magnetisation_tensor_mean,magnetisation_tensor_variance, magnetisation_tensor
    
    def faster_magnet_suscep(self):

        #need fastermagnetisation beforehand
        ##this is for absolute magnet but should be magentic vector
        magnet_suscep = (self.magent_tensor_mean_full_2-torch.pow(self.magent_tensor_mean_full,2))/self.temperature
      
        return magnet_suscep
    
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

        index_y = torch.tensor([0,2],device=device)

        all_xn = torch.index_select(next_n,dim=2, index=index)

        all_yn = torch.index_select(next_n,dim=2, index=index_y)


        energy_per_link = torch.cos(all_xn[:,:,0]-all_xn[:,:,1]) #possible full not just one direction ?

        energy_y_per_link = torch.cos(all_yn[:,:,0]-all_yn[:,:,1])

        current_per_link = torch.sin(all_xn[:,:,0]-all_xn[:,:,1]) 

        sum_energy_per_link = torch.sum(energy_per_link+energy_y_per_link,dim=1)

        sum_current_per_link = torch.pow(torch.sum(current_per_link,dim=1),2)

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
    
    def faster_error_calculation(self,full_observable_tensor_list,full_spin_tensor, sample_nmb,sample_size): #error for [SH, MS, HM]

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
            current_magnet, current_magnet_var, magnet_full = self.faster_mean_magnetisation(spin_lattice_tensor=current_spins)
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

        spin_lattice_halo = F.pad(spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        plaquette_temp = self.plaquette_maker(spin_lattice_halo).view(spin_lattice_tensor.size(0),4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquette_temp, 1, 2)

        all_plaquettes =  torch.index_select(plaquettes,dim=2, index=self.plaq_index)

        changed_plaquette_angles = torch.sub(all_plaquettes[:,:,:],all_plaquettes[:,:,0].view(-1,self.lattice_size*self.lattice_size,1))

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
    
    def set_full_defect_positions(self,defect_lattices,maximal_defect_nmb=2):
        full_defect_position_list = []
        defect_distance_list = []
        

        for defect_lattice in defect_lattices:
            defect_position,defect_distance = self.defect_position_tensor(maximal_defect_number=maximal_defect_nmb,input_defect_lattice = defect_lattice)
            full_defect_position_list.append(defect_position)
            defect_distance_list.append(defect_distance)

        self.full_defect_positions = torch.stack(full_defect_position_list,dim=0)
        

        self.full_defect_distance = torch.stack(defect_distance_list, dim =0)

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
        
        correlation_function_per_distance.pop('{0}'.format(0.0))
        for idx,spin in enumerate(spin_field):
            correlation_function_per_distance_per_spin = {'{0}'.format(np.sqrt(d_x**2+d_y**2)):[] for d_x in np.arange(0,maximum_distance_x_y+1,1) for d_y in np.arange(0,maximum_distance_x_y+1,1)}
            
            
            for distances_x in possible_spin_distances:
                
                for distances_y in possible_spin_distances:

                    full_distance = np.sqrt((distances_x+1)**2+(distances_y+1)**2)

                
                    correlation_function_per_distance_per_pos = []

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


        return correlation_mean_per_temp
    
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


        for x_idx in range(self.lattice_size):#self.lattice_size*self.lattice_size

            full_corr_distances_xrot = torch.roll(full_corr_distances,x_idx,dims=1)


            for y_idx in range(self.lattice_size):



                correlation_lattices = torch.cos((torch.repeat_interleave(spin_fields[:,y_idx,x_idx].view(-1,1),self.lattice_size*self.lattice_size,dim=1).view(-1,self.lattice_size,self.lattice_size)-spin_fields[:,:,:])).view(-1,self.lattice_size,self.lattice_size)

                full_corr_distances_fullrot = torch.roll(full_corr_distances_xrot,y_idx,dims=0)



                for distance in torch.unique(corr_distances.flatten()):

                    distance_dep_corr = torch.masked_select(correlation_lattices,(full_corr_distances_fullrot == distance).view(-1,self.lattice_size,self.lattice_size)).view(correlation_lattices.size(0),-1)

                    correlation_function_per_distance['{0}'.format(distance)].append(torch.mean(distance_dep_corr,dim=1))

     

        correlation_function_dict_mean_per_distance = { distance_key: torch.flatten(torch.mean(torch.stack(correlation_function_per_distance[distance_key],dim=1),dim=1)) for distance_key in list(correlation_function_per_distance.keys())}


        return correlation_function_dict_mean_per_distance
