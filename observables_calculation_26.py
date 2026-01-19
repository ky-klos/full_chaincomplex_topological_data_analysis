import numpy as np
import random


#from skimage.io import imread
#import matplotlib.pyplot as plt
#%matplotlib inline ~ magic function backend for IPython : output of plotting
#commands displayed inline within frontends directly below code, that it produced

#from sklearn.model_selection import train_test_split



import torch


import h5py

#import pickle
import pandas as pd

#from matplotlib.collections import LineCollection

#import seaborn as sb

#import plotly.graph_objects as go

from observables_analysis_class_26 import Observables
from filtration_function import TopologicalAnalysis
from basis_data_production import prepare_data_basis,save_data_basis
##from helper_functions import transformation_pos
##from training_september_25_fixed_sum_fourier import CircularUpscaleConv2d, ResidualBlock, AdaIN, UnetGenerator, CircularPad2d, SumPool2d, InceptionConv2d_mini

#import time

import os

##from XY_model_slow_sep_23_numba import XYSystem


spin_configurations = []
defect_configurations_real = []
temperatures=[]
generator_epochs = []
vortex_numbers= []
lattice_size = 16
#temp = 0.1
gen_temp = 0
#magnetisations = []
mag_suscept = []
#energys = []
specific_heat_list = []
helicity_modulus_list = []
correlation_function = []
temperature = []
#current_per_link_list = []
#energy_per_link_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)


def return_defect_distances(maximal_distance):
    distances = []
    for i in range(maximal_distance):
        for j in range(maximal_distance):
            dist = torch.sqrt(i**2+j**2)
            distances.append(dist)

    return distances

##compare_defect_lattice means here that we take the identical defect_positions for different temperatures
def calculation_observables(data_type,output_data_path,generator_info=None,gen_folder= None,sim_data_type='full_pinned',larger=False,defect_nmb=2,compare_defect_lattice=False,shifte_to_side_test=False,c_depth=0,noise_size = 1.0, training_data_nmb=350000,nmb_label = 1,maximal_distance= 5,new_data=False,data_path='/localscratch/kyklos/Pytorch_cuda_code/iona/iona_oct_23/Full_training_oct_23/', epochs=[0,5,10,15,20,25,30], generated_data_atributes=[64,0.01,0,5,1,0],simulation_data_path  ='/localscratch/kyklos/Pytorch_cuda_code/iona/iona_oct_23/Full_training_oct_23/full_training_data_gan_low_temp_2_corrected.h5', temperature = 0.1, lattice_size= 16,further_simulated=False,samplesize=1000,rotation=False,later_rotation=True):
    with torch.no_grad():
        oservable_data_spins, oservable_data_defects, path_name, name_list = prepare_data_basis(temperature,lattice_size,device,data_type,simulation_data_path,sim_data_type,new_data,compare_defect_lattice,samplesize,defect_nmb,larger,further_simulated,data_path,generated_data_atributes,training_data_nmb,noise_size,nmb_label,epochs,generator_info,gen_folder)
        nmb_distances = int(len(name_list))
        max_distances = max(name_list)
        
        correspodning_data_outname = 'more_data_basis_data_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_batch_gen.h5'.format(data_type,sim_data_type,nmb_distances,max_distances, defect_nmb,samplesize,noise_size,1)

        os.makedirs(output_data_path+ path_name, exist_ok=True)
        save_data_basis(output_data_path=output_data_path,path_name=path_name,correspodning_data_outname=correspodning_data_outname,data_type=data_type,name_list=name_list,epochs=epochs,samplesize=sample_size,observable_data_spins=oservable_data_spins,observable_data_defects=oservable_data_defects,lattice_size=lattice_size,new_data=True)



        output_vorticity_list_full = []
        output_defect_lattice_list_full = []
        output_defect_diff_list_full = []
        output_magentisation_list_full = []
        output_energy_list_full = []
        output_magnet_suscep_list_full = []
        output_magnet_suscep_alternative_list_full = []
        output_specific_heat_list_full = []
        output_helicity_modulus_list_full = []
        output_magnet_suscep_error_list_full = []
        output_magnet_suscep_alternative_error_list_full = []
        output_specific_heat_error_list_full = []
        output_helicity_modulus_error_list_full =[]
        output_vorticity_error_list_full = []
        output_defect_diff_error_list_full = []
        output_magentisation_error_list_full = []
        output_energy_error_list_full = []
        output_corr_mean_list_full = []
        output_corr_var_list_full = []


        output_vorticity_full_list = []
        output_magentisation_full_list = []
        output_energy_full_list = []
        output_defect_diff_full_list = []
        output_correlationfunction_full_list = []

        output_energy_distribution_local_full = []
        
        #name_list = [4.0]


        for idx,types in enumerate(name_list):
            
            ##print('check_defect_nmb',defect_nmb,temperature, torch.all(torch.sum(oservable_data_defects[idx][0].view(-1,lattice_size*lattice_size),dim=1)==0.))

            #print('check',oservable_data_spins[idx][0].size(0))





            ##print(types,oservable_data_spins[idx][0].size(0))

            if int(oservable_data_spins[idx][0].size(0)) == 10000:
                    sample_nmb = 45
                    sample_size = 200
            elif int(oservable_data_spins[idx][0].size(0)) == 1000:
                    sample_nmb = 9
                    sample_size = 100
            elif int(oservable_data_spins[idx][0].size(0)) == 900:
                    sample_nmb = 8
                    sample_size = 100
            elif int(oservable_data_spins[idx][0].size(0)) == 5000:
                    sample_nmb = 35
                    sample_size = 140
            elif int(oservable_data_spins[idx][0].size(0)) == 50000:
                    sample_nmb = 98
                    sample_size = 500
            elif int(oservable_data_spins[idx][0].size(0)) == 20000:
                    sample_nmb = 54
                    sample_size = 350
            elif int(oservable_data_spins[idx][0].size(0)) == 30000:
                    sample_nmb = 64
                    sample_size = 450

            defect_number_tensor = torch.sum(torch.abs(oservable_data_defects[idx][0]).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)
            unique_defect_lattices = torch.unique(oservable_data_defects[idx][0], dim=0)
  
            unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)
            simulation_observables = Observables(lattice_size=lattice_size, temperature = temperature)
            
            print(maximal_defect_nmb)
            simulation_observables.set_full_defect_positions(defect_lattices=oservable_data_defects[idx][0],maximal_defect_nmb=maximal_defect_nmb)
            



            output_vorticity_list = []
            output_defect_lattice_list = []
            output_defect_diff_list = []
            output_magentisation_list = []
            output_energy_list = []
            output_magnet_suscep_list = []
            output_magnet_suscep_alternative_list = []
            output_specific_heat_list = []
            output_helicity_modulus_list = []
            output_magnet_suscep_error_list = []
            output_magnet_suscep_alternative_error_list = []
            output_specific_heat_error_list = []
            output_helicity_modulus_error_list =[]
            output_vorticity_error_list = []
            output_defect_diff_error_list = []
            output_magentisation_error_list = []
            output_energy_error_list = []
            output_corr_mean_list = []
            output_corr_var_list = []

            output_vorticity_full_list_per_name = []
            output_magentisation_full_list_per_name = []
            output_energy_full_list_per_name = []
            output_defect_diff_full_list_per_name = []
            output_correlationfunction_full_list_per_name = []

            output_energy_distribution_local_per_name = []




            for e_idx,epoch in enumerate(epochs):
                print(e_idx)

                defect_lattices, vorticity, vorticity_error, full_vorticity = simulation_observables.faster_vorticity(spin_lattice_tensor=oservable_data_spins[idx][e_idx])

                #print('check v')
                magentisation, magentisation_error, full_magentisation = simulation_observables.faster_mean_magentisation(oservable_data_spins[idx][e_idx])

                #print('check m')
                magentic_suscep = simulation_observables.faster_magnet_suscep()
                magentic_suscep_alternative = magentic_suscep#simulation_observables.check_sucept_corr_function(oservable_data_spins[idx][e_idx])

                energy, energy_error, full_energy = simulation_observables.faster_mean_energy(oservable_data_spins[idx][e_idx])

                energy_distribution_local = simulation_observables.faster_mean_local_energy(oservable_data_spins[idx][e_idx])

                output_energy_distribution_local_per_name.append(energy_distribution_local)

                #print('check e')
                specific_heat = simulation_observables.faster_specfifc_heat()
            
                simulation_observables.set_defects_lattices(sim_defect_lattices=oservable_data_defects[idx][e_idx], gen_defect_lattices=defect_lattices)
                defect_diff, defect_diff_error, full_defect_diff = simulation_observables.defect_distance()
                
             
            
                helicity_modulus = simulation_observables.faster_get_helicity_modulus(oservable_data_spins[idx][e_idx])

                observables_tensor = [specific_heat,magentic_suscep,magentic_suscep_alternative,helicity_modulus]

                sh_error,ms_error,ms_alternative_error,hm_error = simulation_observables.faster_error_calulcation(full_observable_tensor_list=observables_tensor,full_spin_tensor=oservable_data_spins[idx][e_idx], sample_nmb=sample_nmb,sample_size=sample_size)

           

                data_correlation_function = simulation_observables.correlation_function_even_faster_new_try(spin_fields=oservable_data_spins[idx][e_idx])

                

                
                
                distance_parameter = [float(key) for key in data_correlation_function.keys()]

                distance_parameter.sort() 

                mean_correlation_data_sorted = [torch.flatten(torch.mean(data_correlation_function['{}'.format(k)])) for k in distance_parameter]

                full_correlation_data_sorted = [data_correlation_function['{}'.format(k)] for k in distance_parameter]

                var_correlation_data_sorted = [torch.flatten(torch.mean(torch.pow(data_correlation_function['{}'.format(k)],2.0))-torch.pow(torch.mean(data_correlation_function['{}'.format(k)]),2.0)) for k in distance_parameter]

                output_vorticity_list.append(vorticity)
                output_defect_lattice_list.append(defect_lattices)
                output_defect_diff_list.append(defect_diff)
                output_magentisation_list.append(magentisation)
                output_energy_list.append(energy)
                output_magnet_suscep_list.append(magentic_suscep)
                output_magnet_suscep_alternative_list.append(magentic_suscep_alternative)
                output_specific_heat_list.append(specific_heat)
                output_helicity_modulus_list.append(helicity_modulus)
                output_magnet_suscep_error_list.append(ms_error)
                output_magnet_suscep_alternative_error_list.append(ms_alternative_error)
                output_specific_heat_error_list.append(sh_error)
                output_helicity_modulus_error_list.append(hm_error)
                output_vorticity_error_list.append(vorticity_error)
                output_defect_diff_error_list.append(defect_diff_error)
                output_magentisation_error_list.append(magentisation_error)
                output_energy_error_list.append(energy_error)
                output_corr_mean_list.append(torch.stack(mean_correlation_data_sorted, dim=0))
                output_corr_var_list.append(torch.stack(var_correlation_data_sorted, dim =0))



                output_vorticity_full_list_per_name.append(full_vorticity)
                output_magentisation_full_list_per_name.append(full_magentisation)
                output_energy_full_list_per_name.append(full_energy)
                output_defect_diff_full_list_per_name.append(full_defect_diff)
                output_correlationfunction_full_list_per_name.append(torch.stack(full_correlation_data_sorted, dim=0))
            #print(output_vorticity_list)
            output_vorticity_list_full.append(torch.stack(output_vorticity_list, dim =0))
            output_defect_lattice_list_full.append(torch.stack(output_defect_lattice_list, dim =0))
            output_defect_diff_list_full.append(torch.stack(output_defect_diff_list, dim =0))
            output_magentisation_list_full.append(torch.stack(output_magentisation_list, dim =0))
            output_energy_list_full.append(torch.stack(output_energy_list, dim =0))
            output_magnet_suscep_list_full.append(torch.stack(output_magnet_suscep_list, dim =0))
            output_magnet_suscep_alternative_list_full.append(torch.stack(output_magnet_suscep_alternative_list, dim =0))
            output_specific_heat_list_full.append(torch.stack(output_specific_heat_list, dim =0))
            output_helicity_modulus_list_full.append(torch.stack(output_helicity_modulus_list, dim =0))
            output_magnet_suscep_error_list_full.append(torch.stack(output_magnet_suscep_error_list, dim =0))
            output_magnet_suscep_alternative_error_list_full.append(torch.stack(output_magnet_suscep_alternative_error_list, dim =0))
            output_specific_heat_error_list_full.append(torch.stack(output_specific_heat_error_list, dim =0))
            output_helicity_modulus_error_list_full.append(torch.stack(output_helicity_modulus_error_list, dim =0))
            output_vorticity_error_list_full.append(torch.stack(output_vorticity_error_list, dim =0))
            output_defect_diff_error_list_full.append(torch.stack(output_defect_diff_error_list, dim =0))
            output_magentisation_error_list_full.append(torch.stack(output_magentisation_error_list, dim =0))
            output_energy_error_list_full.append(torch.stack(output_energy_error_list, dim =0))
            output_corr_mean_list_full.append(torch.stack(output_corr_mean_list, dim =0))
            output_corr_var_list_full.append(torch.stack(output_corr_var_list, dim =0))

            output_vorticity_full_list.append(torch.stack(output_vorticity_full_list_per_name, dim =0))
            output_magentisation_full_list.append(torch.stack(output_magentisation_full_list_per_name, dim =0))
            output_energy_full_list.append(torch.stack(output_energy_full_list_per_name, dim =0))
            output_defect_diff_full_list.append(torch.stack(output_defect_diff_full_list_per_name, dim =0))
            output_correlationfunction_full_list.append(torch.stack(output_correlationfunction_full_list_per_name, dim =0))

            output_energy_distribution_local_full.append(torch.stack(output_energy_distribution_local_per_name,dim=0))




        outname = 'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_batch_gen.csv'.format(data_type,sim_data_type,nmb_distances,max_distances, defect_nmb,samplesize,noise_size,1)

        outname_full = 'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_full_not_mean_compare_{}_batch_gen.h5'.format(data_type,sim_data_type,nmb_distances,max_distances, defect_nmb,samplesize,noise_size,1)



        data_observables = {'Epoch':np.array(epochs).flatten()}

        magnetisation_mean = {}
        magnetisation_error = {}

        vorticity_mean = {}
        vorticity_error = {}

        defect_diff_mean = {}
        defect_diff_error = {}

        energy_mean = {}
        energy_error = {}

        ms_mean = {}
        ms_alternative_mean = {}
        ms_error = {}
        ms_alternative_error = {}

        hm_mean = {}
        hm_error = {}

        sh_mean = {}
        sh_error = {}

        correlation_function_mean = {}
        correlation_function_var = {}
        
        print(torch.stack(output_magentisation_list_full, dim=0).shape)

        for i,name in enumerate(name_list):

            magnetisation_mean.update({'Mag_mean_{}'.format(name):torch.flatten(torch.stack(output_magentisation_list_full, dim=0)[i]).cpu().numpy()})
            magnetisation_error.update({'Mag_error_{}'.format(name):torch.flatten(torch.stack(output_magentisation_error_list_full, dim=0)[i]).cpu().numpy()})
            vorticity_mean.update({'Vor_mean_{}'.format(name):torch.flatten(torch.stack(output_vorticity_list_full, dim=0)[i]).cpu().numpy()})
            vorticity_error.update({'Vor_error_{}'.format(name):torch.flatten(torch.stack(output_vorticity_error_list_full, dim=0)[i]).cpu().numpy()})
            defect_diff_mean.update({'Defect_Diff_mean_{}'.format(name):torch.flatten(torch.stack(output_defect_diff_list_full, dim=0)[i]).cpu().numpy()})
            defect_diff_error.update({'Defect_Diff_error_{}'.format(name):torch.flatten(torch.stack(output_defect_diff_error_list_full, dim=0)[i]).cpu().numpy()})
            energy_mean.update({'Energy_mean_{}'.format(name):torch.flatten(torch.stack(output_energy_list_full, dim=0)[i]).cpu().numpy()})
            energy_error.update({'Energy_error_{}'.format(name):torch.flatten(torch.stack(output_energy_error_list_full, dim=0)[i]).cpu().numpy()})
            ms_mean.update({'MS_mean_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_list_full, dim=0)[i]).cpu().numpy()})
            ms_alternative_mean.update({'MS_alternative_mean_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_alternative_list_full, dim=0)[i]).cpu().numpy()})
            ms_error.update({'MS_error_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_error_list_full, dim=0)[i]).cpu().numpy()})
            ms_alternative_error.update({'MS_alternative_error_{}'.format(name):torch.flatten(torch.stack(output_magnet_suscep_alternative_error_list_full, dim=0)[i]).cpu().numpy()})
            hm_mean.update({'HM_mean_{}'.format(name):torch.flatten(torch.stack(output_helicity_modulus_list_full, dim=0)[i]).cpu().numpy()})
            hm_error.update({'HM_error_{}'.format(name):torch.flatten(torch.stack(output_helicity_modulus_error_list_full, dim=0)[i]).cpu().numpy()})
            sh_mean.update({'SH_mean_{}'.format(name):torch.flatten(torch.stack(output_specific_heat_list_full, dim=0)[i]).cpu().numpy()})
            sh_error.update({'SH_error_{}'.format(name):torch.flatten(torch.stack(output_specific_heat_error_list_full, dim=0)[i]).cpu().numpy()})
            correlation_function_mean.update({'Corr_mean_{}_{}'.format(dist,name):torch.flatten(torch.stack(output_corr_mean_list_full, dim=0)[i,:,k]).cpu().numpy() for k,dist in enumerate(distance_parameter)})
            correlation_function_var.update({'Corr_var_{}_{}'.format(dist,name):torch.flatten(torch.stack(output_corr_var_list_full, dim=0)[i,:,k]).cpu().numpy() for k,dist in enumerate(distance_parameter)})

        data_observables.update(magnetisation_mean)
        data_observables.update(magnetisation_error)
        data_observables.update(vorticity_mean)
        data_observables.update(vorticity_error)
        data_observables.update(defect_diff_mean)
        data_observables.update(defect_diff_error)
        data_observables.update(energy_mean)
        data_observables.update(energy_error)
        data_observables.update(ms_mean)
        data_observables.update(ms_error)
        data_observables.update(ms_alternative_mean)
        data_observables.update(ms_alternative_error)
        data_observables.update(hm_mean)
        data_observables.update(hm_error)
        data_observables.update(sh_mean)
        data_observables.update(sh_error)
        data_observables.update(correlation_function_mean)
        data_observables.update(correlation_function_var)

        data_observables_data_basis = {}


        



        df_type_observables = pd.DataFrame.from_dict(data_observables)
        
        df_type_observables.to_csv(output_data_path+path_name +outname, index= False)

        #print('distance',name_list)


        with h5py.File(output_data_path+path_name +outname_full,'w') as h5f:

                    

            for dist_idx,dist in enumerate(name_list):

                print(dist)

                    #print(oservable_data_spins[dist_idx])

                dataset_vorticity_compare = h5f.create_dataset('vorticity_{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_magent_compare = h5f.create_dataset('magentisation{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_energy_compare = h5f.create_dataset('energy_{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_defect_diff_compare = h5f.create_dataset('defect_diff_{}'.format(dist), shape=(int(len(epochs)),samplesize,1), dtype='float')
                dataset_corr_compare = h5f.create_dataset('corr_funct_{}'.format(dist), shape=(int(len(epochs)),samplesize,int(len(distance_parameter))), dtype='float')

                dataset_energy_local = h5f.create_dataset('local_energy_{}'.format(dist), shape=(int(len(epochs)),samplesize,lattice_size*lattice_size), dtype='float')

                for e_dix,epoch in enumerate(epochs):

                        #print(output_correlationfunction_full_list[dist_idx][e_dix].size())


                    dataset_energy_local[e_dix] = output_energy_distribution_local_full[dist_idx][e_dix].view(1,-1,lattice_size*lattice_size).cpu().numpy()
                    dataset_vorticity_compare[e_dix] = output_vorticity_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                    dataset_magent_compare[e_dix] = output_magentisation_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                    dataset_energy_compare[e_dix] = output_energy_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                    dataset_defect_diff_compare[e_dix] = output_defect_diff_full_list[dist_idx][e_dix].view(1,-1,1).cpu().numpy()
                        #print('check')
                    dataset_corr_compare[e_dix] = output_correlationfunction_full_list[dist_idx][e_dix].view(1,-1,int(len(distance_parameter))).cpu().numpy()
                        



if __name__ == "__main__":
    pass
