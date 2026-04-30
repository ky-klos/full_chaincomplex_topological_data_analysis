import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from numpy import pi
import matplotlib.colors

import torch

import h5py

import pickle
import pandas as pd

from observables_analysis_class_26 import Observables


from helper_functions import acc,heatmap_error,faster_error_calulcation_binder




def loading_observables(input_observables,fixed_data,simulation_data_path,output_data_path,temperatures,data_type_list,generator_att,input_att,maximal_distance_correlation,samplesize,defect_nmbs,lattice_size,sim_data_type='full_pinned',input_save_path=None):
    
    sim_save_end = 'batch_gen'
    gen_save_end = 'batch_gen'

    output_path = './' #path of the model observables output
    noise_size = input_att['noise_size']

    training_data_nmb = input_att['training_data_nmb']


    dataset_list_energy = []
    dataset_list_magnetisation = []


    plot_observables_distribution_dictionary = {}
    plot_observables_mean_dictionary = {}
    plot_observables_var_dictionary = {}
    for d_idx,data_name in enumerate(data_type_list):

        plot_observables_distribution_dictionary[data_name] = {}
        plot_observables_mean_dictionary[data_name] = {}
        plot_observables_var_dictionary[data_name] = {}
        
        distance ={torch.sqrt(d_x**2+d_y**2):[] for d_x in torch.arange(0,maximal_distance_correlation+1,1) for d_y in torch.arange(0,maximal_distance_correlation+1,1)}
        distance_parameter = [float(key) for key in distance.keys()]
        distance_parameter.sort()
        
        plot_observables_distribution_dictionary[data_name]['Temperatures'] = temperatures
        plot_observables_distribution_dictionary[data_name]['defect_nmbs'] = defect_nmbs
        plot_observables_distribution_dictionary[data_name]['Labels'] = {}
        
        
        
        plot_observables_distribution_dictionary[data_name]['local_Energy'] = {}
        plot_observables_distribution_dictionary[data_name]['reduced_specific_heat'] = {}
        plot_observables_distribution_dictionary[data_name]['Epochs'] = {}
        plot_observables_distribution_dictionary[data_name]['name_list'] = {}
        
        for o in input_observables:
            plot_observables_distribution_dictionary[data_name][o] = {}
            plot_observables_mean_dictionary[data_name][o] = {}
            plot_observables_var_dictionary[data_name][o] = {}
            
        
        for t_idx,t in enumerate(temperatures):
            


            plot_observables_distribution_dictionary[data_name]['Labels'][t] = {}
            plot_observables_distribution_dictionary[data_name]['local_Energy'][t] = {}
            plot_observables_distribution_dictionary[data_name]['reduced_specific_heat'][t] = {}
            plot_observables_distribution_dictionary[data_name]['Epochs'][t] = {}
            plot_observables_distribution_dictionary[data_name]['name_list'][t] = {}
            for o in input_observables:
                plot_observables_distribution_dictionary[data_name][o][t] = {}
                plot_observables_mean_dictionary[data_name][o][t] = {}
                plot_observables_var_dictionary[data_name][o][t] = {}
                
            
            
            for d in defect_nmbs:
                
                plot_observables_distribution_dictionary[data_name]['Labels'][t][d] = []
                


                if fixed_data:
                    if sim_data_type=='full_pinned' or sim_data_type=='new_full_pinned':

                        with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(t,lattice_size,d,samplesize), 'r') as h5f:

                            name_list = np.array(h5f['distances'])
                    else:
                         with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(sim_data_type,t,lattice_size,d,samplesize), 'r') as h5f:

                            name_list = np.array(h5f['distances'])                       
                            
                    number_distances = int(len(name_list))
    
                    maximal_distance = np.max(np.array(name_list))
                    
                if input_save_path is None:
                    save_path = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/res_critic/temp_{}/'.format(noise_size,training_data_nmb,*generator_att,temperatures[0]) 
                    save_path = output_path+'temp_{}/'.format(temperatures[0])
                else:
                    save_path = input_save_path


    
                if data_name == 'training_data':
                    
                    if sim_data_type=='full_pinned':
                        plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Training Data')
                        train_path = 'training_data_full_distance_temp_{}_new/'.format(t)
                        observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.csv'.format(data_name,number_distances,maximal_distance,d,samplesize,noise_size,1), delimiter=',', skipinitialspace= True, header = 0)
                    elif sim_data_type=='new_full_pinned':
                        plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Training Data')
                        train_path = 'training_data_full_distance_temp_{}_new_{}/'.format(t,sim_data_type)
                        observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_full_pinned_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_batch_gen.csv'.format(data_name,number_distances,maximal_distance,d,samplesize,noise_size,1), delimiter=',', skipinitialspace= True, header = 0)                    
                    
                    else:    
                        plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Simulated Data')
                        train_path = 'training_data_full_distance_temp_{}_new_{}/'.format(t,sim_data_type)
                        observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_{}.csv'.format(data_name,'full_pinned',number_distances,maximal_distance,d,samplesize,1.0,1,sim_save_end), delimiter=',', skipinitialspace= True, header = 0)
                if data_name == 'generated_data':

                    plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Generated Data')
                    train_path = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/res_critic/temp_{}/'.format(noise_size,training_data_nmb,*generator_att,t) 
                    #train_path = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/'.format(noise_size,training_data_nmb,*generated_data_atributes) 
                    train_path = output_path+'temp_{}/'.format(t)
                    
                    observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_{}.csv'.format(data_name,'full_pinned',number_distances,maximal_distance,d,samplesize,noise_size,1,gen_save_end), delimiter=',', skipinitialspace= True, header = 0)
                if data_name == 'random_data':

                    plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Random Data')
                    train_path = 'random_data_comparison/'
                    observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_nmb_distances{}_max_distance_{}_samplesize{}_noisesize_{}.csv'.format(data_name,number_distances,maximal_distance,samplesize,noise_size), delimiter=',', skipinitialspace= True, header = 0)
                if data_name == 'real_data':

                    plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Real Data')
                    train_path = 'training_data_full_distance_temp_{}_new_real_data/'.format(t)
                    observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_nmb_distances{}_max_distance_{}_samplesize{}_noisesize_{}.csv'.format(data_name,number_distances,maximal_distance,samplesize,noise_size), delimiter=',', skipinitialspace= True, header = 0)
                if data_name == 'zero_temp':

                    plot_observables_distribution_dictionary[data_name]['Labels'][t][d].append('Analytic Solution')
                    train_path = 'zero_temperature_solution_comparison_new/'
                    observables = pd.read_csv(output_data_path+train_path+'more_data_observables_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}.csv'.format(data_name,number_distances,maximal_distance,d,samplesize,noise_size,1), delimiter=',', skipinitialspace= True, header = 0)
            
                if data_name == 'training_data':
                    if sim_data_type == 'full_pinned':
                        with h5py.File(output_data_path+train_path+'more_data_observables_output_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_full_not_mean_compare_{}.h5'.format(data_name,number_distances,maximal_distance,d,samplesize,noise_size,1), 'r') as h5f:

                            plot_observables_distribution_dictionary[data_name]['Magnetisation'][t][d] = np.array([np.array(h5f['magentisation{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Energy'][t][d] = np.array([np.array(h5f['energy_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Vorticity'][t][d] = np.array([np.array(h5f['vorticity_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Defect_Diff'][t][d] = np.array([np.array(h5f['defect_diff_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['local_Energy'][t][d] = np.array([np.array(h5f['local_energy_{}'.format(name)]) for name in name_list])
                    elif sim_data_type == 'new_full_pinned':
                        with h5py.File(output_data_path+train_path+'more_data_observables_output_{}_full_pinned_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_full_not_mean_compare_{}_batch_gen.h5'.format(data_name,number_distances,maximal_distance,d,samplesize,noise_size,1), 'r') as h5f:

                            plot_observables_distribution_dictionary[data_name]['Magnetisation'][t][d] = np.array([np.array(h5f['magentisation{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Energy'][t][d] = np.array([np.array(h5f['energy_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Vorticity'][t][d] = np.array([np.array(h5f['vorticity_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Defect_Diff'][t][d] = np.array([np.array(h5f['defect_diff_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['local_Energy'][t][d] = np.array([np.array(h5f['local_energy_{}'.format(name)]) for name in name_list])                    
                    
                    else:  
                        with h5py.File(output_data_path+train_path+'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_full_not_mean_compare_{}_{}.h5'.format(data_name,'full_pinned',number_distances,maximal_distance,d,samplesize,1.0,1,sim_save_end), 'r') as h5f:

                            plot_observables_distribution_dictionary[data_name]['Magnetisation'][t][d] = np.array([np.array(h5f['magentisation{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Energy'][t][d] = np.array([np.array(h5f['energy_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Vorticity'][t][d] = np.array([np.array(h5f['vorticity_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['Defect_Diff'][t][d] = np.array([np.array(h5f['defect_diff_{}'.format(name)]) for name in name_list])
                            plot_observables_distribution_dictionary[data_name]['local_Energy'][t][d] = np.array([np.array(h5f['local_energy_{}'.format(name)]) for name in name_list])
                        
                elif data_name == 'generated_data':
                    with h5py.File(output_data_path+train_path+'more_data_observables_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_full_not_mean_compare_{}_{}.h5'.format(data_name,'full_pinned',number_distances,maximal_distance,d,samplesize,noise_size,1,gen_save_end), 'r') as h5f:

                        plot_observables_distribution_dictionary[data_name]['Magnetisation'][t][d] = np.array([np.array(h5f['magentisation{}'.format(name)]) for name in name_list])
                        plot_observables_distribution_dictionary[data_name]['Energy'][t][d] = np.array([np.array(h5f['energy_{}'.format(name)]) for name in name_list])
                        plot_observables_distribution_dictionary[data_name]['Vorticity'][t][d] = np.array([np.array(h5f['vorticity_{}'.format(name)]) for name in name_list])
                        plot_observables_distribution_dictionary[data_name]['Defect_Diff'][t][d] = np.array([np.array(h5f['defect_diff_{}'.format(name)]) for name in name_list])
                        plot_observables_distribution_dictionary[data_name]['local_Energy'][t][d] = np.array([np.array(h5f['local_energy_{}'.format(name)]) for name in name_list])
                        
                if data_name == 'training_data' or data_name == 'generated_data':

                    dataset_magent = pd.DataFrame(plot_observables_distribution_dictionary[data_name]['Magnetisation'][t][d][:,-1,:,-1].reshape(-1,samplesize)).T #justlast one
                    dataset_energy = pd.DataFrame(plot_observables_distribution_dictionary[data_name]['Energy'][t][d][:,-1,:,-1].reshape(-1,samplesize)).T


                    mean_e = np.mean(np.sort(plot_observables_distribution_dictionary[data_name]['Energy'][t][d].reshape(-1,samplesize),axis=1) , axis=1)#[plot_energy_distribution[d_idx]<-470]
                    mean_e2 = np.mean(np.sort(plot_observables_distribution_dictionary[data_name]['Energy'][t][d].reshape(-1,samplesize),axis=1)**2, axis=1)

                    plot_observables_distribution_dictionary[data_name]['reduced_specific_heat'][t][d] = (mean_e2-mean_e**2)/(t**2)


                    dataset_magnet_melt = dataset_magent.melt(var_name='nmb_distances',value_name='Magnetisation')
                    dataset_magnet_melt['Dataset'] = data_name
                    dataset_magnet_melt['distances'] = np.repeat(name_list.reshape(-1,1),axis=1,repeats=samplesize).flatten()
                    dataset_enegery_melt = dataset_energy.melt(var_name='nmb_distances',value_name='Energy')
                    dataset_enegery_melt['Dataset'] = data_name
                    dataset_enegery_melt['distances'] = np.repeat(name_list.reshape(-1,1),axis=1,repeats=samplesize).flatten()

                    print('energy', dataset_enegery_melt.dtypes)

                    dataset_list_energy.append(dataset_enegery_melt)
                    dataset_list_magnetisation.append(dataset_magnet_melt)


                
                plot_observables_distribution_dictionary[data_name]['Epochs'][t][d] = np.array(observables['Epoch'])
                print(observables['Epoch'])
                plot_observables_distribution_dictionary[data_name]['name_list'][t][d] = np.array(name_list)
                
                if data_name == 'real_data' or ( data_name=='training_data' and (sim_data_type=='specific_pinned_wo_sort' or sim_data_type=='full_pinned_wo_sort')):
                    plot_observables_mean_dictionary[data_name]['Magnetisation'][t][d] = np.array([observables['Mag_mean_{}'.format(0.0)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Magnetisation'][t][d] = np.array([observables['Mag_error_{}'.format(0.0)] for name in name_list])
                    
                    plot_observables_mean_dictionary[data_name]['Energy'][t][d] = np.array([observables['Energy_mean_{}'.format(0.0)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Energy'][t][d] = np.array([observables['Energy_error_{}'.format(0.0)] for name in name_list])
                    
                    if 'Magnet_suscep_alternative' in input_observables:
                        plot_observables_mean_dictionary[data_name]['Magnet_suscep_alternative'][t][d] = np.array([observables['MS_alternative_mean_{}'.format(0.0)] for name in name_list])
                        plot_observables_var_dictionary[data_name]['Magnet_suscep_alternative'][t][d] = np.array([observables['MS_alternative_error_{}'.format(0.0)] for name in name_list])
                    else:
                        plot_observables_mean_dictionary[data_name]['Magnet_suscep'][t][d] = np.array([observables['MS_mean_{}'.format(0.0)] for name in name_list])
                        plot_observables_var_dictionary[data_name]['Magnet_suscep'][t][d] = np.array([observables['MS_error_{}'.format(0.0)] for name in name_list])
                        

                    
                    plot_observables_mean_dictionary[data_name]['Helicity_modulus'][t][d] = np.array([observables['HM_mean_{}'.format(0.0)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Helicity_modulus'][t][d] = np.array([observables['HM_error_{}'.format(0.0)] for name in name_list])
                    
                    plot_observables_mean_dictionary[data_name]['Specific_heat'][t][d] = np.array([observables['SH_mean_{}'.format(0.0)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Specific_heat'][t][d] = np.array([observables['SH_error_{}'.format(0.0)] for name in name_list])
                    
                    plot_observables_mean_dictionary[data_name]['Vorticity'][t][d] = np.array([observables['Vor_mean_{}'.format(0.0)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Vorticity'][t][d] = np.array([observables['Vor_error_{}'.format(0.0)] for name in name_list])
                    
                    
                    
                    plot_observables_mean_dictionary[data_name]['Defect_Diff'][t][d] = np.array([observables['Defect_Diff_mean_{}'.format(0.0)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Defect_Diff'][t][d] = np.array([observables['Defect_Diff_error_{}'.format(0.0)] for name in name_list])
                    
                    
                    
                    


                else:

                
                    plot_observables_mean_dictionary[data_name]['Magnetisation'][t][d] = np.array([observables['Mag_mean_{}'.format(name)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Magnetisation'][t][d] = np.array([observables['Mag_error_{}'.format(name)] for name in name_list])
                    
                    plot_observables_mean_dictionary[data_name]['Energy'][t][d] = np.array([observables['Energy_mean_{}'.format(name)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Energy'][t][d] = np.array([observables['Energy_error_{}'.format(name)] for name in name_list])
                    
                    if 'Magnet_suscep_alternative' in input_observables:
                        plot_observables_mean_dictionary[data_name]['Magnet_suscep_alternative'][t][d] = np.array([observables['MS_alternative_mean_{}'.format(name)] for name in name_list])
                        plot_observables_var_dictionary[data_name]['Magnet_suscep_alternative'][t][d] = np.array([observables['MS_alternative_error_{}'.format(name)] for name in name_list])
                    else:
                        plot_observables_mean_dictionary[data_name]['Magnet_suscep'][t][d] = np.array([observables['MS_mean_{}'.format(name)] for name in name_list])
                        plot_observables_var_dictionary[data_name]['Magnet_suscep'][t][d] = np.array([observables['MS_error_{}'.format(name)] for name in name_list])
                        

                    
                    plot_observables_mean_dictionary[data_name]['Helicity_modulus'][t][d] = np.array([observables['HM_mean_{}'.format(name)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Helicity_modulus'][t][d] = np.array([observables['HM_error_{}'.format(name)] for name in name_list])
                    
                    plot_observables_mean_dictionary[data_name]['Specific_heat'][t][d] = np.array([observables['SH_mean_{}'.format(name)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Specific_heat'][t][d] = np.array([observables['SH_error_{}'.format(name)] for name in name_list])
                    
                    plot_observables_mean_dictionary[data_name]['Vorticity'][t][d] = np.array([observables['Vor_mean_{}'.format(name)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Vorticity'][t][d] = np.array([observables['Vor_error_{}'.format(name)] for name in name_list])
                    
                    
                    
                    plot_observables_mean_dictionary[data_name]['Defect_Diff'][t][d] = np.array([observables['Defect_Diff_mean_{}'.format(name)] for name in name_list])
                    plot_observables_var_dictionary[data_name]['Defect_Diff'][t][d] = np.array([observables['Defect_Diff_error_{}'.format(name)] for name in name_list])
                    
                    
                    
                    


    return save_path, plot_observables_distribution_dictionary,plot_observables_mean_dictionary, plot_observables_var_dictionary,dataset_list_energy,dataset_list_magnetisation




def plotting_observables_2D(heatmap,input_att,output_data_path,sim_data_type='new_full_pinned',data_wo_sorting=False,difference=None,comparison='temperatures',observables= ['Magnet_suscep','Magnetisation', 'Vorticity'],comparison_example ='temperature',axis_comparison = 'distance',plot_type='2D', distribution=False,samplesize=1000,fixed_data = True,full_epoch= False,data_type_list = ['training_data','generated_data','random_data','zero_temp'], generated_data_atributes=[[64,0.01,0,5,1,0]],simulation_data_path  ='/localscratch/kyklos/Pytorch_cuda_code/iona/iona_oct_23/Full_training_oct_23/full_training_data_gan_low_temp_2_corrected.h5', temperatures = [0.1],defect_nmbs=[2],  lattice_size= 16,correlation=True,input_save_path=None,full=False):
    simulation_observables = Observables(lattice_size=lattice_size, temperature = temperatures[0])

    maximal_distance_correlation = int(simulation_observables.max_corr_distance(max_corr = int(lattice_size/2)))
    
    
    out = loading_observables(observables,fixed_data,simulation_data_path,output_data_path,temperatures,data_type_list,generated_data_atributes,input_att,maximal_distance_correlation,samplesize,defect_nmbs,lattice_size,input_save_path=input_save_path,sim_data_type=sim_data_type)

    return out




if __name__ == "__main__":
    generated_data_atributes_list=[
    ['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'sum', 'conv_transposed', 0, 'leaky_Relu',0.01,None,0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0),0,None,0,0.5, False,'{}_avg_3_3_fourier_log_10'.format(5)]]
    full_temps = [0.1]
    temps = [0.1]
    out = []
    for tidx,t in enumerate(temps):
        out.append(plotting_observables_2D(input_save_path=None,sim_data_type='full_pinned',data_wo_sorting=False,heatmap=True,full=True,input_att={'noise_size':1.,'training_data_nmb':350000},output_data_path='',difference=None,comparison='temperatures',observables= ['Vorticity','Defect_Diff','Magnetisation', 'Magnet_suscep','Energy','Specific_heat','Helicity_modulus'],comparison_example ='temperature',axis_comparison = 'distance',plot_type='2D', distribution=True,samplesize=1000,fixed_data = True,full_epoch= True,data_type_list = ['training_data','generated_data'], generated_data_atributes=generated_data_atributes_list[0],simulation_data_path  ='./', temperatures = full_temps,defect_nmbs=[2],  lattice_size= 16,correlation=False))
        
        
    out[0][1]['generated_data'].keys()

    energy = out[0][1]['generated_data']['Energy']
    magnet = out[0][1]['generated_data']['Magnetisation']

    energy_train = out[0][1]['training_data']['Energy']
    magnet_train = out[0][1]['training_data']['Magnetisation']
    name_list =  out[0][1]['training_data']['name_list'][0.1][2]

    
    DISTANCES = [i for i in range(14)]  
    temps = sorted(energy.keys())

    stats = {name: {ds: {d: [] for d in temps} for ds in ['gen', 'train']} 
            for name in ['mean', 'std', 'var', 'skew', 'kurtosis', 'Binder cumulant','Binder cumulant error', 'q10', 'q90']}
    
    from scipy.stats import kurtosis 
    from scipy.stats import skew as scipy_skew

    for d in DISTANCES:    
        for t in temps:
            inner_k = list(magnet[t].keys())[0]
            g = magnet[t][inner_k][d, -1]
            tr = magnet_train[t][inner_k][d, -1]
            for data, label in [(g, 'gen'), (tr, 'train')]:
                stats['mean'][label][t].append(np.mean(data))
                stats['std'][label][t].append(np.std(data))
                stats['var'][label][t].append(np.var(data))
                stats['skew'][label][t].append(
                    np.mean(((data - np.mean(data)) / (np.std(data) + 1e-12))**3))
                stats['kurtosis'][label][t].append(
                    np.mean(((data - np.mean(data)) / (np.std(data) + 1e-12))**4))
                binder = 1-np.mean(data**4)/(3*(np.mean(data**2)**2 + 1e-12))
                binder_error = faster_error_calulcation_binder(full_magentization=data,full_binder_values=binder)
                stats['Binder cumulant'][label][t].append(binder)
                stats['Binder cumulant error'][label][t].append(binder_error)
                stats['q10'][label][t].append(np.percentile(data, 10))
                stats['q90'][label][t].append(np.percentile(data, 90))

    stat_names = ['mean', 'std', 'var', 'skew', 'kurtosis', 'Binder cumulant', 'q10', 'q90']
    colors = plt.cm.tab20(np.linspace(0, 1, len(DISTANCES)))

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs = axs.flatten()
    binder_acc_full_t = []
    for i, stat in enumerate(stat_names):
        for j, d in enumerate(temps):
            if stat == 'Binder cumulant':
                axs[i].errorbar(x=name_list, y=stats[stat]['gen'][d],yerr= stats['Binder cumulant error']['gen'][d],fmt='o-', color=colors_gen[j], label=f'gen T={d}')
                axs[i].errorbar(x=name_list, y=stats[stat]['train'][d], yerr= stats['Binder cumulant error']['train'][d],fmt='s--', color=colors_train[j], label=f'train T={d}')
                
                binder_acc=acc(binder_train_list=np.array(stats[stat]['train'][d]), binder_gen_list=np.array(stats[stat]['gen'][d]))
                binder_acc_full_t.append(binder_acc)
            else:
                
                axs[i].plot(name_list, stats[stat]['gen'][d], 'o-', color=colors_gen[j], label=f'gen T={d}')
                axs[i].plot(name_list, stats[stat]['train'][d], 's--', color=colors_train[j], label=f'train T={d}')
        axs[i].set_ylabel(stat)
        axs[i].set_xlabel('Distance')
        if i == 0:
            axs[i].legend(fontsize=7)

    plt.suptitle(f'Temperatures: {temps}')
    plt.tight_layout()
    plt.savefig(output_path+ "summary_stats_multi_dist_magent_binder_per_dist.pdf", dpi=300)
    plt.close()
    
    binder_ = {}
    for t,d in enumerate(temps):
        binder_['Binder cumulant_temp_{}'.format(d)] = stats['Binder cumulant']['gen'][d]
        binder_['Binder cumulant_error_temp_{}'.format(d)] = stats['Binder cumulant error']['gen'][d]
    df_binder = pd.DataFrame.from_dict(binder_)
    df_binder.to_csv('Binder.csv')
    
    heatmap_error(binder_acc=np.array(binder_acc_full_t),temperatures=temps,names=name_list)