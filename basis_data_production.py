from filtration_function import TopologicalAnalysis
import torch
import h5py
import numpy as np
import random
import pandas as pd
from XY_model_numba import XYSystem
from observables_analysis_class_26 import Observables
from helper_functions import transformation_pos

def load_in_generator_network(data_path,epoch,temperature_label,input_defects,lattice_size,
                              noise_factor= 1.0, device="cuda"):
    """
    Loads a trained generator model and generates data based on input defects and temperature labels.
    Returns the generated data.
    """
    load_model = torch.load(data_path 
                            + 'Generator_check_{}.pth'.format(epoch),map_location=torch.device(device), weights_only = False)
    load_model.change_noise_factor(noise_factor)
    

    input_defects = input_defects.float().view(-1,1,lattice_size,lattice_size)
    generated_data_list = []
    for b in range(0,int(input_defects.shape[0]),10):
        generated_data_sub = load_model(input_defects[b:b+10],labels=temperature_label[b:b+10])
        generated_data_list.append(generated_data_sub)


    generated_data = torch.cat(generated_data_list,dim=0)
    return generated_data


def check_defect_type_config(simulated_data_path,temperature, lattice_size,data_type='full_pinned',defect_nmb=2,transformed= False,device='cpu', samplesize = 1000,compare_defect_lattice=False,reference_temp = 0.1,new=True):
    with torch.no_grad():
        
        zeroth_spin_dict = {}
        zeroth_defect_dict = {}
        
        if new:
            
            if data_type == 'full_pinned':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_testing_data_gan_defect_distance_select_compare_0_rotation_0_later_rot_1_latticesize_16_defectnmb_2_samplesize350000_temperature_{}.h5'.format(temperature), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_distances = torch.from_numpy(np.array(h5f['distances'])).to(device).view(-1)


                    print('check input',sim_distances.shape)

                defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                
                


                

                for dist in torch.unique(sim_distances):

                    current_defect_distances = sim_defects[sim_distances==dist]
                    current_spin_distances = sim_spins[sim_distances==dist]

                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    sample_idx = random.sample(spin_data_index,samplesize)
                    
                    distances_dictionary_spins[dist.item()] = current_spin_distances[sample_idx]
                    distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                    
                    sizes.append(int(sim_spins[spin_data_index].size(0)))
                    
                if transformed:
                
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)
                        
            if data_type == 'new_full_pinned':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_1_dist_sorted_0_defect_nmb_{}.h5'.format(data_type,temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)



                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]


                defect_number_tensor = torch.sum(torch.abs(reduced_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)
                defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                topological_analysis_tool.set_full_defect_positions(defect_lattices=reduced_defects,maximal_defect_nmb=maximal_defect_nmb)
                defect_distances = topological_analysis_tool.return_full_defect_distances()
                
                ##print(defect_distances)
                print(torch.unique(defect_distances,return_counts= True))
                
                
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                


                
                #print(torch.any(sim_distances == 5.0))
                for dist in torch.unique(defect_distances):
                    current_defect_distances = reduced_defects[defect_distances==dist]
                    #print(current_defect_distances.shape)
                    current_spin_distances = reduced_spins[defect_distances==dist]

                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    if int(current_defect_distances.size(0)) < samplesize:
                        continue
                    else:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist.item()] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                        sizes.append(int(reduced_spins[spin_data_index].size(0)))
                        
                if transformed:
                
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)
                        
            elif data_type == 'real_data':#need to have 10,000 sample size
                
                temperature_idx = int((temperature*10)-1.)
                batch = 10000
                lazy_count = 200000*temperature_idx
                    
                spin_data_index = [i for i in range(int(200000))]
                sample_idx = random.sample(spin_data_index,samplesize)                    
                distances_dictionary_spins_list = []
                distances_dictionary_defects_list = []
                sizes = []
                with open('observables_test_idx_{}_{}_{}_{}.txt'.format(data_type,samplesize,temperature,defect_nmb), 'w') as f:
                    for i in sample_idx:
                        f.write(f"{i}\n")
                print(sample_idx)
                zeroth_spin_dict = None
                zeroth_defect_dict = None
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_1_dist_sorted_0_defect_nmb_{}.h5'.format(data_type,'full',defect_nmb), 'r') as h5f:
                    
                    ##sim_temperatures = torch.from_numpy(np.array(h5f['temperatures'])[200000*temperature_idx:200000*temperature_idx+200000]).to(device).view(-1)
                    ##print(torch.all(sim_temperatures==(temperature*10)))
                    sim_spins = torch.from_numpy(np.array(h5f['spins'][lazy_count:lazy_count+200000])).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'][lazy_count:lazy_count+200000])).view(-1,lattice_size,lattice_size)
                    print(sim_spins.shape)
                    ##sim_distances = torch.zeros(sim_spins.size(0),dvice=device)
                    
                distances_dictionary_spins[0.] = sim_spins[sample_idx].to(device)
                distances_dictionary_defects[0.] = sim_defects[sample_idx].to(device)
                    ##for i in range(200000*temperature_idx,200000*temperature_idx+200000,batch):
                        
                print('test')
                        
                        
                        

                        
                        #print('check input',sim_distances.shape)
                        
                        ##defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                        ##maximal_defect_nmb=torch.max(defect_number_tensor)
                        

                        ##check_observables = Observables(lattice_size=lattice_size, temperature = temperature)
                        ##for dist in torch.unique(sim_distances):
                        ##    current_defect_distances = sim_defects[sim_distances==dist]
                            #print(current_defect_distances.shape)
                        ##    current_spin_distances = sim_spins[sim_distances==dist]
                            

                            ##sizes.append(int(sim_spins[spin_data_index].size(0)))

                            ##distances_dictionary_spins[dist.item()][] = current_spin_distances[sample_idx]
                            ##distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                            
                            
                if transformed:
                        
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)

                ##for key in distances_dictionary_spins:
                ##    distances_dictionary_spins[key] = torch.stack(distances_dictionary_spins_list,dim=0)     
                ##    distances_dictionary_defects[key] = torch.stack(distances_dictionary_defects_list,dim=0) 
            elif data_type == 'specific_pinned' or data_type == 'new_larger_specific_pinned' or data_type == 'new_specific_pinned':#need to have 10,000 sample size
                
                with h5py.File(simulated_data_path+'/Training_data/'+'latticesize_{}_full_{}_data_gan_{}_rotation_0_dist_sorted_0_defect_nmb_{}.h5'.format(lattice_size,data_type,temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_defects = torch.from_numpy(np.array(h5f['zeroth_defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_spins = torch.from_numpy(np.array(h5f['zeroth_spins'])).to(device).view(-1,lattice_size,lattice_size)
                    


                    print('check input',sim_spins.shape)
                    
                print(sum(torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1)!=0))
                
                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]
                reduced_zeroth_spins = zeroth_spins[reduction_idx]
                reduced_zeroth_defects = zeroth_defects[reduction_idx]

                defect_number_tensor = torch.sum(torch.abs(reduced_zeroth_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                topological_analysis_tool.set_full_defect_positions(defect_lattices=reduced_zeroth_defects,maximal_defect_nmb=maximal_defect_nmb)
                defect_distances = topological_analysis_tool.return_full_defect_distances()
                
                ##print(defect_distances)
                print(torch.unique(defect_distances,return_counts= True))
                
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                check_observables = Observables(lattice_size=lattice_size, temperature = temperature)
                for dist in torch.unique(defect_distances):
                    current_defect_distances = reduced_defects[defect_distances==dist]
                    #print(current_defect_distances.shape)
                    current_spin_distances = reduced_spins[defect_distances==dist]
                    
                    current_zeroth_defect = reduced_zeroth_defects[defect_distances==dist]
                    current_zeroth_spin =reduced_zeroth_spins[defect_distances==dist]
                    
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    
                    if int(current_defect_distances.size(0)) < samplesize:
                        continue
                    else:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist.item()] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist.item()] = current_defect_distances[sample_idx]
                        
                        zeroth_spin_dict[dist.item()] = current_zeroth_spin[sample_idx]
                        zeroth_defect_dict[dist.item()] = current_zeroth_defect[sample_idx]
                        sizes.append(int(reduced_spins[spin_data_index].size(0)))
                    
                if transformed:
                
                    distances_dictionary_spins,distances_dictionary_defects,changed_distances = transformation_pos(defects=distances_dictionary_defects, spins=distances_dictionary_spins, lattice_size=lattice_size,defect_nmb=2)
                        
            elif data_type == 'specific_pinned_wo_sort' or data_type == 'larger_specific_pinned_wo_sort':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_1_dist_sorted_0_defect_nmb_{}.h5'.format('specific_pinned',temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_defects = torch.from_numpy(np.array(h5f['zeroth_defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_spins = torch.from_numpy(np.array(h5f['zeroth_spins'])).to(device).view(-1,lattice_size,lattice_size)
                    
                    
                print(sum(torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1)!=0))
                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]
                reduced_zeroth_spins = zeroth_spins[reduction_idx]
                reduced_zeroth_defects = zeroth_defects[reduction_idx]
                spin_data_index = [i for i in range(int(reduced_spins.size(0)))]
                sample_idx = random.sample(spin_data_index,samplesize)  
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                
                distances_dictionary_spins[0.] = reduced_spins[sample_idx]
                distances_dictionary_defects[0.] = reduced_defects[sample_idx]
                        
                zeroth_spin_dict[0.] = reduced_zeroth_spins[sample_idx]
                zeroth_defect_dict[0.] = reduced_zeroth_defects[sample_idx]
            elif data_type == 'new_specific_pinned_wo_sort':
                with h5py.File(simulated_data_path+'/Training_data/'+'full_{}_data_gan_{}_rotation_0_dist_sorted_0_defect_nmb_{}.h5'.format('new_specific_pinned',temperature,defect_nmb), 'r') as h5f:
                    sim_spins = torch.from_numpy(np.array(h5f['spins'])).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(np.array(h5f['defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_defects = torch.from_numpy(np.array(h5f['zeroth_defects'])).to(device).view(-1,lattice_size,lattice_size)
                    zeroth_spins = torch.from_numpy(np.array(h5f['zeroth_spins'])).to(device).view(-1,lattice_size,lattice_size)
                    
                    
                print(sum(torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1)!=0))
                reduction_idx = torch.sum(sim_defects.view(-1,lattice_size*lattice_size),dim=1) == 0
                reduced_spins = sim_spins[reduction_idx]
                reduced_defects =sim_defects[reduction_idx]
                reduced_zeroth_spins = zeroth_spins[reduction_idx]
                reduced_zeroth_defects = zeroth_defects[reduction_idx]
                spin_data_index = [i for i in range(int(reduced_spins.size(0)))]
                sample_idx = random.sample(spin_data_index,samplesize)  
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                
                distances_dictionary_spins[0.] = reduced_spins[sample_idx]
                distances_dictionary_defects[0.] = reduced_defects[sample_idx]
                        
                zeroth_spin_dict[0.] = reduced_zeroth_spins[sample_idx]
                zeroth_defect_dict[0.] = reduced_zeroth_defects[sample_idx]



        else:
            with h5py.File(simulated_data_path+'/Training_data/'+'full_training_data_gan_{}_corrected.h5'.format(temperature), 'r') as h5f:
                sim_spins = torch.from_numpy(h5f['spins'][350000:700000]).to(device).view(-1,lattice_size,lattice_size)
                sim_defects = torch.from_numpy(h5f['defects'][350000:700000]).to(device).view(-1,lattice_size,lattice_size)


                print('check input')

            defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)
            print(maximal_defect_nmb)
                    ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
                    ##print(unique_defect_lattices.size())
                    ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)

            topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


            topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
            defect_distances = topological_analysis_tool.return_full_defect_distances()

            print(defect_distances)

            distances_dictionary_spins = {}
            distances_dictionary_defects= {}


            if compare_defect_lattice:
                with h5py.File(simulated_data_path+'/observables_output/'+'defect_for_analysis_full_distances_{}_old.h5'.format(reference_temp,samplesize), 'r') as h5f:

                    name_list = np.array(h5f['distances'])

                for dist_idx, dist in enumerate(name_list):

                


                    with h5py.File(simulated_data_path+'/observables_output/'+'defect_for_analysis_full_distances_{}_old.h5'.format(reference_temp,samplesize), 'r') as h5f:#_sample_{}

                            #sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        sim_basis_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                            #full_spin_list.append(sim_spins)
                            #full_defect_list.append(sim_basis_defects)

                    distances_dictionary_spins[dist] = []
                    distances_dictionary_defects[dist] = []

                    spin_list_unique = []
                    defect_list_unique = []

                    unique_defects, unique_defects_count = torch.unique(sim_basis_defects.view(-1,lattice_size,lattice_size),dim=0, return_counts= True)

                        #print(dist, torch.sum(unique_defects_count))

                    current_defect_distances = sim_defects[defect_distances==name_list[dist_idx]]
                    current_spin_distances = sim_spins[defect_distances==name_list[dist_idx]]

                    ##print('size', current_defect_distances.shape)
                    ##print('size', current_spin_distances.shape)


                    ##print(torch.sum(unique_defects_count))

                    ##print(unique_defects_count[0])

                    if dist_idx == 0:

                        spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]

                        sample_idx = random.sample(spin_data_index,samplesize)

                        distances_dictionary_spins[dist] = current_spin_distances[sample_idx]

                        distances_dictionary_defects[dist] = current_defect_distances[sample_idx]

                    else:

                        size_test = []

                        other_size_test = []



                        for unique_idx,basis_defect in enumerate(unique_defects):

                                #print(current_defect_distances==basis_defect)

                            unique_indexing = torch.all((current_defect_distances==basis_defect).view(-1, lattice_size*lattice_size),dim=1)

                            unique_input_spins = current_spin_distances[unique_indexing]

                            unique_input_defects = current_defect_distances[unique_indexing]

                                #print(unique_idx)

                                #print('spins', unique_input_spins.shape)

                            ##other_size_test.append(unique_input_defects.size(0))
                                #print('unique_spins', unique_input_spins[:unique_defects_count[unique_idx]].shape)

                            ##size_test.append(unique_defects_count[unique_idx])
                            ##if unique_defects_count[unique_idx] > unique_input_spins.size(0):
                            ##    print('Error',unique_defects_count[unique_idx], unique_input_spins.size(0))


                            spin_list_unique.append(unique_input_spins[:unique_defects_count[unique_idx]].view(-1,lattice_size,lattice_size))
                            defect_list_unique.append(unique_input_defects[:unique_defects_count[unique_idx]].view(-1,lattice_size,lattice_size))



                        distances_dictionary_spins[dist] = torch.cat(spin_list_unique,dim=0)
                        distances_dictionary_defects[dist] =  torch.cat(defect_list_unique,dim=0)
                        ##print('sum test', sum(size_test))
                        ##print('sum test', sum(other_size_test))
                        ##print(torch.cat(spin_list_unique,dim=0).shape)




                #print('works?')



            
            else:

                    #print(defect_distances.size())


                    ##make  uniform distributed data of the following type:
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                for dist in torch.unique(defect_distances):
                        ##print(dist)
                    current_defect_distances = sim_defects[defect_distances==dist]
                    current_spin_distances = sim_spins[defect_distances==dist]
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    sample_idx = random.sample(spin_data_index,samplesize)
                    distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                    distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                    sizes.append(int(sim_spins[spin_data_index].size(0)))

                print('defects',sizes)


        return distances_dictionary_spins, distances_dictionary_defects, zeroth_spin_dict, zeroth_defect_dict
        
    
def defect_distance_distributions(defect_distances,correspon_defects, samplesize):
    
    unique_distances, unique_distance_nmb = torch.unique(defect_distances, return_counts= True)
    
    new_unique_distance_nmb = torch.clone(unique_distance_nmb)
    
    new_distance_with_counts = {}
    
    for d_idx,d in unique_distances:
        if unique_distance_nmb[d_idx] < samplesize:
            if (d_idx-1) > 0:
                lower_uniq_nmb = new_unique_distance_nmb[d_idx-1]
    
        
def check_defect_type_config_larger_lattice(simulated_data_path,temperature, lattice_size,batch = True,device='cuda', samplesize = 1000, defect_number = 2):
    
    

    with torch.no_grad():
        
        
        
        if batch:
            
            distances_dictionary_spins = {}
            distances_dictionary_defects = {}
            
            batchsize = 51000
            
            for b in range(10):
            
            
                with h5py.File(simulated_data_path+'Training_data/'+'full_training_data_gan_{}_T_{}_lattice_{}_defect.h5'.format(temperature,lattice_size,defect_number), 'r') as h5f:
                ##with h5py.File(simulated_data_path+'lattice_{}_num_pairs{}_temp{}.h5'.format(lattice_size,defect_number//2,temperature), 'r') as h5f:
                    sim_spins = torch.from_numpy(h5f['spins'][b*batchsize:b*batchsize+batchsize]).to(device).view(-1,lattice_size,lattice_size)
                    sim_defects = torch.from_numpy(h5f['defects'][b*batchsize:b*batchsize+batchsize]).to(device).view(-1,lattice_size,lattice_size)


                    print('check input', sim_defects.shape)

                    defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                    maximal_defect_nmb=torch.max(defect_number_tensor)
                    print(maximal_defect_nmb)
                        ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
                        ##print(unique_defect_lattices.size())
                        ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)
                    if  maximal_defect_nmb >= 2:


                        topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                        topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
                        defect_distances = topological_analysis_tool.return_full_defect_distances()

                    else:
                        defect_distances = torch.zeros([sim_defects.size(0)], device=device)

                    print('distances', torch.unique(defect_distances, return_counts= True))


                        #print(defect_distances.size())


                        ##make  uniform distributed data of the following type:

                    sizes = []
                    for dist in torch.unique(defect_distances):
                        #print(dist)
                        
                        current_defect_distances = sim_defects[defect_distances==dist]
                        current_spin_distances = sim_spins[defect_distances==dist]
                        spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                        if int(current_defect_distances.size(0)) > 100:
                            sample_idx = random.sample(spin_data_index,100)
                            if distances_dictionary_spins.get(dist.item()) is not None:
                                print('check')
                                distances_dictionary_spins[dist.item()].append(current_spin_distances[sample_idx])
                                distances_dictionary_defects[dist.item()].append(current_defect_distances[sample_idx])

                            else:

                                distances_dictionary_spins[dist.item()] = [current_spin_distances[sample_idx]]
                                distances_dictionary_defects[dist.item()] = [current_defect_distances[sample_idx]]
                            sizes.append(int(sim_spins[spin_data_index].size(0)))
                            
                    print(list(distances_dictionary_spins.keys()))

            
            
        else:
                    


            #with h5py.File(simulated_data_path+'full_training_data_gan_{}_T_{}_lattice_{}_defect.h5'.format(temperature,lattice_size,defect_number), 'r') as h5f:
            with h5py.File(simulated_data_path+'full_data_gan_{}_rotation_{}_dist_sorted_{}_defect_nmb_{}.h5'.format(temperature,0,0,defect_number), 'r') as h5f:
                sim_spins = torch.from_numpy(h5f['spins'][:]).to(device).view(-1,lattice_size,lattice_size)
                sim_defects = torch.from_numpy(h5f['defects'][:]).to(device).view(-1,lattice_size,lattice_size)
                


                #print('check input', sim_defects.shape)

                defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
                maximal_defect_nmb=torch.max(defect_number_tensor)
                print(maximal_defect_nmb)
                    ##unique_defect_lattices = torch.unique(sim_defects, dim=0)
                    ##print(unique_defect_lattices.size())
                    ##unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)
                if  maximal_defect_nmb >= 2:


                    topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


                    topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
                    defect_distances = topological_analysis_tool.return_full_defect_distances()

                else:
                    defect_distances = torch.zeros([sim_defects.size(0)], device=device)

                print('distances', torch.unique(defect_distances, return_counts= True))


                    #print(defect_distances.size())


                    ##make  uniform distributed data of the following type:
                distances_dictionary_spins = {}
                distances_dictionary_defects = {}
                sizes = []
                #if defect_number == 4:
                    
                #reduced_defect_distances
                for dist in torch.unique(defect_distances):
                    #print(dist)
                    
                    current_defect_distances = sim_defects[defect_distances==dist]
                    current_spin_distances = sim_spins[defect_distances==dist]
                    spin_data_index = [i for i in range(int(current_defect_distances.size(0)))]
                    if int(current_defect_distances.size(0)) >= samplesize:
                        sample_idx = random.sample(spin_data_index,samplesize)
                        distances_dictionary_spins[dist] = current_spin_distances[sample_idx]
                        distances_dictionary_defects[dist] = current_defect_distances[sample_idx]
                        sizes.append(int(sim_spins[spin_data_index].size(0)))

                #print('defects',sizes)


        return distances_dictionary_spins, distances_dictionary_defects
    
    

def output_analysis_basis(simulated_data_path,temperature,lattice_size,data_type='full_pinned',transformed=False,batch= True,compare_defect_lattice=False,device='cpu',samplesize=1000, larger=True,defect_nmb=2,new=True):

    if larger:
        spin_dict, defect_dict= check_defect_type_config_larger_lattice(batch = batch,simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,device=device,samplesize=samplesize,defect_number=defect_nmb)

    else:

        spin_dict, defect_dict, zeroth_spin_dict, zeroth_defect_dict = check_defect_type_config(transformed= transformed,simulated_data_path=simulated_data_path,temperature=temperature, lattice_size=lattice_size,data_type=data_type,defect_nmb=defect_nmb,device=device,samplesize=samplesize,compare_defect_lattice=compare_defect_lattice,new=new)
    #name_list = ['full', 'no_defect', 'small_defects', 'mid_defects', 'large_defects']

    output_spin_dict = {}
    output_defect_dict = {}
    print('check',[dist for dist in spin_dict])

    with h5py.File(simulated_data_path + 'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(data_type,temperature,lattice_size,defect_nmb,samplesize),'w') as h5f:

        dataset_distances = h5f.create_dataset('distances', shape=(int(len(spin_dict)),), dtype='float')
        if batch:
            dataset_distances[:] = np.array([dist for dist in spin_dict])
        else:
            dataset_distances[:] = np.array([dist for dist in spin_dict])
            

        for dist in spin_dict:
            
            if batch:
                output_spin_dict[dist] = torch.cat(spin_dict[dist], dim = 0)
                output_defect_dict[dist] = torch.cat(defect_dict[dist], dim = 0)
            else:
                output_spin_dict[dist] = spin_dict[dist]
                output_defect_dict[dist] = defect_dict[dist]
                
            if transformed:
                samplesize = int(samplesize/2)
                
                            
            dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            
            if data_type == 'specific_pinned' or data_type == 'new_larger_specific_pinned' or data_type == 'specific_pinned_wo_sort' or data_type == 'new_specific_pinned' or data_type == 'new_specific_pinned_wo_sort':
                dataset_zeroth_spins_compare = h5f.create_dataset('spin_lattices_zeroth{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
                dataset_zeroth_defects_compare = h5f.create_dataset('defects_lattices_zeroth{}'.format(dist), shape=(samplesize,lattice_size*lattice_size), dtype='float')
            
            if batch:
                
                print(torch.cat(spin_dict[dist], dim = 0).shape)
            
                dataset_spins_compare[:] = torch.cat(spin_dict[dist], dim = 0).view(-1,lattice_size*lattice_size).cpu().numpy()
                
                dataset_defects_compare[:] = torch.cat(defect_dict[dist],dim=0).view(-1,lattice_size*lattice_size).cpu().numpy()
                
            else:
                
                dataset_spins_compare[:] = spin_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                
                dataset_defects_compare[:] = defect_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                
                if data_type == 'specific_pinned' or data_type == 'new_larger_specific_pinned'or data_type == 'specific_pinned_wo_sort'or data_type == 'new_specific_pinned' or data_type == 'new_specific_pinned_wo_sort':
                    
                    dataset_zeroth_spins_compare[:] = zeroth_spin_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                
                    dataset_zeroth_defects_compare[:] = zeroth_defect_dict[dist].view(-1,lattice_size*lattice_size).cpu().numpy()[:samplesize]
                

                
    return output_spin_dict, output_defect_dict

def load_in_data_basis(temperature, lattice_size, device, data_type, simulation_data_path, sim_data_type, new_data, compare_defect_lattice, samplesize, defect_nmb, larger, further_simulated, data_path,configuration_attributes_dict , training_data_nmb, noise_size, nmb_label, epochs, output_data_path, gen_folder=None,new=True):
        
    if new:
        oservable_data_spins, oservable_data_defects, path_name, name_list =prepare_data_basis(temperature=temperature, lattice_size=lattice_size, device=device, data_type=data_type, simulation_data_path=simulation_data_path, sim_data_type=sim_data_type, new_data=new_data, compare_defect_lattice=compare_defect_lattice, samplesize=samplesize, defect_nmb=defect_nmb, larger=larger, further_simulated=further_simulated, data_path=data_path, configuration_attributes_dict=configuration_attributes_dict, training_data_nmb=training_data_nmb, noise_size=noise_size, nmb_label=nmb_label, epochs=epochs, gen_folder=gen_folder)
        
    else:    
        oservable_data_spins = []
        oservable_data_defects = []

        if sim_data_type == 'full_pinned':
            file_ending = ''
        else:
            file_ending = 'batch_gen'

            #name_list = return_defect_distances(maximal_distance=maximal_distance)
        if defect_nmb > 2 or lattice_size > 16 or temperature > 0.1:##sim_data_type,
            with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                name_list = np.array(h5f['distances'])
                            
        else:
            with h5py.File(simulation_data_path+ 'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:

                name_list = np.array(h5f['distances'])
        print(name_list)

        nmb_distances = int(len(name_list))
        max_distances = max(name_list)

        if data_type == 'training_data' and  sim_data_type == 'full_pinned':
            path_name = 'training_data_full_distance_temp_{}_new/'.format(temperature)
        elif data_type == 'training_data' and  sim_data_type != 'full_pinned':
            path_name = 'training_data_full_distance_temp_{}_new_{}/'.format(temperature,sim_data_type)




        elif data_type == 'generated_data':



            if gen_folder is None:
                path_name = datapath_generation(configuration_attributes_dict,file_type='analysis',analysis_config =[noise_size,training_data_nmb])

            else:
                path_name = gen_folder

                
                
            path_name += 'temp_{}/'.format(temperature)

        elif data_type == 'random_data':

            path_name = 'random_data_comparison/'
        elif data_type == 'zero_temp':

            path_name = 'zero_temperature_solution_comparison_new/'
        elif data_type == 'real_data':
            path_name = 'training_data_full_distance_temp_{}_new_{}/'.format(temperature,data_type)



        with h5py.File(output_data_path+path_name+'more_data_basis_data_output_{}_{}_nmb_distances{}_max_distance_{}_defect_nmb_{}_samplesize{}_noisesize_{}_compare_{}_{}.h5'.format(data_type,sim_data_type,nmb_distances,max_distances,defect_nmb,samplesize,noise_size,1,file_ending), 'r') as h5f:

                        
            for dist in name_list:
                spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][-1]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][-1]).to(device).view(-1,lattice_size,lattice_size)

                oservable_data_spins.append(spins)
                oservable_data_defects.append(defects)
    return oservable_data_spins, oservable_data_defects, path_name, name_list

def datapath_generation(configuration_attributes_dict,file_type='model',analysis_config = [0,350000]):
    c_cfg = configuration_attributes_dict['critic_config']
    g_cfg = configuration_attributes_dict['generator_config']
    d_cfg = configuration_attributes_dict['data_config']
    t_cfg = configuration_attributes_dict['training_config']
    # Alias for readability (optional, but highly recommended)
    cfg = configuration_attributes_dict

    if file_type == 'model':

        full_path_name = ""
    elif file_type == 'analysis':

        full_path_name = "noise_factor_{}_train_nmb_{}".format(analysis_config[0],analysis_config[1])

    # 1. General & Optimizer
    full_path_name += f"/{d_cfg['temperature']}_temp_{d_cfg['temperature_label_nmb']}_labels_{t_cfg['optimizer']}_{t_cfg['optimizer_parameter']}"

    full_path_name += f"_change_epoch_{t_cfg['change_epoch']}_gp_factor_{t_cfg['alpha_gradient_penalty']}_nmb_crit_gen_{t_cfg['nmb_crit_gen']}"

    # 2. Generator Arch
    full_path_name += f"/gen_down_{g_cfg['downsampling']}_up_{g_cfg['upsampling']}_batchnorm_{g_cfg['batchnorm']}{g_cfg['activation_function']}_{g_cfg['activation_function_slope']}"

    # 3. Critic Arch
    full_path_name += f"/crit_norm_{c_cfg['norm']}_input_noise_{c_cfg['input_noise']}_dropout_{c_cfg['dropout']}{c_cfg['activation_function']}_{c_cfg['activation_function_slope']}"

    # 4. Learning Rates
    full_path_name += f"/gen_LR_start_{t_cfg['gen_LR_start']}_changed_{t_cfg['gen_LR_changed']}_crit_LR_start_{t_cfg['crit_LR_start']}_changed_{t_cfg['crit_LR_changed']}"

    # 5. Blur
    full_path_name += f"/gaussian_blur_{t_cfg['gaussian_blur']}"

    # 6. Depth & Defects
    full_path_name += f"/maxfeat_{g_cfg['max_feature_map']}_batchsize_{d_cfg['batchsize']}_depth_{g_cfg['add_depth']}_{t_cfg['defect_loss_type']}_{t_cfg['defect_loss_factor']}"

    # 7. Critic Depth
    full_path_name += f"/critic_4depth_maxfeat_{c_cfg['max_feature_map']}_depth_{c_cfg['add_depth']}"

    # 8. Regularization & Noise (Note the trailing slash / at the end)
    full_path_name += f"/gen_normlayer_{g_cfg['norm_layer']}_AdaInstart_{g_cfg['adaIn_start']}_gen_regul_{g_cfg['adaIn_regularizer']}_extranoise_{t_cfg['extra_noise']}_inception_5_avg_{g_cfg['conv_after_upsample']}_fourier_{c_cfg['fourier_log']}/"

    # 9. Conditionals
    if g_cfg['res']:
        full_path_name += "/res_gen_"

    if c_cfg['res']:
        full_path_name += "res_critic/"



    return full_path_name
    
def prepare_data_basis(temperature, lattice_size, device, data_type, simulation_data_path, sim_data_type, new_data, compare_defect_lattice, samplesize, defect_nmb, larger, further_simulated, data_path, configuration_attributes_dict, training_data_nmb, noise_size, nmb_label, epochs,gen_folder=None):
    with torch.no_grad():

        
        if new_data :
            if lattice_size == 32:
                batch = True
                device_check = device
            else:
                batch = False
                device_check = device
            spin_dict, defect_dict = output_analysis_basis(batch=batch,simulated_data_path=simulation_data_path,temperature=temperature,lattice_size=lattice_size,data_type=sim_data_type,compare_defect_lattice=compare_defect_lattice,device=device_check,samplesize=samplesize,defect_nmb=defect_nmb,larger=larger)
            full_spin_list = []
            full_defect_list = []
            name_list = []
            for dist in spin_dict:
                full_spin_list.append(spin_dict[dist])
                full_defect_list.append(defect_dict[dist])
                name_list.append(dist)
                
            print('data finished')
            
                
        
        else:
            full_spin_list = []
            full_defect_list = []
            
            if defect_nmb > 2 or lattice_size > 16 :
                #with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(sim_data_type,temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                    name_list = np.array(h5f['distances'])
                            
                    for dist in name_list:
                        sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k 
                        sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)

                        full_spin_list.append(sim_spins)
                        full_defect_list.append(sim_defects)
                        
                        ##this should 
                
            else:

            #name_list = return_defect_distances(maximal_distance=maximal_distance)
                #with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_sample_{}.h5'.format(temperature,samplesize), 'r') as h5f:

                with h5py.File(simulation_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                ##with h5py.File(simulation_data_path+'full_training_data_gan_defect_distance_select_compare_{}_rotation_{}_later_rot_{}_latticesize_{}_defectnmb_{}_samplesize{}_temperature_{}.h5'.format(int(compare_defect_lattice),int(rotation),int(later_rotation),lattice_size,defect_nmb,training_data_nmb,temperature), 'r') as h5f:

                    name_list = np.array(h5f['distances'])
                            
                    for dist in name_list:
                        sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)##for constraint_sampling more like 80 k other -1, L,L,2
                        sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)
                        full_spin_list.append(sim_spins)
                        full_defect_list.append(sim_defects)
                        print(sim_spins.shape)
        print(name_list)


        

        defect_nmber = full_defect_list

        if data_type == 'training_data':


            oservable_data_spins = []

            oservable_data_defects = []

            path_name = 'training_data_full_distance_temp_{}_new_{}/'.format(temperature,sim_data_type)

            for idx, name in enumerate(name_list):
                ##transformed_spin = full_spin_list[idx].reshape(-1,lattice_size*lattice_size,2)
                
                
                ##data_norm = torch.sqrt((torch.square(transformed_spin[:,:,1])+torch.square(transformed_spin[:,:,0])))+1e-12
                            ##print(generated_data_norm.size())
                ##transformed_spin_angle = torch.arctan2(transformed_spin[:,:,1]/data_norm,transformed_spin[:,:,0]/data_norm)
                
                ##print(transformed_spin_angle.shape)
                oservable_data_spins.append([full_spin_list[idx].reshape(-1,lattice_size,lattice_size)])
                oservable_data_defects.append([full_defect_list[idx].reshape(-1,lattice_size,lattice_size)])
                
        elif data_type == 'vision_trafo_data':
            
            full_spin_list = []
            full_defect_list = []
            
            with h5py.File(simulation_data_path+'/VIT_check/'+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,defect_nmb,samplesize), 'r') as h5f:
                ##with h5py.File(simulation_data_path+'full_training_data_gan_defect_distance_select_compare_{}_rotation_{}_later_rot_{}_latticesize_{}_defectnmb_{}_samplesize{}_temperature_{}.h5'.format(int(compare_defect_lattice),int(rotation),int(later_rotation),lattice_size,defect_nmb,training_data_nmb,temperature), 'r') as h5f:

                name_list = np.array(h5f['distances'])
                            
                for dist in name_list:
                    sim_spins = torch.from_numpy(h5f['spin_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size,2)##for constraint_sampling more like 80 k other -1, L,L,2
                    sim_defects = torch.from_numpy(h5f['defects_lattices_{}'.format(dist)][:]).to(device).view(-1,lattice_size,lattice_size)
                    full_spin_list.append(sim_spins)
                    full_defect_list.append(sim_defects)
                    
                    print('check',sim_defects.shape)
            
            oservable_data_spins = []

            oservable_data_defects = []

            path_name = '/VIT_check/training_data_full_distance_temp_{}_new/'.format(temperature)

            for idx, name in enumerate(name_list):
                transformed_spin = full_spin_list[idx].reshape(-1,lattice_size*lattice_size,2)
                
                
                data_norm = torch.sqrt((torch.square(transformed_spin[:,:,1])+torch.square(transformed_spin[:,:,0])))+1e-12
                            ##print(generated_data_norm.size())
                transformed_spin_angle = torch.arctan2(transformed_spin[:,:,1]/data_norm,transformed_spin[:,:,0]/data_norm)
                
                ##print(transformed_spin_angle.shape)
                oservable_data_spins.append([transformed_spin_angle.reshape(-1,lattice_size,lattice_size)])
                oservable_data_defects.append([full_defect_list[idx].reshape(-1,lattice_size,lattice_size)])


        elif data_type == 'generated_data':

            oservable_data_spins = []
            oservable_data_defects = []

            ##res_critic = generator_info['res_critic']
            ##res_gen = generator_info['res_gen']
            ##extra_info = generator_info['extra_info']
            ##c_depth = generator_info['c_depth']
            
            if gen_folder is None:
                path_name = datapath_generation(configuration_attributes_dict)
                ##path_name = 'noise_factor_{}_train_nmb_{}_{}_temp_{}_labels_{}_{}_change_epoch_{}_gp_factor_{}_nmb_crit_gen_{}/gen_down_{}_up_{}_batchnorm_{}{}_{}/crit_norm_{}_input_noise_{}_dropout_{}{}_{}/gen_LR_start_{}_changed_{}_crit_LR_start_{}_changed_{}/gaussian_blur_{}/maxfeat_{}_batchsize_{}_depth_{}_{}_{}/critic_4depth_maxfeat_4_depth_{}/gen_normlayer_{}_AdaInstart_{}_gen_regul_{}_extranoise_{}_inception_{}/'.format(noise_size,training_data_nmb,*generated_data_atributes,c_depth, *extra_info) 
            else:
                path_name = gen_folder
            
            ##if res_critic and not res_gen:
            ##    path_name += 'res_critic/'
                
            ##    data_path += 'res_critic/'
            ##elif res_critic and res_gen:
            ##    path_name += 'residual_blocksres_critic/'
                
            ##    data_path += 'residual_blocksres_critic/'
                
                
            path_name += 'temp_{}/'.format(temperature)
            print(path_name)
            if nmb_label == 0:
                temperature_label = None
            elif nmb_label == 1 or nmb_label == 3:

                if temperature <= 0.3:
                    factor = 0.
                elif temperature <= 0.6:
                    factor = 1.
                else:
                    factor  = 2.0

                temperature_label = (torch.ones(samplesize).view(-1,1)*factor).long()
                
            elif nmb_label == 9:
                temperature_label = (torch.sub(torch.ones(samplesize, device = device).view(-1,1)*temperature*10.,1.)).long()
            else:
                temperature_label = (torch.ones(samplesize, device = device).view(-1,1)*temperature*10.).long()
                
           
            
            for input_defects in full_defect_list:

                
                generated_angles_list_per_epoch = []
                generated_defects_list_per_epoch = []

                for epoch in epochs:


                    generated_data = load_in_generator_network(data_path=data_path,
                                                               epoch=epoch,temperature_label=temperature_label,
                                                               input_defects=input_defects,
                                                               lattice_size=lattice_size,
                                                               noise_factor= noise_size,
                                                               device=device)


                    generated_data_x = generated_data[:,0,:,:].view(-1,lattice_size,lattice_size)
                    generated_data_y = generated_data[:,1,:,:].view(-1,lattice_size,lattice_size)
                    generated_data_norm = torch.sqrt((torch.square(generated_data_x)+torch.square(generated_data_y)))+1e-12
                    
                    generated_data_x_normed = torch.div(generated_data_x,generated_data_norm)
                    generated_data_y_normed = torch.div(generated_data_y,generated_data_norm)
                    generated_angles = torch.arctan2(generated_data_y_normed,generated_data_x_normed)
                    if further_simulated == True:

                        generated_angles_add_sim_list = []
                        for add_sim_idx,spin_angles in enumerate(generated_angles):
                            spin_system = XYSystem(temperature = temperature,defect_config=full_defect_list[idx][add_sim_idx].cpu().numpy().reshape(lattice_size*lattice_size),input_analytic_solution=spin_angles.cpu().numpy().reshape(lattice_size*lattice_size),width = lattice_size)
                            spin_system.set_input_configuration()
                            spin_system.multible_sweep_alternative(10)
                            spin_after_add_sim = torch.from_numpy(spin_system.spin_config).view(lattice_size,lattice_size).cuda()
                            generated_angles_add_sim_list.append(spin_after_add_sim)
                        generated_angles_list_per_epoch.append(torch.stack(generated_angles_add_sim_list,dim=0))
                    else:
                        generated_angles_list_per_epoch.append(generated_angles)

                    generated_defects_list_per_epoch.append(input_defects)
    

                oservable_data_spins.append(generated_angles_list_per_epoch)
                oservable_data_defects.append(generated_defects_list_per_epoch)
        elif data_type == 'full_data':

            print(samplesize)

            spin_data_prefix = 'full_cluster_spin_data__'
            defect_data_prefix = 'full_cluster_defect_data_'

            path_name = 'not_fixed_data_full_temp_{}/'.format(temperature)

            spin_configurations = []
            defect_configurations = []
            temperatures = []
            
            for count in range(50):
                training_angles = pd.read_csv(data_path + spin_data_prefix + '{0}_{1}_{2}.csv'.format(temperature,lattice_size, count), delimiter = ',', skipinitialspace = True , header = 0)
                training_defects = pd.read_csv(data_path + defect_data_prefix + '{0}_{1}_{2}.csv'.format(temperature,lattice_size, count), delimiter = ',', skipinitialspace = True , header = 0)

                spin_configuration = [training_angles['Spin_angles{0}'.format(i)] for i in range(1000)]
                defect_configuration = [training_defects['Defects{0}'.format(i)] for i in range(1000)]

                spin_configurations.append(np.array(spin_configuration).reshape((1000,-1)))
                defect_configurations.append(np.array(defect_configuration).reshape((1000,-1)))
                temperatures.append(np.array([temperature for i in range(1000)]).reshape((1000,-1)))


            full_index = [i for i in range(50000)]
            sample_index = sorted(random.sample(full_index,samplesize))
            spin_configurations = np.array(spin_configurations).reshape((50000,-1))
            defect_configurations = np.array(defect_configurations).reshape((50000,-1))
            temperatures = np.array(temperatures).reshape((50000))

            full_data_spins = torch.from_numpy(spin_configurations[sample_index]).to(device).view(-1,lattice_size,lattice_size)
            full_data_defects = torch.from_numpy(defect_configurations[sample_index]).to(device).view(-1,lattice_size,lattice_size)

            #print(full_data_defects[0])

            defect_number_tensor = torch.sum(torch.abs(full_data_defects).view(-1,lattice_size*lattice_size),dim=1)
            maximal_defect_nmb=torch.max(defect_number_tensor)



            name_list = ['full']

            oservable_data_spins = [[full_data_spins]]
            oservable_data_defects = [[full_data_defects]]

            



        elif data_type == 'random_data':

            path_name = 'random_data_comparison/'

            oservable_data_spins = []

            oservable_data_defects = []

            for idx, name in enumerate(name_list):
                
                oservable_data_spins.append([(torch.rand((samplesize,lattice_size,lattice_size)))*2.*np.pi])

                oservable_data_defects.append([full_defect_list[idx]])

        elif data_type == 'zero_temp':

            path_name = 'zero_temperature_solution_comparison_new/'

            oservable_data_spins = []

            oservable_data_defects = []

            lattice_spacing = 1

            spin_positions = np.array([[[i*lattice_spacing,j*lattice_spacing] for j in range(lattice_size)] for i in range(lattice_size)])

            for input_defects in full_defect_list:

                analys_sol_per_defects = []

                for defects in input_defects:
                    analy_spins = torch.from_numpy(analysitcally_calculation(spin_position=spin_positions,defects=defects.cpu().numpy(),lattice_size=lattice_size)).to(device)
                    analys_sol_per_defects.append(analy_spins)
                oservable_data_spins.append([torch.stack(analys_sol_per_defects,dim=0)])
                oservable_data_defects.append([input_defects])

        print('finish load')
        return oservable_data_spins, oservable_data_defects, path_name, name_list
    
def analysitcally_calculation(spin_position,defects,lattice_size):

    vortex_numb = vortex_number(defects,lattice_size)

    #print(vortex_number)

    defect_positions = defect_position(defects = defects,number_of_defects=vortex_numb, lattice_size= lattice_size)

    #print(defect_positions)

    analytic_solution_list = []

    #print(spin_position.shape)

    #analytic_solution = np.zeros([lattice_size,lattice_size])

    for i in range(lattice_size):#should be x direction
        for j in range(lattice_size):#should be y direction
            #print(defect_positions[v,2])

            #print(defect_positions[0,1].shape)

            #print(spin_position[i,j,1].shape)



            analystic_angles = [(defect_positions[v,2])*np.arctan2(np.asarray(spin_position[i,j,1]-(defect_positions[v,1]+0.5)), np.asarray(spin_position[i,j,0]-(defect_positions[v,0]+0.5))) for v in range(vortex_numb)]

            #print(analystic_angles)

            analytic_angle = np.sum(analystic_angles)

            analytic_solution_list.append(analytic_angle)

    analytic_solution = np.asarray(analytic_solution_list).reshape(lattice_size,lattice_size)
    #print(analytic_solution)

    return analytic_solution

def vortex_number(defects,lattice_size):
    """
    Return the number of vortices in the lattice
    """

    defect_array = defects.reshape(lattice_size*lattice_size)
    vortex_counter = 0
    for vortex in defect_array:
        if vortex == 1 or vortex == -1:
            vortex_counter += 1

    return vortex_counter

def defect_position(defects,number_of_defects, lattice_size):
    """
    Return the position of the defects and their value in a vector form
    Return shape: number_of_defects x 3 ( x position, y position, defect value)
    """

    position_vector = np.zeros((number_of_defects,3))
    counter = 0


    for i in range(lattice_size):
        for j in range(lattice_size):
            if defects[i,j] != 0:
                position_vector[counter,0] += i
                position_vector[counter,1] += j
                position_vector[counter,2] += np.copy(defects[i,j])
                counter +=1

    return position_vector

#TODO: not used
def analysitcally_calculation_tensor(spin_position,defects,lattice_size):#tensor

    vortex_numb = torch.sum(torch.abs(defects.view(lattice_size*lattice_size)),dim=0)

    #print(vortex_numb)

    #defect_positions = defect_position(defects = defects,number_of_defects=vortex_numb, lattice_size= lattice_size)

    analytic_solution_list = []

    #print(spin_position.shape)

    defect_pos_tensor = torch.nonzero(defects,as_tuple=True)

    #print(defect_pos_tensor[1][0].item())
    #print(spin_position[:,:,1])
    #print(defects[defect_pos_tensor[0][0],defect_pos_tensor[1][0]])

    analytic_solution = sum([defects[defect_pos_tensor[0][v],defect_pos_tensor[1][v]]*torch.arctan2(spin_position[:,:,1].float()-(defect_pos_tensor[1][v]+0.5),spin_position[:,:,0]-(defect_pos_tensor[0][v]+0.5)) for v in range(int(vortex_numb.item()))])

    print(analytic_solution)

    #torch.arctan2(np.asarray(spin_position[:,:,1]-(defect_positions[v,1]+0.5)), np.asarray(spin_position[i,j,0]-(defect_positions[v,0]+0.5))



    #analytic_solution = np.zeros([lattice_size,lattice_size])

    ##for i in range(lattice_size):#should be x direction
    ##    for j in range(lattice_size):#should be y direction

    ##        analystic_angles = [(defect_positions[v,2])*np.arctan2(np.asarray(spin_position[i,j,1]-(defect_positions[v,1]+0.5)), np.asarray(spin_position[i,j,0]-(defect_positions[v,0]+0.5))) for v in range(vortex_numb)]

     ##       analytic_angle = np.sum(analystic_angles)

     ##       analytic_solution_list.append(analytic_angle)

    ##analytic_solution = np.asarray(analytic_solution_list).reshape(lattice_size,lattice_size)

    return analytic_solution

def save_data_basis(output_data_path,path_name,correspodning_data_outname,data_type,name_list,epochs,samplesize,observable_data_spins,observable_data_defects,lattice_size,new_data=True):
    #Open h5py file and save data
    with h5py.File(output_data_path+path_name +correspodning_data_outname,'w') as h5f:
        if data_type != 'full_data':
            dataset_distances = h5f.create_dataset('distances', shape=(int(len(name_list)),), dtype='float')
            if new_data:
                dataset_distances[:] = np.array([dist for dist in name_list])#
            else :
                dataset_distances[:] = np.array([dist for dist in name_list])

        for dist_idx,dist in enumerate(name_list):

            #Create h5f datasets
            dataset_spins_compare = h5f.create_dataset('spin_lattices_{}'.format(dist), shape=(int(len(epochs)),samplesize,lattice_size*lattice_size), dtype='float')
            dataset_defects_compare = h5f.create_dataset('defects_lattices_{}'.format(dist), shape=(int(len(epochs)),samplesize,lattice_size*lattice_size), dtype='float')

            for e_dix,epoch in enumerate(epochs):
                #Write data to disk
                dataset_spins_compare[e_dix] = observable_data_spins[dist_idx][e_dix].view(1,-1,lattice_size*lattice_size).cpu().numpy()
                dataset_defects_compare[e_dix] = observable_data_defects[dist_idx][e_dix].view(1,-1,lattice_size*lattice_size).cpu().numpy()



def check_defect_type_config_small(simulated_data_path,temperature, lattice_size,device='cpu'):
    with torch.no_grad():

        

        with h5py.File(simulated_data_path+'full_training_data_gan_{}_corrected.h5'.format(temperature), 'r') as h5f:
            sim_spins = torch.from_numpy(h5f['spins'][350000:430000]).to(device).view(-1,lattice_size,lattice_size)
            sim_defects = torch.from_numpy(h5f['defects'][350000:430000]).to(device).view(-1,lattice_size,lattice_size)


        print('check input')

        
        

        defect_number_tensor = torch.sum(torch.abs(sim_defects).view(-1,lattice_size*lattice_size),dim=1)
        maximal_defect_nmb=torch.max(defect_number_tensor)
        ##print(maximal_defect_nmb)
        unique_defect_lattices = torch.unique(sim_defects, dim=0)
        ##print(unique_defect_lattices.size())
        unique_defect_numbers = torch.unique(defect_number_tensor,dim=0)

        topological_analysis_tool = TopologicalAnalysis(lattice_size=lattice_size,device=device)


        topological_analysis_tool.set_full_defect_positions(defect_lattices=sim_defects,maximal_defect_nmb=maximal_defect_nmb)
        defect_distances = topological_analysis_tool.return_full_defect_distances()

        ##make  uniform distributed data of the following type:
        no_distances = defect_distances== 0.0
        small_distances = torch.logical_and(defect_distances > 0.0 ,defect_distances <= 2.0 )
        mid_distances = torch.logical_and(defect_distances > 2.0 ,defect_distances < 5.0 )
        large_distances = defect_distances == 5.0 

        full_sim_data_index = [i for i in range(int(sim_defects.size(0)))]
        no_defect_sim_data_index = [i for i in range(int(sim_defects[no_distances].size(0)))]
        small_defect_sim_data_index = [i for i in range(int(sim_defects[small_distances].size(0)))]
        mid_defect_sim_data_index = [i for i in range(int(sim_defects[mid_distances].size(0)))]
        large_defect_sim_data_index = [i for i in range(int(sim_defects[large_distances].size(0)))]

        no_spins = sim_spins[no_distances]
        small_spins = sim_spins[small_distances]
        mid_spins = sim_spins[mid_distances]
        large_spins = sim_spins[large_distances]

        no_defect_spins = sim_defects[no_distances]
        small_defect_spins = sim_defects[small_distances]
        mid_defect_spins = sim_defects[mid_distances]
        large_defect_spins = sim_defects[large_distances]

        ##print('no', sim_defects[no_distances].size())
        ##print('small',sim_defects[small_distances].size())
        ##print('mid',sim_defects[mid_distances].size())
        ##print('large',sim_defects[large_distances].size())



        ##small_def = sim_defects[small_distances]
        ##unique_small_defect_lattices = torch.unique(small_def, dim=0)
        ##print('small unique',unique_small_defect_lattices.size())

        ##mid_def = sim_defects[mid_distances]
        ##unique_mid_defect_lattices = torch.unique(mid_def, dim=0)
        ##print('mid unique',unique_mid_defect_lattices.size())

        ##large_def = sim_defects[large_distances]
        ##unique_large_defect_lattices = torch.unique(large_def, dim=0)
        ##print('large unique',unique_large_defect_lattices.size())




        ##counting = torch.unique(defect_distances,return_counts= True)
        ##print(defect_distances)
        ##print(counting)

        ##here comes the random sampling :

        full_sampling_idx = random.sample(full_sim_data_index,10000)
        no_defect_sampling_idx = random.sample(no_defect_sim_data_index,1000)
        small_defect_sampling_idx = random.sample(small_defect_sim_data_index,1000)
        mid_defect_sampling_idx = random.sample(mid_defect_sim_data_index,1000)
        large_defect_sampling_idx = random.sample(large_defect_sim_data_index,1000)

        full_sample_spin = sim_spins[full_sampling_idx]
        full_sample_defect = sim_defects[full_sampling_idx]
        no_defects_sample_spin = no_spins[no_defect_sampling_idx]
        no_defects_sample_defect = no_defect_spins[no_defect_sampling_idx]
        small_defects_sample_spin = small_spins[small_defect_sampling_idx]
        small_defects_sample_defect = small_defect_spins[small_defect_sampling_idx]
        mid_defects_sample_spin = mid_spins[mid_defect_sampling_idx]
        mid_defects_sample_defect = mid_defect_spins[mid_defect_sampling_idx]
        large_defects_sample_spin = large_spins[large_defect_sampling_idx]
        large_defects_sample_defect = large_defect_spins[large_defect_sampling_idx]

        print(no_defects_sample_spin.size())

        full_spin_list = [full_sample_spin,no_defects_sample_spin,small_defects_sample_spin,mid_defects_sample_spin,large_defects_sample_spin]

        full_defect_list = [full_sample_defect,no_defects_sample_defect,small_defects_sample_defect,mid_defects_sample_defect,large_defects_sample_defect]

        return full_spin_list, full_defect_list
        #random.sample(,1000)
