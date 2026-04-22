

from filtration_function import TopologicalAnalysis

import torch

import h5py
import time 

import pandas as pd

from basis_data_production import datapath_generation

import numpy as np

from numba import jit
from numba.typed import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)


@jit(nopython=True)
def minkowski_with_edges_numba(cluster_list):

    #liste mit anzahl 3,2,1 in dim 2 von unterliste
    #now also with 4
    result_before = List()
    result = List()
    for arr in (cluster_list):
        d = 0
        
        
        result_before.append(np.sum(np.logical_and(arr[:,2]>=2,arr[:,2]<4))-np.sum(arr[:,2]==1))
        result.append(np.sum(arr[:,2]>=2)-np.sum(arr[:,2]==1))

    return result_before,result


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

                    
def get_topological_measures_mean(temperature,extra_idx, output_data_path,configuration_attributes_dict,training_data_nmb, simulated_data_path, device, less_index,epsilon_size=2000,sim_data_type='full_pinned',basis_sample_size=1000,number_defects=2,noise_size=1.0,samplesize=100,lattice_size=16,fixed_data=True,data_type_list = ['training_data','generated_data','random_data','zero_temp'], name_idxs = None, epoch=-1):

    
    if fixed_data:
        
        before_sim_load = time.time()
         
        if sim_data_type == 'full_pinned':
            if number_defects > 2 or lattice_size > 16 or temperature > 0.1:
                with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,number_defects,basis_sample_size), 'r') as h5f:
                    name_list = np.array(h5f['distances'])

            else:
                with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temperature,lattice_size,number_defects,basis_sample_size), 'r') as h5f:
                    name_list = np.array(h5f['distances'])
        else:
            with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(sim_data_type,temperature,lattice_size,number_defects,basis_sample_size), 'r') as h5f:
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
            path = datapath_generation(configuration_attributes_dict,file_type='analysis',analysis_config = [noise_size,training_data_nmb])
                
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


            with h5py.File(output_data_path+ path_name +'{}_{}_Plaquette_edge_graph_rep_before_epsilon_comparison_{}_temp_{}_nmb_defects_{}_latticesize_{}_epoch_{}.h5'.format(name,sim_data_type,data_type,temperature,number_defects,lattice_size,epoch),'r') as h5f:
                plaquettes = torch.from_numpy(np.array(h5f['Plaquette'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]

                edges_x = torch.from_numpy(np.array(h5f['Edge x'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
                edges_y = torch.from_numpy(np.array(h5f['Edge y'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
                defects = torch.from_numpy(np.array(h5f['Defects'])).view(-1,lattice_size,lattice_size).to(torch.float32)[less_index]
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

            prev_plaq = None
            prev_edge = None

            dictionary_persistence_objects = {}

            dictionary_persistence_holes = {}

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

                edges_x_per_epsilon = full_list_edges_x[n_idx][data_idx] <= (angular_filt*epsilon)

                edges_y_per_epsilon = full_list_edges_y[n_idx][data_idx] <= (angular_filt*epsilon)

                full_state_percolation, full_bond_percolation = topological_analysis_tool.percolation_faster(plaquettes= plaquette_per_epsilon.view(-1,lattice_size*lattice_size), edges_x=edges_x_per_epsilon.view(-1,lattice_size*lattice_size), edges_y=edges_y_per_epsilon.view(-1,lattice_size*lattice_size))
                plq_percolation_list.append((full_state_percolation).cpu())
                edge_percolation_list.append((full_bond_percolation).cpu())

                full_edge_matrix_per_epsilon = topological_analysis_tool.make_full_edge_matrix_with_definite_holes(epsilon=0.00175*epsilon, edge_x=full_list_edges_x[n_idx][data_idx].view(-1,lattice_size*lattice_size), edge_y=full_list_edges_y[n_idx][data_idx].view(-1,lattice_size*lattice_size),plaq=full_list_plaquettes[n_idx][data_idx].view(-1,lattice_size*lattice_size))
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


                            mean_cluster_euler_plq_list_per_epsilon = []
                            mean_cluster_area_plq_list_per_epsilon = []
                            mean_cluster_perimeter_plq_list_per_epsilon = []
                            


                            if int(len(cluster_list)) > 0:



                                for cluster in cluster_list :
                                    cluster_tensor = torch.tensor(cluster, device=device)
                                    cluster_lattices_zero = torch.zeros(lattice_size,lattice_size, device=device)
                                    cluster_lattices_zero[cluster_tensor[:,0],cluster_tensor[:,1]]=1
                                    euler_nmb = topological_analysis_tool.euler_charactistics_faster(cluster_lattices_zero)
                                    mean_cluster_euler_plq_list_per_epsilon.append(torch.tensor([euler_nmb],dtype=float,device=device))

                                    area_clusters, perimeter_clusters = topological_analysis_tool.minkowski_measure_new_faster(cluster_lattices_zero.view(1,lattice_size,lattice_size))
                                    mean_cluster_area_plq_list_per_epsilon.append(area_clusters)
                                    mean_cluster_perimeter_plq_list_per_epsilon.append(perimeter_clusters)

                                full_euler_plq_list.append(torch.flatten(torch.sum(torch.stack(mean_cluster_euler_plq_list_per_epsilon))))
                                mean_cluster_euler_plq_list.append(torch.flatten(torch.mean(torch.stack(mean_cluster_euler_plq_list_per_epsilon))))
                                full_cluster_nmb_plq_list.append(torch.flatten(torch.tensor([float(len(cluster_list))],device=device)))
                                per_epsilon_cluster_plq_area[idx] = torch.stack(mean_cluster_area_plq_list_per_epsilon)
                                per_epsilon_cluster_plq_perimeter[idx] = torch.stack(mean_cluster_perimeter_plq_list_per_epsilon)
                                per_epsilon_cluster_plq_euler[idx] = torch.stack(mean_cluster_euler_plq_list_per_epsilon)


                                
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

                        full_diameter_list_per_epsilon = []
                        full_radius_list_per_epsilon = []

                        hole_counter = []

                        vertex_in_cluster_counter = []

                        if int(len(cluster_edge_list)) > 0:
                            full_euler_list_per_epsilon,full_euler_list_per_epsilon_correct = minkowski_with_edges_numba([np.array(c) for c in cluster_edge_list])

                            cluster_lattices_list = []
                            for cluster_idx,cluster in enumerate(cluster_edge_list):
                                
                                clusterHash = str(cluster)
                                if clusterHash not in cluster_calc_dict:
                                    cluster_calc_dict[clusterHash] = topological_analysis_tool.diameter_radius_graph(cluster)
                                diameter,radius = cluster_calc_dict[clusterHash]
                                
                                full_diameter_list_per_epsilon.append(torch.tensor([diameter],dtype=float,device=device))
                                full_radius_list_per_epsilon.append(torch.tensor([radius],dtype=float,device=device))

                                cluster_tensor = torch.tensor(cluster)

                                hole_counter.append((cluster_tensor[:,2]==4).sum())
                                vertex_in_cluster_counter.append((cluster_tensor[:,2]==2).sum())


                            full_euler_edge_list.append(torch.flatten(torch.mean(torch.tensor(full_euler_list_per_epsilon).float())))
                            full_euler_edge_list_correct.append(torch.flatten(torch.mean(torch.tensor(full_euler_list_per_epsilon_correct).float())))
                            full_diameter_edge_list.append(torch.flatten(torch.mean(torch.stack(full_diameter_list_per_epsilon).float())))
                            full_radius_edge_list.append(torch.flatten(torch.mean(torch.stack(full_radius_list_per_epsilon).float())))
                            # euler charactitics via betti numbers, should give same as all vertices - all edges + all faces
                            objects_nmb = (lattice_size**2-sum(vertex_in_cluster_counter)) + int(len(cluster_edge_list)) 
                            
                            betti_number_euler_charactistic_per_epsilon.append(torch.flatten(torch.tensor(objects_nmb - sum(hole_counter),device=device)).float())
                            per_epsilon_cluster_edge_euler[idx] =torch.tensor(full_euler_list_per_epsilon_correct).float()
                            per_epsilon_cluster_edge_diameter[idx] = torch.stack(full_diameter_list_per_epsilon).float()
                            per_epsilon_cluster_edge_radius[idx] = torch.stack(full_radius_list_per_epsilon).float()
                            
                        else:

                            full_euler_edge_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                            full_diameter_edge_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                            full_radius_edge_list.append(torch.flatten(torch.tensor([0.0],device=device)))
                            full_euler_edge_list_correct.append(torch.flatten(torch.tensor([0.0],device=device)))
                            per_epsilon_cluster_edge_euler[idx] =torch.flatten(torch.tensor([0.0],device=device))
                            per_epsilon_cluster_edge_diameter[idx] = torch.flatten(torch.tensor([0.0],device=device))
                            per_epsilon_cluster_edge_radius[idx] = torch.flatten(torch.tensor([0.0],device=device))

                            betti_number_euler_charactistic_per_epsilon.append(torch.flatten(torch.tensor([lattice_size**2],device=device)).float())


                    edge_epsilon_euler_list.append((torch.stack(full_euler_edge_list, dim =0)).cpu())
                    edge_epsilon_euler_list_correct.append((torch.stack(full_euler_edge_list_correct, dim =0)).cpu())

                    edge_epsilon_diameter_list.append((torch.stack(full_diameter_edge_list, dim =0)).cpu())
                    edge_epsilon_radius_list.append((torch.stack(full_radius_edge_list, dim =0)).cpu())
                    edge_cluster_nmb_list.append((torch.stack(full_cluster_nmb_edge_list, dim =0)).cpu())

                    full_cluster_edge_euler[epsilon] =per_epsilon_cluster_edge_euler
                    full_cluster_edge_diameter[epsilon] = per_epsilon_cluster_edge_diameter
                    full_cluster_edge_radius[epsilon] = per_epsilon_cluster_edge_radius



                    betti_number_euler_charactistic.append((torch.stack(betti_number_euler_charactistic_per_epsilon, dim =0)).cpu())

                else:
                    edge_epsilon_euler_list.append(edge_epsilon_euler_list[-1])
                    edge_epsilon_euler_list_correct.append(edge_epsilon_euler_list_correct[-1])
                    edge_epsilon_diameter_list.append(edge_epsilon_diameter_list[-1])
                    edge_epsilon_radius_list.append(edge_epsilon_radius_list[-1])
                    edge_cluster_nmb_list.append(edge_cluster_nmb_list[-1])

                    full_cluster_edge_euler[epsilon] = full_cluster_edge_euler[epsilon-1]
                    full_cluster_edge_diameter[epsilon] = full_cluster_edge_diameter[epsilon-1]
                    full_cluster_edge_radius[epsilon] = full_cluster_edge_radius[epsilon-1]
                   
                    betti_number_euler_charactistic.append(betti_number_euler_charactistic[-1])

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
                                'plq_percol':torch.mean(torch.stack(plq_percolation_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'edge_percol':torch.mean(torch.stack(edge_percolation_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'full_euler_new_wo_faces':torch.mean(torch.stack(edge_epsilon_euler_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'full_euler_old_w_faces':torch.mean(torch.stack(edge_epsilon_euler_list_correct,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'bettinumber_euler':torch.mean(torch.stack(betti_number_euler_charactistic,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'full_diameter':torch.mean(torch.stack(edge_epsilon_diameter_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(),
                                'full_radius':torch.mean(torch.stack(edge_epsilon_radius_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy(), 
                                'full_nmb_edge_clusters':torch.mean(torch.stack(edge_cluster_nmb_list,dim=0).view(epsilon_size,samplesize),dim=1).cpu().numpy()}
            
            df_tda_measurements = pd.DataFrame.from_dict(tda_measurements)

          
            df_tda_measurements.to_csv(output_data_path+path_name_list[data_idx]+full_output_names[n_idx][data_idx],index=False)

           
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


if __name__ == '__main__':  
    import argparse 
    import random
    lattice_size = 16
    generated_data_atributes_list=[
         ['full',10,'adam',(0.9, 0.999), 40, 1.0, 10, 'sum', 'conv_transposed', 0, 'leaky_Relu',0.01,None,0.0,0.0,'leaky_Relu',0.01,0.0001,0.0001,0.0001,0.0001, (0.0, 0.0), 4, 64, 0, 'L2', (100.0 ,0.0)]]

    temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    extra_infos = [[None, 0, 0.5, False,'{}_avg'.format(5)]]#
    def fixed_idx(basic_sample_size, reduced_samplesize, temperature, new=True):
        if new :
            less_index = random.sample([i for i in range(basic_sample_size)],reduced_samplesize)
            with open('topological_idx_{}_{}.txt'.format(reduced_samplesize,temperature), 'w') as f:
                for i in less_index:
                    f.write(f"{i}\n")
            return less_index
        else:
            less_idx = []
            file =open('topological_idx_{}_{}.txt'.format(reduced_samplesize,temperature))
            for i in range(reduced_samplesize):
                less_idx.append(int(file.readline()[:-1]))
                
            print(less_idx)
            return less_idx
    parser = argparse.ArgumentParser(description='Process some integers.')


    parser.add_argument('--name_idxs', type=int, nargs='+', help='an integer for name idxs to calculate')
    args = parser.parse_args()
    
    for att in generated_data_atributes_list:
    
        for temp in temps:
            less_index = fixed_idx(basic_sample_size=1000, reduced_samplesize= 100, temperature = temp, new=False)
            get_topological_measures_mean(extra_idx = 0,less_index =less_index,res_criritc= True,extra_info = extra_infos[0],output_data_path='./observables_output/new/',generated_data_attributes=att,basis_sample_size=1000,number_defects=2,noise_size = 1.0, training_data_nmb=350000,simulated_data_path='./observables_output/new/', device='cuda', temperature=temp,samplesize=100,lattice_size=lattice_size,fixed_data=True,data_type_list = ['generated_data'], name_idxs = args.name_idxs)


