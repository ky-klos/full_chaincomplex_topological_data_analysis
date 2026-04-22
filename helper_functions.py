import numpy as np
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

two_pi = 2.0*np.pi

from collections import Counter


def transformation_pos(defects, spins, distances,lattice_size=16,defect_nmb=2):#defect and spins should be lists
    
    transformed_spins = []
    transformed_defects = []
    
    topological_tool = TopologicalAnalysis(lattice_size=lattice_size,device='cpu')
    
    changed_distances = np.zeros(int(len(distances)))
    

    
    for d_idx, dist in enumerate(distances):
        
        topological_tool.set_full_defect_positions(defect_lattices=torch.from_numpy(defects[d_idx]),maximal_defect_nmb=defect_nmb,individual=True)
        full_defect_distance_x,full_defect_distance_y = topological_tool.get_defect_distance_xy()
        
        spins_per_distance = []
        defects_per_distance = []
        if torch.any(full_defect_distance_x==full_defect_distance_y) or torch.any(full_defect_distance_x==full_defect_distance_y*(-1.)):#fulldiag
            if torch.sum(full_defect_distance_x==full_defect_distance_y) +torch.sum(full_defect_distance_x==full_defect_distance_y*(-1.))>= 500:
                
                changed_distances[d_idx] = 1.
                
                spins_per_distance.append(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x>0.0)])
                defects_per_distance.append(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x>0.0)])
                
                spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x<0.0)],k=2,dims=[1,2]))
                defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y,full_defect_distance_x<0.0)],k=2,dims=[1,2]))
                
                spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_x<0.0)],k=1,dims=[1,2]))
                defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_x<0.0)],k=1,dims=[1,2]))
                
                spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_y<0.0)],k=3,dims=[1,2]))
                defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==full_defect_distance_y*(-1.),full_defect_distance_y<0.0)],k=3,dims=[1,2]))
                #for case: full_defect_distance_x==full_defect_distance_y*(-1)
                
                
            else:
                
                raise ValueError('Not enough data for that')

        if torch.any(full_defect_distance_x==0.) or torch.any(full_defect_distance_y==0.):#horizontal/vertical
            
            changed_distances[d_idx] = 1.
            
            spins_per_distance.append(spins[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x>=0.)])
            defects_per_distance.append(defects[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x>=0.)])
            spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x<0.)],k=2,dims=[1,2]))
            defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_y==0.,full_defect_distance_x<0.)],k=2,dims=[1,2]))
            spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y>0.)],k=1,dims=[1,2]))
            defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y>0.)],k=1,dims=[1,2]))
            spins_per_distance.append(torch.rot90(spins[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y<0.)],k=3,dims=[1,2]))
            defects_per_distance.append(torch.rot90(defects[d_idx][torch.logical_and(full_defect_distance_x==0.,full_defect_distance_y<0.)],k=3,dims=[1,2]))
            
        transformed_spins.append(torch.stack(spins_per_distance))
        transformed_defects.append(torch.stack(defects_per_distance))
        
    return transformed_spins,transformed_defects,changed_distances
  