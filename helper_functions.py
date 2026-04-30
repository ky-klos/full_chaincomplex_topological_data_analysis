import numpy as np
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

two_pi = 2.0*np.pi

from collections import Counter
def mean_squared_error(mean, var_tensor, sample_nmb):

    sigma_squared = np.sum(np.power((var_tensor-mean),2))/(sample_nmb-1.0)
    mean_squared_error = np.sqrt(sigma_squared)/np.sqrt(sample_nmb)
    return mean_squared_error
    
    
def faster_error_calulcation_binder(full_magentization,full_binder_values, sample_nmb=9,sample_size=100): 

    error_index = [i for i in range(0,int(full_magentization.shape[0]))]


    binder_error_list = []

    for sample in range(sample_nmb):
        observables_indices = random.sample(error_index,sample_size)
        current_magent = full_magentization[observables_indices]

        current_binder = 1-np.mean(current_magent**4)/(3*(np.mean(current_magent**2)**2))
        binder_error_list.append(current_binder)


    binder_error = mean_squared_error(mean=full_binder_values, var_tensor=np.stack(binder_error_list), sample_nmb=sample_nmb)

       


    return binder_error

def heatmap_error(binder_acc,temperatures,names):

    import seaborn as sb
    mpl.rcParams['ytick.labelsize'] = 40
    mpl.rcParams['xtick.labelsize'] = 40


    fig, axs = plt.subplots(1,figsize=(20, 15*4+5), dpi=250)
    full_error = binder_acc.reshape(int(len(temperatures)),-1) 

    if np.min(full_error) < 0.:

        fmt='.1e'
    else:
        fmt='.2f'

    cbar = True
    ax = sb.heatmap(full_error, 
                        cmap='Spectral',  # Colormap
                        annot=True,  # Display values on the heatmap
                        cbar=cbar,
                        fmt=fmt,  # Format of the values
                        linewidths=0.5,  # Width of the lines between cells
                        linecolor='white',  # Color of the lines between cells
                        cbar_kws={'label': 'Agreement Score','location':'top'},
                        vmin=0, vmax=1,
                        xticklabels =[],
                        yticklabels =temperatures,
                        annot_kws = {'fontsize':20},
                        ax=axs
                        )

    axs.title.set_text('U')
    axs.set_ylabel('Temperatures '+ r'$\mathbf{T}$')
    xtickslabels = np.round(names,2)
    axs.set_xlabel('Defect Pair Distances '+ r'$\mathbf{d_{\nu}}$')
    axs.set_xticks(range(len(xtickslabels)), labels=xtickslabels,
              rotation=45)
    plt.tight_layout()
    plt.savefig('heatmap_binder.pdf') 
    plt.close()

def acc(binder_train_list, binder_gen_list):
    #acc per tempereature
    accurarcy = 1. - np.sqrt(np.power(binder_train_list-binder_gen_list,2)/np.power((np.max(binder_train_list)-np.min(binder_train_list)),2))
    return accurarcy   

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
  
