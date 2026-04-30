import numpy as np
import h5py
import pandas as pd
import random
import seaborn as sb
import torch
from torch.autograd import Variable
import torch.nn as nn
from wgan_loss_function_jan_24 import XY_loss_function
import matplotlib.pyplot as plt
def check_vortex_error(simulated_data_path,temp,lattice_size = 16,defect_nmb=2,samplesize=1000):

    extra_loss = XY_loss_function(lattice_size=lattice_size, device= 'cpu')
    extra_loss.helper_plaquette()
    with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temp,lattice_size,defect_nmb,samplesize), 'r') as h5f:

        name_list = np.array(h5f['distances'])

    spin_list = []
    defect_list = []
    vortex_loss_list = []
    for dist_idx, dist in enumerate(name_list):


        with h5py.File(simulated_data_path+'defect_for_analysis_full_distances_{}_latticesize_{}_defect_nmb_{}_sample_{}.h5'.format(temp,lattice_size,defect_nmb,samplesize), 'r') as h5f:

            sim_spins = torch.from_numpy(np.array(h5f['spin_lattices_{}'.format(dist)][:])).view(-1,lattice_size,lattice_size)
            sim_basis_defects = torch.from_numpy(np.array(h5f['defects_lattices_{}'.format(dist)][:])).view(-1,lattice_size,lattice_size)

            spin_list.append(sim_spins.view(-1,lattice_size,lattice_size).cpu().numpy())
            defect_list.append(sim_basis_defects.view(-1,lattice_size,lattice_size).cpu().numpy())

            #make sim_spins to vector field
            sim_spin_vector_field = torch.stack([torch.cos(sim_spins.view(-1,1,lattice_size,lattice_size)),torch.sin(sim_spins.view(-1,1,lattice_size,lattice_size))],dim=1).view(-1,2,lattice_size,lattice_size)

            vortex_loss_diff = extra_loss.vortex_loss_check(generated_spins=sim_spin_vector_field, defect_config=sim_basis_defects.view(-1,1,lattice_size,lattice_size))

            vortex_loss_list.append(torch.mean(vortex_loss_diff.view(-1,lattice_size*lattice_size),dim=1).cpu().numpy())

    return name_list,vortex_loss_list

def plot_distance(simulated_data_path,temps,lattice_size = 16,defect_nmb=2,samplesize=1000):
    full_vortex_loss_list = []
    vortex_dataset_list = []
    for temp in temps:

        name_list,vortex_loss_list = check_vortex_error(simulated_data_path,temp,lattice_size,defect_nmb,samplesize)
        full_vortex_loss_list.append(vortex_loss_list)
        dataset_vortex = pd.DataFrame(np.array(vortex_loss_list).reshape(-1,samplesize)).T
        dataset_vortex_melt = dataset_vortex.melt(var_name='nmb_distances',value_name='Vortex_loss')
        dataset_vortex_melt['Temp'] = temp
        dataset_vortex_melt['distances'] = np.repeat(name_list.reshape(-1,1),axis=1,repeats=samplesize).flatten()
        vortex_dataset_list.append(dataset_vortex_melt)
    

    full_data_vortex_loss = pd.concat(vortex_dataset_list)
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    for idx, temp in enumerate(temps):
        axs.plot(name_list,np.mean(np.array(full_vortex_loss_list[idx]),axis=1),'o-',label='Temp {}'.format(temp))

        axs.set_ylabel(r'$\Delta \nu$')
    axs.set_xlabel(r'$d_{\nu}$')
    axs.legend()
    plt.savefig(simulated_data_path+ "vortex_loss_accuracy.pdf", dpi=300)
    print(np.min(np.array(full_vortex_loss_list)),np.max(np.array(full_vortex_loss_list)))
    df_vortex = pd.DataFrame.from_dict({'Vortex_diff_temp_{}'.format(t): np.mean(np.array(full_vortex_loss_list[idx]),axis=1) for idx,t in enumerate(temps)})
    df_vortex.to_csv('vortex_loss_accuracy.csv')
    

plot_distance(simulated_data_path='./',temps=[0.1],lattice_size = 16,defect_nmb=2,samplesize=1000)