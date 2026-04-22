import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

import sys

from collections import OrderedDict
from skimage.measure import euler_number

import copy
two_pi = 2*np.pi
lattice_spacing = 1


np.set_printoptions(threshold=sys.maxsize)

class TopologicalAnalysis():
    def __init__(self,device,lattice_size=5, epsilon= 0.0,batchsize=100):

        

        self.device = device
        torch.set_default_device(self.device)
        self.lattice_spacing = 1
        self.lattice_size = lattice_size
        self.num_spins = lattice_size**2
        self.saved_L = 1
        self.saved_N = 1
        L,N = self.lattice_size,self.num_spins
        self.nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                    (i // L) * L + (i - 1) % L, (i - L) % N) \
                                            for i in list(range(N))}#right,down,left,up
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
        self.nn_y_direction = {i : ((i + L) % N)\
                                            for i in list(range(N))}#down, up: ,(i - L) % N

        self.nn_cluster = {i : ((i // L) * L + (i - 1) % L,(i - L) % N)\
                                            for i in list(range(N))}  #left nn, up
        
        self.nn_cluster_edge_h = {i : ((i // L) * L + (i - 1) % L,(i - L) % N)\
                                            for i in list(range(N))}  #left (edge x) up (edge y)
        
        self.nn_cluster_edge_v = {i : ((i // L) * L + (i - 1) % L,(i - L) % N)\
                                            for i in list(range(N))}  #same (edge x) up (edge y) left (edge x) diag up left (edge x)
        
        self.nn_cluster_boundary = {i : ((i // L) * L + (i + 1) % L,(i + L) % N)\
                                            for i in list(range(N))}  #right, down nn

        self.epsilon = epsilon

        self.clusters_per_epsilon = {}

        self.two_pi = 2*np.pi

        # these folowwing two dictionary will have indice in 2 dim lattice*2 (because of edge matrix) as key and array of birth depenent on epsilon each epsilon with depth of batch size

        #where are possible edges
        self.edge_ordering = {batch_idx: OrderedDict() for batch_idx in range(batchsize)}
        # where are possible plaquettes
        self.plaquette_ordering = {batch_idx: OrderedDict() for batch_idx in range(batchsize)}


    
    def defect_position_tensor(self,maximal_defect_number,input_defect_lattice= None):
        """
        Returns a tensor of defect positions and the average distance between defects
        Returns: defect_position_vector: Tensor of shape (maximal_defect_number,3) with each row being (x_position,y_position,defect_type)
                    defect_distance: average distance between closest defects (mean distance between min distance of vortices to each anti-vortices)
        """

        counter = 0

        if input_defect_lattice == None:
            defect_lattice = self.defect_config
        else:
            defect_lattice = input_defect_lattice

        defect_position_vector = torch.zeros(int(maximal_defect_number),3,device = self.device)

        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if defect_lattice[i,j] != 0:
                    defect_position_vector[counter,0] += i
                    defect_position_vector[counter,1] += j
                    defect_position_vector[counter,2] += torch.clone(defect_lattice[i,j])
                    counter +=1

        if counter == 0:
            defect_distance = torch.tensor(0)
           
        elif counter == 2:
            defect_distance = torch.sqrt(torch.pow((defect_position_vector[0,0]-defect_position_vector[1,0]),2) +torch.pow((defect_position_vector[0,1]-defect_position_vector[1,1]),2))

        else:
            vortices = defect_position_vector[defect_position_vector[:,2]== 1]
            anti_vortices = defect_position_vector[defect_position_vector[:,2]== -1]
            defect_distances_list = torch.zeros(int(counter/2))

     
            #for each anti-vortex find distance to closest vortex
            for c in range(int(counter/2)):
                defect_distances_list[c] = torch.min(torch.sqrt(torch.pow((vortices[:,0]-anti_vortices[c,0]),2) +torch.pow((vortices[:,1]-anti_vortices[c,1]),2)))

            defect_distance = torch.mean(defect_distances_list)
        
        return defect_position_vector, defect_distance
    

    def defect_position_tensor_simult(self,maximal_defect_number,input_defect_lattices= None):


        

        if input_defect_lattices == None:
            defect_lattices = self.defect_config
        else:
            defect_lattices = input_defect_lattices

        counter = torch.zeros(defect_lattices.size(0))

        defect_position_vector = torch.zeros(defect_lattices.size(0),int(maximal_defect_number),3,device = self.device)

        defect_distances_vector = torch.zeros(defect_lattices.size(0))


        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if torch.any(defect_lattices[:,i,j] != 0):

                    current_counter = counter[defect_lattices[:,i,j] != 0]
                    current_defects = defect_lattices[defect_lattices[:,i,j] != 0]

                    non_zero_index = torch.nonzero(defect_lattices[:,i,j] != 0)

                    

                    defect_position_vector[non_zero_index,current_counter,torch.ones(non_zero_index.shape())*0] += i
                    defect_position_vector[non_zero_index,current_counter,torch.ones(non_zero_index.shape())*1] += j
                    defect_position_vector[non_zero_index,current_counter,torch.ones(non_zero_index.shape())*2] += torch.clone(current_defects[:,i,j])
                    counter[defect_lattices[:,i,j] != 0] += 1


        if torch.any(counter == 2):
            defect_distances_vector[counter == 2] = torch.sqrt(torch.pow((defect_position_vector[counter == 2,0,0]-defect_position_vector[counter == 2,1,0]),2) +torch.pow((defect_position_vector[counter == 2,0,1]-defect_position_vector[counter == 2,1,1]),2))
        if torch.any(counter == 0):
            defect_distances_vector[counter == 2] = torch.zeros_like(counter[counter == 2])
        if torch.any(counter > 2):

            vortices = defect_position_vector[defect_position_vector[counter > 2,:,2]== 1]
            anti_vortices = defect_position_vector[defect_position_vector[counter > 2,:,2]== -1]

            defect_distances_vector[counter > 2]
            defect_distances_list = torch.zeros(int(counter/2))

           

            for c in range(int(counter/2)):


                defect_distances_list[c] = torch.min(torch.sqrt(torch.pow((vortices[:,0]-anti_vortices[c,0]),2) +torch.pow((vortices[:,1]-anti_vortices[c,1]),2)))

            defect_distance = torch.mean(defect_distances_list)
        

       

        return defect_position_vector, defect_distance
    
    def set_full_defect_positions(self,defect_lattices,maximal_defect_nmb=2,individual=False):
        full_defect_position_list = []
        defect_distance_list = []
        defect_distance_list_x = []
        defect_distance_list_y = []
        

        for defect_lattice in defect_lattices:
            defect_position,defect_distance = self.defect_position_tensor(maximal_defect_number=maximal_defect_nmb,input_defect_lattice = defect_lattice)
            if individual:
                vortices = defect_position[defect_position[:,2]== 1]
                anti_vortices = defect_position[defect_position[:,2]== -1]

                defect_distances_list_x = torch.zeros(int(maximal_defect_nmb/2))
                defect_distances_list_y = torch.zeros(int(maximal_defect_nmb/2))
                
               
                for c in range(int(maximal_defect_nmb/2)):
                    if int(vortices.size(0))>0:


                        defect_distances_list_y[c] = torch.min((vortices[:,0]-anti_vortices[c,0]))
                        defect_distances_list_x[c] = torch.min((vortices[:,1]-anti_vortices[c,1]))
                        
                    else:
                        defect_distances_list_y[c] = torch.tensor([0.])
                        defect_distances_list_x[c] = torch.tensor([0.])
                    

                defect_distance_list_x.append(torch.mean(defect_distances_list_x))
                defect_distance_list_y.append(torch.mean(defect_distances_list_y))

            full_defect_position_list.append(defect_position)
            defect_distance_list.append(defect_distance)


        self.full_defect_positions = torch.stack(full_defect_position_list,dim=0)
        self.full_defect_distance = torch.stack(defect_distance_list, dim =0)
        
        if individual:
            self.full_defect_distance_x = torch.stack(defect_distance_list_x, dim =0)
            self.full_defect_distance_y = torch.stack(defect_distance_list_y, dim =0)
        else:
            self.full_defect_distance_x = None
            self.full_defect_distance_y = None
    
    
    def return_full_defect_distances(self):
        return self.full_defect_distance
    

    def make_full_edge_matrix_with_definite_holes(self, epsilon, edge_x, edge_y, plaq,ordering=False):

        # 2 vertex, 1 edge, 4 hole , 3 filled hole

        full_edges = torch.zeros((edge_x.size(0),self.lattice_size*2,self.lattice_size*2))
        connection_constraint_x = edge_x<= epsilon
        connection_constraint_y = edge_y<= epsilon
        plaq_constraint = plaq <= epsilon
        
        

        for idx in range(self.lattice_size*self.lattice_size):
            
            if ordering:
                
                full_edges[:,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[:,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 2
                full_edges[:,((idx//self.lattice_size)*2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[:,((idx//self.lattice_size)*2+2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2

            
            if connection_constraint_x[:,idx].sum() > 0:
                connection_constraint_x_idx = connection_constraint_x[:,idx].nonzero()
                
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 1
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[connection_constraint_x_idx,(idx//self.lattice_size)*2,((idx%self.lattice_size)*2+2)%(self.lattice_size*2)] = 2

                
            if connection_constraint_y[:,idx].sum() > 0:
                connection_constraint_y_idx = connection_constraint_y[:,idx].nonzero()
                

                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2+2)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 2
                
                full_edges[connection_constraint_y_idx,((idx//self.lattice_size)*2+1)%(self.lattice_size*2),((idx%self.lattice_size)*2)%(self.lattice_size*2)] = 1

                
            if plaq_constraint[:,idx].sum() > 0:
                plaq_constraint_idx = plaq_constraint[:,idx].nonzero()
                full_edges[plaq_constraint_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 3


           

            if  torch.logical_and((connection_constraint_x[:,idx]>0),(connection_constraint_y[:,idx]>0)).sum() > 0:
                

                
                possible_hole = torch.logical_and(torch.logical_and((connection_constraint_x[:,idx]>0),(connection_constraint_y[:,idx]>0)),torch.logical_and((connection_constraint_x[:,(idx+self.lattice_size)%(self.lattice_size**2)] >0),(connection_constraint_y[:,(idx // self.lattice_size) * self.lattice_size + (idx + 1) % self.lattice_size]> 0)))
                
                
                if possible_hole.sum() > 0:
                    
                    plaq_hole_idx = torch.logical_and(plaq_constraint[:,idx] == 0,possible_hole).nonzero()

                    full_edges[plaq_hole_idx,(idx//self.lattice_size)*2+1,((idx%self.lattice_size)*2+1)%(self.lattice_size*2)] = 4


        return full_edges


    def simplices_ordering(self,full_edge_matrix_list,epsilon):
        

        self.full_edge_matrix_list = full_edge_matrix_list


        for idx in range(int(full_edge_matrix_list.size(0))):

            new_lattice_size = self.lattice_size*2



            edge_indices = torch.nonzero((full_edge_matrix_list[idx].flatten()==1))
            plq_indices = torch.nonzero((full_edge_matrix_list[idx].flatten()==3))#because other 3 is filled and  empty and here it is important when ist 3 born
            vertex_indices = torch.nonzero((full_edge_matrix_list[idx].flatten()==2))


            for edg_idx in edge_indices:

                
                if edg_idx.item() not in self.edge_ordering[idx].keys():
                    
                    if torch.sum(((edg_idx//new_lattice_size)*new_lattice_size + (edg_idx+1)%(new_lattice_size))==vertex_indices) == 1:
                        self.edge_ordering[idx][edg_idx.item()] = torch.tensor([epsilon,(edg_idx//new_lattice_size)*new_lattice_size + (edg_idx+1)%(new_lattice_size),(edg_idx//new_lattice_size)*new_lattice_size + (edg_idx-1)%(new_lattice_size)])#birth epsilon and boundary: here two vertices
                    else: #needs to be boundary of y edge
                        self.edge_ordering[idx][edg_idx.item()] = torch.tensor([epsilon,(edg_idx+new_lattice_size)%(new_lattice_size**2),(edg_idx-new_lattice_size)%(new_lattice_size**2)])#modulu should not be nessescary since new lattice was made with implicit bc
                else:
                    continue

            for plq_idx in plq_indices:
                if plq_idx.item() not in self.plaquette_ordering[idx].keys():

                    self.plaquette_ordering[idx][plq_idx.item()]= torch.tensor([epsilon,(plq_idx-1)%(new_lattice_size**2),(plq_idx+new_lattice_size)%(new_lattice_size**2),(plq_idx+1 - new_lattice_size*(((plq_idx+1)%(2*new_lattice_size))==0))%(new_lattice_size**2),(plq_idx-new_lattice_size)%(new_lattice_size**2)])#birth epsilon and boundary: here four edges


            self.vertex_indices = vertex_indices.flatten()
            
  
    def boundary_matrices(self):
        
        edge_boundary_matrix = torch.zeros((int(len(list(self.edge_ordering.keys()))),(int(torch.numel(self.vertex_indices))),int(len(list(self.edge_ordering[0].keys())))))#last dimenion shoul be maximum of edges ()

        plaq_boundary_matrix = torch.zeros((int(len(list(self.plaquette_ordering.keys()))),int(len(list(self.edge_ordering[0].keys()))),int(len(list(self.plaquette_ordering[0].keys())))))

        for batch_idx in self.edge_ordering.keys():
            edge_indices = list(self.edge_ordering[batch_idx].keys())#these are the edge positions 
            plq_indices = list(self.plaquette_ordering[batch_idx].keys())

            
            for e_idx,edge_matrix_indices in enumerate(edge_indices):

                boundary_indices = self.edge_ordering[batch_idx][edge_matrix_indices][1:]

                coresponding_matrix_indice = (torch.logical_or(self.vertex_indices == boundary_indices[0],self.vertex_indices == boundary_indices[1])).nonzero()

                edge_boundary_matrix[batch_idx,coresponding_matrix_indice,e_idx] = 1

            for p_idx,plaquette_matrix_indices in enumerate(plq_indices):
                boundary_indices = self.plaquette_ordering[batch_idx][plaquette_matrix_indices][1:]

                for b_idx in boundary_indices:

                    sub_b_idx = (torch.tensor(edge_indices)==b_idx).nonzero(as_tuple=True)

                    plaq_boundary_matrix[batch_idx,sub_b_idx,p_idx] = 1

        self.edge_boundary_matrix = edge_boundary_matrix
        self.plaq_boundary_matrix = plaq_boundary_matrix

        torch.set_printoptions(threshold=sys.maxsize)


    def low_idx(self, transposed_boundary_matrix, boundary, column=False):

        if boundary == 2 :
            low_indices = transposed_boundary_matrix.nonzero().view(-1,boundary,2)[:,-1,1]
            if column:
                low_cloumn = transposed_boundary_matrix.nonzero().view(-1,boundary,2)[:,-1,0]
        elif boundary == 4:

            nonzeros = transposed_boundary_matrix.nonzero()
            low_indices = [0 for i in range(transposed_boundary_matrix.size(0))]#this includes all 
            
            for (x,y) in nonzeros:
                low_indices[x]= max(low_indices[x],y)#wrong?

        else:
            raise ValueError('Boundary unkown')
        
        if column:
            return low_indices, low_cloumn
        else:

            return low_indices
            
        

    def lowest_one(self, boundary_matrix, boundary, output= False,reduction=True):

        #either max vaule of one or if everythin is zero 

        transposed_boundary_matrix = torch.t(boundary_matrix)

        if output:

                transposed_boundary_matrix[torch.all(torch.t(boundary_matrix)==0, dim =1),:2] = -1

                low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)


        else:

            if reduction:

                if torch.any(torch.all(torch.t(boundary_matrix)==0, dim =1)):

                    transposed_boundary_matrix[torch.all(torch.t(boundary_matrix)==0, dim =1),:2] = -1

                    low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)




                else:

                    low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)

            else:

                transposed_boundary_matrix[transposed_boundary_matrix==-1] = 0#should filter out zero rows
                
                low_indices = self.low_idx(transposed_boundary_matrix=transposed_boundary_matrix, boundary=boundary)


    
        return torch.tensor(low_indices)




    def reduced_boundary_matrices(self):


        temp_edge_boudnary_matrix = torch.clone(self.edge_boundary_matrix)
        temp_plaq_boundary_matrix  = torch.clone(self.plaq_boundary_matrix)

        


        for batch in range(int(temp_edge_boudnary_matrix.size(0))):

            temp_edge_low_indices = self.lowest_one(temp_edge_boudnary_matrix[batch],boundary=2)
            temp_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4)

           
            plq_check_unique, plq_check_counts = torch.unique(temp_plq_low_indices,return_counts=True)



            while torch.any(plq_check_counts>1):

                

                for j in range(int(temp_plaq_boundary_matrix.size(2))):#should be columns

                    i=0

                    
                    while i < j:
                        
                        if (temp_plq_low_indices[i] == temp_plq_low_indices[j]) and torch.logical_and(torch.all(temp_plaq_boundary_matrix[batch,:,j] != -1),torch.all(temp_plaq_boundary_matrix[batch,:,i] != -1)):
                            temp_plaq_boundary_matrix[batch,:,j]=((temp_plaq_boundary_matrix[batch,:,j]+temp_plaq_boundary_matrix[batch,:,i])==1).long()
                            
                            temp_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4)
                   
                    
                        i += 1

           

                check_plq_low_indices = self.lowest_one(temp_plaq_boundary_matrix[batch],boundary=4,reduction=False)
                
                plq_check_unique, plq_check_counts = torch.unique(check_plq_low_indices,return_counts=True)


            edge_check_unique, edge_check_counts = torch.unique(temp_edge_low_indices,return_counts=True)

           

            while torch.any(edge_check_counts>1):

                
                for j in range(int(temp_edge_boudnary_matrix.size(2))):#should be columns

                    i=0

                    while i < j:
                    
                        if (temp_edge_low_indices[i] == temp_edge_low_indices[j]) and torch.logical_and(torch.all(temp_edge_boudnary_matrix[batch,:,j] != -1),torch.all(temp_edge_boudnary_matrix[batch,:,i] != -1)):
                            temp_edge_boudnary_matrix[batch,:,j]=((temp_edge_boudnary_matrix[batch,:,j]+temp_edge_boudnary_matrix[batch,:,i])==1).long()
                            temp_edge_low_indices = self.lowest_one(temp_edge_boudnary_matrix[batch],boundary=2)
                   
                        i += 1


                check_edge_low_indices = self.lowest_one(temp_edge_boudnary_matrix[batch],boundary=2,reduction=False)
                
                edge_check_unique, edge_check_counts = torch.unique(check_edge_low_indices,return_counts=True)

               

        return temp_edge_boudnary_matrix,temp_plaq_boundary_matrix
    
    def get_persistent_info_of_boundary_matrix(self, reduced_boundary_matrix_edge,reduced_boundary_matrix_plq, homology_group = 0):

        birth_position_and_epsilon = []

        death_position_and_epsilon = []

        unpaired_just_brith_pos_and_epslion = []

        

        copy_vertex_ordering = torch.clone(self.vertex_indices)

        for batch in range(int(reduced_boundary_matrix_edge.size(0))):

            if homology_group == 0:


               

                lowest_indices = self.lowest_one(reduced_boundary_matrix_edge[batch],boundary=2,reduction=False)

                birth = self.vertex_indices[lowest_indices] #should give back vertex indices epsilon is zero in our case

                unpaired = torch.from_numpy(np.delete(copy_vertex_ordering.flatten().cpu().numpy(), lowest_indices.flatten().cpu().numpy()))# does this work ?

                

                unpaired_epsilon = torch.zeros_like(unpaired)# torch.tensor([unpaired[batch][u.item()][0] for u in unpaired])

                death = torch.tensor(list(self.edge_ordering[batch].keys()))[torch.logical_not(torch.all(torch.t(reduced_boundary_matrix_edge[batch])==0, dim =1))]
                
                death_epsilon = torch.tensor([self.edge_ordering[batch][d.item()][0] for d in death])

                birth_position_and_epsilon.append(torch.cat((birth,torch.zeros_like(birth)),dim=0).view(2,-1).cpu().numpy())
                
                death_position_and_epsilon.append(torch.cat((death,death_epsilon), dim=0).view(2,-1).cpu().numpy())

                #unpaired_just_brith_pos_and_epslion.append(unpaired)
                unpaired_just_brith_pos_and_epslion.append(torch.cat((unpaired,unpaired_epsilon), dim=0).view(2,-1).cpu().numpy())

               

            elif homology_group == 1:

                copy_edge_ordering = copy.deepcopy(self.edge_ordering[batch])

                lowest_indices = self.lowest_one(reduced_boundary_matrix_plq[batch],boundary=4,reduction=False)[torch.logical_not(torch.all(torch.t(reduced_boundary_matrix_plq[batch])==0, dim =1))]#need to exclude zero columns
                
                print(self.lowest_one(reduced_boundary_matrix_plq[batch],boundary=4,reduction=False)[torch.all(torch.t(reduced_boundary_matrix_plq[batch])==0, dim =1)])
               
                birth =  torch.tensor(list(self.edge_ordering[batch].keys()))[lowest_indices]
                
                ##unpaired differently:
                
                unpaired_birth = torch.tensor(list(self.edge_ordering[batch].keys()))[torch.all(torch.t(reduced_boundary_matrix_edge[batch])==0, dim =1)] #since h_2 is not tested should be 
                
                fully_unpaired = []
                
                for u in unpaired_birth: 
                    
                    if (torch.any(u == birth)):
                        continue
                    else:
                        fully_unpaired.append(u)
                
               
                
                print('after',list(copy_edge_ordering.keys()))

                unpaired_epsilon = torch.tensor([[self.edge_ordering[batch][u.item()][0],u.item()] for u in fully_unpaired]).cpu()

                brith_epsilon = torch.tensor([self.edge_ordering[batch][b.item()][0] for b in birth])


                death = torch.tensor(list(self.plaquette_ordering[batch].keys()))[torch.logical_not(torch.all(torch.t(reduced_boundary_matrix_plq[batch])==0, dim =1))]

                death_epsilon = torch.tensor([self.plaquette_ordering[batch][d.item()][0] for d in death])

                birth_position_and_epsilon.append(torch.stack([birth,brith_epsilon], dim= 0).cpu().numpy())

                death_position_and_epsilon.append(torch.stack([death,death_epsilon],dim=0).cpu().numpy())

                unpaired_just_brith_pos_and_epslion.append(torch.t(unpaired_epsilon).cpu().numpy())



            else:
                raise ValueError('Working currently only with two dimensions')
            
        return np.stack(birth_position_and_epsilon,axis=0), np.stack(death_position_and_epsilon,axis=0),np.stack(unpaired_just_brith_pos_and_epslion, axis=0)
        
   
    def diameter_radius_graph(self, full_cluster_list):


        L = (2*self.lattice_size)
        N = L*L

        if self.saved_L != L or self.saved_N != N:
            self.saved_L = L
            self.saved_N = N

            self.nn_x_direction_larger = {i : ((i // L) * L + (i + 1) % L)\
                                                for i in list(range(N))}#right
            self.nn_y_direction_larger = {i : ((i + L) % N)\
                                                for i in list(range(N))}#down, up: ,(i - L) % N
            
            self.nn_x_direction_larger_second = {i : ((i // L) * L + (i + 2) % L)\
                                                for i in list(range(N))}#right
            self.nn_y_direction_larger_second = {i : ((i + 2*L) % N)\
                                                for i in list(range(N))}#down, up: ,(i - L) % N

        cluster_tensor = torch.tensor(full_cluster_list).view(-1,3)

        cluster_lattice = torch.zeros(self.lattice_size*2,self.lattice_size*2)
        cluster_lattice[cluster_tensor[:,0].int(),cluster_tensor[:,1].int()]=cluster_tensor[:,2]

        cluster_lattice = cluster_lattice.flatten()

        

        vertex_indices = torch.nonzero(cluster_lattice==2)


        graph_list_right = [[idx.item(),self.nn_x_direction_larger_second[idx.item()]] for idx in vertex_indices if cluster_lattice[self.nn_x_direction_larger[idx.item()]]==1.]
        
        graph_list_down = [[idx.item(),self.nn_y_direction_larger_second[idx.item()]] for idx in vertex_indices if cluster_lattice[self.nn_y_direction_larger[idx.item()]]==1]

        graph_list = graph_list_right+graph_list_down

        graph = nx.Graph(graph_list)

        return nx.diameter(graph),nx.radius(graph)

    
    def saw_function_tensor(self, spin_lattice_tensor):

        first_saw = torch.where(torch.abs(spin_lattice_tensor-self.two_pi)< torch.abs(spin_lattice_tensor),spin_lattice_tensor-self.two_pi, spin_lattice_tensor)

        second_saw = torch.where(torch.abs(spin_lattice_tensor+self.two_pi)< torch.abs(spin_lattice_tensor), spin_lattice_tensor+self.two_pi, first_saw)

        return second_saw
    
    def edge_matrix_batch_faster(self, spin_lattice_tensor):

        plaquette_maker = nn.Unfold(kernel_size=(2,2), stride= 1)
        plaq_index = torch.tensor([0,1,2],device=self.device)


        spin_lattice_halo = F.pad(spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        plaquette_temp = plaquette_maker(spin_lattice_halo).view(spin_lattice_tensor.size(0),4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquette_temp, 1, 2)

        all_plaquettes =  torch.index_select(plaquettes,dim=2, index=plaq_index)

        angel_intervaled_plaquette_angles_1 = torch.where(torch.logical_and(all_plaquettes < 0.0,all_plaquettes > -self.two_pi), all_plaquettes + self.two_pi,all_plaquettes)

        angel_intervaled_plaquette_angles = torch.where(torch.abs(all_plaquettes) >= self.two_pi ,angel_intervaled_plaquette_angles_1%self.two_pi,angel_intervaled_plaquette_angles_1)

        ## angle diff

        nn_angle_distance_x=torch.abs(self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,0] - angel_intervaled_plaquette_angles[:,:,1])))
        nn_angle_distance_y=torch.abs(self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,0] - angel_intervaled_plaquette_angles[:,:,2])))

        edges_x = (nn_angle_distance_x<= self.epsilon).view(-1,self.lattice_size,self.lattice_size).float()
        edges_y = (nn_angle_distance_y<= self.epsilon).view(-1,self.lattice_size,self.lattice_size).float()

        self.edges_x_batch = edges_x
        self.edges_y_batch = edges_y

        return nn_angle_distance_x.view(-1,self.lattice_size,self.lattice_size),nn_angle_distance_y.view(-1,self.lattice_size,self.lattice_size)## careful here return before compared to epsilon##edges_x, edges_y

    def plaquettes_matrix_batch_faster(self, spin_lattice_tensor):

        plaquette_maker = nn.Unfold(kernel_size=(2,2), stride= 1)
        plaq_index = torch.tensor([0,2,3,1,0],device=self.device)

        spin_lattice_halo = F.pad(spin_lattice_tensor.view(-1,1,self.lattice_size,self.lattice_size), (0,1,0,1), 'circular')

        plaquette_temp = plaquette_maker(spin_lattice_halo).view(spin_lattice_tensor.size(0),4,self.lattice_size*self.lattice_size)

        plaquettes = torch.transpose(plaquette_temp, 1, 2)

        all_plaquettes =  torch.index_select(plaquettes,dim=2, index=plaq_index)

        changed_plaquette_angles = torch.sub(all_plaquettes[:,:,:],all_plaquettes[:,:,0].view(-1,self.lattice_size*self.lattice_size,1))

        angel_intervaled_plaquette_angles_1 = torch.where(torch.logical_and(changed_plaquette_angles < 0.0,changed_plaquette_angles > -self.two_pi), changed_plaquette_angles + self.two_pi,changed_plaquette_angles)

        angel_intervaled_plaquette_angles = torch.where(torch.abs(changed_plaquette_angles) >= self.two_pi ,angel_intervaled_plaquette_angles_1%self.two_pi,angel_intervaled_plaquette_angles_1)

        ## angle diff

        full_diff_list =[self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,1] - angel_intervaled_plaquette_angles[:,:,0])) 
        ,self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,2] - angel_intervaled_plaquette_angles[:,:,1])) 
        ,self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,3] - angel_intervaled_plaquette_angles[:,:,2])) 
        ,self.saw_function_tensor((angel_intervaled_plaquette_angles[:,:,4] - angel_intervaled_plaquette_angles[:,:,3]))]

        full_diff_tensor = torch.stack(full_diff_list,dim=2)

        max_diff_tensor = torch.max(torch.abs(full_diff_tensor),dim=2)[0]

        plaquette_matrices= max_diff_tensor.view(-1,self.lattice_size,self.lattice_size) <= self.epsilon

        self.plaquettes = plaquette_matrices

        return max_diff_tensor.view(-1,self.lattice_size,self.lattice_size)## careful here return before compared to epsilon plaquette_matrices


    
    def get_key_from_value(self,value, dic):
        return [k for k,v in dic.items() if np.any(value in v)]
    

    
    def cluster(self,index):#only for state percolation
        cluster_nmb = 0
        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = []
        states = self.plaquettes_batch[index]
        clusters_position_states = {}
       
        for idx in range(int(self.lattice_size*self.lattice_size)):
                
            if states[idx] == 1:
                cluster_values_array = sorted({x for v in clusters_position_states.values() for x in v})
                
                if idx not in cluster_values_array:
                   
                    before_neigbour_indices_boundary = [cluster_index for cluster_index in self.nn_cluster_boundary[idx]]#right,down

                    before_neigbour_indices = [cluster_index for cluster_index in self.nn_cluster[idx]]
                   
                    if before_neigbour_indices[0] in cluster_values_array or before_neigbour_indices[1] in cluster_values_array:
                        #should be put in corresponding cluster
                        
                        if before_neigbour_indices[0] in cluster_values_array and before_neigbour_indices[1] in cluster_values_array:
                            cluster_number_one = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                #connect both clusters 
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                clusters_position_states['{0}'.format(earlier_cluster[0])].append(idx)
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                cluster_nmb -= 1
                            else:
                                clusters_position_states['{0}'.format(cluster_number_one[0])].append(idx)
                        elif before_neigbour_indices[0] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[0], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                        elif before_neigbour_indices[1] in cluster_values_array:
                            cluster_number = self.get_key_from_value(before_neigbour_indices[1], clusters_position_states)
                            clusters_position_states['{0}'.format(cluster_number[0])].append(idx)
                    else:
                        cluster_nmb += 1
                        clusters_position_states['{0}'.format(cluster_nmb)] = []
                        clusters_position_states['{0}'.format(cluster_nmb)].append(idx)

                    cluster_values_array_2 = sorted({x for v in clusters_position_states.values() for x in v})

                    

                    if idx % self.lattice_size == (self.lattice_size-1) or int(idx/self.lattice_size) == (self.lattice_size-1):
                        
                        #the very lower edge case
                        if idx == (self.lattice_size*self.lattice_size)-1 and before_neigbour_indices_boundary[0] in cluster_values_array_2 and before_neigbour_indices_boundary[1] in cluster_values_array_2:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)
                            cluster_number_three = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)

                            if cluster_number_one != cluster_number_two and cluster_number_one != cluster_number_three and cluster_number_two!= cluster_number_three:
                                cluster_number_list = cluster_number_one+cluster_number_two+cluster_number_three

                                earlier_cluster = min(cluster_number_one,cluster_number_two,cluster_number_three)

                                later_cluster = max(cluster_number_one,cluster_number_two,cluster_number_three)
                                middle_cluster = [cluster_number for cluster_number in cluster_number_list if cluster_number != earlier_cluster[0] and cluster_number != later_cluster[0]]

                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]+clusters_position_states['{0}'.format(cluster_number_three[0])]

                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                                clusters_position_states['{0}'.format(middle_cluster[0])] = []
                            elif cluster_number_one != cluster_number_two and cluster_number_one == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]

                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one == cluster_number_two and cluster_number_one != cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]

                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            elif cluster_number_one != cluster_number_two and cluster_number_two == cluster_number_three:
                                earlier_cluster = min(cluster_number_one,cluster_number_three)
                                later_cluster = max(cluster_number_one,cluster_number_three)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_three[0])]

                                clusters_position_states['{0}'.format(later_cluster[0])] = []

                            else:
                                continue

                        #the right edge full

                        elif idx % self.lattice_size == (self.lattice_size-1) and before_neigbour_indices_boundary[0] in cluster_values_array_2:
                            
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[0], clusters_position_states)

                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        #lowest line (edge) of lattice 
                        elif int(idx/self.lattice_size) == (self.lattice_size-1) and before_neigbour_indices_boundary[1] in cluster_values_array_2:
                            cluster_number_one = self.get_key_from_value(idx, clusters_position_states)
                            cluster_number_two = self.get_key_from_value(before_neigbour_indices_boundary[1], clusters_position_states)
                            if cluster_number_one != cluster_number_two:
                                earlier_cluster = min(cluster_number_one,cluster_number_two)
                                later_cluster = max(cluster_number_one,cluster_number_two)
                                clusters_position_states['{0}'.format(earlier_cluster[0])] = clusters_position_states['{0}'.format(cluster_number_one[0])] + clusters_position_states['{0}'.format(cluster_number_two[0])]
                                clusters_position_states['{0}'.format(later_cluster[0])] = []
                        else:
                            continue
                else:
                    continue


        self.elements_per_clusters = clusters_position_states

        self.clusters_per_epsilon['{0}'.format(self.epsilon)] = clusters_position_states

        return clusters_position_states
   
 
    def percolation_faster(self,plaquettes, edges_x, edges_y):

        filled_states = torch.sum(plaquettes == 1.0, dim = 1).float()

        empty_states = torch.sum(plaquettes == 0.0, dim = 1).float()

        filled_bonds_x = torch.sum(edges_x == 1.0, dim = 1).float()
        empty_bonds_x = torch.sum(edges_x == 0.0, dim = 1).float()
        filled_bonds_y = torch.sum(edges_y == 1.0, dim = 1).float()
        empty_bonds_y = torch.sum(edges_y == 0.0, dim = 1).float()

        filled_bonds = filled_bonds_x+filled_bonds_y
        empty_bonds = empty_bonds_x +empty_bonds_y


        state_percolation = filled_states/(empty_states+filled_states)

        bond_percolation = filled_bonds/(empty_bonds+filled_bonds)

  
        return state_percolation,bond_percolation

    

    def translation_of_pb_overlapping_cluster_tensor(self, cluster_lattice):
  

        rot_nmb_d = 0
        rot_nmb_r = 0
        while torch.sum(torch.logical_and(cluster_lattice[0,:],cluster_lattice[self.lattice_size-1,:])) > 0 and rot_nmb_d< self.lattice_size:
            
            cluster_lattice = torch.roll(cluster_lattice, 1 , 0)
            rot_nmb_d+=1


        while torch.sum(torch.logical_and(cluster_lattice[:,0],cluster_lattice[:,self.lattice_size-1])) > 0 and rot_nmb_r< self.lattice_size:
            cluster_lattice = torch.roll(cluster_lattice, 1 ,1)
            rot_nmb_r+=1

        return cluster_lattice

    

    def euler_charactistics_faster(self, cluster_lattices):#takes directly a lattice with zeros and ones for each cluster

        transformed_cluster = self.translation_of_pb_overlapping_cluster_tensor(cluster_lattices).cpu().numpy()

        padded_cluster_lattice = np.pad(transformed_cluster, 3, mode='constant')

        e = euler_number(padded_cluster_lattice,connectivity=1) #full euler characteristic.

        return e

    
    def minkowski_measure_new_faster(self,clusters_zero_one_tensors):# should take a tensor of all the clusters for one exmaple and one epsilon

        areas_of_clusters = []
        perimeters_of_clusters = []
       

        areas_of_clusters = torch.sum(clusters_zero_one_tensors.view(-1,self.lattice_size*self.lattice_size), dim = 1 )

        n_maker = nn.Unfold(kernel_size=(3,3), stride=1)

        cluster_halo = F.pad(clusters_zero_one_tensors.view(-1,1,self.lattice_size,self.lattice_size),pad = (1,1,1,1),mode = 'circular')

        nn_cluster_temp = n_maker(cluster_halo).view(clusters_zero_one_tensors.size(0),9,self.lattice_size*self.lattice_size)

        nn_cluster_transposed = torch.transpose(nn_cluster_temp,1,2)

        indices_cluster = torch.tensor([4,1,3,5,7], device=self.device)

        nn_cluster_indices = torch.index_select(nn_cluster_transposed, dim=2, index=indices_cluster)
        
        perimeter_cluster_start = nn_cluster_indices[nn_cluster_indices[:,:,0] == 1].view(clusters_zero_one_tensors.size(0),-1,5)

        perimeters_of_clusters = torch.sum(torch.sum(perimeter_cluster_start[:,:,0:]==0,dim=2),dim=1).float()


        return torch.flatten(areas_of_clusters),torch.flatten(perimeters_of_clusters)
    

