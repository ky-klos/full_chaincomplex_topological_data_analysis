


from numba import jit
from numba.typed import List



#TODO: not used
def array_to_lattice_position(pos,lattice_size):
    

    x_pos = pos%lattice_size
    y_pos = int(pos)//lattice_size
    #print(pos)
    #print('x: ', x_pos)
    #print('y: ', y_pos)

    return int(x_pos), int(y_pos)

#TODO: not used
def array_to_lattice_multipble(positions, lattice_size):
    x_positions, y_positions = [], []

    for pos in positions:
        #print(pos)
        x_pos,y_pos = array_to_lattice_position(pos=pos,lattice_size=lattice_size)
        x_positions.append(x_pos)
        y_positions.append(y_pos)

    return x_positions, y_positions



#TODO: not used
def cluster_faster(input_plaquette_tensor,device):

    rows, cols = input_plaquette_tensor.size()

    clusters = []

    visited = torch.zeros_like(input_plaquette_tensor, dtype=bool, device= device)

    def explore_clusters(row, col, cluster,cluster_counter):
            
        if not (0<=row<rows) or not (0<= col < cols) or visited[row,col] or input_plaquette_tensor[row, col] == 0:
            return
        visited[row,col] = True
        cluster.append(torch.tensor([row,col,cluster_counter+1], device = device))

        for i, j in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_row=(row +i) % rows
            new_col = (col+j) % cols
            explore_clusters(new_row,new_col, cluster,cluster_counter)

    cluster_counter = 0

    for i in range(rows):
        for j in range(cols):
            if input_plaquette_tensor[i,j] == 1 and not visited[i,j]:
                current_cluster = []
                explore_clusters(i,j,current_cluster,cluster_counter)
                clusters.append(torch.stack(current_cluster, dim=0))
                cluster_counter += 1


    return clusters

#TODO: not used
@jit(nopython=True)
def run_bfs_edge_numba(img, x, y, visited):
    height, width = img.shape
    cluster = List()
    cluster.append((x, y, img[x, y]))
    queue = List()
    queue.append((x, y))
    
    while queue:
        x, y = queue.pop(0)
        
        for dx in (-1, 1):
            x_new = (x + dx + height) % height
            
            if img[x_new, y] >= 1 and (x_new, y) not in visited:
                visited.add((x_new, y))
                queue.append((x_new, y))
                cluster.append((x_new, y, img[x_new, y]))
        
        for dy in (-1, 1):
            y_new = (y + dy + width) % width
            
            if img[x, y_new] >= 1 and (x, y_new) not in visited:
                visited.add((x, y_new))
                queue.append((x, y_new))
                cluster.append((x, y_new, img[x, y_new]))
    
    return cluster

#TODO: not used
@jit(nopython=True)
def cluster_image_bfs_edge_numba(img):
    height, width = img.shape
    visited = set()
    cluster_list = List()
    
    for x_coord in range(height):
        for y_coord in range(width):
            if (x_coord, y_coord) not in visited and img[x_coord, y_coord] >= 1:
                visited.add((x_coord, y_coord))
                cluster = run_bfs_edge_numba(img, x_coord, y_coord, visited)
                cluster_list.append(cluster)
    
    return cluster_list

#TODO: not used
def cluster_faster_edges(input_edge_tensor_x, input_edge_tensor_y,device):

    rows, cols = input_plaquette_tensor.size()

    clusters = []

    visited = torch.zeros_like(input_plaquette_tensor, dtype=bool, device= device)

    def explore_clusters(row, col, cluster,cluster_counter):
            
        if not (0<=row<rows) or not (0<= col < cols) or visited[row,col] or input_plaquette_tensor[row, col] == 0:
            return
        visited[row,col] = True
        cluster.append(torch.tensor([row,col,cluster_counter+1], device = device))

        for i, j in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_row=(row +i) % rows
            new_col = (col+j) % cols
            explore_clusters(new_row,new_col, cluster,cluster_counter)

    cluster_counter = 0

    for i in range(rows):
        for j in range(cols):
            if input_plaquette_tensor[i,j] == 1 and not visited[i,j]:
                current_cluster = []
                explore_clusters(i,j,current_cluster,cluster_counter)
                clusters.append(torch.stack(current_cluster, dim=0))
                cluster_counter += 1


    return clusters

def run_edges(edge_x,edge_y, x, y, visited,type):#type 0 for x, 1 for y
 
    size = edge_x.shape
    cluster = [[x,y,type]]
    queue = [(x, y)]
 
    while queue:
        x, y = queue.pop(0)
       
        # check neighbors with periodic boundary
        for dx in [-1,1]:
            x_new = (x + dx + size[0]) % size[0]
 
            if edge_x[x_new][y] == 1 and (x_new, y) not in visited:
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
 

#TODO: not used
def cluster_image_edges(edge_x, edge_y):
 
    size = edge_x.shape
 
    visited = set()
    cluster_list = []
 
    for x_coord in range(size[0]):
        for y_coord in range(size[1]):
 
            if edge_x[x_coord][y_coord] == 1:
 
                if (x_coord, y_coord,0) in visited:
                    continue
                else:
                    visited.add((x_coord, y_coord,0))#0 means here x
                    cluster = run_edges(edge_x,edge_y, x_coord, y_coord, visited,edge_type=0)
                    cluster_list.append(cluster)

            if edge_y[x_coord][y_coord] == 1:
 
                if (x_coord, y_coord,1) in visited:
                    continue
                else:
                    visited.add((x_coord, y_coord, 1))
                    cluster = run_edges(edge_x,edge_y, x_coord, y_coord, visited,edge_type=1)
                    cluster_list.append(cluster)
 
 
    return cluster_list


#TODO: not used
def cluster_faster_alternative(input_plaquette_tensor):

    batchsize, rows, cols = input_plaquette_tensor.size()

    print(batchsize)
    print(rows)
    print(cols)

    reshape_tensor = input_plaquette_tensor.view(batchsize,-1)

    ##row_indices = torch.arange(rows).view(-1,1).repeat(1,cols).view(-1)
    ##col_indices = torch.arange(cols).repeat(rows)

    ##indices = torch.arange(reshape_tensor.numel()).view(1,-1)
    indices = reshape_tensor.nonzero(as_tuple=True)[1]


    print(indices)

    row_idx = (indices/cols).long()

    col_idx = (indices%cols).long()

    non_zero_mask = reshape_tensor != 0

    print(non_zero_mask)

    row_idx = (row_idx-1) % rows

    col_idx = (col_idx-1) % cols

    n_row=(row_idx.repeat(4,1)+torch.tensor([-1,1,0,0]).view(-1,1))%rows

    n_col = (col_idx.repeat(4,1)+torch.tensor([0,0,-1,1]).view(-1,1))%cols

    ##neighbor_indices = n_row*cols+n_col

    neighbor_indices = row_idx*cols+col_idx

    print(neighbor_indices)

    ##neighbor_mask = reshape_tensor[:,neighbor_indices]*non_zero_mask.view(batchsize,-1)

    ##print(neighbor_mask.size())

    ##clusters = neighbor_mask.nonzero(as_tuple=True)[1]

    clusters, indices = torch.unique(neighbor_indices, return_inverse= True)

    unique_clusters = [indices[indices==i] for i in range(len(clusters))]

    return unique_clusters

#TODO: not used
def cluster_faster_other(input_plaquette_tensor, lattice_size=5):

    spins_halo= F.pad(input_plaquette_tensor.float().view(-1,1,lattice_size,lattice_size),(1,1,1,1),'circular')

    nn_maker = nn.Unfold(kernel_size=(3,3),stride=1)

    nn_temp = nn_maker(spins_halo).view(input_plaquette_tensor.size(0),9,lattice_size*lattice_size)

    next_n = torch.transpose(nn_temp, 1,2)

    index = torch.tensor([4,1,3,5,7])

    all_nn = torch.index_select(next_n,dim=2, index=index)

    object_cluster = all_nn!=0

    object_cluster_indices = object_cluster.nonzero(as_tuple=True)

    ##print(object_cluster_indices[:,2])

    #check if zero i


    print('object_cluster_indices',object_cluster_indices)

    return object_cluster_indices



