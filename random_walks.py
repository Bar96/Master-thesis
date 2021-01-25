import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import math
import random
import time

###########################################################################################################
                                        #RANDOM WALKS
###########################################################################################################


def build_alias_table(probs):
    ''' Compute the probabilities table and the alias table 
        of a descrete distribution.
    
    Parameters
    ----------
    probs : list
        list containing the probabilites of the distribution.

    Returns
    -------
    alias : numpy array
        alias table.
    q : numpy array
        probabilities table.

    '''
    n = len(probs) #number of probabilities in the distribution
    alias = np.zeros(n, dtype=np.int32)
    q = np.zeros(n, dtype=np.float32)
    
    smaller = []
    larger = []
    for i, p in enumerate(probs):
        q[i] = n*p
        if q[i] < 1.0:
            smaller.append(i) 
        else:
            larger.append(i)
        
    while len(smaller) > 0 and len(larger) > 0:
        #extract one element from the small and one from the large
        small = smaller.pop()
        large = larger.pop()
        alias[small] = large
        q[large] = q[large] + q[small] - 1.0 #update the prob of the large element
        
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
            
    return alias, q




def preprocess(network, verbose=False):
    '''For each node of the network compute the list of its neighbors and 
    the alias table

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.
    verbose : boolean, optional
        print the passages. The default is False.

    Returns
    -------
    neighbors : list 
        list where each element is the list of neighbors of the corresponding node.
    alias_tables : list
        list where each element contains the probabilities and the alias table
        of the corresponding node.

    '''
    neighbors = [] 
    alias_tables = []
    for i in range(network.shape[0]):
        line = [] #list to store the indices of the neighbors of node i 
        unnormalized_probs = []
        for j in range(network.shape[0]):
            if network[i][j] > 0:
                line.append(j)
                unnormalized_probs.append(network[i][j])
        neighbors.append(line)
        
        #normalize the probabilities between 0 and 1
        line_sum = sum(unnormalized_probs)
        if line_sum != 0:
            probs = [p/line_sum for p in unnormalized_probs]
        else:
            probs = [] #the node is isolated
        alias_tables.append(build_alias_table(probs))
        
        if i % 500 == 0 and verbose:
            print('preprocessing: '+str(i)+'/'+str(network.shape[0]))
    network = None
    return neighbors, alias_tables

  


def get_next_node(alias, q):
	'''Sample the next node using the alias method

    Parameters
    ----------
    alias : numpy array
        alias table.
    q : numpy array
        probabilities table.

    Returns
    -------
    integer
        index of the next node sampled.

    '''
	rand_idx = random.randint(0, len(alias)-1)
	rand = random.random()
	if rand < q[rand_idx]:
		return rand_idx
	else:
		return alias[rand_idx]      




def sequential_random_walks(network, walk_per_node=10, walk_length=20, restart_prob=0, seed=None, verbose=False):
    '''For each node of the network generate a series of random walks

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.
    walk_per_node : integer, optional
        number of walks per node to generate. The default is 10.
    walk_length : integer, optional
        Lenght of the walks to generate. The default is 20.
    restart_prob : float, optional
        probability to restart the walk. The default is 0.
    seed : integer, optional
        optional seed. The default is None.
    verbose : boolean, optional
        print the passages. The default is False.

    Returns
    -------
    walks : list
        list where each element is a random walk.

    '''
    if seed != None:
        random.seed(seed)
    else:
        random.seed()
        
    num_nodes = network.shape[0]
    
    neighbors, alias_tables = preprocess(network, verbose)
    network = None #the network is no more needed
    walks = []
    
    for h in range(num_nodes):
        for n in range(walk_per_node):
            if not len(neighbors[h]) > 0: #the node has no neighbors
                #use a list of -1 when it's not possible to compute the walk
                walks.append(['-1' for i in range(walk_length)]) 
                continue
            walk = []   
            current_node = h
            walk.append(str(current_node))
            for i in range(walk_length-1):
                restart = random.random()
                if restart < restart_prob and current_node != h: #check also if we are already in the starting node
                    walk.append(str(h))
                    current_node = h
                else:
                    index = get_next_node(alias_tables[current_node][0], alias_tables[current_node][1])
                    walk.append(str(neighbors[current_node][index]))
                    current_node = neighbors[current_node][index]
            walks.append(walk)
        if h % 1000 == 0:
            print(str(h)+'/'+str(num_nodes))
   
    return walks    



def save_walks_as_npy(walks, filename):
    '''Convert a series of random walks into a bidimensional numpy array where
    each row is a walk and then save it

    Parameters
    ----------
    walks : list
        list of random walks.
    filename : string
        name of the file save the walks.

    Returns
    -------
    walks_np : numpy array
        numpy array of the walks.

    '''
    walks_np = np.array(walks)
    with open(filename, 'wb') as f:
        np.save(f, walks_np)
    return walks_np




def build_probability_matrix(random_walks, walks_per_node, num_nodes):
    '''Given a series of random walks for each node, compute the probability matrix. 
    the element i,j of the matrix represent the probability to find node j on a random
    walk that start from node i

    Parameters
    ----------
    random_walks : list
        list of random walks.
    walks_per_node : integer
        number of walks per node.
    num_nodes : integer
        number of nodes.

    Returns
    -------
    probs_matrix : numpy array
        probability matrix.

    '''    
    probs = []
    count = [0 for i in range(num_nodes)] #counter for each node
   
    i = 0
    for walk in random_walks:
        if walk[0] != '-1': 
            for k, node in enumerate(walk):
                if k == 0:  #don't count the starting node of the walk
                    continue
                else: count[int(node)] += 1 #update the counter of the corresponding node
        i+=1
        if i == walks_per_node: #stop the count, compute the probabilities 
            count_sum = sum(count)
            if count_sum > 0:
                probs.append([x/count_sum for x in count])
            else:
                probs.append(count)
            count = [0 for i in range(num_nodes)]
            i = 0
    
    probs_matrix = np.array(probs, dtype=np.float32)
    
    return probs_matrix
        
        
        
       

#####################################################################################################
                            #BIASED RANDOM WALKS
#####################################################################################################
        
def build_graph(network):
    '''
    
    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.

    Returns
    -------
    graph : list
        list where each element i contains the neighbors of the corresponding node.
        Each neighborhood is stored as a dictionary where the keys are the indices 
        of the neighbors and the values are the weights of the arc between the node
        and the neighbor.
    '''
    graph = []
    for i in range(network.shape[0]):
        neighbors = {}
        for j in range(network.shape[1]):
            if network[i][j] > 0:
                neighbors[j] = network[i][j]
        graph.append(neighbors)
    network = None
    return graph

    


def build_shared_graph(network):
    '''Allocate a shared memory block and copy inside it the list of neighbors 
    for each node and the adjaceny matrix with boolean values

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.

    Returns
    -------
    shape : shape
        shape of the adjacency matrix.

    '''
    mask = np.zeros(network.shape, dtype=np.bool) #adjacency matrix with boolean values
    graph = [] #list of lists where element i contains the list of neighbors of node i. Each neighbor
               #is a tuple [j, w] where j is the index of the neighbor and w is the weight of arc i,j
    max_lenght = -1
    
    #compute the neighborhoods and the mask
    for i in range(network.shape[0]):
        neighbors =[]
        for j in range(network.shape[1]):
            if network[i][j] > 0: #if there is an arc between node i and j
                mask[i][j] = True
                neighbors.append([j, network[i][j]])
        graph.append(neighbors)
        if len(neighbors) > max_lenght: #keep track of the largest neighborhood
            max_lenght = len(neighbors)
    network = None #network is no more required
    
    #add [-1, -1] to the neighborhoods until they all have lenght equal to the 
    #largest one, in this way graph become a matrix. This is necessary since in 
    #the shared memory it's not possible to store a list of lists
    for i in range(len(graph)):
        if len(graph[i]) < max_lenght:
            while len(graph[i]) < max_lenght:
                graph[i].append([-1, -1])

    graph_np = np.array(graph).astype(np.float32)
    graph = None
    
    #allocate shaerd memory
    shm1 = shared_memory.SharedMemory(create=True, name='graph', size=graph_np.nbytes)
    shm2 = shared_memory.SharedMemory(create=True, name='mask', size=mask.nbytes)
    
    shared_graph = np.ndarray(graph_np.shape, dtype=graph_np.dtype, buffer=shm1.buf)
    shared_mask = np.ndarray(mask.shape, dtype=mask.dtype, buffer=shm2.buf)
    
    #copy the data inside the shared memory
    shared_graph[:, :, :] = graph_np[:, :, :]
    shared_mask[:, :] = mask[:, :]
    shape = graph_np.shape
    
    #close the access to the shared memory block
    shm1.close()
    shm2.close()
    graph_np = None
    mask = None    
    return shape
    
    
    
    
def biased_random_walks(graph_shape, starts, walk_length, p, q, results_list, index, seed_list, verbose):
    '''Function to be executed by a worker. For each node in the starts list generate a 
    biased random walk. Data required to compute the walks are provided by the shared memory.
    Results are stored in results_list that is a Manager object.

    Parameters
    ----------
    graph_shape : shape
        shape of the graph matrix stored in the shared memory.
    starts : list
        list of starting nodes. To generate more than one walk per node,
        the node is repeated in the list
    walk_length : integer
        lenght of the walks.
    p : float
        p parameter.
    q : float
        q parameter.
    results_list : Manager
        Manager object of the multiproccessing module to store the results.
    index : integer
        id of the worker that is executing the function.
    seed_list : list
        list containing a seed for each worker.
    verbose : boolean
        print the results.

    Returns
    -------
    None.

    '''
    print('process: '+str(index)+' starts')
    if len(seed_list) > index: 
        random.seed(seed_list[index])
    else: 
        random.seed()
        
    #access the shared memory
    shm1 = shared_memory.SharedMemory(name='graph')
    shm2 = shared_memory.SharedMemory(name='mask')
    
    shared_graph = np.ndarray(graph_shape, dtype=np.float32, buffer=shm1.buf)
    shared_mask = np.ndarray((graph_shape[0], graph_shape[0]), dtype=np.bool, buffer=shm2.buf)
        
    walks = []
    for n, starting_node in enumerate(starts):
        walk = []
        
        if shared_graph[starting_node][0][0] == -1:
            walks.append(['-1' for m in range(walk_length)])
            continue
        
        walk.append(str(starting_node))
        
        #compute the distribution of the first node
        #since there is no previous node at the start, p and q are't used
        unnormalized_probs = []
        i = 0
        while i < shared_graph[starting_node].shape[0] and shared_graph[starting_node][i][0] > -1: #the graph is padded with -1
            unnormalized_probs.append(shared_graph[starting_node][i][1])
            i+=1
        sum_tot = sum(unnormalized_probs)
        probs = [x / sum_tot for x in unnormalized_probs]
        
        #sample the next node and add it to the walk
        rand = random.random()
        curr_sum = 0
        for i, prob in enumerate(probs):
            curr_sum += prob
            if rand < curr_sum:
                walk.append(str(int(shared_graph[starting_node][i][0])))
                break
        
        while len(walk) < walk_length:
            curr_node = int(walk[-1])
            prev_node = int(walk[-2])
    
            #compute the distribution of the current node
            unnormalized_probs = []  
            k = 0
            while k < shared_graph[curr_node].shape[0] and shared_graph[curr_node][k][0] > -1:
                elem = shared_graph[curr_node][k] #elem is a neighbor of the current node
                if elem[0] == prev_node:
                    unnormalized_probs.append((1/p)*elem[1])
                elif shared_mask[int(elem[0])][prev_node] == True: #check if the node and prev_node are neighbors
                    unnormalized_probs.append(elem[1])
                else:
                    unnormalized_probs.append((1/q)*elem[1])
                k+=1
            
            #sample the next node
            sum_tot = sum(unnormalized_probs)
            rand = random.random()
            curr_sum = 0
            for i, prob in enumerate(unnormalized_probs):
                curr_sum += prob / sum_tot
                if rand < curr_sum:
                    walk.append(str(int(shared_graph[curr_node][i][0])))
                    break
          
        walks.append(walk)
        if n%100 == 0 and verbose:
            print('process'+str(index)+': '+str(n)+'/'+str(len(starts)))
    
    #save results in the manager object and close the connection with the shared memory
    results_list.append([index, walks])
    shm1.close()
    shm2.close()
        
    


def parallel_biased_random_walks(network, walk_per_node=10, walk_length=20, 
                                 p=1, q=1, seed_list=[], num_worker=mp.cpu_count(), verbose=False):
    '''Compute biased random walks using different workers

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network
    walk_per_node : integer, optional
        number of walks per node. The default is 10.
    walk_length : integer, optional
        lenght of the walks. The default is 20.
    p : float, optional
        p parameter. The default is 1.
    q : float, optional
        q parameter. The default is 1.
    seed_list : list, optional
        list of seeds, one for each worker. The default is [].
    num_worker : integer, optional
        number of workers. The default is the number of cpu.
    verbose : boolean, optional
        print the results. The default is False.

    Returns
    -------
    res : list
        list of results.

    '''
   
    #generate the starts list 
    starts = []
    for i in range(network.shape[0]):
        for j in range(walk_per_node):
            starts.append(i)
    
    #split the starts into a numeber of batches equal to the number of workers
    batches = []
    batch_size = math.ceil(len(starts) / num_worker) 

    start = 0
    end = batch_size
    for i in range(num_worker-1):
        batches.append(starts[start:end])
        start = end
        end += batch_size
    batches.append(starts[start:])
    
    start = time.time()
    #allocate the shared memory and build the shared graph
    graph_shape = build_shared_graph(network)
    network = None
    end = time.time()
    print('preprocess complete in: '+str(end-start))
    
    manager = mp.Manager() #Manager object to store the results
    results_list = manager.list()
    #initialize the workers
    processes = [mp.Process(target=biased_random_walks, 
                args=(graph_shape, batches[i], walk_length, 
                      p, q, results_list, i, seed_list, verbose)) for i in range(num_worker)]
    #start the workers
    for p in processes: p.start()
    for p in processes: p.join()
    
    #delete the shared memory
    shm1 = shared_memory.SharedMemory(name='graph')
    shm2 = shared_memory.SharedMemory(name='mask')
    shm1.close()
    shm1.unlink()
    shm2.close()
    shm2.unlink()
    
    #sort the results
    results = [elem for elem in results_list]
    results.sort()
    
    res = []
    for r in results:
        res.extend(r[1])
    return res    
    



def sequential_biased_random_walks(net, walk_per_node=10, 
                                   walk_length=20,  p=1, q=1, seed=None, verbose=True):
    '''Compute the biased random walks with a sequential algorithm (only one worker)

    Parameters
    ----------
    net : numpy array
        adjacency matrix of the network
    walk_per_node : integer, optional
        number of walks per node. The default is 10.
    walk_length : integer, optional
        lenght of the walks. The default is 20.
    p : float, optional
        p paramter. The default is 1.
    q : float, optional
        q paramter. The default is 1.
    seed : integer, optional
        seed. The default is None.
    verbose : boolean, optional
        print the results. The default is True.

    Returns
    -------
    walks : list
        list of walks.

    '''    
    if seed != None: 
        random.seed(seed)
    else: 
        random.seed()
    
    network = build_graph(net) 
    net = None
    print('preprocessing complete')
    
    walks = []
    for starting_node in range(len(network)):
        for k in range(walk_per_node):
            walk = []
            walk.append(starting_node)
            
            #compute the distribution of the first node 
            neighbors = list(network[starting_node].keys())
            unnormalized_probs = list(network[starting_node].values())
            sum_tot = sum(unnormalized_probs)
            probs = [x / sum_tot for x in unnormalized_probs]
            
            #sample the next node
            rand = random.random()
            curr_sum = 0
            for i, prob in enumerate(probs):
                curr_sum += prob
                if rand < curr_sum:
                    walk.append(neighbors[i])
                    break
    
            while len(walk) < walk_length:
                curr_node = walk[-1]
                prev_node = walk[-2]
                
                #compute the distribution of the current node
                neighbors = list(network[curr_node].keys())
                unnormalized_probs = []  
                for key, value in network[curr_node].items():
                    if key == prev_node:
                        unnormalized_probs.append((1/p)*value)
                    elif key in network[prev_node]:
                        unnormalized_probs.append(value)
                    else:
                        unnormalized_probs.append((1/q)*value)
            
                #sample the next ndoe
                sum_tot = sum(unnormalized_probs)
                rand = random.random()
                curr_sum = 0
                for i, prob in enumerate(unnormalized_probs):
                    curr_sum += prob / sum_tot
                    if rand < curr_sum:
                        walk.append(neighbors[i])
                        break
            walks.append(walk)
        
        if starting_node%500 == 0 and verbose:
            print(str(starting_node)+'/'+str(len(network)))

    return walks




def compute_alias_edges(network, neighbors, prev, curr, p, q):
    '''Compute the alias table of the current node given the previous
    node of the walk and the parameters p and q

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.
    neighbors : list
        list where the element i is the list of neighbors 
        of node i.
    prev : integer
        previus node of the wlak.
    curr : integer
        current node of the walk.
    p : float
        p paramater.
    q : float
        q parameter.

    Returns
    -------
    alias, q
        alias table and probability table.

    '''
    unnormalized_probs = []
    
    for n in neighbors[curr]:
        if n == prev:
            unnormalized_probs.append((network[curr][n])/p)
        elif network[n][prev] > 0:
            unnormalized_probs.append(network[curr][n])
        else:
            unnormalized_probs.append((network[curr][n])/q)
            
    probs_sum = sum(unnormalized_probs)
    probs = [float(k)/probs_sum for k in unnormalized_probs]
    return build_alias_table(probs)




def preprocess_biased(network, p, q, verbose=False):
    '''For each node of the network compute the list of its neighbors and 
    the alias table. For each edge compute the alias table using one vertex
    as previous node and the other one as current node

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.
    p : float
        p parameter.
    q : floar
        q paramter.
    verbose : boolean, optional
        print the passages. The default is False.

    Returns
    -------
    neighbors : list 
        list where each element is the list of neighbors of the corresponding node.
    alias_tables : list
        list where each element contains the probabilities and the alias tables
        of the corresponding node.
    edges_alias_tables: dictionary
        ddictionary with edges as keys and the corresponding probabilites 
        and alias tables as values

    '''
    neighbors = [] 
    alias_tables = []
    edges = []
    for i in range(network.shape[0]):
        line = [] #list to store the indices of the neighbors of node i 
        unnormalized_probs = []
        for j in range(network.shape[0]):
            if network[i][j] > 0:
                line.append(j)
                unnormalized_probs.append(network[i][j])
                edges.append([i, j])
        neighbors.append(line)
        
        #normalize the probabilities between 0 and 1
        line_sum = sum(unnormalized_probs)
        if line_sum != 0:
            probs = [p/line_sum for p in unnormalized_probs]
        else:
            probs = [] #the node is isolated
        alias_tables.append(build_alias_table(probs))
        
        if i % 500 == 0 and verbose:
            print('preprocessing: '+str(i)+'/'+str(network.shape[0]))
            
    edges_alias_tables = {}
    for k, edge in enumerate(edges):
        edges_alias_tables[(edge[0], edge[1])] = compute_alias_edges(network, neighbors, edge[0], edge[1], p, q)
        if i % 5000 == 0 and verbose:
            print('preprocessing: '+str(k)+'/'+str(len(edges)))
    
    network = None
    return neighbors, alias_tables, edges_alias_tables



    
def biased_random_walks_small(network, p=1, q=1, walk_length=5, walk_per_node=5, seed=None, verbose=False):
    '''Compute biased random walks using the alias method. This
    function works only for small networks

    Parameters
    ----------
    network : numpy array
        adjacency matrix of the network.
    p : float, optional
        p parameter. The default is 1.
    q : float, optional
        q parameter. The default is 1.
    walk_length : integer, optional
        lenght of the walks. The default is 5.
    walk_per_node : integer, optional
        walk per node. The default is 5.
    seed : integer, optional
        random seed. The default is None.
    verbose : boolean, optional
        flag to print the results. The default is False.

    Returns
    -------
    walks : list
        list where each element is a random walks.

    '''
    if seed != None:
        random.seed(seed)
    else:
        random.seed()
        
    num_nodes = network.shape[0]
    
    neighbors, alias_tables, edges_alias_tables = preprocess_biased(network, p, q, verbose)
    network = None #the network is no more needed
    walks = []
    
    for h in range(num_nodes):
        for n in range(walk_per_node):
            if not len(neighbors[h]) > 0: #the node has no neighbors
                #use a list of -1 when it's not possible to compute the walk
                walks.append(['-1' for i in range(walk_length)]) 
                continue
            
            walk = []   
            walk.append(str(h))
            
            index = get_next_node(alias_tables[h][0], alias_tables[h][1])
            walk.append(str(neighbors[h][index]))
        
            while len(walk) < walk_length:
                curr_node = int(walk[-1])
                prev_node = int(walk[-2])
                index = get_next_node(edges_alias_tables[(prev_node, curr_node)][0], 
                                      edges_alias_tables[(prev_node, curr_node)][1])
                walk.append(str(neighbors[curr_node][index]))
            walks.append(walk)
        if h % 1000 == 0:
            print(str(h)+'/'+str(num_nodes))
        
    return walks 