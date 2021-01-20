import random
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix


subnet_dim = 5000
network = mmread('coexpedia_network_mouse.mtx').astype('float32').toarray()
netgenes = np.load('coexpedia_gene_names_mouse.npy')
	

subnet_binary = np.zeros([subnet_dim, subnet_dim], dtype=np.int16)
subnet = np.zeros([subnet_dim, subnet_dim], dtype=np.float32)
subnet_netgenes = []

index_list = [i for i in range(network.shape[0])]
sel_indices = []
for i in range(subnet_dim):
	rnd_idx = random.randint(0, len(index_list)-1)
	elem = index_list[rnd_idx]
	sel_indices.append(elem)
	index_list.remove(elem)

sel_indices = sorted(sel_indices)

for i in range(subnet_dim):
	subnet_netgenes.append(netgenes[sel_indices[i]])
	for j in range(i+1, subnet_dim):
		idx1 = sel_indices[i]
		idx2 = sel_indices[j]        
		if network[idx1][idx2] > 0:
		    subnet_binary[i][j] = 1
		    subnet_binary[j][i] = 1
		    subnet[i][j] = network[idx1][idx2]
		    subnet[j][i] = network[idx1][idx2]

subnet_binary = csr_matrix(subnet_binary)
subnet = csr_matrix(subnet)
subnet_netgenes = np.array(subnet_netgenes)
mmwrite('net/coexpedia_subnet_binary1.mtx', subnet_binary)
mmwrite('net/coexpedia_subnet1.mtx', subnet)
np.save('net/genes1.npy', subnet_netgenes)
np.save('seq/genes1.npy', subnet_netgenes)
