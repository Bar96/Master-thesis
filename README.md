# Master-thesis

-The folder R contains all the code (mostly R code) to generate simulated datasets, in particular:
	-the folder splatter contains the modified version of the splatter algorithm
	-generate_rnd_subnet.py is a python script to produce a network required to splatter
	-generate_dataset.R is a R script that executes splatter and generates the datastes
	
-The folder netNMFsc contains the code for the netNMF-sc algorithm

-gcn.py contains the functions to create a graph convolutional autoencoder

-random_walks.py contains the functions to generate standard and biased random walks

-simulation.py is an example of the use of netNMF-sc and the cell clustering of simulated data

-gcn_simulation.py is an example of the use of GCNs and the gene clustering
