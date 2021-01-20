from netNMFsc.utils import reorder, normalize

import numpy as np
import matplotlib.pyplot as plt
import pandas
import gcn
from scipy.io import mmread
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn import cluster

if __name__ == "__main__":      
    
    #load the network
    print('loading data...')
    netgenes = np.load('data/network/coexpedia_gene_names_mouse.npy')
    network = mmread('data/network/coexpedia_network_mouse').astype('float32').toarray()
    
    dataframe = pandas.read_csv('data/dataset/mESC.csv')
    dataframe = dataframe.drop('Unnamed: 0', axis=1)
    genes = [col for col in dataframe.columns]
    genes = np.array(genes)
    X = dataframe.to_numpy().astype('float32') 
    del dataframe
    #normalize and transpose
    X = normalize(X)
    X = X.transpose()
    
    #compute the adjacency matrix
    print('computation of the adjacency matrix starts')
    adj = reorder(genes, netgenes, network, 0.0)
   
    #split the dataset
    train_mask, val_mask, test_mask = gcn.prepare_masks(adj.shape[0])

    #build the model
    encoder, autoencoder = gcn.build_autoencoder(X.shape[1], adj.shape[0], conv1_dim=32, conv2_dim=5, verbose=False)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    #preprocssing
    adj = gcn.preprocess(adj)
    validation_data = ([X, adj], X, val_mask)
    
    #train the model
    print('model train starts')
    autoencoder.fit([X, adj], X,
          epochs=2,
          sample_weight=train_mask,
          validation_data=validation_data,
          batch_size=adj.shape[0],
          shuffle=False)

    X_enc = encoder.predict([X, adj], batch_size=adj.shape[0]) #gene representation in the latent space
    X_pred = autoencoder.predict([X, adj], batch_size=adj.shape[0])

    del test_mask
    del train_mask
    del val_mask
    del adj
    
    print('computing the clustering')
    num_cluster = 3
    kmeans = cluster.KMeans(n_clusters=num_cluster).fit(X_enc)
    predicted_labels = kmeans.labels_
    silhouette = metrics.silhouette_score(X_enc, predicted_labels)
    print('Silhouette score: ' + str(silhouette))
    
    points = TSNE(n_components=2, random_state=1).fit_transform(X_enc)
    plt.scatter(points[:, 0], points[:, 1], c=predicted_labels, s=1, cmap='viridis')