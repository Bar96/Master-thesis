from __future__ import print_function
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
from . import utils
import copy, argparse, os, math, random, time
from scipy import io, linalg
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.linalg import blas
import warnings
import pandas as pd
from numpy import dot, multiply

from math import sqrt
#import warnings
import numbers
#import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.extmath import safe_min
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast

EPSILON = np.finfo(np.float32).eps  # smallest representable number such that 1.0 + EPSILON != 1.0.

INTEGER_TYPES = (numbers.Integral, np.integer)


class netNMFGD:
    '''
    Performs netNMF-sc with gradient descent using Tensorflow
    '''

    def __init__(self, distance="KL", d=None, N=None, alpha=10, n_inits=1, tol=1e-2, max_iter=20000, n_jobs=1,
                 weight=0.1, parallel_backend='multiprocessing', normalize=True, sparsity=0.75, lr=0.0001, use_prob_matrix=False):
        """
            d:          number of dimensions
            N:          Network (weighted adjacency matrix)
            alpha:      regularization parameter
            n_inits:    number of runs to make with different random inits (in order to avoid being stuck in local minima)
            n_jobs:     number of parallel jobs to run, when n_inits > 1
            tol:        stopping criteria
            max_iter:   stopping criteria
            use_prob_matrix: flag to use the probability matrix in substitution of the laplacian
        """
        self.X = None
        self.M = None
        self.d = d
        self.N = N
        self.alpha = alpha
        self.n_inits = n_inits
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.normalize = normalize
        self.sparsity = sparsity
        self.weight = weight
        self.distance = distance
        self.lr = lr
        self.use_prob_matrix = use_prob_matrix


    def _init(self, X):
        temp_H = np.random.randn(self.d, X.shape[1]).astype(np.float32)
        temp_W = np.random.randn(X.shape[0], self.d).astype(np.float32)
        temp_H = np.array(temp_H, order='F')
        temp_W = np.array(temp_W, order='F')
        return abs(temp_H), abs(temp_W)


    def _fit(self, X):
        import tensorflow as tf
        tf.compat.v1.disable_v2_behavior()      #added this line
        temp_H, temp_W = self._init(X)
        conv = False

        mask = tf.constant(self.M.astype(np.float32)) 
        eps = tf.constant(np.float32(1e-8))
        A = tf.constant(X.astype(np.float32)) + eps  
        H = tf.Variable(temp_H.astype(np.float32))
        W = tf.Variable(temp_W.astype(np.float32))
        #print(np.max(mask), np.min(mask), np.sum(mask))
        WH = tf.matmul(W, H)
        if self.weight < 1:
            WH = tf.multiply(mask, WH)
        WH += eps
        L_s = tf.constant(self.L.astype(np.float32))  # initialize L, the laplacian of the gene co expression network
        alpha_s = tf.constant(np.float32(self.alpha))

        if self.distance == 'frobenius':
            cost0 = tf.reduce_sum(tf.pow(A - WH, 2))
            costL = alpha_s * tf.linalg.trace(tf.matmul(tf.transpose(W), tf.matmul(L_s, W)))  # alpha*Tr(W^T L W)
        elif self.distance == 'KL':
            cost0 = tf.reduce_sum(tf.multiply(A, tf.math.log(tf.divide(A, WH))) - A + WH)   #NB correction: divide instead of div 
            costL = alpha_s * tf.linalg.trace(tf.matmul(tf.transpose(W), tf.matmul(L_s, W))) #NB correction: tf.linalg.trace instead of tf.trace
        else:
            raise ValueError('Select frobenius or KL for distance')

        if self.alpha > 0:
            cost = cost0 + costL
        else:
            cost = cost0

        lr = self.lr
        decay = 0.95

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)
        learning_rate = tf.compat.v1.train.exponential_decay(lr, global_step, self.max_iter, decay, staircase=True)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=.1)
        train_step = optimizer.minimize(cost0, global_step=global_step)

        init = tf.compat.v1.global_variables_initializer()
        # Clipping operation. This ensure that W and H learnt are non-negative
        clip_W = tf.compat.v1.assign(W, tf.maximum(tf.zeros_like(W), W))
        clip_H = tf.compat.v1.assign(H, tf.maximum(tf.zeros_like(H), H))
        clip = tf.group(clip_W, clip_H)

        c = np.inf
        with tf.compat.v1.Session() as sess:
            print('The learning starts')                   # Added this line
            sess.run(init)
            for i in range(self.max_iter):
                sess.run(train_step)
                sess.run(clip)
                if i % 300 == 0:
                    c2 = sess.run(cost)
                    e = c - c2
                    c = c2
                    if i % 1000 == 0:
                        print(i, c, e)
                    if e < self.tol:
                        conv = True
                        break
                #print('iteration: '+str(i))
            learnt_W = sess.run(W)
            learnt_H = sess.run(H)
        tf.compat.v1.reset_default_graph()    # NB added compat.v1

        return {
            'conv': conv,
            'obj': c,
            'H': learnt_H,
            'W': learnt_W
        }


    def load_10X(self, direc=None, genome='mm10'):
        if direc.endswith('hdf5') or direc.endswith('h5'):
            X, genenames = utils.import_10X_hdf5(direc, genome)
        else:
            X, genenames = utils.import_10X_mtx(direc)
        self.X = X
        self.genes = genenames


    def load_network(self, net=None, genenames=None, sparsity=.75):
        if net:
            if net.endswith('.txt'):
                network = utils.import_network_from_gene_pairs(net)
                #network, netgenes = utils.import_network_from_gene_pairs(net, genenames) #NB this function returns only the network
            else:
                network, netgenes = utils.import_network(net, genenames, sparsity)
        network = utils.network_threshold(network, sparsity)
        self.N = network
        self.netgenes = netgenes


    def fit_transform(self, X=None):
        if type(X) == np.ndarray:
            self.X = X
        
        if type(self.genes) == np.ndarray and type(
                self.netgenes) == np.ndarray:  # if imported data from file reorder network to match genes in X
            assert type(self.X) == np.ndarray
            assert type(self.N) == np.ndarray
            if not self.use_prob_matrix: #if the prob matrix is not used, reorder the network
                network = utils.reorder(self.genes, self.netgenes, self.N, self.sparsity)
                self.N = network
                self.netgenes = self.genes
        
        if self.normalize:
            print('library size normalizing...')
            self.X = utils.normalize(self.X)
        # self.X = utils.log_transform(self.X)
        M = np.ones_like(self.X)
        M[self.X == 0] = self.weight
        self.M = M
        if self.d is None:
            self.d = min(X.shape)
            print('rank set to:', self.d)
        if self.use_prob_matrix: #the probability matrix is used
            self.L = self.N
        else: 
            if self.N is not None: #otherwise compute the Laplacian
                if np.max(abs(self.N)) > 0:
                    self.N = self.N / np.max(abs(self.N))
                N = self.N
                D = np.sum(abs(self.N), axis=0) * np.eye(self.N.shape[0])  
                print(np.count_nonzero(N), 'edges')
                self.D = D
                self.N = N
                self.L = self.D - self.N
                assert utils.check_symmetric(self.L)
            else:
                self.N = np.eye(X.shape[0])
                self.D = np.eye(X.shape[0])
                self.L = self.D - self.N
        
        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(  # execute n_inits instances of the problem in parallel
            delayed(self._fit)(self.X) for x in range(self.n_inits))            # delayed(function)(arguments) 
        best_results = {"obj": np.inf, "H": None, "W": None}
        for r in results:                                                       #find the best result among the different instances executed
            if r['obj'] < best_results['obj']:
                best_results = r
        if 'conv' not in best_results:
            warn("Did not converge after {} iterations. Error is {}. Try increasing `max_iter`.".format(self.max_iter,
                                                                                                        best_results[
                                                                                                            'e']))
        return (best_results['W'], best_results['H'])
