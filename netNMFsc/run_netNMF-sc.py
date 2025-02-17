# run netNMF-sc from command line and save outputs to specified directory
from __future__ import print_function
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
import copy, argparse, os, math, random, time
from scipy import sparse, io, linalg
from scipy.sparse import csr_matrix
import warnings, os
from . import plot

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
   
    if args.method == 'GD':
        from .classes import netNMFGD
        operator = netNMFGD(d=args.dimensions, alpha=args.alpha, n_inits=1, tol=args.tol, max_iter=args.max_iters, n_jobs=1)
    elif args.method == 'MU':
        from .classes import netNMFMU   
        operator = netNMFMU(d=args.dimensions, alpha=args.alpha, n_inits=1, tol=args.tol, max_iter=args.max_iters, n_jobs=1)
    operator.load_10X(direc=args.tenXdir,genome='mm10')
    operator.load_network(net=args.network,genenames=args.netgenes,sparsity=args.sparsity)
    W, H = operator.fit_transform()
    k,clusters = plot.select_clusters(H,max_clusters=20)
    plot.tSNE(H,clusters,fname=args.direc + '/netNMFsc_tsne')
    os.system('mkdir -p %s'%(args.direc))
    np.save(os.path.join(args.direc,'W.npy'),W)
    np.save(os.path.join(args.direc,'H.npy'),H)
    return
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="either 'GD for gradient descent or MU for multiplicative update",
                        type=str, default='GD')
    parser.add_argument("-f", "--filename", help="path to data file (.npy or .mtx)", type=str, default='matrix.mtx')
    parser.add_argument("-g", "--gene_names", help="path to file containing gene names (.npy or .tsv)", type=str,
                        default='gene_names.tsv')
    parser.add_argument("-net", "--network", help="path to network file (.npy or .mtx)", type=str, default='')
    parser.add_argument("-netgenes", "--netgenes", help="path to file containing gene names for network (.npy or .tsv)",
                        type=str, default='')
    parser.add_argument("-org", "--organism", help="mouse or human", type=str, default='human')
    parser.add_argument("-id", "--idtype", help="ensemble, symbol, or entrez", type=str, default='ensemble')
    parser.add_argument("-netid", "--netidtype", help="ensemble, symbol, or entrez", type=str, default='entrez')
    parser.add_argument("-n", "--normalize", help="normalize data? 1 = yes, 0 = no", type=int, default=0)
    parser.add_argument("-sparse", "--sparsity", help="sparsity for network", type=float, default=0.99)
    parser.add_argument("-mi", "--max_iters", help="max iters for netNMF-sc", type=int, default=1500)
    parser.add_argument("-t", "--tol", help="tolerence for netNMF-sc", type=float, default=1e-2)
    parser.add_argument("-d", "--direc", help="directory to save files", default='')
    parser.add_argument("-D", "--dimensions", help="number of dimensions to apply shift", type=int, default=10)
    parser.add_argument("-a", "--alpha", help="lambda param for netNMF-sc", type=float, default=1.0)
    parser.add_argument("-x", "--tenXdir",
                        help="data is from 10X. Only required to provide directory containing matrix.mtx, genes.tsv, barcodes.tsv files",
                        type=str, default='')
    args = parser.parse_args()
    main(args)
    
