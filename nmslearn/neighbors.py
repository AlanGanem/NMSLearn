# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks_dev/neighbors.ipynb (unless otherwise specified).

__all__ = ['sparsify', 'hstack', 'vstack', 'stack', 'NMSLibSklearnWrapper', 'FastJaccardNN', 'FastL2NN', 'FastKLDivNN',
           'sparsify', 'hstack', 'vstack', 'stack', 'NMSLibSklearnWrapper', 'FastJaccardNN', 'FastL2NN', 'FastKLDivNN']

# Cell
from pathlib import Path
import time

import numpy as np
from scipy import sparse

import nmslib

from sklearn.base import BaseEstimator, TransformerMixin

# Cell

def sparsify(*arrs):
    '''
    makes input arrs sparse
    '''
    arrs = list(arrs)
    for i in range(len(arrs)):
        if not sparse.issparse(arrs[i]):
            arrs[i] = sparse.csr_matrix(arrs[i])

    return arrs

def _robust_stack(blocks, stack_method = 'stack', **kwargs):

    if any(sparse.issparse(i) for i in blocks):
        stacked = getattr(sparse, stack_method)(blocks, **kwargs)
    else:
        stacked = getattr(np, stack_method)(blocks, **kwargs)
    return stacked

def hstack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'hstack', **kwargs)

def vstack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'vstack', **kwargs)

def stack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'stack', **kwargs)

# Cell
class NMSLibSklearnWrapper(BaseEstimator):
    '''
    Generic wrapper for nmslib nearest neighbors under sklearn NN API.
    for distance types avalible, refer to https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
    '''
    def __init__(
        self,
        #init index params
        nmslib_method='hnsw',
        nmslib_space='jaccard_sparse',
        nmslib_data_type=nmslib.DataType.OBJECT_AS_STRING,
        nmslib_dtype = nmslib.DistType.FLOAT,
        nmslib_space_params = {},
        #index creation params
        index_time_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        #
        n_neighbors = 30,
        verbose = False,
        #x_prep_function
        X_prep_function = None
    ):


        self.nmslib_method = nmslib_method
        self.nmslib_space=nmslib_space
        self.nmslib_data_type=nmslib_data_type
        self.nmslib_space_params = nmslib_space_params
        self.nmslib_dtype = nmslib_dtype
        #index creation params
        self.index_time_params = index_time_params
        self.query_time_params = query_time_params
        #
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        #x_prep_function
        self.X_prep_function = X_prep_function

        pass

    def _preprocess_X(self, X):
        '''
        encodes sparse rows into str of id of nonzero columns
        '''
        if not self.X_prep_function is None:
            X = self.X_prep_function(X)

        return X


    def _instantiate_index(self,):
        '''
        method for instantiating index.
        usefull for pickling
        '''

        index = nmslib.init(
            method = self.nmslib_method,
            space = self.nmslib_space,
            data_type = self.nmslib_data_type,
            space_params = self.nmslib_space_params,
            dtype = self.nmslib_dtype,
        )

        return index

    def fit(self, X, y = None, **kwargs):
        '''
        instantiates and creates index
        '''
        #instantiate index
        index = self._instantiate_index()
        # preprocess X
        X_prep = self._preprocess_X(X)
        #add points to index
        index.addDataPointBatch(X_prep)
        # Create an index
        index.createIndex(self.index_time_params, self.verbose)
        #handle None for y (data to save under indexes)
        if y is None:
            y = np.ones((X.shape[0], 0)) #empty array
        # save states
        self.index_ = index
        self.y_ = y
        self.X_ = X
        return self

    def partial_fit(self, X, y = None, **kwargs):
        '''
        adds new datapoints to index and y.
        estimator needs to be fit prior to calling partial fit,
        so first call fit in the first batch of data, then call partial fit
        passing the subsequent batches
        '''
        #assume index is already instantiated
        # preprocess X
        X_prep = self._preprocess_X(X)
        #add points to index
        self.index_.addDataPointBatch(X_prep)
        # Create an index
        self.index_.createIndex(self.index_time_params, self.verbose)
        #handle None for y (data to save under indexes)
        if y is None:
            y = np.ones((X.shape[0], 0)) #empty array
        # save states
        self.y_ = vstack([self.y_, y])
        self.X_ = vstack([self.X_, X])
        return self

    def kneighbors(self, X = None, n_neighbors = None, return_distance = True, query_time_params = None, n_jobs = 4):
        '''
        query neighbors, if X is None, will return the neighbors of each point in index
        '''
        if query_time_params is None:
            query_time_params = self.query_time_params
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self.X_

        #preprocess X
        X = self._preprocess_X(X)

        self.index_.setQueryTimeParams(query_time_params)

        # Querying
        start = time.time()
        nbrs = self.index_.knnQueryBatch(X, k = n_neighbors, num_threads = n_jobs)
        end = time.time()

        if self.verbose:
            try:
                query_qty = len(X)
            except:
                query_qty = X.shape[0]
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end-start, float(end-start)/query_qty, n_jobs*float(end-start)/query_qty))

        if return_distance:
            distances = [nb[1] for nb in nbrs]
            nbrs = [nb[0] for nb in nbrs]
            return distances, nbrs
        else:
            nbrs = [nb[0] for nb in nbrs]
            return nbrs

    def __getstate__(self,):
        '''
        creates binary file for index and then saves into object attribute to be pickled alongside
        other attributes.
        '''
        #read tempfiles with binaries to save binary str inside object
        tempfile_name = fr'.~nmslib_index_{str(int(time.time()*1e7))}'
        self.index_.saveIndex(tempfile_name, save_data = True)

        with open(tempfile_name, 'rb') as f:
            fb = f.read()
        with open(tempfile_name+'.dat', 'rb') as f:
            fb_dat = f.read()

        #save binary as attribute (index and data)
        self.index_ = (fb,fb_dat)
        #delete tempfiles
        Path(tempfile_name).unlink()
        Path(tempfile_name+'.dat').unlink()
        return self.__dict__

    def __setstate__(self,d):
        '''
        sets state during unpickling.
        instantiates index and loads binary index
        '''
        self.__dict__ = d

        #write tempfiles with binaries to load from index.loadIndex
        tempfile_name = fr'.~nmslib_index_{str(int(time.time()*1e7))}'
        with open(tempfile_name, 'wb') as f:
            f.write(self.index_[0])

        with open(tempfile_name+'.dat', 'wb') as f:
            f.write(self.index_[1])

        index = self._instantiate_index()
        index.loadIndex(tempfile_name, load_data = True)
        #sets self.index_
        self.index_ = index
        #delete tempfile
        Path(tempfile_name).unlink()
        Path(tempfile_name+'.dat').unlink()
        return

# Cell
def _preprocess_sparse_to_idx_str(X):
    '''
    encodes sparse rows into str of id of nonzero columns
    '''
    #ensure is sparse
    X = sparse.csr_matrix(X)
    indptr = X.indptr
    cols = X.tocoo().col.astype(str)
    id_strs = [*(' '.join(cols[slice(*indptr[i:i+2])]) for i in range(len(indptr)-1))]
    return id_strs


class FastJaccardNN(NMSLibSklearnWrapper):

    def __init__(
        self,
        n_neighbors = 30,
        index_time_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        verbose = False,
    ):

        super().__init__(
            #jaccard init params
            nmslib_method='hnsw',
            nmslib_space='jaccard_sparse',
            nmslib_data_type=nmslib.DataType.OBJECT_AS_STRING,
            nmslib_dtype = nmslib.DistType.FLOAT,
            nmslib_space_params = {},
            #other params
            X_prep_function = _preprocess_sparse_to_idx_str,
            n_neighbors = n_neighbors,
            index_time_params = index_time_params,
            query_time_params = query_time_params,
            verbose = verbose,
        )
        return

class FastL2NN(NMSLibSklearnWrapper):

    def __init__(
        self,
        n_neighbors = 30,
        index_time_params = {'M': 30, 'indexThreadQty': 8, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        verbose = False,
    ):

        super().__init__(
            #jaccard init params
            nmslib_method='hnsw',
            nmslib_space='l2',
            nmslib_data_type=nmslib.DataType.DENSE_VECTOR,
            nmslib_dtype = nmslib.DistType.FLOAT,
            nmslib_space_params = {},
            #other params
            X_prep_function = None,
            n_neighbors = n_neighbors,
            index_time_params = index_time_params,
            query_time_params = query_time_params,
            verbose = verbose,
        )
        return


class FastKLDivNN(NMSLibSklearnWrapper):

    def __init__(
        self,
        n_neighbors = 30,
        index_time_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        verbose = False,
    ):


        super().__init__(
            #kldib init params
            nmslib_method='sw-graph',
            nmslib_space='kldivgenfast',
            nmslib_data_type=nmslib.DataType.DENSE_VECTOR,
            nmslib_dtype = nmslib.DistType.FLOAT,
            nmslib_space_params = {},
            #other params
            X_prep_function = None,#_preprocess_sparse_to_idx_str,
            n_neighbors = n_neighbors,
            index_time_params = index_time_params,
            query_time_params = query_time_params,
            verbose = verbose,
        )
        return


# Cell
from pathlib import Path
import time

import numpy as np
from scipy import sparse

import nmslib

from sklearn.base import BaseEstimator, TransformerMixin

# Cell

def sparsify(*arrs):
    '''
    makes input arrs sparse
    '''
    arrs = list(arrs)
    for i in range(len(arrs)):
        if not sparse.issparse(arrs[i]):
            arrs[i] = sparse.csr_matrix(arrs[i])

    return arrs

def _robust_stack(blocks, stack_method = 'stack', **kwargs):

    if any(sparse.issparse(i) for i in blocks):
        stacked = getattr(sparse, stack_method)(blocks, **kwargs)
    else:
        stacked = getattr(np, stack_method)(blocks, **kwargs)
    return stacked

def hstack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'hstack', **kwargs)

def vstack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'vstack', **kwargs)

def stack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'stack', **kwargs)

# Cell
class NMSLibSklearnWrapper(BaseEstimator):
    '''
    Generic wrapper for nmslib nearest neighbors under sklearn NN API.
    for distance types avalible, refer to https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
    '''
    def __init__(
        self,
        #init index params
        nmslib_method='hnsw',
        nmslib_space='jaccard_sparse',
        nmslib_data_type=nmslib.DataType.OBJECT_AS_STRING,
        nmslib_dtype = nmslib.DistType.FLOAT,
        nmslib_space_params = {},
        #index creation params
        index_time_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        #
        n_neighbors = 30,
        verbose = False,
        #x_prep_function
        X_prep_function = None
    ):


        self.nmslib_method = nmslib_method
        self.nmslib_space=nmslib_space
        self.nmslib_data_type=nmslib_data_type
        self.nmslib_space_params = nmslib_space_params
        self.nmslib_dtype = nmslib_dtype
        #index creation params
        self.index_time_params = index_time_params
        self.query_time_params = query_time_params
        #
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        #x_prep_function
        self.X_prep_function = X_prep_function

        pass

    def _preprocess_X(self, X):
        '''
        encodes sparse rows into str of id of nonzero columns
        '''
        if not self.X_prep_function is None:
            X = self.X_prep_function(X)

        return X


    def _instantiate_index(self,):
        '''
        method for instantiating index.
        usefull for pickling
        '''

        index = nmslib.init(
            method = self.nmslib_method,
            space = self.nmslib_space,
            data_type = self.nmslib_data_type,
            space_params = self.nmslib_space_params,
            dtype = self.nmslib_dtype,
        )

        return index

    def fit(self, X, y = None, **kwargs):
        '''
        instantiates and creates index
        '''
        #instantiate index
        index = self._instantiate_index()
        # preprocess X
        X_prep = self._preprocess_X(X)
        #add points to index
        index.addDataPointBatch(X_prep)
        # Create an index
        index.createIndex(self.index_time_params, self.verbose)
        #handle None for y (data to save under indexes)
        if y is None:
            y = np.ones((X.shape[0], 0)) #empty array
        # save states
        self.index_ = index
        self.y_ = y
        self.X_ = X
        return self

    def partial_fit(self, X, y = None, **kwargs):
        '''
        adds new datapoints to index and y.
        estimator needs to be fit prior to calling partial fit,
        so first call fit in the first batch of data, then call partial fit
        passing the subsequent batches
        '''
        #assume index is already instantiated
        # preprocess X
        X_prep = self._preprocess_X(X)
        #add points to index
        self.index_.addDataPointBatch(X_prep)
        # Create an index
        self.index_.createIndex(self.index_time_params, self.verbose)
        #handle None for y (data to save under indexes)
        if y is None:
            y = np.ones((X.shape[0], 0)) #empty array
        # save states
        self.y_ = vstack([self.y_, y])
        self.X_ = vstack([self.X_, X])
        return self

    def kneighbors(self, X = None, n_neighbors = None, return_distance = True, query_time_params = None, n_jobs = 4):
        '''
        query neighbors, if X is None, will return the neighbors of each point in index
        '''
        if query_time_params is None:
            query_time_params = self.query_time_params
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self.X_

        #preprocess X
        X = self._preprocess_X(X)

        self.index_.setQueryTimeParams(query_time_params)

        # Querying
        start = time.time()
        nbrs = self.index_.knnQueryBatch(X, k = n_neighbors, num_threads = n_jobs)
        end = time.time()

        if self.verbose:
            try:
                query_qty = len(X)
            except:
                query_qty = X.shape[0]
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end-start, float(end-start)/query_qty, n_jobs*float(end-start)/query_qty))

        if return_distance:
            distances = [nb[1] for nb in nbrs]
            nbrs = [nb[0] for nb in nbrs]
            return distances, nbrs
        else:
            nbrs = [nb[0] for nb in nbrs]
            return nbrs

    def __getstate__(self,):
        '''
        creates binary file for index and then saves into object attribute to be pickled alongside
        other attributes.
        '''
        #read tempfiles with binaries to save binary str inside object
        tempfile_name = fr'.~nmslib_index_{str(int(time.time()*1e7))}'
        self.index_.saveIndex(tempfile_name, save_data = True)

        with open(tempfile_name, 'rb') as f:
            fb = f.read()
        with open(tempfile_name+'.dat', 'rb') as f:
            fb_dat = f.read()

        #save binary as attribute (index and data)
        self.index_ = (fb,fb_dat)
        #delete tempfiles
        Path(tempfile_name).unlink()
        Path(tempfile_name+'.dat').unlink()
        return self.__dict__

    def __setstate__(self,d):
        '''
        sets state during unpickling.
        instantiates index and loads binary index
        '''
        self.__dict__ = d

        #write tempfiles with binaries to load from index.loadIndex
        tempfile_name = fr'.~nmslib_index_{str(int(time.time()*1e7))}'
        with open(tempfile_name, 'wb') as f:
            f.write(self.index_[0])

        with open(tempfile_name+'.dat', 'wb') as f:
            f.write(self.index_[1])

        index = self._instantiate_index()
        index.loadIndex(tempfile_name, load_data = True)
        #sets self.index_
        self.index_ = index
        #delete tempfile
        Path(tempfile_name).unlink()
        Path(tempfile_name+'.dat').unlink()
        return

# Cell
def _preprocess_sparse_to_idx_str(X):
    '''
    encodes sparse rows into str of id of nonzero columns
    '''
    #ensure is sparse
    X = sparse.csr_matrix(X)
    indptr = X.indptr
    cols = X.tocoo().col.astype(str)
    id_strs = [*(' '.join(cols[slice(*indptr[i:i+2])]) for i in range(len(indptr)-1))]
    return id_strs


class FastJaccardNN(NMSLibSklearnWrapper):

    def __init__(
        self,
        n_neighbors = 30,
        index_time_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        verbose = False,
    ):

        super().__init__(
            #jaccard init params
            nmslib_method='hnsw',
            nmslib_space='jaccard_sparse',
            nmslib_data_type=nmslib.DataType.OBJECT_AS_STRING,
            nmslib_dtype = nmslib.DistType.FLOAT,
            nmslib_space_params = {},
            #other params
            X_prep_function = _preprocess_sparse_to_idx_str,
            n_neighbors = n_neighbors,
            index_time_params = index_time_params,
            query_time_params = query_time_params,
            verbose = verbose,
        )
        return

class FastL2NN(NMSLibSklearnWrapper):

    def __init__(
        self,
        n_neighbors = 30,
        index_time_params = {'M': 30, 'indexThreadQty': 8, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        verbose = False,
    ):

        super().__init__(
            #jaccard init params
            nmslib_method='hnsw',
            nmslib_space='l2',
            nmslib_data_type=nmslib.DataType.DENSE_VECTOR,
            nmslib_dtype = nmslib.DistType.FLOAT,
            nmslib_space_params = {},
            #other params
            X_prep_function = None,
            n_neighbors = n_neighbors,
            index_time_params = index_time_params,
            query_time_params = query_time_params,
            verbose = verbose,
        )
        return


class FastKLDivNN(NMSLibSklearnWrapper):

    def __init__(
        self,
        n_neighbors = 30,
        index_time_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0},
        query_time_params = {'efSearch': 100},
        verbose = False,
    ):


        super().__init__(
            #kldib init params
            nmslib_method='sw-graph',
            nmslib_space='kldivgenfast',
            nmslib_data_type=nmslib.DataType.DENSE_VECTOR,
            nmslib_dtype = nmslib.DistType.FLOAT,
            nmslib_space_params = {},
            #other params
            X_prep_function = None,#_preprocess_sparse_to_idx_str,
            n_neighbors = n_neighbors,
            index_time_params = index_time_params,
            query_time_params = query_time_params,
            verbose = verbose,
        )
        return
