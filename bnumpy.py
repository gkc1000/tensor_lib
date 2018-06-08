import numpy as np
import pyscf.lib

def shape_from_lens(block_lens):
    shape = [len(x) for x in block_lens]
    prod_lens = pyscf.lib.cartesian_prod(block_lens)
    return np.reshape(pyscf.lib.cartesian_prod(block_lens),
                      shape + [prod_lens.shape[1]])

def lens_from_shape(block_shape):
    pass
    
class bndarray(np.ndarray):
    def __new__(subtype, block_shape, block_dtype=float):
        shape = block_shape.shape
        obj = np.ndarray.__new__(subtype, shape, dtype=np.object)
        obj.block_shape = block_shape
        obj.block_dtype = block_dtype
        
        for ix in np.ndindex(shape):
            obj[ix] = np.empty(block_shape[ix], block_dtype)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            for ix in np.ndindex(obj.shape):
                self.block_shape=obj[ix].shape

def zeros(block_lens, dtype=float):
    block_shape = make_block_shape(block_lens)
    obj = bndarray(block_shape, dtype)
    for ix in np.ndindex(obj.shape):
        obj[ix] = np.zeros(obj.block_shape[ix], dtype)
    return obj

def eye(block_N, dtype=float):
    block_shape = make_block_shape([block_N,block_N])
    obj = bndarray(block_shape, dtype)
    for i in range(obj.shape[0]):
        obj[i,i] = np.eye(obj.block_shape[i,i], dtype)
    return obj

def npshape(block_shape):
    block_lens=

def asndarray(ba):
    pass
