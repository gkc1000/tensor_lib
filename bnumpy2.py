#!/usr/bin/env python
import numpy as np

class bndarray(np.ndarray):
    def __new__(subtype, shape, block_shape, block_dtype=float):
        if len(shape)!=len(block_shape):
            raise RuntimeError
        obj = np.ndarray.__new__(subtype, shape, dtype=np.object)
        obj.block_shape = tuple(block_shape)
        obj.block_dtype = block_dtype

        for ix in np.ndindex(obj.shape):
            obj[ix]=np.empty(block_shape, dtype=block_dtype)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            self.block_shape = obj.block_shape
            self.block_dtype = obj.block_dtype

    def transpose(self, axes=None):
        obj = np.ndarray.transpose(self, axes)
        for ix in np.nditer(obj, op_flags='readwrite'):
            ix[...] = ix.transpose(axes)
        return obj

def zeros(shape, block_shape, dtype=float):
    obj = bndarray(shape, block_shape, dtype)
    for ix in np.ndindex(obj.shape):
        obj[ix] = np.zeros(obj.block_shape, dtype)
    return obj

def eye(N, block_N, dtype=float):
    shape = [N, N]
    block_shape = [block_N, block_N]
    obj = zeros(shape, block_shape, dtype)
    for i in range(obj.shape[0]):
        obj[i,i] = np.eye(block_N, dtype=dtype)
    return obj

def asndarray(ba):
    shape = np.multiply(ba.shape, ba.block_shape)
    a = np.empty(shape, dtype=ba.block_dtype)
    for ix in np.ndindex(ba.shape):
        start = np.multiply(ix, ba.block_shape)
        end = np.add(start , ba.block_shape)
        slices = [slice(s,e) for (s,e) in zip(start,end)]
        a[slices] = ba[ix]
    return a
    



# if __name__ == '__main__':
#     print "\nmain program\n"
#     a = bndarray((3,1,2), (2,2,2)) 
#     print a
#     print asndarray(a)

