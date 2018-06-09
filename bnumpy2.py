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

    def reshape(self, newshape, block_shape=None):
        obj=np.reshape(self, newshape)
        if block_shape is not None:
            for ix in np.ndindex(obj.shape):
                obj[ix] = np.reshape(obj[ix], block_shape)
        return obj
        
    def transpose(self, axes=None):
        obj = self.copy()
        obj = np.ndarray.transpose(obj, axes)
        if axes == None:
            obj.block_shape = self.block_shape[::-1]
        else:
            obj.block_shape = self.block_shape[axes]
        
        for ix in np.ndindex(obj.shape):
            obj[ix] = obj[ix].transpose(axes)
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

# should be in bnumpy/random
def random(shape, block_shape, dtype=float):
    obj = bndarray(shape, block_shape, dtype)
    for ix in np.ndindex(obj.shape):
        obj[ix] = np.random.random(obj.block_shape)
    return obj
    
    
def asndarray(ba):
    shape = np.multiply(ba.shape, ba.block_shape)
    a = np.empty(shape, dtype=ba.block_dtype)
    for ix in np.ndindex(ba.shape):
        start = np.multiply(ix, ba.block_shape)
        end = np.add(start, ba.block_shape)
        slices = [slice(s,e) for (s,e) in zip(start,end)]
        a[slices] = ba[ix]
    return a
    



def test():
     print "\nmain program\n"
     #a = bndarray((3,1,2), (2,4,7)) 
     #a = bndarray((1,2), (2,4))
     a = random((1,2), (2,4))
     b = a.transpose()
     
     print asndarray(a.transpose()).shape
     print asndarray(b).shape
     print asndarray(a).transpose().shape

     at = asndarray(a).transpose()
     print np.allclose(at,asndarray(b))
     
     #print a.transpose().shape
     #print asndarray(a).shape
     

if __name__ == '__main__':
     test()
