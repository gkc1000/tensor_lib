#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys


def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

class KnowValues(unittest.TestCase):
    def test_dot_1d(self):
        a1 = numpy.ones(4)
        self.assertTrue(allclose(ctf.dot(ctf.astensor(a1), a1), numpy.dot(a1, a1)))
        self.assertTrue(allclose(ctf.dot(a1+1j, ctf.astensor(a1)), numpy.dot(a1+1j, a1)))
        a2 = ctf.astensor(a1).dot(a1+0j)
        self.assertTrue(a2.dtype == numpy.complex128)
        #self.assertTrue(ctf.astensor(a1).dot(a1+0j).dtype == numpy.complex)

    def test_dot_2d(self):
        a1 = numpy.random.random(4)
        a2 = numpy.random.random((4,3))
        self.assertTrue(ctf.dot(ctf.astensor(a1), ctf.astensor(a2)).shape == (3,))
        self.assertTrue(allclose(ctf.dot(a1, ctf.astensor(a2)), numpy.dot(a1, a2)))
        self.assertTrue(ctf.dot(ctf.astensor(a2).T(), a1).shape == (3,))
        self.assertTrue(allclose(ctf.dot(ctf.astensor(a2).T(), a1), numpy.dot(a2.T, a1)))

        with self.assertRaises(ValueError):
            ctf.dot(a2, a2)
        self.assertTrue(allclose(ctf.dot(ctf.astensor(a2).T(), a2), numpy.dot(a2.T, a2)))
        self.assertTrue(allclose(ctf.astensor(a2).dot(a2.T), a2.dot(a2.T)))

    def test_tensordot(self):
        a0 = numpy.random.random((2,2,2))
        self.assertTrue(allclose(ctf.tensordot(a0, a0), numpy.tensordot(a0, a0)))
        self.assertTrue(allclose(ctf.tensordot(a0, a0, 1), numpy.tensordot(a0, a0, 1)))
        self.assertTrue(allclose(ctf.tensordot(a0, a0, [[1,0],[1,0]]), numpy.tensordot(a0, a0, [[1,0],[1,0]])))
        self.assertTrue(allclose(ctf.tensordot(a0, a0, [[0,1],[1,0]]), numpy.tensordot(a0, a0, [[0,1],[1,0]])))
        self.assertTrue(allclose(ctf.tensordot(a0, a0, [[2,1,0],[1,0,2]]), numpy.tensordot(a0, a0, [[2,1,0],[1,0,2]])))
        with self.assertRaises(IndexError):
            ctf.tensordot(a0, a0, [[2,1,0,3],[0,1,2,3]])


if __name__ == "__main__":
    numpy.random.seed(5330);
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for dot")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    ctf.MPI_Stop()
    print result
    sys.exit(not result)
