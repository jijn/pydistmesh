# encoding: utf-8
"""Distance functions implemented in C."""

#-----------------------------------------------------------------------------
#  Copyright (C) 2004-2012 Per-Olof Persson
#  Copyright (C) 2012 Bradley Froehle

#  Distributed under the terms of the GNU General Public License. You should
#  have received a copy of the license along with this program. If not,
#  see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Cython imports
#-----------------------------------------------------------------------------

cimport numpy as cnp
import numpy as np

# Define NumPy's API version to avoid compatibility issues
cdef extern from "numpy/arrayobject.h":
    void import_array()

# Initialize NumPy
import_array()


cdef extern from "src/distance_functions.c":
    double _dellipse "dellipse" (double x0, double y0, double a, double b)
    double _dellipsoid "dellipsoid" (double x0, double y0, double z0,
            double a, double b, double z)
    double _dsegment "dsegment" (double x0, double y0, double p1x, double p1y,
            double p2x, double p2y)

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def dellipse(cnp.ndarray[cnp.float64_t, ndim=2] p, cnp.ndarray[cnp.float64_t, ndim=1] axes):
    """
    d = dellipse(p, axes)

    Parameters
    ----------
    p : array, shape (np, 2)
        points
    axes : array, shape (2,)

    Returns
    -------
    d = array, shape (np, )
        distance from each point to the ellipse
    """
    cdef double a, b
    cdef Py_ssize_t n, i
    
    a, b = axes[0], axes[1]
    
    n = p.shape[0]
    assert p.shape[1] == 2, "array should have shape (np, 2)"
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] D = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        D[i] = _dellipse(p[i, 0], p[i, 1], a, b)
    
    return D

def dellipsoid(cnp.ndarray[cnp.float64_t, ndim=2] p, cnp.ndarray[cnp.float64_t, ndim=1] axes):
    """
    d = dellipsoid(p, axes)

    Parameters
    ----------
    p : array, shape (np, 3)
        points
    axes : array, shape (3,)

    Returns
    -------
    d = array, shape (np, )
        distance from each point to the ellipsoid
    """
    cdef double a, b, c
    cdef Py_ssize_t n, i
    
    a, b, c = axes[0], axes[1], axes[2]
    
    n = p.shape[0]
    assert p.shape[1] == 3, "array should have shape (np, 3)"
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] D = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        D[i] = _dellipsoid(p[i, 0], p[i, 1], p[i, 2], a, b, c)
    
    return D

def dsegment(cnp.ndarray[cnp.float64_t, ndim=2] p, cnp.ndarray[cnp.float64_t, ndim=2] v):
    """
    d = dsegment(p, v)

    Parameters
    ----------
    p : array, shape (np, 2)
        points
    v : array, shape (nv, 2)
        vertices of a closed array, whose edges are v[0]..v[1],
        ... v[nv-2]..v[nv-1]

    Output
    ------
    ds : array, shape (np, nv-1)
        distance from each point to each edge
    """
    cdef double p1x, p1y, p2x, p2y
    cdef Py_ssize_t n, nv1, i, iv
    
    n = p.shape[0]
    assert p.shape[1] == 2, "array should have shape (np, 2)"
    
    nv1 = v.shape[0] - 1
    assert v.shape[1] == 2, "array should have shape (nv, 2)"
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] DS = np.empty((n, nv1), dtype=np.float64)
    
    for iv in range(nv1):
        p1x, p1y = v[iv, 0], v[iv, 1]
        p2x, p2y = v[iv + 1, 0], v[iv + 1, 1]
        
        for i in range(n):
            DS[i, iv] = _dsegment(p[i, 0], p[i, 1], p1x, p1y, p2x, p2y)
    
    return DS
