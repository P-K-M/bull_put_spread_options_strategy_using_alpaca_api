# bellman_ford_dll.py

import ctypes
import bellman_ford

bellman_ford_dll = ctypes.CDLL("path_to_bellman_ford_dll.dll")

bellman_ford_dll.Bellman_Ford.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int)
]
bellman_ford_dll.Bellman_Ford.restype = None

bellman_ford_dll.Negative_Cycle.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]
bellman_ford_dll.Negative_Cycle.restype = ctypes.c_bool

def Bellman_Ford(vertices, currency_matrix, tickers, arches):
    vertices_c = ctypes.c_int(vertices)
    currency_matrix_c = (ctypes.c_double * len(currency_matrix))(*currency_matrix)
    tickers_c = (ctypes.c_char_p * len(tickers))(*[bytes(t, 'utf-8') for t in tickers])
    arches_c = (ctypes.c_int * len(arches))(*[val for sublist in arches for val in sublist])

    bellman_ford_dll.Bellman_Ford(
        vertices_c,
        currency_matrix_c,
        tickers_c,
        arches_c
    )

def Negative_Cycle(dist, path, arches):
    dist_c = (ctypes.c_double * len(dist))(*dist)
    path_c = (ctypes.c_int * len(path))(*path)
    arches_c = (ctypes.c_int * len(arches))(*[val for sublist in arches for val in sublist])

    return bellman_ford_dll.Negative_Cycle(dist_c, path_c, arches_c)