import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
from itertools import combinations

# ================= bitstring primitives: one site index, one state ================
# Every one is a one-liner, but naming them fixes the argument order (site first,
# state second) and lets the Hamiltonian below read as physics, not bit twiddling.

def basis_states(L, N):
    """Occupation-number basis of the N-particle sector, as sorted bitstrings."""
    states = np.array(sorted(sum(1 << j for j in sites)
                             for sites in combinations(range(L), N)), dtype=np.int64)
    return states, {int(s): i for i, s in enumerate(states)}


def occ(j, s):
    """Is site j occupied in the bitstring state s?

    Example: s = 13, bin(s) = '0b1101', so occ(0, s) = 1 but occ(1, s) = 0.
    """
    return (s >> j) & 1


def c(j, s):
    """Remove a particle at site j: if the jth bit is 1 it is set to 0, else nothing.

    For that `else' the caller needs an external guardrail -- see h_fermion_ring.
    """
    return s & ~(1 << j)


def cdag(j, s):
    """Add a particle at site j: if the jth bit is 0 it is set to 1, else nothing.

    For that `else' the caller needs an external guardrail -- see h_fermion_ring.
    """
    return s | (1 << j)


def count_below(j, s):
    """Number of particles on sites with index strictly below j (Jordan-Wigner string).

    (1 << j) is 2^j; (1 << j) - 1 is 2^j - 1 = 2^0 + ... + 2^(j-1), i.e. a mask of
    ones on every site below j.  For j = 3 that is 1000 - 1 = 111.  The & then keeps
    only those sites of s, and .count("1") counts the particles among them.
    """
    return bin(s & ((1 << j) - 1)).count("1")


# solution: implementation of h_fermion_ring according to the interface above
def h_fermion_ring(L, N, t=1.0, dtype=np.complex128, boundary='periodic'):
    """Hopping matrix for spinless fermions on an L-site ring threaded by flux, N-particle sector (CSR scipy matrix).
    Pure kinetic energy: -t sum_j c^dag_{j+1} c_j + h.c.), with
    Jordan-Wigner signs.
    """

    # the matrix will be specified by triplets (value, (row, column))
    vals, rows, cols = [], [], []

    states, _index = basis_states(L, N)

    dim = len(states)

    for st in states: #for each state
        for src in range(L): # for every site
            if occ(src,st) == 1: # if there is an electron to hop
                s1 = c(src,st)
                sgn1 = (-1) ** count_below(src, st)

                for direction in (-1,1):
                    if (src+direction < 0 or src+direction >= L) and boundary=='open':
                        continue
                    #end
                    dst = (src+direction)%L
                    amp = -t
                    if occ(dst,st) == 0: # if the site is free
                        sgn2 = (-1) ** count_below(dst, s1)
                        s2 = cdag(dst, s1)

                        # H[s2, st] = amp*sgn1*sgn2
                        vals.append(amp*sgn1*sgn2)
                        rows.append(_index[s2])
                        cols.append(_index[st])
                    #end if
                #next direction
            #end if
        #next src
    #next state

    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()
    H.sum_duplicates()

    return H.astype(dtype), states
#end function

# Solution a): implementation of the Hubbard model

def hubbard_operator(L, Nup, Ndn, t=1.0, U=0.0, boundary='periodic'):
    """
        Matrix-free Fermi-Hubbard H in the (Nup, Ndn) sector.
        Returns (LinearOperator, double-occupancy diagonal, (D_up, D_dn)).
    """
    
    # thanks to the ordering of creation operators in the basis
    # the minus signs are the same for QUADRATIC operators (not for c, cdag, individually, though)
    # the hopping Hamiltonian thus splits to the tensor product of :
    Hup = h_fermion_ring(L, Nup, t=t, dtype=np.complex128, boundary=boundary)[0]
    Hdn = h_fermion_ring(L, Ndn, t=t, dtype=np.complex128, boundary=boundary)[0]

    # the double occupancy matrix is diagonal ---> we'll make it a sparse matrix
    index, value = [], []

    states_up, _index_up = basis_states(L,Nup)
    states_dn, _index_dn = basis_states(L,Ndn)
    Du = len(states_up)
    Dd = len(states_dn)

    for s1 in states_up:
        for s2 in states_dn:
            val = np.bitwise_count(s1 & s2)
            if val != 0: # the matrix is sparse, so add only if nonzero
                index.append( _index_up[s1]*Dd + _index_dn[s2])
                value.append( val )
            #end if
        #next s2
    #next s1
    docc = sp.coo_matrix((value, (index, index)), shape=(Du*Dd, Du*Dd), dtype=np.complex128).tocsr()
    
    def matvec(x):
        P = x.reshape(Du, Dd)
        return (Hup @ P + (Hdn @ P.T).T).ravel() + (U * docc) * x
    #end function

    return spla.LinearOperator((Du * Dd, Du * Dd), matvec=matvec, dtype=np.complex128), docc, (Du, Dd)
#end function


def hubbard_dense(L, Nup, Ndn, **kw):
    """The same H, assembled explicitly.  Only for validation at small L."""
    op, docc, (Du, Dd) = hubbard_operator(L, Nup, Ndn, **kw)
    D = Du * Dd
    return np.column_stack([op.matvec(e) for e in np.eye(D, dtype=np.complex128)])
#end function