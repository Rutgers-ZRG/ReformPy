import numpy as np
import numba
from numba import jit, types, int32, float64
from scipy.optimize import linear_sum_assignment

def get_rcovdata():
    dat = \
    [[ 0  , "X" , 1.0],
    [ 1  , "H"  , 0.37],
    [ 2  , "He" , 0.32],
    [ 3  , "Li" , 1.34],
    [ 4  , "Be" , 0.90],
    [ 5  , "B"  , 0.82],
    [ 6  , "C"  , 0.77],
    [ 7  , "N"  , 0.75],
    [ 8  , "O"  , 0.73],
    [ 9  , "F"  , 0.71],
    [ 10 , "Ne" , 0.69],
    [ 11 , "Na" , 1.54],
    [ 12 , "Mg" , 1.30],
    [ 13 , "Al" , 1.18],
    [ 14 , "Si" , 1.11],
    [ 15 , "P"  , 1.06],
    [ 16 , "S"  , 1.02],
    [ 17 , "Cl" , 0.99],
    [ 18 , "Ar" , 0.97],
    [ 19 , "K"  , 1.96],
    [ 20 , "Ca" , 1.74],
    [ 21 , "Sc" , 1.44],
    [ 22 , "Ti" , 1.36],
    [ 23 , "V"  , 1.25],
    [ 24 , "Cr" , 1.27],
    [ 25 , "Mn" , 1.39],
    [ 26 , "Fe" , 1.25],
    [ 27 , "Co" , 1.26],
    [ 28 , "Ni" , 1.21],
    [ 29 , "Cu" , 1.38],
    [ 30 , "Zn" , 1.31],
    [ 31 , "Ga" , 1.26],
    [ 32 , "Ge" , 1.22],
    [ 33 , "As" , 1.19],
    [ 34 , "Se" , 1.16],
    [ 35 , "Br" , 1.14],
    [ 36 , "Kr" , 1.10],
    [ 37 , "Rb" , 2.11],
    [ 38 , "Sr" , 1.92],
    [ 39 , "Y"  , 1.62],
    [ 40 , "Zr" , 1.48],
    [ 41 , "Nb" , 1.37],
    [ 42 , "Mo" , 1.45],
    [ 43 , "Tc" , 1.56],
    [ 44 , "Ru" , 1.26],
    [ 45 , "Rh" , 1.35],
    [ 46 , "Pd" , 1.31],
    [ 47 , "Ag" , 1.53],
    [ 48 , "Cd" , 1.48],
    [ 49 , "In" , 1.44],
    [ 50 , "Sn" , 1.41],
    [ 51 , "Sb" , 1.38],
    [ 52 , "Te" , 1.35],
    [ 53 , "I"  , 1.33],
    [ 54 , "Xe" , 1.30],
    [ 55 , "Cs" , 2.25],
    [ 56 , "Ba" , 1.98],
    [ 57 , "La" , 1.80],
    [ 58 , "Ce" , 1.63],
    [ 59 , "Pr" , 1.76],
    [ 60 , "Nd" , 1.74],
    [ 61 , "Pm" , 1.73],
    [ 62 , "Sm" , 1.72],
    [ 63 , "Eu" , 1.68],
    [ 64 , "Gd" , 1.69],
    [ 56 , "Tb" , 1.68],
    [ 66 , "Dy" , 1.67],
    [ 67 , "Ho" , 1.66],
    [ 68 , "Er" , 1.65],
    [ 69 , "Tm" , 1.64],
    [ 70 , "Yb" , 1.70],
    [ 71 , "Lu" , 1.60],
    [ 72 , "Hf" , 1.50],
    [ 73 , "Ta" , 1.38],
    [ 74 , "W"  , 1.46],
    [ 75 , "Re" , 1.59],
    [ 76 , "Os" , 1.28],
    [ 77 , "Ir" , 1.37],
    [ 78 , "Pt" , 1.28],
    [ 79 , "Au" , 1.44],
    [ 80 , "Hg" , 1.49],
    [ 81 , "Tl" , 1.48],
    [ 82 , "Pb" , 1.47],
    [ 83 , "Bi" , 1.46],
    [ 84 , "Po" , 1.45],
    [ 85 , "At" , 1.47],
    [ 86 , "Rn" , 1.42],
    [ 87 , "Fr" , 2.23],
    [ 88 , "Ra" , 2.01],
    [ 89 , "Ac" , 1.86],
    [ 90 , "Th" , 1.75],
    [ 91 , "Pa" , 1.69],
    [ 92 , "U"  , 1.70],
    [ 93 , "Np" , 1.71],
    [ 94 , "Pu" , 1.72],
    [ 95 , "Am" , 1.66],
    [ 96 , "Cm" , 1.66],
    [ 97 , "Bk" , 1.68],
    [ 98 , "Cf" , 1.68],
    [ 99 , "Es" , 1.65],
    [ 100, "Fm" , 1.67],
    [ 101, "Md" , 1.73],
    [ 102, "No" , 1.76],
    [ 103, "Lr" , 1.61],
    [ 104, "Rf" , 1.57],
    [ 105, "Db" , 1.49],
    [ 106, "Sg" , 1.43],
    [ 107, "Bh" , 1.41],
    [ 108, "Hs" , 1.34],
    [ 109, "Mt" , 1.29],
    [ 110, "Ds" , 1.28],
    [ 111, "Rg" , 1.21],
    [ 112, "Cn" , 1.22]]

    return dat

@jit('(float64)(int32, int32)', nopython=True)
def kron_delta(i, j):
    if i == j:
        m = 1.0
    else:
        m = 0.0
    return m

# @jit('(boolean)(float64[:,:], float64, float64)', nopython=True)
def check_symmetric(A, rtol = 1e-05, atol = 1e-08):
    return np.allclose(A, A.T, rtol = rtol, atol = atol)

# @jit('(boolean)(float64[:,:])', nopython=True)
def check_pos_def(A):
    eps = np.finfo(float).eps
    B = A + eps*np.identity(len(A))
    if np.array_equal(B, B.T):
        try:
            np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

@jit('(int32)(float64[:,:], float64)', nopython=True)
def get_ixyz(lat, cutoff):
    lat = np.ascontiguousarray(lat)
    lat2 = np.dot(lat, np.transpose(lat))
    vec = np.linalg.eigvals(lat2)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    ixyz = np.int32(ixyz)
    # return np.sqrt(1.0/max(np.linalg.eigvals(np.dot(lat, np.transpose(lat)))))*cutoff + 1
    return ixyz

# @jit(nopython=True)
def readvasp(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float)
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    nat = sum(typt)
    pos = np.array(buff[7:7 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    rxyz = np.dot(pos, lat)
    # rxyz = pos
    return lat, rxyz, types

# @jit(nopython=True)
def read_types(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    return types

# @jit('Tuple((float64[:,:], float64[:,:]))(int32, float64[:,:], \
#       float64[:], float64[:])', nopython=True)
@jit(nopython=True)
def get_gom(lseg, rxyz, alpha, amp):
    # s orbital only lseg == 1
    nat = len(rxyz)
    if lseg == 1:
        om = np.zeros((nat, nat), dtype = np.float64)
        mamp = np.zeros((nat, nat), dtype = np.float64)
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                t1 = alpha[iat] * alpha[jat]
                t2 = alpha[iat] + alpha[jat]
                om[iat][jat] = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)
                mamp[iat][jat] = amp[iat]*amp[jat]

    else:
        # for both s and p orbitals
        om = np.zeros((4*nat, 4*nat), dtype = np.float64)
        mamp = np.zeros((4*nat, 4*nat), dtype = np.float64)
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                t1 = alpha[iat] * alpha[jat]
                t2 = alpha[iat] + alpha[jat]

                # <s_i | s_j>
                sij = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)
                om[4*iat][4*jat] = sij
                mamp[4*iat][4*jat] = amp[iat]*amp[jat]

                # <s_i | p_j>
                stv = 2.0 * (1/np.sqrt(alpha[jat])) * (t1/t2) * sij
                om[4*iat][4*jat+1] = stv * d[0]
                om[4*iat][4*jat+2] = stv * d[1]
                om[4*iat][4*jat+3] = stv * d[2]

                mamp[4*iat][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat][4*jat+3] = amp[iat]*amp[jat]
                # <p_i | s_j>
                stv = -2.0 * (1/np.sqrt(alpha[iat])) * (t1/t2) * sij
                om[4*iat+1][4*jat] = stv * d[0]
                om[4*iat+2][4*jat] = stv * d[1]
                om[4*iat+3][4*jat] = stv * d[2]

                mamp[4*iat+1][4*jat] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat] = amp[iat]*amp[jat]

                # <p_i | p_j>
                # stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                stv = 2.0 * np.sqrt(t1)/t2 * sij
                sx = -2.0*t1/t2

                for i_pp in range(3):
                    for j_pp in range(3):
                        om[4*iat+i_pp+1][4*jat+j_pp+1] = stv * (sx * d[i_pp] * d[j_pp] + \
                                                                kron_delta(i_pp, j_pp))

                for i_pp in range(3):
                    for j_pp in range(3):
                        mamp[4*iat+i_pp+1][4*jat+j_pp+1] = amp[iat]*amp[jat]

                '''
                om[4*iat+1][4*jat+1] = stv * (sx * d[0] * d[0] + 1.0)
                om[4*iat+1][4*jat+2] = stv * (sx * d[1] * d[0]      )
                om[4*iat+1][4*jat+3] = stv * (sx * d[2] * d[0]      )
                om[4*iat+2][4*jat+1] = stv * (sx * d[0] * d[1]      )
                om[4*iat+2][4*jat+2] = stv * (sx * d[1] * d[1] + 1.0)
                om[4*iat+2][4*jat+3] = stv * (sx * d[2] * d[1]      )
                om[4*iat+3][4*jat+1] = stv * (sx * d[0] * d[2]      )
                om[4*iat+3][4*jat+2] = stv * (sx * d[1] * d[2]      )
                om[4*iat+3][4*jat+3] = stv * (sx * d[2] * d[2] + 1.0)

                mamp[4*iat+1][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat+1][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat+1][4*jat+3] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat+3] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat+3] = amp[iat]*amp[jat]
                '''

    # for i in range(len(om)):
    #     for j in range(len(om)):
    #         if abs(om[i][j] - om[j][i]) > 1e-6:
    #             print ("ERROR", i, j, om[i][j], om[j][i])
    '''
    if check_symmetric(om*mamp) and check_pos_def(om*mamp):
        return om, mamp
    else:
        raise Exception("Gaussian Overlap Matrix is not symmetric and positive definite!")
    '''
    return (om, mamp)

# @jit('(float64[:,:,:,:])(int32, float64[:,:], float64[:], \
#       float64[:], float64[:,:], float64[:], int32)', nopython=True)
@jit(nopython=True)
def get_dgom(lseg, gom, amp, damp, rxyz, alpha, icenter):
    nat = len(rxyz)
    if lseg == 1:
        # s orbital only lseg == 1
        di = np.empty(3, dtype = np.float64)
        dj = np.empty(3, dtype = np.float64)
        dc = np.empty(3, dtype = np.float64)
        dgom = np.zeros((nat, 3, nat, nat), dtype = np.float64)
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                t1 = alpha[iat] * alpha[jat]
                t2 = alpha[iat] + alpha[jat]
                tt = 2.0 * t1 / t2
                dic = rxyz[iat] - rxyz[icenter]
                djc = rxyz[jat] - rxyz[icenter]

                pij = amp[iat] * amp[jat]
                dipj = damp[iat] * amp[jat]
                djpi = damp[jat] * amp[iat]

                for k in range(3):
                    di[k] = -pij * tt * gom[iat][jat] * d[k] + dipj * gom[iat][jat] * dic[k]
                    dj[k] = +pij * tt * gom[iat][jat] * d[k] + djpi * gom[iat][jat] * djc[k]
                    dc[k] = -dipj * gom[iat][jat] * dic[k] - djpi * gom[iat][jat] * djc[k]

                    dgom[iat][k][iat][jat] += di[k]
                    dgom[jat][k][iat][jat] += dj[k]
                    dgom[icenter][k][iat][jat] += dc[k]
    else:
        # for both s and p orbitals
        dss_i = np.empty(3, dtype = np.float64)
        dss_j = np.empty(3, dtype = np.float64)
        dss_c = np.empty(3, dtype = np.float64)
        dsp_i = np.empty((3,3), dtype = np.float64)
        dsp_j = np.empty((3,3), dtype = np.float64)
        dsp_c = np.empty((3,3), dtype = np.float64)
        dps_i = np.empty((3,3), dtype = np.float64)
        dps_j = np.empty((3,3), dtype = np.float64)
        dps_c = np.empty((3,3), dtype = np.float64)
        dpp_i = np.empty((3,3,3), dtype = np.float64)
        dpp_j = np.empty((3,3,3), dtype = np.float64)
        dpp_c = np.empty((3,3,3), dtype = np.float64)
        dgom = np.zeros((nat, 3, 4*nat, 4*nat), dtype = np.float64)
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                t1 = alpha[iat] * alpha[jat]
                t2 = alpha[iat] + alpha[jat]
                tt = 2.0 * t1 / t2
                dic = rxyz[iat] - rxyz[icenter]
                djc = rxyz[jat] - rxyz[icenter]

                pij = amp[iat] * amp[jat]
                dipj = damp[iat] * amp[jat]
                djpi = damp[jat] * amp[iat]

                # <s_i | s_j>
                for k_ss in range(3):
                    dss_i[k_ss] = -pij * tt * gom[4*iat][4*jat] * d[k_ss] + dipj * \
                    gom[4*iat][4*jat] * dic[k_ss]
                    dss_j[k_ss] = +pij * tt * gom[4*iat][4*jat] * d[k_ss] + djpi * \
                    gom[4*iat][4*jat] * djc[k_ss]
                    dss_c[k_ss] = -dipj * gom[4*iat][4*jat] * dic[k_ss] - djpi * \
                    gom[4*iat][4*jat] * djc[k_ss]

                    dgom[iat][k_ss][4*iat][4*jat] += dss_i[k_ss]
                    dgom[jat][k_ss][4*iat][4*jat] += dss_j[k_ss]
                    dgom[icenter][k_ss][4*iat][4*jat] += dss_c[k_ss]

                # <s_i | p_j>
                for k_sp in range(3):
                    for i_sp in range(3):
                        dsp_i[k_sp][i_sp] = +(1/np.sqrt(alpha[jat])) * pij * tt * \
                        kron_delta(k_sp, i_sp) * gom[4*iat][4*jat] - \
                        (1/np.sqrt(alpha[jat]))* pij * tt ** 2 * \
                        np.multiply(d[k_sp], d[i_sp]) * gom[4*iat][4*jat] + \
                        dipj * gom[4*iat][4*jat+i_sp+1] * dic[k_sp]

                        dsp_j[k_sp][i_sp] = -(1/np.sqrt(alpha[jat])) * pij * tt * \
                        kron_delta(k_sp, i_sp) * gom[4*iat][4*jat] + \
                        (1/np.sqrt(alpha[jat]))* pij * tt ** 2 * \
                        np.multiply(d[k_sp], d[i_sp]) * gom[4*iat][4*jat] + \
                        djpi * gom[4*iat][4*jat+i_sp+1] * djc[k_sp]

                        dsp_c[k_sp][i_sp] = -dipj * gom[4*iat][4*jat+i_sp+1] * dic[k_sp] - \
                        djpi * gom[4*iat][4*jat+i_sp+1] * djc[k_sp]

                        dgom[iat][k_sp][4*iat][4*jat+i_sp+1] += dsp_i[k_sp][i_sp]
                        dgom[jat][k_sp][4*iat][4*jat+i_sp+1] += dsp_j[k_sp][i_sp]
                        dgom[icenter][k_sp][4*iat][4*jat+i_sp+1] += dsp_c[k_sp][i_sp]

                # <p_i | s_j>
                for k_ps in range(3):
                    for i_ps in range(3):
                        dps_i[k_ps][i_ps] = -(1/np.sqrt(alpha[iat])) * pij * tt * \
                        kron_delta(k_ps, i_ps) * gom[4*iat][4*jat] + \
                        (1/np.sqrt(alpha[iat]))* pij * tt ** 2 * \
                        np.multiply(d[k_ps], d[i_ps]) * gom[4*iat][4*jat] + \
                        dipj * gom[4*iat+i_ps+1][4*jat] * dic[k_ps]

                        dps_j[k_ps][i_ps] = +(1/np.sqrt(alpha[iat])) * pij * tt * \
                        kron_delta(k_ps, i_ps) * gom[4*iat][4*jat] - \
                        (1/np.sqrt(alpha[iat]))* pij * tt ** 2 * \
                        np.multiply(d[k_ps], d[i_ps]) * gom[4*iat][4*jat] + \
                        djpi * gom[4*iat+i_ps+1][4*jat] * djc[k_ps]

                        dps_c[k_ps][i_ps] = -dipj * gom[4*iat+i_ps+1][4*jat] * dic[k_ps] - \
                        djpi * gom[4*iat+i_ps+1][4*jat] * djc[k_ps]

                        dgom[iat][k_ps][4*iat+i_ps+1][4*jat] += dps_i[k_ps][i_ps]
                        dgom[jat][k_ps][4*iat+i_ps+1][4*jat] += dps_j[k_ps][i_ps]
                        dgom[icenter][k_ps][4*iat+i_ps+1][4*jat] += dps_c[k_ps][i_ps]

                # <p_i | p_j>
                for k_pp in range(3):
                    for i_pp in range(3):
                        for j_pp in range(3):
                            dpp_i[k_pp][i_pp][j_pp] = -(1/np.sqrt(alpha[iat]*alpha[jat])) * \
                            pij * tt ** 2 * d[k_pp] * (kron_delta(i_pp, j_pp) - tt * \
                            np.multiply(d[i_pp], d[j_pp])) * gom[4*iat][4*jat] - \
                            (1/np.sqrt(alpha[iat]*alpha[jat])) * pij * tt ** 2 * \
                            (kron_delta(k_pp, i_pp)*d[j_pp] + kron_delta(k_pp, j_pp)*d[i_pp]) * \
                            gom[4*iat][4*jat] + \
                            dipj * gom[4*iat+i_pp+1][4*jat+j_pp+1] * dic[k_pp]

                            dpp_j[k_pp][i_pp][j_pp] = +(1/np.sqrt(alpha[iat]*alpha[jat])) * \
                            pij * tt ** 2 * d[k_pp] * (kron_delta(i_pp, j_pp) - tt * \
                            np.multiply(d[i_pp], d[j_pp])) * gom[4*iat][4*jat] + \
                            (1/np.sqrt(alpha[iat]*alpha[jat])) * pij * tt ** 2 * \
                            (kron_delta(k_pp, i_pp)*d[j_pp] + kron_delta(k_pp, j_pp)*d[i_pp]) * \
                            gom[4*iat][4*jat] + \
                            djpi * gom[4*iat+i_pp+1][4*jat+j_pp+1] * djc[k_pp]

                            dpp_c[k_pp][i_pp][j_pp] = -dipj * gom[4*iat+i_pp+1][4*jat+j_pp+1] * \
                            dic[k_pp] - djpi * gom[4*iat+i_pp+1][4*jat+j_pp+1] * djc[k_pp]

                            dgom[iat][k_pp][4*iat+i_pp+1][4*jat+j_pp+1] += dpp_i[k_pp][i_pp][j_pp]
                            dgom[jat][k_pp][4*iat+i_pp+1][4*jat+j_pp+1] += dpp_j[k_pp][i_pp][j_pp]
                            dgom[icenter][k_pp][4*iat+i_pp+1][4*jat+j_pp+1] += \
                                                                           dpp_c[k_pp][i_pp][j_pp]


    return dgom

# @jit('(float64[:])(float64[:,:], int32[:])', nopython=True)
def get_fp_nonperiodic(rxyz, znucls):
    rcov = []
    amp = [1.0] * len(rxyz)
    rcovdata = get_rcovdata()
    for x in znucls:
        rcov.append(rcovdata[x][2])
    om, mamp = get_gom(1, rxyz, rcov, amp)
    gom = om*mamp
    fp = np.linalg.eigvals(gom)
    fp = sorted(fp)
    fp = np.array(fp, float)
    return fp

# @jit('(float64)(float64[:], float64[:])', nopython=True)
def get_fpdist_nonperiodic(fp1, fp2):
    d = fp1 - fp2
    return np.sqrt(np.vdot(d, d))

@jit('Tuple((float64[:,:], float64[:,:,:,:]))(float64[:,:], float64[:,:], int32[:], int32[:], \
      boolean, boolean, int32, int32, int32, float64)', nopython=True)
def get_fp(lat, rxyz, types, znucl,
           contract,
           ldfp,
           ntyp,
           nx,
           lmax,
           cutoff):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2

    rcovdata =  [[ 0 ,  1.0],
                [ 1  ,  0.37],
                [ 2  ,  0.32],
                [ 3  ,  1.34],
                [ 4  ,  0.90],
                [ 5  ,  0.82],
                [ 6  ,  0.77],
                [ 7  ,  0.75],
                [ 8  ,  0.73],
                [ 9  ,  0.71],
                [ 10 ,  0.69],
                [ 11 ,  1.54],
                [ 12 ,  1.30],
                [ 13 ,  1.18],
                [ 14 ,  1.11],
                [ 15 ,  1.06],
                [ 16 ,  1.02],
                [ 17 ,  0.99],
                [ 18 ,  0.97],
                [ 19 ,  1.96],
                [ 20 ,  1.74],
                [ 21 ,  1.44],
                [ 22 ,  1.36],
                [ 23 ,  1.25],
                [ 24 ,  1.27],
                [ 25 ,  1.39],
                [ 26 ,  1.25],
                [ 27 ,  1.26],
                [ 28 ,  1.21],
                [ 29 ,  1.38],
                [ 30 ,  1.31],
                [ 31 ,  1.26],
                [ 32 ,  1.22],
                [ 33 ,  1.19],
                [ 34 ,  1.16],
                [ 35 ,  1.14],
                [ 36 ,  1.10],
                [ 37 ,  2.11],
                [ 38 ,  1.92],
                [ 39 ,  1.62],
                [ 40 ,  1.48],
                [ 41 ,  1.37],
                [ 42 ,  1.45],
                [ 43 ,  1.56],
                [ 44 ,  1.26],
                [ 45 ,  1.35],
                [ 46 ,  1.31],
                [ 47 ,  1.53],
                [ 48 ,  1.48],
                [ 49 ,  1.44],
                [ 50 ,  1.41],
                [ 51 ,  1.38],
                [ 52 ,  1.35],
                [ 53 ,  1.33],
                [ 54 ,  1.30],
                [ 55 ,  2.25],
                [ 56 ,  1.98],
                [ 57 ,  1.80],
                [ 58 ,  1.63],
                [ 59 ,  1.76],
                [ 60 ,  1.74],
                [ 61 ,  1.73],
                [ 62 ,  1.72],
                [ 63 ,  1.68],
                [ 64 ,  1.69],
                [ 56 ,  1.68],
                [ 66 ,  1.67],
                [ 67 ,  1.66],
                [ 68 ,  1.65],
                [ 69 ,  1.64],
                [ 70 ,  1.70],
                [ 71 ,  1.60],
                [ 72 ,  1.50],
                [ 73 ,  1.38],
                [ 74 ,  1.46],
                [ 75 ,  1.59],
                [ 76 ,  1.28],
                [ 77 ,  1.37],
                [ 78 ,  1.28],
                [ 79 ,  1.44],
                [ 80 ,  1.49],
                [ 81 ,  1.48],
                [ 82 ,  1.47],
                [ 83 ,  1.46],
                [ 84 ,  1.45],
                [ 85 ,  1.47],
                [ 86 ,  1.42],
                [ 87 ,  2.23],
                [ 88 ,  2.01],
                [ 89 ,  1.86],
                [ 90 ,  1.75],
                [ 91 ,  1.69],
                [ 92 ,  1.70],
                [ 93 ,  1.71],
                [ 94 ,  1.72],
                [ 95 ,  1.66],
                [ 96 ,  1.66],
                [ 97 ,  1.68],
                [ 98 ,  1.68],
                [ 99 ,  1.65],
                [ 100,  1.67],
                [ 101,  1.73],
                [ 102,  1.76],
                [ 103,  1.61],
                [ 104,  1.57],
                [ 105,  1.49],
                [ 106,  1.43],
                [ 107,  1.41],
                [ 108,  1.34],
                [ 109,  1.29],
                [ 110,  1.28],
                [ 111,  1.21],
                [ 112,  1.22]]

    #Modified so that now a float is returned and converted into an int
    ixyzf = get_ixyz(lat, cutoff)
    ixyz = int(ixyzf) + 1
    NC = 2
    wc = cutoff / np.sqrt(2.* NC)
    fc = 1.0 / (2.0 * NC * wc**2)
    nat = len(rxyz)
    cutoff2 = cutoff**2

    n_sphere_list = []
    lfp = np.empty((nat, lseg*nx), dtype = np.float64)
    sfp = []
    dfp = np.zeros((nat, nat, 3, lseg*nx), dtype = np.float64)
    for iat in range(nat):
        rxyz_sphere = []
        rcov_sphere = []
        alpha = []
        ind = [0] * (lseg * nx)
        indori = []
        amp = []
        damp = []
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            rcovjur = rcovdata.copy()
            index11 = int(types[jat] - 1)
            index1 = int(znucl[index11])
            rcovj = rcovjur[index1][1]
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > nx:
                                raise Exception("FP WARNING: Cutoff radius is too large, \
                                                increase nx or decrease cutoff.")
                            # amp.append((1.0-d2*fc)**NC)
                            # nd2 = d2/cutoff2
                            ampt = (1.0-d2*fc)**(NC-1)
                            amp.append(ampt * (1.0-d2*fc))
                            damp.append(-2.0 * fc * NC * ampt)
                            indori.append(jat)
                            # amp.append(1.0)
                            # print (1.0-d2*fc)**NC
                            rxyz_sphere.append([xj, yj, zj])
                            rcov_sphere.append(rcovj)
                            alpha.append(0.5 / rcovj**2)
                            if jat == iat and ix == 0 and iy == 0 and iz == 0:
                                ityp_sphere = 0
                                icenter = n_sphere-1
                            else:
                                ityp_sphere = types[jat]
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere)
        # full overlap matrix
        nid = lseg * n_sphere
        (gom, mamp) = get_gom(lseg, rxyz_sphere, alpha, amp)
        gomamp = gom * mamp
        val, vec = np.linalg.eigh(gomamp)
        # val = np.real(val)
        fp0 = np.zeros(nx*lseg)
        for i in range(len(val)):
            # print (val[i])
            fp0[i] = val[len(val)-1-i]
        # fp0 = fp0/np.linalg.norm(fp0)
        # np.append(lfp, fp0)
        lfp[iat] = fp0
        # pvec = np.real(np.transpose(vec)[0])

        vectmp = np.transpose(vec)
        vecs = []
        for i in range(len(vectmp)):
            vecs.append(vectmp[len(vectmp)-1-i])

        pvec = vecs[0]
        # derivative
        if ldfp:
            dgom = get_dgom(lseg, gom, amp, damp, rxyz_sphere, alpha, icenter)
            # print (dgom[0][0][0])
            dvdr = np.zeros((n_sphere, lseg*n_sphere, 3))
            for iats in range(n_sphere):
                for iorb in range(lseg*n_sphere):
                    vvec = vecs[iorb]
                    for ik in range(3):
                        matt = dgom[iats][ik]
                        vv1 = np.dot(np.conjugate(vvec), matt)
                        vv2 = np.dot(vv1, np.transpose(vvec))
                        dvdr[iats][iorb][ik] = vv2
            for iats in range(n_sphere):
                iiat = indori[iats]
                for iorb in range(lseg*n_sphere):
                    for ik in range(3):
                        dfp[iat][iiat][ik][iorb] += dvdr[iats][iorb][ik]

        # contracted overlap matrix
        if contract:
            nids = l * (ntyp + 1)
            omx = np.zeros((nids, nids))
            for i in range(nid):
                for j in range(nid):
                    # print ind[i], ind[j]
                    omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
            # for i in range(nids):
            #     for j in range(nids):
            #         if abs(omx[i][j] - omx[j][i]) > 1e-6:
            #             print ("ERROR", i, j, omx[i][j], omx[j][i])
            # print omx
            sfp0 = np.linalg.eigvals(omx)
            sfp.append(sorted(sfp0))


    # print ("n_sphere_min", min(n_sphere_list))
    # print ("n_shpere_max", max(n_sphere_list))

    if contract:
        # sfp = np.array(sfp, dtype = np.float64)
        # dfp = np.array(dfp, dtype = np.float64)
        sfp = np.array(sfp)
        return sfp, dfp

    else:
        # lfp = np.array(lfp, dtype = np.float64)
        # dfp = np.array(dfp, dtype = np.float64)
        return lfp, dfp

# @jit(nopython=True)
def get_fpdist(ntyp, types, fp1, fp2, mx=False):
    nat, lenfp = np.shape(fp1)
    fpd = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        MX = np.zeros((nat, nat))
        for iat in range(nat):
            if types[iat] == itype:
                for jat in range(nat):
                    if types[jat] == itype:
                        tfpd = fp1[iat] - fp2[jat]
                        MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd))

        row_ind, col_ind = linear_sum_assignment(MX)
        # print(row_ind, col_ind)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    # fpd = ((fpd+1.0)*np.log(fpd+1.0)-fpd)
    if mx:
        return fpd, col_ind
    else:
        return fpd


# from numba import njit
# @njit

# @jit('Tuple((float64, float64[:,:]))(float64[:,:], float64[:,:,:,:], int32, \
#        int32[:])', nopython=True)
# def get_ef(fp, dfp, ntyp, types):
#     nat, D = fp.shape  # Number of atoms and feature dimension
#     e = 0.0

#     # Initialize force arrays
#     force_0 = np.zeros((nat, 3))
#     force_prime = np.zeros((nat, 3))

#     # Compute norms of fp manually
#     norms = np.empty(nat)
#     for i in range(nat):
#         norms[i] = 0.0
#         for d in range(D):
#             norms[i] += fp[i, d] ** 2
#         norms[i] = np.sqrt(norms[i])

#     # Energy calculation
#     for ityp in range(ntyp):
#         itype = ityp + 1

#         # Collect indices of atoms of the current type
#         indices = np.empty(nat, dtype=np.int32)
#         count = 0
#         for i in range(nat):
#             if types[i] == itype:
#                 indices[count] = i
#                 count += 1
#         n_type_atoms = count

#         if n_type_atoms == 0:
#             continue  # Skip if no atoms of this type

#         # Extract fp and norms for the current type
#         fp_type = np.empty((n_type_atoms, D))
#         norms_type = np.empty(n_type_atoms)
#         for idx in range(n_type_atoms):
#             i = indices[idx]
#             fp_type[idx, :] = fp[i, :]
#             norms_type[idx] = norms[i]

#         # Energy calculation for the current type
#         e0 = 0.0
#         for i in range(n_type_atoms):
#             for j in range(n_type_atoms):
#                 vij = fp_type[i, :] - fp_type[j, :]
#                 t = 0.0
#                 for d in range(D):
#                     t += vij[d] ** 2
#                 e0 += t
#         for i in range(n_type_atoms):
#             e0 += 1.0 / (norms_type[i] ** 2)
#         e += e0

#         # Force calculation for force_0
#         for k in range(nat):
#             # dfp for atoms of this type and atom k
#             dfp_type_k = np.empty((n_type_atoms, 3, D))
#             for idx in range(n_type_atoms):
#                 i = indices[idx]
#                 for l in range(3):
#                     for d in range(D):
#                         dfp_type_k[idx, l, d] = dfp[i, k, l, d]

#             # Compute force_0 contribution
#             for i in range(n_type_atoms):
#                 for j in range(n_type_atoms):
#                     vij = fp_type[i, :] - fp_type[j, :]
#                     for l in range(3):
#                         dvij = np.empty(D)
#                         for d in range(D):
#                             dvij[d] = dfp_type_k[i, l, d] - dfp_type_k[j, l, d]
#                         t = -2.0 * np.dot(vij, dvij)
#                         force_0[k, l] += t

#     # Force calculation for force_prime (over all atoms)
#     for k in range(nat):
#         # dfp for all atoms and atom k
#         dfp_k = np.empty((nat, 3, D))
#         for i in range(nat):
#             for l in range(3):
#                 for d in range(D):
#                     dfp_k[i, l, d] = dfp[i, k, l, d]

#         # Compute numerators and denominators
#         for l in range(3):
#             t_prime = 0.0
#             for i in range(nat):
#                 numerator = 0.0
#                 for d in range(D):
#                     numerator += fp[i, d] * dfp_k[i, l, d]
#                 denominator = norms[i] ** 4
#                 t_prime += 2.0 * numerator / denominator
#             force_prime[k, l] += t_prime

#     # Final force calculation
#     total_force = np.sum(force_0 + force_prime, axis=0) / nat
#     force = (force_0 + force_prime) - total_force

#     return e, force

@jit('Tuple((float64, float64[:,:]))(float64[:,:], float64[:,:,:,:], int32, \
      int32[:])', nopython=True)
def get_ef(fp, dfp, ntyp, types):
    nat = len(fp)
    e = 0.
    fp = np.ascontiguousarray(fp)
    dfp = np.ascontiguousarray(dfp)
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.vdot(vij, vij)
                    e0 += t
            e0 += 1.0/(np.linalg.norm(fp[i]) ** 2)
        # print ("e0", e0)
        e += e0
    # print ("e", e)

    force_0 = np.zeros((nat, 3), dtype = np.float64)
    force_prime = np.zeros((nat, 3), dtype = np.float64)

    for k in range(nat):
        for ityp in range(ntyp):
            itype = ityp + 1
            for i in range(nat):
                for j in range(nat):
                    if  types[i] == itype and types[j] == itype:
                        vij = fp[i] - fp[j]
                        dvij = dfp[i][k] - dfp[j][k]
                        for l in range(3):
                            t = -2 * np.vdot(vij, dvij[l])
                            force_0[k][l] += t
                for m in range(3):
                    t_prime = 2.0 * np.vdot(fp[i],dfp[i][k][m]) / (np.linalg.norm(fp[i]) ** 4)
                    force_prime[k][m] += t_prime
    force = force_0 + force_prime
    force = force - np.sum(force, axis=0)/len(force)
    # return ((e+1.0)*np.log(e+1.0)-e), force*np.log(e+1.0)
    return e, force


@jit('(float64)(float64[:,:], int32, int32[:])', nopython=True)
def get_fpe(fp, ntyp, types):
    nat = len(fp)
    e = 0.
    fp = np.ascontiguousarray(fp)
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.vdot(vij, vij)
                    e0 += t
            e0 += 1.0/(np.linalg.norm(fp[i]) ** 2)
        e += e0
    # return ((e+1.0)*np.log(e+1.0)-e)
    return e

# @jit('(float64[:])(float64[:,:], float64[:,:], int32[:], int32[:], \
#       boolean, int32, int32, int32, float64)', nopython=True)
# def get_stress(lat, rxyz, types, znucl,
#                contract,
#                ntyp,
#                nx,
#                lmax,
#                cutoff):
#     lat = np.ascontiguousarray(lat)
#     rxyz = np.ascontiguousarray(rxyz)
#     pos = np.dot(rxyz, np.linalg.inv(lat))
#     pos = np.ascontiguousarray(pos)
#     rxyz_delta = np.zeros_like(rxyz)
#     cell_vol = np.linalg.det(lat)
#     # stress = np.zeros(6)
#     stress = np.zeros((3,3), dtype = np.float64)
#     step_size = 1.e-5
#     strain_delta_tmp = step_size*(np.random.randint(1, 9999, (3, 3))/9999)
#     # print (strain_delta_tmp)
#     # Make strain tensor symmetric
#     strain_delta = 0.5*(strain_delta_tmp + strain_delta_tmp.T - \
#                         np.diag(np.diag(strain_delta_tmp)))
#     # print (strain_delta)
#     rxyz_ratio = np.eye(3, dtype = np.float64)
#     # rxyz_ratio_new = rxyz_ratio.copy()
#     for m in range(3):
#         for n in range(3):
#             h = strain_delta[m][n]
#             rxyz_ratio_left = np.eye(3, dtype = np.float64)
#             rxyz_ratio_right = np.eye(3, dtype = np.float64)
#             rxyz_ratio_left[m][n] = rxyz_ratio[m][n] - h
#             rxyz_ratio_left[n][m] = rxyz_ratio_left[m][n]
#             rxyz_ratio_right[m][n] = rxyz_ratio[m][n] + h
#             rxyz_ratio_right[n][m] = rxyz_ratio_right[m][n]
#             lat_left = np.dot(lat, rxyz_ratio_left)
#             lat_right = np.dot(lat, rxyz_ratio_right)
#             rxyz_left = np.dot(pos, lat_left)
#             rxyz_right = np.dot(pos, lat_right)
#             ldfp = False
#             fp_left, dfptmp1 = get_fp(lat_left, rxyz_left, types, znucl, \
#                                       contract, ldfp, ntyp, nx, lmax, cutoff)
#             fp_right, dfptmp2 = get_fp(lat_right, rxyz_right, types, znucl, \
#                                        contract, ldfp, ntyp, nx, lmax, cutoff)
#             fp_energy_left = get_fpe(fp_left, ntyp, types)
#             fp_energy_right = get_fpe(fp_right, ntyp, types)
#             stress[m][n] = (fp_energy_right - fp_energy_left)/(2.0*h*cell_vol)
#         #################




#     #################
#     # print (stress)
#     stress_voigt = np.zeros(6, dtype = np.float64)
#     voigt_list = [0, 4, 8, 5, 2, 1]
#     for i in range(6):
#         stress_voigt[i] = stress.ravel()[voigt_list[i]]
#     # stress_voigt = stress.flat[[0, 4, 8, 5, 2, 1]]
#     # stress_voigt = np.array(stress_voigt, dtype = np.float64)
#     # return np.zeros(6, dtype = np.float64)
#     return stress_voigt



# from numba import njit
# @njit

@jit('(float64[:])(float64[:,:], float64[:,:], float64[:,:])', nopython=True)
def get_stress(lat, rxyz, forces):



    """
    Compute the stress tensor analytically using the virial theorem.

    Parameters:
    - lat: (3, 3) array of lattice vectors.
    - rxyz: (nat, 3) array of atomic positions in Cartesian coordinates.
    - forces: (nat, 3) array of forces on each atom.

    Returns:
    - stress_voigt: (6,) array representing the stress tensor in Voigt notation.
    """
    # Ensure inputs are NumPy arrays with correct data types
    lat = np.asarray(lat, dtype=np.float64)
    rxyz = np.asarray(rxyz, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)

    # Compute the cell volume
    cell_vol = np.abs(np.linalg.det(lat))

    # Initialize the stress tensor
    stress_tensor = np.zeros((3, 3), dtype=np.float64)

    # Compute the stress tensor using the virial theorem
    nat = rxyz.shape[0]
    for i in range(nat):
        for m in range(3):
            for n in range(3):
                stress_tensor[m, n] -= forces[i, m] * rxyz[i, n]

    # Divide by the cell volume
    stress_tensor /= cell_vol

    # Ensure the stress tensor is symmetric (if applicable)
    # stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

    # Convert the stress tensor to Voigt notation
    # The Voigt notation order is: [xx, yy, zz, yz, xz, xy]
    stress_voigt = np.array([
        stress_tensor[0, 0],  # xx
        stress_tensor[1, 1],  # yy
        stress_tensor[2, 2],  # zz
        stress_tensor[1, 2],  # yz
        stress_tensor[0, 2],  # xz
        stress_tensor[0, 1],  # xy
    ], dtype=np.float64)

    return stress_voigt


@jit('Tuple((float64, float64))(float64[:,:], float64[:,:], int32[:], int32[:], \
      boolean, int32, int32, int32, float64)', nopython=True)
def get_simpson_energy(lat, rxyz, types, znucl,
                       contract,
                       ntyp,
                       nx,
                       lmax,
                       cutoff):
    lat = np.ascontiguousarray(lat)
    rxyz = np.ascontiguousarray(rxyz)
    rxyz_delta = np.zeros_like(rxyz)
    rxyz_disp = np.zeros_like(rxyz)
    rxyz_left = np.zeros_like(rxyz)
    rxyz_mid = np.zeros_like(rxyz)
    rxyz_right = np.zeros_like(rxyz)
    nat = len(rxyz)
    del_fpe = 0.0
    iter_max = 100
    step_size = 1.e-5
    rxyz_delta = step_size*( np.random.rand(nat, 3).astype(np.float64) - \
                            0.5*np.ones((nat, 3), dtype = np.float64) )
    for i_iter in range(iter_max):
        # rxyz_delta = step_size*( np.random.rand(nat, 3).astype(np.float64) - \
        #                         0.5*np.ones((nat, 3), dtype = np.float64) )
        rxyz_disp += 2.0*rxyz_delta
        rxyz_left = rxyz.copy() + 2.0*i_iter*rxyz_delta
        rxyz_mid = rxyz.copy() + 2.0*(i_iter+1)*rxyz_delta
        rxyz_right = rxyz.copy() + 2.0*(i_iter+2)*rxyz_delta
        ldfp = True
        fp_left, dfp_left = get_fp(lat, rxyz_left, types, znucl, \
                                   contract, ldfp, ntyp, nx, lmax, cutoff)
        fp_mid, dfp_mid = get_fp(lat, rxyz_mid, types, znucl, \
                                   contract, ldfp, ntyp, nx, lmax, cutoff)
        fp_right, dfp_right = get_fp(lat, rxyz_right, types, znucl, \
                                     contract, ldfp, ntyp, nx, lmax, cutoff)
        fpe_left, fpf_left = get_ef(fp_left, dfp_left, ntyp, types)
        fpe_mid, fpf_mid = get_ef(fp_mid, dfp_mid, ntyp, types)
        fpe_right, fpf_right = get_ef(fp_right, dfp_right, ntyp, types)

        rxyz_delta = np.ascontiguousarray(rxyz_delta)
        fpf_left = np.ascontiguousarray(fpf_left)
        fpf_mid = np.ascontiguousarray(fpf_mid)
        fpf_right = np.ascontiguousarray(fpf_right)
        for i_atom in range(nat):
            del_fpe += ( -np.dot(rxyz_delta[i_atom], fpf_left[i_atom]) - \
                        4.0*np.dot(rxyz_delta[i_atom], fpf_mid[i_atom]) - \
                        np.dot(rxyz_delta[i_atom], fpf_right[i_atom]) )/3.0

    rxyz_final = rxyz + rxyz_disp
    ldfp = False
    fp_init, dfptmp1 = get_fp(lat, rxyz, types, znucl, \
                              contract, ldfp, ntyp, nx, lmax, cutoff)
    fp_final, dfptmp2 = get_fp(lat, rxyz_final, types, znucl, \
                               contract, ldfp, ntyp, nx, lmax, cutoff)
    e_init = get_fpe(fp_init, ntyp, types)
    e_final = get_fpe(fp_final, ntyp, types)
    e_diff = e_final - e_init
    return del_fpe, e_diff

