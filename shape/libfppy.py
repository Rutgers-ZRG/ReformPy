import warnings
import numpy as np
import numba
# from numba import jit, njit, int32, float64
from scipy.optimize import linear_sum_assignment

try:
    from numba import jit, float64, int32
    use_numba = True
except ImportError:
    use_numba = False

    # Define dummy decorator and type aliases if Numba is not available
    def jit(*args, **kwargs):
        return lambda func: func

    float64 = int32 = lambda: None

@jit(nopython=True)
def get_rcovdata():
    dat = [
        [0  , 1.0],   # X
        [1  , 0.37],  # H
        [2  , 0.32],  # He
        [3  , 1.34],  # Li
        [4  , 0.90],  # Be
        [5  , 0.82],  # B
        [6  , 0.77],  # C
        [7  , 0.75],  # N
        [8  , 0.73],  # O
        [9  , 0.71],  # F
        [10 , 0.69],  # Ne
        [11 , 1.54],  # Na
        [12 , 1.30],  # Mg
        [13 , 1.18],  # Al
        [14 , 1.11],  # Si
        [15 , 1.06],  # P
        [16 , 1.02],  # S
        [17 , 0.99],  # Cl
        [18 , 0.97],  # Ar
        [19 , 1.96],  # K
        [20 , 1.74],  # Ca
        [21 , 1.44],  # Sc
        [22 , 1.36],  # Ti
        [23 , 1.25],  # V
        [24 , 1.27],  # Cr
        [25 , 1.39],  # Mn
        [26 , 1.25],  # Fe
        [27 , 1.26],  # Co
        [28 , 1.21],  # Ni
        [29 , 1.38],  # Cu
        [30 , 1.31],  # Zn
        [31 , 1.26],  # Ga
        [32 , 1.22],  # Ge
        [33 , 1.19],  # As
        [34 , 1.16],  # Se
        [35 , 1.14],  # Br
        [36 , 1.10],  # Kr
        [37 , 2.11],  # Rb
        [38 , 1.92],  # Sr
        [39 , 1.62],  # Y
        [40 , 1.48],  # Zr
        [41 , 1.37],  # Nb
        [42 , 1.45],  # Mo
        [43 , 1.56],  # Tc
        [44 , 1.26],  # Ru
        [45 , 1.35],  # Rh
        [46 , 1.31],  # Pd
        [47 , 1.53],  # Ag
        [48 , 1.48],  # Cd
        [49 , 1.44],  # In
        [50 , 1.41],  # Sn
        [51 , 1.38],  # Sb
        [52 , 1.35],  # Te
        [53 , 1.33],  # I
        [54 , 1.30],  # Xe
        [55 , 2.25],  # Cs
        [56 , 1.98],  # Ba
        [57 , 1.80],  # La
        [58 , 1.63],  # Ce
        [59 , 1.76],  # Pr
        [60 , 1.74],  # Nd
        [61 , 1.73],  # Pm
        [62 , 1.72],  # Sm
        [63 , 1.68],  # Eu
        [64 , 1.69],  # Gd
        [65 , 1.68],  # Tb
        [66 , 1.67],  # Dy
        [67 , 1.66],  # Ho
        [68 , 1.65],  # Er
        [69 , 1.64],  # Tm
        [70 , 1.70],  # Yb
        [71 , 1.60],  # Lu
        [72 , 1.50],  # Hf
        [73 , 1.38],  # Ta
        [74 , 1.46],  # W
        [75 , 1.59],  # Re
        [76 , 1.28],  # Os
        [77 , 1.37],  # Ir
        [78 , 1.28],  # Pt
        [79 , 1.44],  # Au
        [80 , 1.49],  # Hg
        [81 , 1.48],  # Tl
        [82 , 1.47],  # Pb
        [83 , 1.46],  # Bi
        [84 , 1.45],  # Po
        [85 , 1.47],  # At
        [86 , 1.42],  # Rn
        [87 , 2.23],  # Fr
        [88 , 2.01],  # Ra
        [89 , 1.86],  # Ac
        [90 , 1.75],  # Th
        [91 , 1.69],  # Pa
        [92 , 1.70],  # U
        [93 , 1.71],  # Np
        [94 , 1.72],  # Pu
        [95 , 1.66],  # Am
        [96 , 1.66],  # Cm
        [97 , 1.68],  # Bk
        [98 , 1.68],  # Cf
        [99 , 1.65],  # Es
        [100, 1.67],  # Fm
        [101, 1.73],  # Md
        [102, 1.76],  # No
        [103, 1.61],  # Lr
        [104, 1.57],  # Rf
        [105, 1.49],  # Db
        [106, 1.43],  # Sg
        [107, 1.41],  # Bh
        [108, 1.34],  # Hs
        [109, 1.29],  # Mt
        [110, 1.28],  # Ds
        [111, 1.21],  # Rg
        [112, 1.22],  # Cn
    ]

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
# def readvasp(vp):
#     buff = []
#     with open(vp) as f:
#         for line in f:
#             buff.append(line.split())

#     lat = np.array(buff[2:5], float)
#     try:
#         typt = np.array(buff[5], int)
#     except:
#         del(buff[5])
#         typt = np.array(buff[5], int)
#     nat = sum(typt)
#     pos = np.array(buff[7:7 + nat], float)
#     types = []
#     for i in range(len(typt)):
#         types += [i+1]*typt[i]
#     types = np.array(types, int)
#     rxyz = np.dot(pos, lat)
#     # rxyz = pos
#     return lat, rxyz, types

# # @jit(nopython=True)
# def read_types(vp):
#     buff = []
#     with open(vp) as f:
#         for line in f:
#             buff.append(line.split())
#     try:
#         typt = np.array(buff[5], int)
#     except:
#         del(buff[5])
#         typt = np.array(buff[5], int)
#     types = []
#     for i in range(len(typt)):
#         types += [i+1]*typt[i]
#     types = np.array(types, int)
#     return types

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

# @jit(nopython=True)
@jit('(float64)(float64[:,:], float64[:,:], float64)', nopython=True)
def count_atoms_within_cutoff(lat, rxyz, cutoff):
    natoms = len(rxyz)
    count = 0

    ixyzf = get_ixyz(lat, cutoff)
    ixyz = int(ixyzf) + 1

    for iat in range(natoms):
        xi, yi, zi = rxyz[iat]

        for jat in range(natoms):
            if jat == iat:
                continue  # Skip the same atom

            for ix in range(-ixyz, ixyz + 1):
                for iy in range(-ixyz, ixyz + 1):
                    for iz in range(-ixyz, ixyz + 1):

                        xj = rxyz[jat][0] + ix * lat[0][0] + iy * lat[1][0] + iz * lat[2][0]
                        yj = rxyz[jat][1] + ix * lat[0][1] + iy * lat[1][1] + iz * lat[2][1]
                        zj = rxyz[jat][2] + ix * lat[0][2] + iy * lat[1][2] + iz * lat[2][2]

                        d2 = (xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2

                        if d2 < cutoff ** 2:
                            count += 1
                            break  # Only need to count one image per jat

    return count

def recommend_cutoff_and_nx(lat, rxyz, initial_cutoff=5.0, max_cutoff=10.0, step=0.5):
    """
    Recommend a suitable `cutoff` and `nx` value based on lattice and atomic positions.

    Parameters:
    lat: numpy.ndarray
        Lattice matrix.
    rxyz: numpy.ndarray
        Positions of atoms in the unit cell.
    initial_cutoff: float
        Starting value for the cutoff radius.
    max_cutoff: float
        Maximum cutoff radius to try.
    step: float
        Step size to increase the cutoff.

    Returns:
    tuple:
        Recommended cutoff and nx.
    """
    cutoff = initial_cutoff
    best_cutoff = initial_cutoff
    best_nx = 0

    lat = np.array(lat, dtype = np.float64)
    rxyz = np.array(rxyz, dtype = np.float64)

    while cutoff <= max_cutoff:
        cutoff = np.float64(cutoff)
        max_atoms_in_sphere = count_atoms_within_cutoff(lat, rxyz, cutoff)
        if max_atoms_in_sphere > best_nx:
            best_nx = max_atoms_in_sphere
            best_cutoff = cutoff

        cutoff += step

    return best_cutoff, best_nx

@jit(nopython=True)
def shrink_cutoff(lat, rxyz, types, znucl, iat, nx, initial_cutoff):
    # Start with a cutoff slightly smaller than the current value
    step = 0.5  # Amount to shrink the cutoff
    cutoff = initial_cutoff - step

    while True:
        n_sphere = 0
        cutoff2 = cutoff ** 2
        ixyzf = get_ixyz(lat, cutoff)
        ixyz = int(ixyzf) + 1

        for jat in range(len(rxyz)):
            for ix in range(-ixyz, ixyz + 1):
                for iy in range(-ixyz, ixyz + 1):
                    for iz in range(-ixyz, ixyz + 1):
                        xj = rxyz[jat][0] + ix * lat[0][0] + iy * lat[1][0] + iz * lat[2][0]
                        yj = rxyz[jat][1] + ix * lat[0][1] + iy * lat[1][1] + iz * lat[2][1]
                        zj = rxyz[jat][2] + ix * lat[0][2] + iy * lat[1][2] + iz * lat[2][2]
                        d2 = (xj - rxyz[iat][0]) ** 2 + (yj - rxyz[iat][1]) ** 2 + (zj - rxyz[iat][2]) ** 2

                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > nx:
                                break

        if n_sphere <= nx:
            return cutoff
        else:
            cutoff -= step  # Reduce the cutoff further if necessary

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

    # Modified the way to get rcov
    rcovdata = get_rcovdata()

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
        neighbors = []
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
                            neighbors.append((d2, jat, [xj, yj, zj], rcovj))  # Collect all neighbors
        # Sort neighbors by distance (d2)
        neighbors.sort(key=lambda x: x[0])
        # Chop neighbors to `nx` if `n_sphere` exceeds `nx`
        if n_sphere > nx:
            neighbors = neighbors[:nx]

        # Process each neighbor
        for n, (d2, jat, rxyz_j, rcovj) in enumerate(neighbors):
            ampt = (1.0 - d2 * fc) ** (NC - 1)
            amp.append(ampt * (1.0 - d2 * fc))
            damp.append(-2.0 * fc * NC * ampt)
            indori.append(jat)
            rxyz_sphere.append(rxyz_j)
            rcov_sphere.append(rcovj)
            alpha.append(0.5 / rcovj**2)

            if jat == iat and n == 0:
                ityp_sphere = 0
                icenter = n
            else:
                ityp_sphere = types[jat]

            for il in range(lseg):
                if il == 0:
                    ind[il + lseg * n] = ityp_sphere * l
                else:
                    ind[il + lseg * n] = ityp_sphere * l + 1

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

        lfp[iat] = fp0

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
                    omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
            sfp0 = np.linalg.eigvals(omx)
            sfp.append(sorted(sfp0))

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
def get_fp_dist(fp1, fp2, types, mx=False):
# def get_fp_dist(ntyp, types, fp1, fp2, mx=False):
    ntyp = len(set(types))
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



def get_lfp(cell,
            cutoff=4.0,
            log=True,
            orbital='s',
            natx=300):
    '''
    cell : tuple (lattice, rxyz, types, znucl)
    '''
    (lat, rxyz, types, znucl) = _expand_cell(cell)
    if log is True:
        wlog = 1
    else:
        wlog = 0
    if orbital == 's':
        lmax = 0
    elif orbital == 'sp':
        lmax = 1
    else:
        print ('Warning: wrong type of orbital')
        lmax = 0

    contract = False
    ldfp = False
    nx = natx
    ntyp = len(set(types))
    lat = np.array(lat, dtype = np.float64)
    rxyz = np.array(rxyz, dtype = np.float64)
    types = np.int32(types)
    znucl =  np.int32(znucl)
    ntyp =  np.int32(ntyp)
    nx = np.int32(nx)
    lmax = np.int32(lmax)
    cutoff = np.float64(cutoff)
    lfp, _ = get_fp(lat, rxyz, types, znucl, contract, ldfp, ntyp, nx, lmax, cutoff)
    return np.array(lfp)


def get_sfp(cell,
            cutoff=4.0,
            log=True,
            orbital='s',
            natx=300):
    '''
    cell : tuple (lattice, rxyz, types, znucl)
    '''
    lat, rxyz, types, znucl = _expand_cell(cell)
    if log is True:
        wlog = 1
    else:
        wlog = 0
    if orbital == 's':
        lmax = 0
    elif orbital == 'sp':
        lmax = 1
    else:
        print ('Warning: wrong type of orbital')
        lmax = 0

    contract = True
    ldfp = False
    nx = natx
    ntyp = len(set(types))
    lat = np.array(lat, dtype = np.float64)
    rxyz = np.array(rxyz, dtype = np.float64)
    types = np.int32(types)
    znucl =  np.int32(znucl)
    ntyp =  np.int32(ntyp)
    nx = np.int32(nx)
    lmax = np.int32(lmax)
    cutoff = np.float64(cutoff)
    sfp, _ = get_fp(lat, rxyz, types, znucl, contract, ldfp, ntyp, nx, lmax, cutoff)
    return np.array(sfp)

def get_dfp(cell,
            cutoff=4.0,
            log=True,
            orbital='s',
            natx=300):
    '''
    cell : tuple (lattice, rxyz, types, znucl)
    '''
    (lat, rxyz, types, znucl) = _expand_cell(cell)
    if log is True:
        wlog = 1
    else:
        wlog = 0
    if orbital == 's':
        lmax = 0
    elif orbital == 'sp':
        lmax = 1
    else:
        print ('Warning: wrong type of orbital')
        lmax = 0

    contract = False
    ldfp = True
    nx = natx
    ntyp = len(set(types))
    lat = np.array(lat, dtype = np.float64)
    rxyz = np.array(rxyz, dtype = np.float64)
    types = np.int32(types)
    znucl =  np.int32(znucl)
    ntyp =  np.int32(ntyp)
    nx = np.int32(nx)
    lmax = np.int32(lmax)
    cutoff = np.float64(cutoff)
    lfp, dfp = get_fp(lat, rxyz, types, znucl, contract, ldfp, ntyp, nx, lmax, cutoff)
    return np.array(lfp), np.array(dfp)




# def get_nfp(rxyz, types, znucl):
#     nfp = fp.fp_nonperiodic(rxyz, types, znucl)
#     return np.array(nfp)

def get_nfp_dist(fp1, fp2):
    d = fp1 - fp2
    return np.sqrt(np.dot(d, d))

def _expand_cell(cell):
    lat = np.array(cell[0], float)
    rxyz = np.array(cell[1], float)
    types = np.array(cell[2], int)
    znucl = np.array(cell[3], int)
    if len(rxyz) != len(types) or len(set(types)) != len(znucl):
        raise ValueError('Something wrong with rxyz / types / znucl.')
    return lat, rxyz, types, znucl


# @jit('Tuple((float64, float64[:,:]))(float64[:,:], float64[:,:,:,:], int32, \
#       int32[:])', nopython=True)
# def get_ef(fp, dfp, ntyp, types):
#     nat = len(fp)
#     e = 0.
#     fp = np.ascontiguousarray(fp)
#     dfp = np.ascontiguousarray(dfp)
#     for ityp in range(ntyp):
#         itype = ityp + 1
#         e0 = 0.
#         for i in range(nat):
#             for j in range(nat):
#                 if types[i] == itype and types[j] == itype:
#                     vij = fp[i] - fp[j]
#                     t = np.vdot(vij, vij)
#                     e0 += t
#             e0 += 1.0/(np.linalg.norm(fp[i]) ** 2)
#         # print ("e0", e0)
#         e += e0
#     # print ("e", e)

#     force_0 = np.zeros((nat, 3), dtype = np.float64)
#     force_prime = np.zeros((nat, 3), dtype = np.float64)

#     for k in range(nat):
#         for ityp in range(ntyp):
#             itype = ityp + 1
#             for i in range(nat):
#                 for j in range(nat):
#                     if  types[i] == itype and types[j] == itype:
#                         vij = fp[i] - fp[j]
#                         dvij = dfp[i][k] - dfp[j][k]
#                         for l in range(3):
#                             t = -2 * np.vdot(vij, dvij[l])
#                             force_0[k][l] += t
#                 for m in range(3):
#                     t_prime = 2.0 * np.vdot(fp[i],dfp[i][k][m]) / (np.linalg.norm(fp[i]) ** 4)
#                     force_prime[k][m] += t_prime
#     force = force_0 + force_prime
#     force = force - np.sum(force, axis=0)/len(force)
#     # return ((e+1.0)*np.log(e+1.0)-e), force*np.log(e+1.0)
#     return e, force


# @jit('(float64)(float64[:,:], int32, int32[:])', nopython=True)
# def get_fpe(fp, ntyp, types):
#     nat = len(fp)
#     e = 0.
#     fp = np.ascontiguousarray(fp)
#     for ityp in range(ntyp):
#         itype = ityp + 1
#         e0 = 0.
#         for i in range(nat):
#             for j in range(nat):
#                 if types[i] == itype and types[j] == itype:
#                     vij = fp[i] - fp[j]
#                     t = np.vdot(vij, vij)
#                     e0 += t
#             e0 += 1.0/(np.linalg.norm(fp[i]) ** 2)
#         e += e0
#     # return ((e+1.0)*np.log(e+1.0)-e)
#     return e


# @njit
# def get_stress(lat, rxyz, forces):
#     """
#     Compute the stress tensor analytically using the virial theorem.

#     Parameters:
#     - lat: (3, 3) array of lattice vectors.
#     - rxyz: (nat, 3) array of atomic positions in Cartesian coordinates.
#     - forces: (nat, 3) array of forces on each atom.

#     Returns:
#     - stress_voigt: (6,) array representing the stress tensor in Voigt notation.
#     """
#     # Ensure inputs are NumPy arrays with correct data types
#     lat = np.asarray(lat, dtype=np.float64)
#     rxyz = np.asarray(rxyz, dtype=np.float64)
#     forces = np.asarray(forces, dtype=np.float64)

#     # Compute the cell volume
#     cell_vol = np.abs(np.linalg.det(lat))

#     # Initialize the stress tensor
#     stress_tensor = np.zeros((3, 3), dtype=np.float64)

#     # Compute the stress tensor using the virial theorem
#     nat = rxyz.shape[0]
#     for i in range(nat):
#         for m in range(3):
#             for n in range(3):
#                 stress_tensor[m, n] -= forces[i, m] * rxyz[i, n]

#     # Divide by the cell volume
#     stress_tensor /= cell_vol

#     # Ensure the stress tensor is symmetric (if applicable)
#     # stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

#     # Convert the stress tensor to Voigt notation
#     # The Voigt notation order is: [xx, yy, zz, yz, xz, xy]
#     stress_voigt = np.array([
#         stress_tensor[0, 0],  # xx
#         stress_tensor[1, 1],  # yy
#         stress_tensor[2, 2],  # zz
#         stress_tensor[1, 2],  # yz
#         stress_tensor[0, 2],  # xz
#         stress_tensor[0, 1],  # xy
#     ], dtype=np.float64)

#     return stress_voigt


# @jit('Tuple((float64, float64))(float64[:,:], float64[:,:], int32[:], int32[:], \
#       boolean, int32, int32, int32, float64)', nopython=True)
# def get_simpson_energy(lat, rxyz, types, znucl,
#                        contract,
#                        ntyp,
#                        nx,
#                        lmax,
#                        cutoff):
#     lat = np.ascontiguousarray(lat)
#     rxyz = np.ascontiguousarray(rxyz)
#     rxyz_delta = np.zeros_like(rxyz)
#     rxyz_disp = np.zeros_like(rxyz)
#     rxyz_left = np.zeros_like(rxyz)
#     rxyz_mid = np.zeros_like(rxyz)
#     rxyz_right = np.zeros_like(rxyz)
#     nat = len(rxyz)
#     del_fpe = 0.0
#     iter_max = 100
#     step_size = 1.e-5
#     rxyz_delta = step_size*( np.random.rand(nat, 3).astype(np.float64) - \
#                             0.5*np.ones((nat, 3), dtype = np.float64) )
#     for i_iter in range(iter_max):
#         # rxyz_delta = step_size*( np.random.rand(nat, 3).astype(np.float64) - \
#         #                         0.5*np.ones((nat, 3), dtype = np.float64) )
#         rxyz_disp += 2.0*rxyz_delta
#         rxyz_left = rxyz.copy() + 2.0*i_iter*rxyz_delta
#         rxyz_mid = rxyz.copy() + 2.0*(i_iter+1)*rxyz_delta
#         rxyz_right = rxyz.copy() + 2.0*(i_iter+2)*rxyz_delta
#         ldfp = True
#         fp_left, dfp_left = get_fp(lat, rxyz_left, types, znucl, \
#                                    contract, ldfp, ntyp, nx, lmax, cutoff)
#         fp_mid, dfp_mid = get_fp(lat, rxyz_mid, types, znucl, \
#                                    contract, ldfp, ntyp, nx, lmax, cutoff)
#         fp_right, dfp_right = get_fp(lat, rxyz_right, types, znucl, \
#                                      contract, ldfp, ntyp, nx, lmax, cutoff)
#         fpe_left, fpf_left = get_ef(fp_left, dfp_left, ntyp, types)
#         fpe_mid, fpf_mid = get_ef(fp_mid, dfp_mid, ntyp, types)
#         fpe_right, fpf_right = get_ef(fp_right, dfp_right, ntyp, types)

#         rxyz_delta = np.ascontiguousarray(rxyz_delta)
#         fpf_left = np.ascontiguousarray(fpf_left)
#         fpf_mid = np.ascontiguousarray(fpf_mid)
#         fpf_right = np.ascontiguousarray(fpf_right)
#         for i_atom in range(nat):
#             del_fpe += ( -np.dot(rxyz_delta[i_atom], fpf_left[i_atom]) - \
#                         4.0*np.dot(rxyz_delta[i_atom], fpf_mid[i_atom]) - \
#                         np.dot(rxyz_delta[i_atom], fpf_right[i_atom]) )/3.0

#     rxyz_final = rxyz + rxyz_disp
#     ldfp = False
#     fp_init, dfptmp1 = get_fp(lat, rxyz, types, znucl, \
#                               contract, ldfp, ntyp, nx, lmax, cutoff)
#     fp_final, dfptmp2 = get_fp(lat, rxyz_final, types, znucl, \
#                                contract, ldfp, ntyp, nx, lmax, cutoff)
#     e_init = get_fpe(fp_init, ntyp, types)
#     e_final = get_fpe(fp_final, ntyp, types)
#     e_diff = e_final - e_init
#     return del_fpe, e_diff

