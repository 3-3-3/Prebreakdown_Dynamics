import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, besselpoly, jn_zeros


class Prebreakdown:
    def __init__(s, r_min, r_max, z_min, z_max, points_r, points_z, v_bound=lambda r,z : 0):
        s.dr = (r_max - r_min) / points_r; s.dz = (z_max - z_min) / points_z
        s.rr, s.zz = np.meshgrid(np.linspace(r_min,r_max,points_r), np.linspace(z_min, z_max, points_z))
        s.V = np.zeros(s.rr.shape)

        #set boundary conditions from function v_bound
        s.V[0,:] = v_bound(s.rr[0,:], s.zz[0,:])
        s.V[:,0] = v_bound(s.rr[:,0], s.zz[:,0])
        s.V[-1,:] = v_bound(s.rr[-1,:], s.zz[-1,:])
        s.V[:,-1] = v_bound(s.rr[:,-1], s.zz[:,-1])

        s.a = np.ones(s.V.shape); s.b = np.ones(s.V.shape);
        s.c = (s.rr + s.dr / 2) / s.rr; s.d = (s.rr - s.dr / 2) / s.rr

        s.e = -4 * np.ones(s.V.shape);
        s.f = np.zeros(s.V.shape)

    def sor(s, sp_r, iter):
        #successive overrelaxation
        #see e.g. Numerical Recipes Chapter 20
        #a,b,c,d,f: finite differencing coefficients
        j_max = s.rr.shape[0]
        l_max = s.rr.shape[1]
        omega = 1
        for n in range(iter):
            if n % 10 == 0:
                print(n)
            jsw = 1
            for ipass in range(2): #odd-even ordering
                lsw = jsw
                for j in range(1,j_max-1):
                    for l in range(lsw, l_max-1, 2):
                        res = s.a[j][l] * s.V[j+1][l] + s.b[j][l] * s.V[j-1][l] + s.c[j][l] * s.V[j][l + 1] \
                                + s.d[j][l] * s.V[j][l - 1] + s.e[j][l] * s.V[j][l] - s.f[j][l] #residual at nth step

                        #anorm += np.abs(res)
                        s.V[j][l] -= omega * res / s.e[j][l]
                    lsw = 3 - jsw #(3 - 1 --> 2, 3 - 2 --> 1 etc.)

                jsw = 3 - jsw

                if n == 0 and ipass == 0:
                    omega = 1 / (1 - 0.5 * sp_r ** 2)
                else:
                    omega = 1 / (1 - 0.25 * sp_r ** 2 * omega)


def B(n, a, V_0):
    a_n = jn_zeros(0, n) #first n zeros of first order Bessel equation
    return 2 * V_0 / (jv(1, a_n) ** 2 * np.cosh(a_n)) * besselpoly(a_n / 2, 1, 0)

def cylindrical_check(n, rr, zz, a, V_0):
    b = B(n, a, V_0)
    a_n = jn_zeros(0, n)

    e = np.zeros(rr.shape)
    for i in range(1,n):
        e += b[i] * jv(0, a_n[i] / a * rr) * np.cosh(a_n[i] / a * zz)

    return e
