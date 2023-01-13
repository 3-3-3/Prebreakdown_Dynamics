import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, besselpoly, jn_zeros
import scipy.constants as pc


class Prebreakdown:
    def __init__(s, r_min, r_max, z_min, z_max, points_r, points_z, dt, nu=0, v_bound=lambda r,z : 0):
        s.dr = (r_max - r_min) / points_r; s.dz = (z_max - z_min) / points_z
        s.rr, s.zz = np.meshgrid(np.linspace(r_min,r_max,points_r), np.linspace(z_min, z_max, points_z))
        s.V = np.zeros(s.rr.shape) #Potential
        s.n = np.zeros(s.rr.shape) #electron density
        s.u_r = np.zeros(s.rr.shape) #electron velocity density in r
        s.u_z = np.zeros(s.rr.shape) #electron velocity density in z

        s.n_old = s.n; s.u_r_old = s.u_r; s.u_z_old = s.u_z #old densities for leapfrog method.
        s.nu = nu #neutral collision frequency
        s.dt = dt #time step


        #set boundary conditions from function v_bound
        s.V[0,:] = v_bound(s.rr[0,:], s.zz[0,:])
        s.V[:,0] = v_bound(s.rr[:,0], s.zz[:,0])
        s.V[-1,:] = v_bound(s.rr[-1,:], s.zz[-1,:])
        s.V[:,-1] = v_bound(s.rr[:,-1], s.zz[:,-1])

        #finite difference coefficients for solving Poisson's equation with sor
        s.a = np.ones(s.V.shape); s.b = np.ones(s.V.shape);
        s.c = (s.rr + s.dr / 2) / s.rr; s.d = (s.rr - s.dr / 2) / s.rr
        s.e = -4 * np.ones(s.V.shape)

    def fluid(s):
        #solve fluid equation using leapfrog
        j_max = s.rr.shape[0]
        l_max = s.rr.shape[0]
        u_r_new = s.u_r; u_z_new = s.u_z

        for j in range(1, j_max - 1):
            for l in range(1, l_max - 1):
                u_r_new[j][l] = s.u_r_old[j][l] - s.u_r[j][l] * s.dt / s.dr * (s.u_r[j+1][l] - s.u_r[j-1][l]) \
                            - s.u_z[j][l] * s.dt / s.dz * (s.u_r[j][l+1] - s.u_r[j][l-1]) - s.nu * s.u_r[j][l] * s.dt \
                            - pc.e / pc.m_e * s.dt / s.dr * (s.V[j+1][l] - s.V[j-1][l])

                u_z_new[j][l] = s.u_z_old[j][l] - s.u_r[j][l] * s.dt / s.dr * (s.u_z[j+1][l] - s.u_z[j-1][l]) \
                            - s.u_z[j][l] * s.dt / s.dz * (s.u_z[j][l+1] - s.u_z[j][l-1]) - s.nu * s.u_z[j][l] \
                            - pc.e / pc.m_e * s.dt / s.dr * (s.V[j][l+1] - s.V[j][l-1])

            s.u_r_old = s.u_r; s.u_z_old = s.u_z
            s.u_r = u_r_new; s.u_z = u_r_new

    def continuity(s):
        #solve continuity equation using leapfrog
        j_max = s.rr.shape[0]
        l_max = s.rr.shape[0]
        n_new = s.n

        for j in range(1, j_max - 1):
            for l in range(1, l_max - 1):
                #loop over all gridpoints (except boundaries)
                n_new[j][l] = s.n_old[j][l] \
                    - 1 / s.rr[j][l] * s.dt / s.dr \
                    * (s.rr[j+1][l] * s.n[j+1][l] * s.u_r[j+1][l] - s.rr[j-1][l] * s.n[j-1][l] * s.u_r[j-1][l]) \
                    - s.dt / s.dz * (s.n[j][l+1] * s.u_z[j][l+1] - s.n[j][l-1] * s.u_z[j][l-1])

        s.n_old = s.n
        s.n = n_new

    def sor(s, sp_r, iter):
        #successive overrelaxation to solve Poisson's equation
        #see e.g. Numerical Recipes Chapter 20
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
                                + s.d[j][l] * s.V[j][l - 1] + s.e[j][l] * s.V[j][l] - pc.e * s.dz * s.dr * s.n[j][l] #residual at nth step

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
