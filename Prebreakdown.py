import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, besselpoly, jn_zeros
import scipy.constants as pc


class Prebreakdown:
    def __init__(s, N_r, N_z, Dr, Dz, dt, nu=0,
                    V_top=lambda r : 0, V_bottom=lambda r : 0,
                    V_r=lambda z : 0):
        '''
        r_min (usually 0): Minimum radial value in domain
        r_max: Maximum radial value in domain
        z_min, z_max: minimum and maximum z values
        points_r, points_z: Number of points to use in r and z directions
        dt: time step for continuity and fluid equations
        V_top: function describing top boundary (z=z_max) for potential
        V_bottom: function describing bottom boundary (z=z_min) for potential
        V_min: function describing boundary at r=0 for potential
        V_max: function describing boundary at r=r_max for potential
        '''
        s.dr = Dr; s.dz = Dz
        s.N_r = N_r; s.N_z = N_z
        s.r_max = s.dr * s.N_r; s.z_max = s.dz * N_z
        s.rr, s.zz = np.meshgrid(np.linspace(0, s.r_max, s.N_r),
                                    np.linspace(0, s.z_max, s.N_z))
        s.V = np.zeros(s.rr.shape) #Potential
        s.n = np.zeros(s.rr.shape) #electron density
        s.u_r = np.zeros(s.rr.shape) #electron velocity density in r
        s.u_z = np.zeros(s.rr.shape) #electron velocity density in z

        s.n_old = s.n; s.u_r_old = s.u_r; s.u_z_old = s.u_z #old densities for leapfrog method.
        s.nu = nu #neutral collision frequency
        s.dt = dt #time step

        #finite difference coefficients for solving Poisson's equation with sor
        #see derivation notes
        s.a = np.ones(s.V.shape); s.b = np.ones(s.V.shape);
        s.c = (s.rr + s.dr / 2) / s.rr; s.d = (s.rr - s.dr / 2) / s.rr
        s.e = -4 * np.ones(s.V.shape)
        s.f = np.zeros(s.V.shape) #source term e/e_0 * Dz * Dr * n
        #finite difference coefficients for solving Poisson's equation at the r=0 boundary
        s.a_b = 1; s.b_b = 1; s.c_b = 4; s.e_b = -6


        #set boundary conditions
        for i in range(s.zz[:,0].size):
            s.V[i,0] = V_r(s.zz[i,0])
            s.V[i,-1] = V_r(s.zz[i,0])

        for i in range(s.rr[0,:].size):
            s.V[0,i] = V_bottom(s.rr[0,i])
            s.V[-1,i] = V_top(s.rr[-1,i])

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

    def sor(s, sp_r, iterations=1000, EPS=1e-10):
        #successive overrelaxation
        #see e.g. Numerical Recipes Chapter 20
        #a,b,c,d,f: finite differencing coefficients
        j_max = s.rr.shape[0]
        l_max = s.rr.shape[1]
        omega = 1
        anorm_i = 0 #magnitude of initial residual vector
                    #to use as a criterion for stopping

        anorm = 0

        for j in range(1,j_max-1):
            for l in range(l_max-2,0,-1):
                res = s.a[j,l] * s.V[j+1, l] + s.b[j, l] * s.V[j-1, l] + s.c[j, l] * s.V[j, l + 1] \
                        + s.d[j, l] * s.V[j, l - 1] + s.e[j, l] * s.V[j, l] - s.f[j, l] #residual at nth step
                anorm_i += np.abs(res)

            res = s.a_b * s.V[j+1,0] + s.b_b * s.V[j-1,0] + s.c_b * s.V[j,1] + s.e_b * s.V[j,0]
            anorm_i += np.abs(res)




        for n in range(iterations):
            if n % 10 == 0:
                print(n)
            lsw = 1
            anorm = 0

            for ipass in range(2): #odd-even ordering
                jsw = lsw
                for j in range(jsw, j_max-1, 2):
                    #First, we loop over j (z-direction) with Dirichlet boundaries on both ends
                    for l in range(l_max-2,0,-1): #Next, we loop backwards over l so that we begin with the Dirichlet condition at L (r_max)
                                                    #and then we end the loop at l=1
                                                    #and finally treat the l=0 case–the von Neuman boundary seperatly, before incrementing j and starting over

                        res = s.a[j,l] * s.V[j+1, l] + s.b[j, l] * s.V[j-1, l] + s.c[j, l] * s.V[j, l + 1] \
                                + s.d[j, l] * s.V[j, l - 1] + s.e[j, l] * s.V[j, l] - s.f[j, l] #residual at nth step

                        anorm += np.abs(res)
                        #weighted average of old potential and new potential
                        s.V[j,l] -= omega * res / s.e[j,l]

                    #implement Neuman/axisymmetric boundary condition for l = 0
                    res = s.a_b * s.V[j+1,0] + s.b_b * s.V[j-1,0] + s.c_b * s.V[j,1] + s.e_b * s.V[j,0]
                    anorm += np.abs(res)
                    s.V[j,0] -= omega * res / s.e[j,l]

                    jsw = 3 - jsw #(3 - 1 --> 2, 3 - 2 --> 1 etc.)

                lsw = 3 - lsw

                if n == 0 and ipass == 0:
                    omega = 1 / (1 - 0.5 * sp_r ** 2)
                else:
                    omega = 1 / (1 - 0.25 * sp_r ** 2 * omega)

                if n > 5:
                    if anorm <= EPS*anorm_i:
                        print(f'Error reduced by factor of {EPS} in {n} iterations')
                        return True
                else:
                    anorm_i = anorm

        print(f'Error reduced by factor of {anorm} in {iter} iterations')
        return False
