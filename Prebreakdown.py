import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, besselpoly, jn_zeros
import scipy.constants as pc
import os
from os.path import join



class Prebreakdown:
    def __init__(s, N_r, N_z, Dr, Dz, dt, nu=0,
                    V_top=lambda r : 0, V_bottom=lambda r : 0,
                    V_r=lambda z : 0, n_bottom=lambda r : 0,
                    u_z_bottom=lambda r : 0,
                    save_dir='/'):
        '''
        r_min (usually 0): Minimum radial value in domain
        r_max: Maximum radial value in domain
        z_min, z_max: minimum and maximum z values
        points_r, points_z: Number of points to use in r and z directions
        dt: time step for continuity and fluid equations
        V_top: function describing top boundary (z=z_max) for potential
        V_bottom: function describing bottom boundary (z=z_min) for potential
        n_bottom: function describing density at z=z_min
        V_min: function describing boundary at r=0 for potential
        V_max: function describing boundary at r=r_max for potential
        '''
        s.time = 0 #time since start of simulation
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

        s.e_c = pc.e
        s.m_e = pc.m_e
        s.nu = 0 #friction coefficient


        #***set Dirichlet boundary conditions***
        #first, set in r
        for i in range(s.zz[:,0].size):
            #potential at r=r_max
            s.V[i,-1] = V_r(s.zz[i,0])
            #density at r=r_max (vanishing)
            s.n[i,-1] = 0
            #velocity at r=r_max (vanishing)
            s.u_r[i,-1] = 0
            s.u_z[i,-1] = 0

        #then, in z
        for i in range(s.rr[0,:].size):
            #potential at z=0, z=z_max
            s.V[0,i] = V_bottom(s.rr[0,i])
            s.V[-1,i] = V_top(s.rr[-1,i])
            #density at z=0 (user defined)
            s.n[0,i] = n_bottom(s.rr[0,i])
            #u_r (vanishing everywhere)

    def fluid(s):
        #solve fluid equation with operator splitting
        #(most of these should be able to be combined)
        #(but makes thinking about stability conditions easier)
        dt_p = s.dt / 4
        u_r_new = s.u_r
        u_z_new = s.u_z

        #first quarter step (field terms)
        for j in range(1,j_max): #loop over all j, excluding boundaries
            for l in range(1,l_max): #loop over all l, excluding boundaries
                u_r_new[j,l] = s.u_r_old[j,l] - s.e_c/s.m_e * dt_p/s.dr * (s.V[j,l+1] - s.V[j,l-1])
                u_z_new[j,l] = s.u_z_old[j,l] - s.e_c/s.m_e * dt_p/s.dz * (s.V[j+1,l] - s.V[j-1,l])

        s.u_r_old = s.u_r
        s.u_z_old = s.u_z
        s.u_r = u_r_new
        s.u_z = u_z_new

        #second quarter step (friction terms)
        for j in range(1,j_max):
            for l in range(1,l_max):
                u_r_new[j,l] = s.u_r_old[j,l] - 2*s.nu*s.u_r[j,l]*dt_p
                u_z_new[j,l] = s.u_z_old[j,l] - 2*s.nu*s.u_z[j,l]*dt_p

        s.u_r_old = s.u_r
        s.u_z_old = s.u_z
        s.u_r = u_r_new
        s.u_z = u_z_new

        #third quarter step (uncoupled convective terms)
        for j in range(1,j_max):
            for l in range(1,l_max):
                u_r_new[j,l] = s.u_r_old[j,l] - s.u_r[j,l]*dt_p/s.dr*(s.u_r[j,l+1] - s.u_r[j,l-1])
                u_z_new[j,l] = s.u_z_old[j,l] - s.u_r[j,l]*dt_p/s.dz*(s.u_z[j+1,l] - s.u_z[j+1,l])

        s.u_r_old = s.u_r
        s.u_z_old = s.u_z
        s.u_r = u_r_new
        s.u_z = u_z_new

        #fourth quarter step (coupled convective terms)
        for j in range(1,j_max):
            for l in range(1,l_max):
                u_r_new[j,l] = s.u_r_old[j,l] - s.u_z[j,l]*dt_p/s.dz*(s.u_z[j+1,l]-s.u_z[j-1,l])
                u_z_new[j,l] = s.u_z_old[j,l] - s.u_r[j,l]*dt_p/s.dr*(s.u_r[j,l+1]-s.u_r[j,l-1])

        s.u_r_old = s.u_r
        s.u_z_old = s.u_z
        s.u_r = u_r_new
        s.u_z = u_z_new



    def continuity(s):
        #solve continuity equation using leapfrog
        #and relevant boundary conitions
        #see derivation notes
        j_max = s.rr.shape[0]
        l_max = s.rr.shape[0]
        n_new = s.n.copy()

        #begin loop over all other non-derichlet points
        for j in range(1, j_max): #loop over j (z)
            if j == (j_max-1): #boundary at z=1
                for l in range(1, l_max - 1): #loop over interior r at boundary
                    #Deal with von Neumann-0 boundary condition at z=1
                    #we assume that the z-velocity and the density are the same on each side of the boundary
                    n_new[j,l] = s.n[j-1,l]


            else:
                #update r=0 boundary
                n_new[j,0] = s.n[j,1]

                for l in range(1, l_max - 1): #loop over all interior points in r
                    n_new[j,l] = s.n_old[j,l] \
                        - 1 / s.rr[j,l] * s.dt / s.dr \
                        * (s.rr[j,l+1] * s.n[j,l+1] * s.u_r[j,l+1] - s.rr[j,l-1] * s.n[j,l-1] * s.u_r[j,l-1]) \
                        - s.dt / s.dz * (s.n[j+1,l] * s.u_z[j+1,l] - s.n[j-1,l] * s.u_z[j-1,l])

                #update r=R boundary
                n_new[j,l_max-1] = s.n[j,l_max-1]

        s.n_old = s.n.copy()
        s.n = n_new.copy()


    def cont_dt_lim(s):
        dt = []
        for j in range(1,s.rr.shape[0]-1):
            for l in range(1,s.rr.shape[1]-1):
                dt.append(s.dz / (s.u_z[j,l] ** 2 + s.u_r ** 2))

        return np.min(dt)

    def fluid_r_lim(s):
        dt = []
        for j in range(1,s.rr.shape[0]-1):
            for l in range(1,s.rr.shape[1]-1):
                dt.append(s.dr / (2 * s.u_r[j,l]))

        return 4*np.min(dt)

    def fluid_z_lim(s):
        dt = []
        for j in range(1,r.shape[0]-1):
            for l in range(1,r.shape[1]-1):
                dt.append(s.dz / (2 * s.u_z[j,l]))

        return 4*np.min(dt)

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
            #if n % 10 == 0:
                #print(n)
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

    def step(s,save=True):
        #Begin by choosing a stable dt for the step
        s.dt = np.min(s.cont_dt_lim(), s.fluid_r_lim(), s.fluid_z_lim())
        #next, resolve the potential
        print('[*] Resolving potential')
        s.sor(0.97)
        #then, update u_r and u_z
        print('[*] Solving fluid equation')
        s.fluid()
        #and finally, update n
        print('[*] Solving continuity equation')
        s.continuity()

        if save:
            s.save()

    def save(s,verbose=True):
        dir = join(save_dir, f'at_{s.time}')
        if verbose:
            print(f'[**] Saving in {dir}')

        try:
            os.mmkdir(dir)
        except FileExistsError:
            pass

        np.save(join(dir,'V.npy'),s.V)
        np.save(join(dir,'n.npy'),s.n)
        np.save(join(dir,'u_r.npy'),s.u_r)
        np.save(join(dir,'u_z.npy'),s.u_z)

    def viz_V(s):
        fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel('Potential')
        ax.plot_surface(s.rr,s.zz,s.V,cmap=cm.bone,linewidth=0,antialiased=False)
        plt.show()
