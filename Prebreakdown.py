import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

        s.n_old = s.n.copy(); s.u_r_old = s.u_r.copy(); s.u_z_old = s.u_z.copy() #old densities for leapfrog method.
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

        #then, in z
        for i in range(s.rr[0,:].size):
            #potential at z=0, z=z_max
            s.V[0,i] = V_bottom(s.rr[0,i])
            s.V[-1,i] = V_top(s.rr[-1,i])

        s.n_bottom = n_bottom
        s.u_z_bottom = u_z_bottom
        s.V_bottom = V_bottom


        s.save_dir = save_dir

    def step(s,update_dt=True,save=True):
        #Begin by choosing a stable dt for the step
        #going to try just using the continuity limit; if that doesn't work, will try some ideas for fluid limit
        if update_dt:
            s.dt = np.min([s.cont_dt_lim(),s.fluid_r_lim(),s.fluid_z_lim()])
        print(f'[*] dt limited to: {s.dt}')
        #next, resolve the potential
        print('[*] Resolving potential')
        s.sor(0.97)
        #then, update u_r and u_z
        print('[*] Solving fluid equation')
        s.fluid()
        #and finally, update n
        print('[*] Solving continuity equation')
        s.continuity()
        #update SOR source term
        s.f = s.e_c/pc.epsilon_0*s.dr*s.dz*s.n

        if save:
            s.save()

    def fluid(s):
        j_max = s.rr.shape[0] - 1
        l_max = s.rr.shape[0] - 1
        u_r_new = np.empty(s.rr.shape)
        u_z_new = np.empty(s.rr.shape)

        for j in range(1,j_max):
            for l in range(1,l_max):
                #update interior points
                u_r_new[j,l] = s.u_r_old[j,l] - s.u_r[j,l]*s.dt/s.dr*(s.u_r[j,l+1]-s.u_r[j,l-1]) - s.u_z[j,l]*s.dt/s.dz*(s.u_r[j+1,l]-s.u_r[j-1,l]) \
                                                    - 2*s.dt*s.nu*s.u_r[j,l] - s.e_c/s.m_e * s.dt/s.dr*(s.V[j,l+1]-s.V[j,l-1])

                u_z_new[j,l] = s.u_z_old[j,l] - s.u_z[j,l]*s.dt/s.dz*(s.u_z[j+1,l]-s.u_z[j-1,l]) - s.u_r[j,l]*s.dt/s.dr*(s.u_z[j,l+1]-s.u_z[j,l-1]) \
                                                    - 2*s.dt*s.nu*s.u_z[j,l] - s.e_c/s.m_e * s.dt/s.dz*(s.V[j+1,l]-s.V[j-1,l])


        for l in range(1,l_max):
            #update boundaries in z
            #First, update boundary at z=0
            #Where u is set at the boundary
            u_r_new[0,l] = s.u_r_old[0,l] - s.u_r[0,l]*s.dt/s.dr*(s.u_r[0,l+1]-s.u_r[0,l-1]) - s.u_z[0,l]*s.dt/s.dz*(s.u_r[1,l]) \
                            - 2*s.dt*s.nu*s.u_r[0,l] - s.e_c/s.m_e * s.dt/s.dr*(s.V[0,l+1]-s.V[0,l-1])

            u_z_new[0,l] = s.u_z_old[0,l] - s.u_z[0,l]*s.dt/s.dz*(s.u_z[1,l]-s.u_z_bottom()) - s.u_r[0,l]*s.dt/s.dr*(s.u_z[0,l+1]-s.u_z[0,l-1]) \
                                                - 2*s.dt*s.nu*s.u_z[0,l] - s.e_c/s.m_e * s.dt/s.dr*(s.V[1,l]-s.V_bottom(s.rr[0,l]))

            #diffuse boundary at z=1
            u_r_new[j_max,l] = s.u_r[j_max-1,l]-s.u_r[j_max,l]*s.dt/s.dr*(s.u_r[j_max,l+1]-s.u_r[j_max,l-1])-2*s.nu*s.u_r[j_max,l]-s.e_c/s.m_e*s.dt/s.dr*(s.V[j_max,l+1]-s.V[j_max,l-1])
            u_z_new[j_max,l] = s.u_z[j_max-1,l]-s.u_z[j_max,l]*s.dt/s.dr*(s.u_z[j_max,l+1]-s.u_z[j_max,l-1])-2*s.nu*s.u_r[j_max,l]

        for j in range(1,j_max):
            #update boundaries in r
            #first, at r=0 (axisymmetric)
            u_r_new[j,0] = s.u_r_old[j,0]-s.u_z[j,0]*s.dt/s.dz*(s.u_r[j+1,0]-s.u_r[j-1,l])-2*s.nu*s.u_r[j,0]
            u_z_new[j,0] = s.u_z_old[j,0]-s.u_z[j,0]*s.dt/s.dz*(s.u_z[j+1,0]-s.u_z[j-1,0])-2*s.nu*s.u_z[j,0] \
                            -s.e_c/s.m_e*s.dt/s.dr*(s.V[j+1,0]-s.V[j-1,0])

            #and the boundary at l=l_max
            u_r_new[j,l_max] = s.u_r[j,l_max-1]-s.u_z[j,l_max]*s.dt/s.dz*(s.u_r[j+1,l_max]-s.u_r[j-1,l_max])-2*s.nu*s.u_r[j,l]
            u_z_new[j,l_max] = s.u_z[j,l_max-1]-s.u_z[j,l_max]*s.dt/s.dz*(s.u_z[j+1,l_max]-s.u_z[j-1,l_max])-2*s.nu*s.u_z[j,l_max]-s.e_c/s.m_e*s.dt/s.dr*(s.V[j+1,l_max]-s.V[j-1,l_max])

        #finally, deal with the corners
        #where BC are mixed
        #at j=0 and l=0 we have:
        u_r_new[0,0] = s.u_r_old[0,0] - s.u_z[0,0]*s.dt/s.dr*(s.u_r[1,0]) - 2*s.dt*s.nu*s.u_r[0,0]
        u_z_new[0,0] = s.u_z_old[0,0] - s.u_z[0,0]*s.dt/s.dr*(s.u_z[1,0]-s.u_z_bottom(s.rr[0,0])) - 2*s.nu*s.u_z[0,0] - s.e_c/s.m_e*s.dt/s.dr*(s.V[1,0]-s.V_bottom(s.rr[0,0]))

        #at j=j_max and l=l_max we have:
        u_r_new[j_max,l_max] = s.u_r[j_max-1,l_max-1] - 2*s.nu*s.u_r[j_max,l_max]
        u_z_new[j_max,l_max] = s.u_z[j_max-1,l_max-1] - 2*s.nu*s.u_r[j_max,l_max]

        #at j=j_max and l=0
        u_r_new[j_max,0] = s.u_r[j_max-1,0] - 2*s.nu*s.u_r[j,0]
        u_z_new[j_max,0] = s.u_z[j_max-1,0] - 2*s.nu*s.u_z[j,0]

        #and finally, at j=0 and l=l_max
        u_r_new[0,l_max] = s.u_r[0,l_max-1] - s.u_z[0,l_max]*s.dt/s.dz*(s.u_r[1,l_max]) - 2*s.nu*s.u_r[0,l_max]
        u_z_new[0,l_max] = s.u_z[0,l_max-1] - s.u_z[0,l_max]*s.dt/s.dz*(s.u_z[1,l_max] - s.u_z_bottom(s.rr[0,l_max])) - 2*s.nu*s.u_z[0,l_max] - s.e_c/s.m_e*s.dt/s.dr*(s.V[1,l_max]-s.V_bottom(s.rr[0,l_max]))

        s.u_r_old = s.u_r.copy()
        s.u_z_old = s.u_z.copy()
        s.u_r = u_r_new.copy()
        s.u_z = u_z_new.copy()



    def continuity(s):
        #solve continuity equation using leapfrog
        #and relevant boundary conitions
        #see derivation notes
        j_max = s.rr.shape[0] - 1
        l_max = s.rr.shape[0] - 1
        n_new = np.empty(s.rr.shape)

        #Upate interior points
        for j in range(1,j_max):
            for l in range(1,l_max):
                n_new[j,l] = s.n_old[j,l] - s.dt/s.dz*(s.n[j+1,l]*s.u_z[j+1,l]-s.n[j-1,l]*s.u_z[j-1,l]) \
                                - 4*s.dt/(s.rr[j,l+1]**2-s.rr[j,l-1]**2)*(s.rr[j,l+1]*s.n[j,l+1]*s.u_r[j,l+1]-s.rr[j,l-1]*s.n[j,l-1]*s.u_r[j,l-1])

        #Update boundaries at j=0 and j=J
        for l in range(1,l_max):
            #boundary at j=0 (set by n_0 and u_z_0 photoelectric effect)
            n_new[0,l] = s.n_old[0,l] - s.dt/s.dz*(s.n[1,l]*s.u_z[1,l]-s.n_bottom(s.rr[0,l])*s.u_z_bottom(s.rr[0,l])) \
                                    - 4*s.dt/(s.rr[0,l+1]**2-s.rr[0,l-1]**2)*(s.rr[0,l+1]*s.n[0,l+1]*s.u_r[0,l+1]-s.rr[0,l-1]*s.n[0,l-1]*s.u_r[0,l-1])
            #boundary condition at j=J (diffuse)
            n_new[j_max,l] = s.n[j_max-1,l] \
                            - 4*s.dt/(s.rr[j_max,l+1]**2-s.rr[j_max,l-1]**2)*(s.rr[j_max,l+1]*s.n[j_max,l+1]*s.u_r[j_max,l+1]-s.rr[j_max,l-1]*s.n[j_max,l-1]*s.u_r[j_max,l-1])

        #Update boundaries at l=0 and l=L
        for j in range(1,j_max):
            #boundary at l=0
             n_new[j,0] = s.n_old[j,0] - s.dt/s.dz*(s.n[j+1,0]*s.u_z[j+1,0]-s.n[j-1,0]*s.u_z[j-1,0]) \
                            - 4*s.dt/s.rr[j,1]*s.n[j,1]*s.u_r[j,1]



             #boundary at l_max (diffuse)
             n_new[j,l_max] = s.n[j,l_max-1] - s.dt/s.dz*(s.n[j+1,l]*s.u_z[j+1,l]-s.n[j-1,l]*s.u_z[j-1,l])
        #boundaries at corners, which we will treat as a combination of two boundary conditions
        n_new[0,0] = s.n_old[0,0] - s.dt/s.dz*(s.n[1,0]*s.u_z[1,0]-s.n_bottom(s.rr[0,0])*s.u_z_bottom(s.rr[0,0])) \
                        - 4*s.dt/s.rr[0,1]*s.n[0,1]*s.u_r[0,1]

        n_new[j_max,0] = s.n[j_max-1,0] - 4*s.dt/s.rr[j_max,1]*s.n[j_max,1]*s.u_r[j_max,1]

        n_new[0,l_max] = s.n[0,l_max-1] - s.dt/s.dz*(s.n[1,l_max]*s.u_z[1,l_max]-s.n_bottom(s.rr[0,l_max])*s.u_z_bottom(s.rr[0,l_max]))

        n_new[j_max,l_max] = s.n[j_max-1,l_max-1]

        s.n_old = s.n.copy()
        s.n = n_new.copy()

    def cont_dt_lim(s):
        dt = []
        for j in range(1,s.rr.shape[0]-1):
            for l in range(1,s.rr.shape[1]-1):
                v_z = np.abs(s.u_z[j+1,l] - s.u_z[j-1,l])
                v_r = np.abs(4/(s.rr[j,l+1]**2-s.rr[j,l-1]**2)*(s.rr[j,l+1]*s.u_r[j,l+1]-s.rr[j,l-1]*s.u_r[j,l-1]))

                dt.append(1 / (2*np.sqrt(2)) * 1/(v_z/s.dz + v_r/s.dr))

        for l in range(1,s.rr.shape[1]-1):
            v_r = np.abs(4/(s.rr[0,l+1]**2-s.rr[0,l-1]**2)*(s.rr[0,l+1]*s.u_r[0,l+1]-s.rr[0,l-1]*s.u_r[0,l-1]))
            v_z = np.abs(s.u_z[1,l] - s.u_z_bottom(s.rr[0,l]))
            dt.append(1 / (2*np.sqrt(2)) * 1/(v_z/s.dz + v_r/s.dr))

        for j in range(1,s.rr.shape[0]-1):
            v_z = np.abs(s.u_z[j+1,0] - s.u_z[j-1,0])
            v_r = np.abs(4*s.u_r[j,1]/s.rr[j,1])
            dt.append(1 / (2*np.sqrt(2)) * 1/(v_z/s.dz + v_r/s.dr))

        v_z = np.abs(s.u_z[1,0] - s.u_z_bottom(s.rr[0,0]))
        v_r = np.abs(4*s.u_r[0,1]/s.rr[0,1])
        dt.append(1 / (2*np.sqrt(2)) * 1/(v_z/s.dz + v_r/s.dr))

        return np.min(dt)

    def fluid_r_lim(s):
        dt = np.empty((s.rr.shape[0]-1,s.rr.shape[1]-1))
        for j in range(1,s.rr.shape[0]-1):
            for l in range(1,s.rr.shape[1]-1):
                dt[j,l] = 1/(np.abs(1/s.dr*(s.u_r[j,l]*(s.u_r[j,l+1]-s.u_r[j,l-1])) + np.abs(1/s.dr*s.u_z[j,l]*(s.u_r[j+1,l]-s.u_r[j-1,l])) + np.abs(s.e_c/s.m_e*1/s.dr*(s.V[j,l+1]-s.V[j,l-1]))))

        return np.min(dt)

    def fluid_z_lim(s):
        dt = np.empty((s.rr.shape[0]-1,s.rr.shape[1]-1))
        for j in range(1,s.rr.shape[0]-1):
            for l in range(1,s.rr.shape[1]-1):
                dt[j,l] = 1/(np.abs(s.u_r[j,l]*1/s.dr*(s.u_z[j,l+1]-s.u_z[j,l-1])) + np.abs(s.u_z[j,l]*1/s.dz*(s.u_z[j+1,l]-s.u_z[j-1,l])) + np.abs(s.e_c/s.m_e*1/s.dz*(s.V[j+1,l]-s.V[j-1,l])))

        return np.min(dt)



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

    def initialize(s):
        #first, we solve for the initial potential
        s.sor(0.97)
        #next, we relax to a steady state solution for the fluid equation given this initial potential

        for _ in range(1000):
            s.dt = np.min([s.fluid_r_lim(),s.fluid_z_lim()])
            s.fluid()

    def save(s,verbose=True):
        dir = join(s.save_dir, f'at_{s.time}')
        if verbose:
            print(f'[**] Saving in {dir}')

        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

        np.save(join(dir,f'V_{s.time}.npy'),s.V)
        np.save(join(dir,f'n_{s.time}.npy'),s.n)
        np.save(join(dir,f'u_r_{s.time}.npy'),s.u_r)
        np.save(join(dir,f'u_z_{s.time}.npy'),s.u_z)

    def u_r_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.u_r, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'$u_r$')

        plt.show()

    def u_z_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.u_z, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'$u_z$')

        plt.show()

    def V_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.V, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'V')

        plt.show()

    def n_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.n, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'n')

        plt.show()
