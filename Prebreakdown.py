#AUTHOR: Liam Keeley, Colorado College
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import matplotlib.animation as animation
import scipy.constants as pc
from scipy.ndimage import gaussian_filter


import os
import time
import string



class Prebreakdown:
    def __init__(s, N_z, N_r, Dz, Dr, dt=1e-15, nu=0, diffusion=0, mobility=0,
                    V_top=lambda r, t : 0, V_bottom=lambda r, t : 0,
                    V_r=lambda z, t : 0, n_bottom=lambda r, t : 0,
                    u_z_bottom=lambda r, t: 0, save_dir='out'):
        '''
        Class for solving the prebreakdown electron dynamics of our spark gap triggered by shining UV-light on the cathode.
        Goal of simulation is to demonstrate that we can boost an applied electric field just below breakdown to one just
        above breakdown and understand the conditions under which this is possible.

        ATTRIBUTES
        -----------------------------------------------------------------
        time: time since beginning of simulation
        dz, dr: distance between gridpoints
        N_z, N_r: number of gridpoints in z and r
        z_max, r_max: maximum point in the computational domain in z and r
        zz, rr: computational grids in z and r
        V: electric potential
        E_fld: electric field
        n: electron density
        n_old: electron density from last time step for use in leapfrog method
        u_r: electron drift velocity in r
        u_r_old: electron drift velocity in r from last time step for use in leapfrog method
        u_z: electron drift velocity in z
        u_z_old: electron drift velocity in z from last time step for use in leapfrog method
        dt: time step
        nu: collision frequency for fluid model
        diffusion: electron diffusion coefficient for drift-diffusion prescription
        mobility: electron mobility coefficient for drift-diffusion prescription




        METHODS
        -----------------------------------------------------------------
        step(s,save,method,verbose): Evolve the simulation by one time step.
        drift_diffusion(s): Update the electron density based on a drift-diffusion prescription.
        continuity(s): Update the electron density based on the continuity equation.
        fluid(s): solve for the electron dift velocities based on the first moment of Vlasov's equation.
        sor(s,sp_r,iterations,EPS): resolve the elctric potential using successive overrelaxation.
        resolve_E_r(s): resolve the r component of the electric field by finite differencing the electric potential
        resolve_E_z(s): resolve the z component of the electric field by finite differencing the potential
        resolve_E_fld(s): resolve both components of the electric field
        E_mag(s): calculate the electric field magnitude at each gridpoint
        drift_diffusion_cfl(s): determine a stable time step for the drift diffusion prescription using the cfl condition
        fluid_cfl(s): determine a stable time step for using the fluid model, also with the cfl condition



        ARGUMENTS
        -----------------------------------------------------------------
        N_z: number of gridpoints to be used in z
        N_r: number of gridpoints to be used in r
        Dz: grid spacing in z
        Dr: grid spacing r
        V_top: function describing top boundary (z=z_max) for potential
        V_bottom: function describing bottom boundary (z=z_min) for potential
        n_bottom: function describing density at z=z_min
        u_z_bottom: function describing electron drift velocity in z at cathode
        V_min: function describing boundary at r=0 for potential
        V_max: function describing boundary at r=r_max for potential
        '''
        s.time = 0
        s.times = []
        s.dz = Dz; s.dr = Dr
        s.N_z = N_z; s.N_r = N_r
        s.z_max = s.dz * s.N_z; s.r_max = s.dr * N_r
        s.rr, s.zz = np.meshgrid(np.linspace(0, s.r_max, s.N_r),
                                    np.linspace(0, s.z_max, s.N_z))

        s.dV = 2 * np.pi * s.dz * s.dr * s.rr #volume element

        s.V = np.zeros(s.rr.shape)
        s.E_fld = (np.zeros(s.rr.shape), np.zeros(s.rr.shape))
        s.n = np.zeros(s.rr.shape); s.n_old = np.zeros(s.rr.shape)
        s.u_r = np.zeros(s.rr.shape); s.u_r_old = np.zeros(s.rr.shape)
        s.u_z = np.zeros(s.zz.shape); s.u_z_old = np.zeros(s.zz.shape)
        s.dt = dt

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
        s.nu = nu

        s.diffusion = diffusion
        s.mobility = mobility


        # -------------------------------------------------------------
        #***set Dirichlet boundary conditions***
        #first, set in r
        for i in range(s.zz[:,0].size):
            #potential at r=r_max
            s.V[i,-1] = V_r(s.zz[i,0],0)

        #then, in z
        for i in range(s.rr[0,:].size):
            #potential at z=0, z=z_max
            s.V[0,i] = V_bottom(s.rr[0,i],0)
            s.V[-1,i] = V_top(s.rr[-1,i],0)
            #n has 2 layers will not be touched so that the potential extends beyond it


        s.n_bottom = n_bottom
        s.u_z_bottom = u_z_bottom
        s.n_old = s.n.copy() #old densities for leapfrog method.


        s.save_dir = save_dir

        try:
            os.mkdir(s.save_dir)
        except FileExistsError:
            pass

        s.E_max = []
        s.iter = 0

    def step(s, method='Drift-Diffusion', smooth=False, smooth_sigma=0, alpha=0, iterations=1000,
                    sp_r=0.97, EPS=1e-10, save=True, verbose=True, save_every=10,substeps=1,resolve_pot_every=1):
        '''
        Take a step, updating the time step dt, electric potential, and electron density. Use either a Drift-Diffusion prescription or Fluid model.

        save: whether or not to save the output.
        method: which method to use for updating dt and the elctron density: one of 'Drift-Diffusion' or 'Fluid'.

        '''
        t = time.time()

        s.iter += 1

        if (s.iter % resolve_pot_every) == 0:
            print('Resolving potential!')
            s.f = s.e_c/pc.epsilon_0*s.dr*s.dz*s.n
            s.sor(sp_r, iterations=iterations, EPS=EPS,verbose=verbose)

        s.resolve_E_fld()
        s.E_max.append(s.E_mag().max())

        if method == 'Drift-Diffusion':
            s.dt = s.drift_diffusion_cfl()
            s.time += s.dt
            s.times.append(s.time)

            if verbose:
                print(f'Time step limited to: {s.dt} at time {s.time}')

            #update the electron density at the cathode
            for i in range(s.rr[0,:].size):
                s.n[1,i] = s.n_bottom(s.rr[1,i], s.time)
            if verbose:
                print('[*] Solving drift-diffusion equation')
            s.drift_diffusion()

        elif method == 'Fluid':
            #update the electron drift velocity at the cathode (should be constant...)
            for k in range(substeps):
                #Take substeps steps before re-resolving potential
                for i in range(s.rr[0,:].size):
                    s.u_z[0,i] = s.u_z_bottom(s.rr[0,i], s.time)
                    s.u_r[0,i] = 0

                s.fluid()
                target_dt = s.fluid_cfl() / substeps
                if verbose and (k==0):
                    print(f'target_dt: {target_dt}')

                ratio = target_dt / s.dt

                if verbose and (k==0):
                    print(f'Old dt: {s.dt}, ratio: {ratio}')

                if ratio < 1.1025 and ratio > 0.9025:
                    s.dt = np.sqrt(ratio) * s.dt
                elif ratio >= 1.025:
                    s.dt = 1.05 * s.dt
                elif ratio <= 0.9025:
                    s.dt = 0.95 * s.dt

                #s.dt = s.fluid_cfl()
                s.time += s.dt
                s.times.append(s.time)

                if verbose and (k==0):
                    print(f'Time step limited to: {s.dt} at time: {s.time}')

                #resolve fluid velocity and update electron density using continuity equation
                s.continuity()

            if smooth:
                s.n = gaussian_filter(s.n, sigma=smooth_sigma)
                s.u_r = gaussian_filter(s.u_r, sigma=smooth_sigma)
                s.u_z = gaussian_filter(s.u_z, sigma=smooth_sigma)

            if save and (s.iter % save_every == 0):
                s.save(verbose=verbose)

            if verbose:
                print(f'Step took: {time.time() - t}')

    def constant_dt_step(s, dt, method='Drift-Diffusion', iterations=1000, sp_r=0.97, EPS=1e-10, save=True, verbose=True, save_every=10):
        '''
        Take a step with a given time step dt and update electric potential, and electron density. Use either a Drift-Diffusion prescription or Fluid model.

        save: whether or not to save the output.
        method: which method to use for updating dt and the elctron density: one of 'Drift-Diffusion' or 'Fluid'.

        '''
        t = time.time()
        for i in range(s.rr[0,:].size):
            s.n[1,i] = s.n_bottom(s.rr[1,i], s.time)
            s.u_z[0,i] = s.u_z_bottom(s.rr[0,i], s.time)
            s.u_r[0,i] = 0

        s.iter += 1
        s.f = s.e_c/pc.epsilon_0*s.dr*s.dz*s.n
        s.sor(sp_r, iterations=iterations, EPS=EPS, verbose=verbose)
        s.resolve_E_fld()
        s.E_max.append(s.E_mag().max())

        if method == 'Drift-Diffusion':
            s.dt = dt
            s.time += s.dt
            s.times.append(s.time)


            #update the electron density at the cathode
            for i in range(s.rr[0,:].size):
                s.n[1,i] = s.n_bottom(s.rr[1,i], s.time)

            if verbose:
                print('[*] Solving drift-diffusion equation')

            s.drift_diffusion()

        elif method == 'Fluid':
            #update the electron density at the cathode
            s.fluid()
            s.dt = dt
            s.time += s.dt
            s.times.append(s.time)

            cfl = s.fluid_cfl()

            if dt > cfl:
                print(f'WARNING: time step is {s.dt}, but cfl step is: {cfl}. Time step is unstable!')



            #resolve fluid velocity and update electron density using continuity equation
            s.continuity()

        if save and (s.iter % save_every == 0):
            s.save(verbose=verbose)

        if verbose:
            print(f'Step took: {time.time() - t}')

    def drift_diffusion(s):
        '''update electron density based on a drift-diffusion prescription using leapfrog and relevant boundary conditions'''
        j_max = s.rr.shape[0] - 1
        l_max = s.rr.shape[0] - 1
        n_new = s.n.copy()

        #First, update all interior points
        for j in range(2,j_max):
            for l in range(1,l_max):
                n_new[j,l] = s.n_old[j,l] + s.mobility*1/s.rr[j,l]*s.dt/s.dr*(s.rr[j,l+1]*s.n[j,l+1]*s.E_fld[1][j,l+1] - s.rr[j,l-1]*s.n[j,l-1]*s.E_fld[1][j,l-1]) \
                                          + s.mobility*s.dt/s.dz*(s.n[j+1,l]*s.E_fld[0][j+1,l] - s.n[j-1,l]*s.E_fld[0][j-1,l]) \
                                          + s.diffusion*2/s.rr[j,l]*s.dt/s.dr*((s.rr[j,l]+s.dr/2)*(s.n[j,l+1]-s.n[j,l])-(s.rr[j,l]-s.dr/2)*(s.n[j,l]-s.n[j,l-1])) \
                                          + s.diffusion*2*s.dt/(s.dz**2)*(s.n[j+1,l] - 2*s.n[j,l] + s.n[j-1,l])

        #Next, update boundaries at j=J (the one at j=0 is constant)
        for l in range(1,l_max):
            #boundary at j=J (diffuse)
            n_new[j_max,l] = s.n[j_max-1,l] \
                                    + s.mobility*1/s.rr[j_max,l]*s.dt/s.dr*(s.rr[j_max,l+1]*s.n[j_max,l+1]*s.E_fld[1][j_max,l+1] - s.rr[j_max,l-1]*s.n[j_max,l-1]*s.E_fld[1][j_max,l-1]) \
                                    + s.diffusion*2/s.rr[j_max,l]*s.dt/s.dr*((s.rr[j_max,l]+s.dr/2)*(s.n[j_max,l+1]-s.n[j_max,l])-(s.rr[j_max,l]-s.dr/2)*(s.n[j_max,l]-s.n[j_max,l-1]))

        #update boundaries at l=0 and l=L
        for j in range(2,j_max):
            #update at l=0 (axisymmetric)
            n_new[j,0] = s.n_old[j,0] + 8*s.diffusion*s.dt/(s.dr**2)*(s.n[j,1]-s.n[j,0]) \
                                        + s.mobility*s.dt/s.dz*(s.n[j+1,0]*s.E_fld[0][j+1,0] - s.n[j-1,0]*s.E_fld[0][j-1,0]) \
                                        + s.diffusion*2*s.dt/(s.dz**2)*(s.n[j+1,0] - 2*s.n[j,0] + s.n[j-1,0])
            #and at l=L (diffuse)
            n_new[j,l_max] = s.n[j,l_max-1] \
                                        + s.mobility*s.dt/s.dz*(s.n[j+1,l_max]*s.E_fld[0][j+1,l_max] - s.n[j-1,0]*s.E_fld[0][j-1,l_max]) \
                                        + s.diffusion*2*s.dt/(s.dz**2)*(s.n[j+1,l_max] - 2*s.n[j,l_max] + s.n[j-1,l_max])

        #Now do the corners
        n_new[j_max,0] = s.n[j_max-1,0] + 8*s.diffusion*s.dt/(s.dr**2)*(s.n[j_max,1]-s.n[j_max,0]) #j=J,l=0
        n_new[j_max,l_max] = s.n[j_max-1,l_max-1]

        s.n_old = s.n.copy()
        s.n = n_new.copy()

    def continuity(s):
        '''
        Update electron density by solving the continuity equation using the leapfrog method and relevant boundary conitions.
        See derivation notes.
        '''
        j_max = s.rr.shape[0] - 1
        l_max = s.rr.shape[0] - 1
        n_new = np.copy(s.n)

        #Upate interior points
        for j in range(2,j_max-1):
            for l in range(1,l_max):
                n_new[j,l] = s.n_old[j,l] - s.dt/s.dz*(s.n[j+1,l]*s.u_z[j+1,l]-s.n[j-1,l]*s.u_z[j-1,l]) \
                                - 4*s.dt/(s.rr[j,l+1]**2-s.rr[j,l-1]**2)*(s.rr[j,l+1]*s.n[j,l+1]*s.u_r[j,l+1]-s.rr[j,l-1]*s.n[j,l-1]*s.u_r[j,l-1])

        #Update boundaries at j=J-1
        for l in range(1,l_max):
            #boundary condition at j=J-1 (diffuse)
            n_new[j_max-1,l] = s.n[j_max-2,l] \
                            - 4*s.dt/(s.rr[j_max-1,l+1]**2-s.rr[j_max-1,l-1]**2)*(s.rr[j_max-1,l+1]*s.n[j_max-1,l+1]*s.u_r[j_max-1,l+1]-s.rr[j_max-1,l-1]*s.n[j_max-1,l-1]*s.u_r[j_max-1,l-1])

            #boundary condition at j=0 (von Neumann)
            n_new[0,l] = s.n_old[0,l] + s.dt * s.n_bottom(s.rr[0,l], s.time) * s.u_z_bottom(s.rr[0,l], s.time)

        #Update boundaries at l=0 and l=L
        for j in range(2,j_max-1):
            #boundary at l=0
             n_new[j,0] = s.n_old[j,0] - s.dt/s.dz*(s.n[j+1,0]*s.u_z[j+1,0]-s.n[j-1,0]*s.u_z[j-1,0]) \
                            - 4*s.dt/s.rr[j,1]*s.n[j,1]*s.u_r[j,1]



             #boundary at l_max (diffuse)
             n_new[j,l_max] = s.n[j,l_max-1] - s.dt/s.dz*(s.n[j+1,l_max]*s.u_z[j+1,l_max]-s.n[j-1,l_max]*s.u_z[j-1,l_max])


        #boundaries at corners, which we will treat as a combination of two boundary conditions
        n_new[j_max-1,0] = s.n[j_max-2,0] - 4*s.dt/s.rr[j_max-1,1]*s.n[j_max-1,1]*s.u_r[j_max-1,1]
        n_new[j_max-1,l_max] = s.n[j_max-2,l_max-1]

        s.n_old = s.n.copy()
        s.n = n_new.copy()

    def fluid(s):
        '''
        Resolve electron velocities using fluid equation, for use with the continuity equation.
        '''
        j_max = s.rr.shape[0] - 1
        l_max = s.rr.shape[0] - 1
        u_r_new = np.copy(s.u_r)
        u_z_new = np.copy(s.u_z)

        for j in range(1,j_max):
            for l in range(1,l_max):
                #update interior points
                u_r_new[j,l] = s.u_r_old[j,l] - 2*s.e_c/s.m_e*s.E_fld[1][j,l]*s.dt - s.u_r[j,l]*(s.u_r[j,l+1]-s.u_r[j,l-1])/s.dr*s.dt \
                                    - s.u_z[j,l]*(s.u_r[j+1,l]-s.u_r[j-1,l])/s.dz*s.dt - 2*s.nu*s.u_r_old[j,l]*s.dt

                u_z_new[j,l] = s.u_z_old[j,l] - 2*s.e_c/s.m_e*s.E_fld[0][j,l]*s.dt - s.u_z[j,l]*(s.u_z[j+1,l]-s.u_z[j-1,l])/s.dz*s.dt \
                                    - s.u_r[j,l]*(s.u_z[j,l+1]-s.u_z[j,l-1])/s.dr*s.dt - 2*s.nu*s.u_z_old[j,l]*s.dt

        for l in range(1,l_max):
            #update boundaries in z
            #First, update boundary at z=0
            #Where u is set at the boundary
            #diffuse boundary at z=1
            u_r_new[j_max,l] = s.u_r[j_max-1,l]-2*s.e_c/s.m_e*s.E_fld[0][j_max,l]*s.dt-s.u_r[j_max,l]*(s.u_r[j_max,l+1]-s.u_r[j_max,l-1])/s.dr*s.dt-2*s.nu*s.u_r_old[j_max,l]*s.dt
            u_z_new[j_max,l] = s.u_z[j_max-1,l] - s.u_z[j_max,l] * (s.u_z[j_max,l+1] - s.u_z[j_max,l-1])/s.dr*s.dt - 2*s.nu*s.u_r_old[j_max,l]*s.dt

        for j in range(1,j_max):
            #update boundaries in r
            #first, at r=0 (axisymmetric)
            u_r_new[j,0] = s.u_r_old[j,0] - s.u_z[j,0]*(s.u_r[j+1,0]-s.u_r[j-1,0])/s.dz*s.dt - 2*s.nu*s.u_r_old[j,0]*s.dt
            u_z_new[j,0] = s.u_z_old[j,0] - 2*s.e_c/s.m_e*s.E_fld[0][j,0]*s.dt - s.u_z[j,0]*(s.u_z[j+1,0]-s.u_z[j-1,0])/s.dz*s.dt - 2*s.nu*s.u_z_old[j,0]*s.dt

            #and the boundary at l=l_max
            u_r_new[j,l_max] = s.u_r[j,l_max-1]-s.u_z[j,l_max]*(s.u_r[j+1,l_max]-s.u_r[j-1,l_max])/s.dz*s.dt-2*s.nu*s.u_r_old[j,l]*s.dt
            u_z_new[j,l_max] = s.u_z[j,l_max-1]-2*s.e_c/s.m_e*s.E_fld[0][j,l_max]*s.dt-s.u_z[j,l_max]*(s.u_z[j+1,l_max]-s.u_z[j-1,l_max])/s.dz*s.dt-2*s.nu*s.u_z_old[j,l_max]*s.dt

        #finally, deal with the corners
        #where BC are mixed
        #at j=j_max and l=l_max we have:
        u_r_new[j_max,l_max] = s.u_r[j_max-1,l_max-1] - 2*s.nu*s.u_r_old[j_max,l_max]*s.dt
        u_z_new[j_max,l_max] = s.u_z[j_max-1,l_max-1] - 2*s.nu*s.u_r_old[j_max,l_max]*s.dt

        #at j=j_max and l=0
        u_r_new[j_max,0] = s.u_r[j_max-1,0] - 2*s.nu*s.u_r_old[j,0]*s.dt
        u_z_new[j_max,0] = s.u_z[j_max-1,0] - 2*s.nu*s.u_z_old[j,0]*s.dt

        s.u_r_old = s.u_r.copy()
        s.u_z_old = s.u_z.copy()
        s.u_r = u_r_new.copy()
        s.u_z = u_z_new.copy()

    def sor(s, sp_r, iterations=1000, EPS=1e-10, verbose=True):
        '''
        Resolve electric potential using successive overrelaxation (see e.g. Numerical Recipes Chapter 20)

        sp_r: spectral radius (ideal amount by which to over/under relax by... 0.97 is usually pretty good)
        iterations: number of iterations to try before giving up
        EPS: target ratio of final error to initial error.
        '''
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
                        if verbose:
                            print(f'Error reduced by factor of {EPS} in {n} iterations and anorm = {anorm}')
                        return True
                else:
                    anorm_i = anorm
        if verbose:
            print(f'Error reduced by factor of {anorm} in {iter} iterations')

        return False

    def J(s,j=1):
        '''
        Calculate the current through a circular area at index j.
        '''
        J = 0

        for l in range(0,s.rr.shape[1]-1):
            J += 2*s.e_c / (s.r_max ** 2) * (s.n[j,l]*s.u_z[j,l])*s.rr[j,l]*s.dr

        return J

    def N(s):
        '''
        Calculate the total number of electrons in the computational domain (easy check to see if different grids are giving the same physics)
        '''
        return np.sum(s.n * s.dV)

    def resolve_E_r(s):
        '''
        Resolve the r component of the electric field by finite differencing the potential in r
        '''
        for j in range(1,s.rr.shape[0]-1):
            for l in range(1,s.rr.shape[1]-1):
                s.E_fld[1][j,l] = -(s.V[j,l+1] - s.V[j,l-1]) / (2 * s.dr)

        s.E_fld[1][:,0] = 0 #By symmetry, the electric field in r is 0 at r=0

    def resolve_E_z(s):
        '''
        Resolve the electric field in the z direction by finite differencing the potential in z
        '''
        for j in range(1,s.rr.shape[0]-1):
            for l in range(0,s.rr.shape[1]-1):
                s.E_fld[0][j,l] = -(s.V[j+1,l] - s.V[j-1,l]) / (2*s.dz)

    def resolve_E_fld(s):
        '''
        Resolve the electric field in both r and z
        '''
        s.resolve_E_r()
        s.resolve_E_z()

    def E_mag(s):
        return np.sqrt(s.E_fld[0]**2 + s.E_fld[1]**2)

    def drift_diffusion_cfl(s):
        '''
        Apply the cfl condition to get a stable time step for the drift diffusion prescription
        '''
        return np.min(s.dz/(s.mobility * np.abs(s.E_fld[0])))

    def fluid_cfl(s):
        '''
        Apply the cfl condition to get a stable time step for the fluid model.
        '''
        return np.min([
                        np.min(np.float64(1)/(np.sqrt((s.u_z/s.dz)**2 + (s.u_r/s.dr)**2))),
                        np.min(np.float64(s.dz)/s.u_z_bottom(s.rr,s.time))
                        ]) / 4 #cast numerators as np.float64 to avoid division by zero error

    def initialize(s):
        s.sor(0.97,iterations=10000,EPS=1e-12)

        s.dt = 1e-15

        s.resolve_E_fld()

        for i in range(s.rr[0,:].size):
            s.n[1,i] = s.n_bottom(s.rr[1,i], s.time)
            s.u_z[0,i] = s.u_z_bottom(s.rr[0,i], s.time)
            s.u_r[0,i] = 0

        s.fluid()
        s.fluid()

    def save(s,verbose=True):
        dir = os.path.join(s.save_dir, f'time_{s.time}_s')
        if verbose:
            print(f'[**] Saving in {dir}')

        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

        np.save(os.path.join(dir,'V.npy'),s.V)
        np.save(os.path.join(dir,'E_z.npy'),s.E_fld[0])
        np.save(os.path.join(dir,'E_r.npy'),s.E_fld[1])
        np.save(os.path.join(dir,'u_z.npy'),s.u_z)
        np.save(os.path.join(dir,'u_r.npy'),s.u_r)
        np.save(os.path.join(dir,'n.npy'),s.n)
        np.save(os.path.join(dir,'rr.npy'),s.rr)
        np.save(os.path.join(dir,'zz.npy'),s.zz)

    def V_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.V, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'V')

        plt.show()

    def V_contour_plot(s):
        fig, ax = plt.subplots()
        ax.contour(s.rr,s.zz,s.V,levels=50,cmap=cm.bone)

        plt.show()

    def n_contour_plot(s):
        fig,ax = plt.subplots()
        ax.contourf(s.rr,s.zz,s.n)
        fig.show()

    def n_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.n, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'n')

        plt.show()

    def n_old_surface_plot(s):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(s.rr, s.zz, s.n_old, cmap=cm.bone,linewidth=0, antialiased=False)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_zlabel(r'n')

        plt.show()

    def E_fld_plot(s):
        plt.quiver(s.zz,s.rr,*s.E_fld)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title('Electric Field')
        plt.show()

    def u_plot(s):
        plt.quiver(s.zz,s.rr,s.u_z,s.u_r)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.show()
