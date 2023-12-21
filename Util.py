#AUTHOR: Liam Keeley, Colorado College
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import matplotlib.animation as animation
import matplotlib.ticker as tkr
from scipy.special import j0, jn_zeros
from Prebreakdown import Prebreakdown
import matplotlib as mpl
import os
import string
import math


def bessel_boundary(V_0,r_max):
    '''
    Return a bessel function which can be used as the boundary condition for V
    Useful because it goes to zero at r=r_max and results can be compared with
    analytical solution of Laplace's equation in cylindrical coordinates
    '''
    alpha = jn_zeros(0,1)[0]
    return lambda r, t : V_0 * j0(alpha * r / r_max) if r <= r_max else 0

def analytic(rr, zz):
    alpha = jn_zeros(0,1)[0]
    return 1/np.sinh(alpha) * j0(alpha * rr) * np.sinh(alpha * zz)

def step_boundary(amp, r_max):
    '''
    A boundary which suddenly goes to zero at r_max. Can be used for either the potential or the electron number density.
    '''
    return lambda r, t : amp if r <= r_max else 0

def gaussian_boundary(power, std):
    return lambda r, t : power / (np.sqrt(2*np.pi)*std) * np.exp(-1/2*(r/std)**2)


def get_time_from_dir_name(dir_name):
    return float(dir_name.strip('_./' + string.ascii_letters))

def dir_name_from_time(time):
    return f'time_{time}_s'

def n_contour_plot(fig, ax, rr, zz, n,vmin=None,vmax=None):
    n_plot = np.ma.masked_where(n <= 0, n)
    cf = ax.contourf(rr,zz,n_plot,levels=35,cmap=cm.viridis,vmin=vmin,vmax=vmax)
    #cf = ax.contour(rr,zz,n_plot,levels=35,colors=['grey'],linewidths=[0.5])
    return fig, ax, cf

def grid(fig, ax, rr, zz):
    ax.scatter(rr,zz,marker='.',s=1,color='lightgrey')

def u_quiver_plot(fig, ax, rr, zz, u_r, u_z, n=None):
    if n is None:
        u_r_plot = np.copy(u_r)
        u_z_plot = np.copy(u_z)

    else:
        u_r_plot = np.ma.masked_where(n <= 0, u_r)
        u_z_plot = np.ma.masked_where(n <= 0, u_z)

    ax.quiver(rr,zz,u_r_plot,u_z_plot)
    return fig, ax

def n_u_plot(fig, ax, rr, zz, n, u_r, u_z):
    grid(fig,ax,rr,zz)
    n_contour_plot(fig,ax,rr,zz,n)
    u_quiver_plot(fig,ax,rr,zz,u_r,u_z,n=n)

    return fig, ax

def n_u_animation(times, base, interval=1, save=False):
    fig, ax = plt.subplots()
    target = os.path.join(base,dir_name_from_time(times[0]))

    rr = np.load(os.path.join(target,'rr.npy'))[:,:40]
    zz = np.load(os.path.join(target,'zz.npy'))[:,:40]

    n = np.load(os.path.join(target,'n.npy'))[:,:40]
    u_r = np.load(os.path.join(target,'u_r.npy'))[:,:40]
    u_z = np.load(os.path.join(target,'u_z.npy'))[:,:40]

    cont = ax.contourf(rr, zz, n)
    cf = ax.contour(rr,zz,n,levels=30,colors=['white'],linewidths=[0.2])

    u_r_plot = np.ma.masked_where(n == 0, u_r)
    u_z_plot = np.ma.masked_where(n == 0, u_z)

    #quiv = ax.quiver(rr,zz,u_r_plot,u_z_plot)

    ax.set_xlabel(r'$r$ (m)')
    ax.set_ylabel(r'$z$ (m)')
    fig.suptitle(r'Evolution of $n_e$')



    nmax = 0
    for time in times:
        target = os.path.join(base,dir_name_from_time(time))
        n = np.load(os.path.join(target,'n.npy'))[:,:40]

        if np.max(n) > nmax:
            nmax = np.max(n)

    print(nmax)
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0,vmax=nmax)


    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label(r'$n_e$', rotation=0)





    def ani_func(time, base, ax, fig, rr, zz, v_max=None):
        nonlocal cont, cbar, cf



        for c in cont.collections:
            c.remove()

        for c in cf.collections:
            c.remove()
        #quiv.remove()

        #load data at time step
        target = os.path.join(base,dir_name_from_time(time))
        n = np.load(os.path.join(target,'n.npy'))[:,:40]
        u_r = np.load(os.path.join(target,'u_r.npy'))[:,:40]
        u_z = np.load(os.path.join(target,'u_z.npy'))[:,:40]


        cont = ax.contourf(rr, zz, n, levels=30, norm=norm)
        cf = ax.contour(rr,zz,n,levels=30,colors=['white'],linewidths=[0.2])


        u_r_plot = np.ma.masked_where(n == 0, u_r)
        u_z_plot = np.ma.masked_where(n == 0, u_z)

        #quiv = ax.quiver(rr,zz,u_r_plot,u_z_plot)
        ax.set_title(f'Time: {time}')

    ani = animation.FuncAnimation(fig, ani_func, frames=times, fargs=(base, ax, fig, rr, zz), blit=False, interval=interval,repeat=False)

    if save:
        writergif = animation.PillowWriter(fps=30)
        ani.save(os.path.join(base,'animation.gif'),writer=writergif)
    plt.show()

def n_E_fld_plot(target,axes,fig,a=0,b=-1,c=0,d=-1,i_r=1,i_z=1):
    rr = np.load(os.path.join(target,'rr.npy'))[a:b,c:d]
    zz = np.load(os.path.join(target,'zz.npy'))[a:b,c:d]
    n = np.load(os.path.join(target,'n.npy'))[a:b,c:d]
    u_r = np.load(os.path.join(target,'u_r.npy'))[a:b,c:d]
    u_z = np.load(os.path.join(target,'u_z.npy'))[a:b,c:d]
    E_r = np.load(os.path.join(target,'E_r.npy'))[a:b,c:d]
    E_z = np.load(os.path.join(target,'E_z.npy'))[a:b,c:d]

    nmin = 0
    nmax = np.max(n)

    norm = mpl.colors.Normalize(vmin=nmin,vmax=nmax)
    cmap = mpl.cm.magma
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes[0])


    n_contour_plot(fig, axes[0], rr, zz, n,vmin=nmin,vmax=nmax)

    E_mag = np.sqrt(E_r ** 2 + E_z ** 2)
    emin = np.min(E_mag)
    emax = np.max(E_mag)

    norm = mpl.colors.Normalize(vmin=emin,vmax=emax)
    cmap = mpl.cm.viridis
    axes[1].quiver(rr[0:-1:i_z,0:-1:i_r],
                   zz[0:-1:i_z,0:-1:i_r],
                   E_r[0:-1:i_z,0:-1:i_r],
                   E_z[0:-1:i_z,0:-1:i_r],
                   E_mag[0:-1:i_z,0:-1:i_r],
                   norm=norm, cmap=cmap,lw=5)

def n_E_fld_pertubation_plot(target,axes,fig,E_r_0,E_z_0,a=0,b=-1,c=0,d=-1,i_r=1,i_z=1):
    rr = np.load(os.path.join(target,'rr.npy'))[a:b,c:d] * 1e3
    zz = np.load(os.path.join(target,'zz.npy'))[a:b,c:d] * 1e3
    n = np.load(os.path.join(target,'n.npy'))[a:b,c:d]
    u_r = np.load(os.path.join(target,'u_r.npy'))[a:b,c:d]
    u_z = np.load(os.path.join(target,'u_z.npy'))[a:b,c:d]
    E_r = np.load(os.path.join(target,'E_r.npy'))[a:b,c:d]
    E_z = np.load(os.path.join(target,'E_z.npy'))[a:b,c:d]

    nmin = 0
    nmax = np.max(n)

    norm = mpl.colors.Normalize(vmin=nmin,vmax=nmax)
    cmap = mpl.cm.viridis
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes[0])



    n_contour_plot(fig, axes[0], rr, zz, n,vmin=nmin,vmax=nmax)

    E_r_pertubation = E_r - E_r_0[a:b,c:d]
    E_z_pertubation = E_z - E_z_0[a:b,c:d]

    E_r_pertubation = E_r_pertubation[45::i_z,::i_r]
    E_z_pertubation = E_z_pertubation[45::i_z,::i_r]

    E_mag = np.sqrt(E_r_pertubation ** 2 + E_z_pertubation ** 2)
    emin = np.min(E_mag)
    emax = np.max(E_mag)

    norm = mpl.colors.Normalize(vmin=emin,vmax=emax)
    cmap = mpl.cm.viridis

    axes[0].tick_params(axis='both', which='major')
    axes[1].tick_params(axis='both', which='major')
    axes[1].quiver(rr[45::i_z,::i_r],
                   zz[45::i_z,::i_r],
                   E_r_pertubation,
                   E_z_pertubation,
                   E_mag,
                   norm=norm, cmap=cmap)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes[1])

def time_evol_plot(targets,base,E_r_0,E_z_0,a=0,b=-1,c=0,d=-1,i_r=1,i_z=1,save=False):
    fig = plt.figure(figsize=(11,12))
    plt.style.use('seaborn-v0_8-paper')
    outer_grid = fig.add_gridspec(4,2,wspace=0.2,hspace=0.4)


    files = os.listdir(base)
    try:
        files.remove('.DS_Store')
    except:
        pass

    try:
        files.remove('animation.gif')
    except:
        pass

    times = [get_time_from_dir_name(f) for f in files]
    times = np.sort(np.array(times))
    #times.sort()

    for i in range(len(targets)):
        m = int(np.floor(i / 2))
        n = i % 2

        inner_grid = outer_grid[m,n].subgridspec(1,2,wspace=0.15)
        axes = inner_grid.subplots(sharey=True)
        #axes[0].ticklabel_format(style='scientific',scilimits=(-4,-4))
        #axes[1].ticklabel_format(style='scientific',scilimits=(-4,-4))

        axes[0].set_ylabel(r'$z$ ($mm$)')
        axes[0].set_xlabel(r'$r$ ($mm$)')
        axes[1].set_xlabel(r'$r$ ($mm$)')

        txt_str = f'{np.round(1e12*targets[i],1)} ps'
        props = {'facecolor':'white'}
        axes[0].text(0.50,0.90,txt_str,transform=axes[0].transAxes,bbox=props,fontsize=8)
        axes[1].text(0.53,0.90,txt_str,transform=axes[1].transAxes,bbox=props,fontsize=8)
        axes[0].set_title(r'$n_e$')
        axes[1].set_title(r'$\vec{E} - \vec{E}_0$')

        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')



        time = find_nearest(times,targets[i])
        print(f'Target: {targets[i]}, time: {time}')
        t = os.path.join(base,dir_name_from_time(time))
        n_E_fld_pertubation_plot(t,axes,fig,E_r_0,E_z_0,a=a,b=b,c=c,d=d,i_r=i_r,i_z=i_z)

        if save:
            fig.savefig('Main_figure.pdf')

def residual_one_by_one(V,V_an,rr,zz,rr_an,zz_an):
    f_per_c = rr_an.shape[0] / rr.shape[0]
    f_per_c = int(f_per_c)

    res = 0
    V_tot = 0


    for j_c in range(1, rr.shape[0]-1):
        for l_c in range(1,zz.shape[0]-1):

            for m in range(f_per_c):
                j_f = j_c*f_per_c + m
                for n in range(f_per_c):
                    l_f = l_c*f_per_c + n
                    print(np.abs(V_an[j_f,l_f] - V[j_c, l_c]) / V_an[j_f,l_f])

                    res += np.abs(V_an[j_f,l_f] - V[j_c, l_c]) / V_an[j_f,l_f]

    return res / ((rr_an.shape[0]) * (rr_an.shape[1])) #One less than total number of points

def integrate(V,rr,zz):
    dz = zz[1,0] - zz[0,0]
    dr = rr[0,1] - rr[0,0]

    V_tot = 0

    for j in range(0,rr.shape[0]-1):
        for l in range(0,zz.shape[1]-1):
            tau = dz * np.pi * ((rr[j,l] + dr / 2) ** 2 - (rr[j,l] - dr / 2) ** 2)
            V_tot += V[j,l] * tau

    return V_tot

def total_error(V,V_f,rr,zz,rr_f,zz_f):
    V_tot = integrate(V,rr,zz)
    V_tot_f = integrate(V_f,rr_f,zz_f)

    return np.abs(V_tot - V_tot_f) / V_tot_f


def residual(V,V_f,rr,zz,rr_f,zz_f):
    f_per_c = int(rr_f.shape[0] / rr.shape[0])
    print(f_per_c)

    #only works on uniform grids...
    dz_f = zz_f[1,0] - zz_f[0,0]
    dr_f = rr_f[0,1] - rr_f[0,0]

    res = 0
    V_tot = 0


    for j_c in range(1, rr.shape[0]-1): #iterate over the course grid
        for l_c in range(1,zz.shape[0]-1):

            for m in range(f_per_c): #and then the fine grid
                j_f = j_c*f_per_c + m
                for n in range(f_per_c):
                    l_f = l_c*f_per_c + n
                    tau = dz_f * np.pi * ((rr_f[j_f,l_f] + dr_f / 2) ** 2 - (rr_f[j_f,l_f] - dr_f / 2) ** 2)

                    res += np.abs(V_f[j_f,l_f] - V[j_c, l_c]) * tau
                    print(f'res: {res}')
                    V_tot += V_f[j_f,l_f] * tau
                    print(f'V_tot: {V_tot}')

    return res / V_tot #One less than total number of points



def find_nearest(array,value):
    '''
    ripped off stackoverflow
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def find_nearest_idx(array,value):
    '''
    ripped off stackoverflow
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def E_max_n_max(base):
    files = os.listdir(base)

    try:
        files.remove('.DS_Store')
    except:
        pass

    try:
        files.remove('animation.gif')
    except:
        pass

    times = [get_time_from_dir_name(f) for f in files]
    times = np.sort(np.array(times))

    E_r_0 = np.load(os.path.join(os.path.join(base,dir_name_from_time(times[0])),'E_r.npy'))
    E_z_0 = np.load(os.path.join(os.path.join(base,dir_name_from_time(times[0])),'E_z.npy'))
    print(f'E_0 = {np.max(np.sqrt(E_r_0 ** 2 + E_z_0 ** 2))}')

    E_max = []
    n_max = []

    min_z = int(E_z_0.shape[1] * 0.80)
    max_r = int(E_r_0.shape[0]*0.25)

    for time in times:
        E_r = np.load(os.path.join(os.path.join(base, dir_name_from_time(time)),'E_r.npy'))
        E_z = np.load(os.path.join(os.path.join(base, dir_name_from_time(time)),'E_z.npy'))
        n = np.load(os.path.join(os.path.join(base, dir_name_from_time(time)),'n.npy'))

        E_mag = np.sqrt((E_r-E_r_0)**2 + (E_z-E_z_0)**2)
        E_max.append(np.max(E_mag[min_z:,:]))
        n_max.append(np.max(n[min_z:,:max_r]))

    return (times, np.array(E_max),np.array(n_max))
