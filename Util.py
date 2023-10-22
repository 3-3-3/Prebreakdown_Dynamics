#AUTHOR: Liam Keeley, Colorado College
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import matplotlib.animation as animation
from scipy.special import j0, jn_zeros
from Prebreakdown import Prebreakdown
import os


def bessel_boundary(V_0,r_max):
    '''
    Return a bessel function which can be used as the boundary condition for V
    Useful because it goes to zero at r=r_max and results can be compared with
    analytical solution of Laplace's equation in cylindrical coordinates
    '''
    alpha = jn_zeros(0,1)[0]
    return lambda r, t : V_0 * j0(alpha * r / r_max) if r <= r_max else 0

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
    cf = ax.contourf(rr,zz,n_plot,levels=25,cmap=cm.magma,vmin=vmin,vmax=vmax)
    cf = ax.contour(rr,zz,n_plot,levels=25,colors=['white'],linewidths=[0.2])
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

    rr = np.load(os.path.join(target,'rr.npy'))
    zz = np.load(os.path.join(target,'zz.npy'))

    n = np.load(os.path.join(target,'n.npy'))
    u_r = np.load(os.path.join(target,'u_r.npy'))
    u_z = np.load(os.path.join(target,'u_z.npy'))

    cont = ax.contourf(rr, zz, n)
    cbar = fig.colorbar(cont)

    u_r_plot = np.ma.masked_where(n == 0, u_r)
    u_z_plot = np.ma.masked_where(n == 0, u_z)

    quiv = ax.quiver(rr,zz,u_r_plot,u_z_plot)

    ax.set_xlabel(r'$r$ (m)')
    ax.set_ylabel(r'$z$ (m)')
    fig.suptitle(r'Evolution of $\vec{u_e}$ and $n_e$')

    def ani_func(time, base, ax, fig, rr, zz):
        nonlocal quiv, cont, cbar
        for c in cont.collections:
            c.remove()
        quiv.remove()
        cbar.remove()

        #load data at time step
        target = os.path.join(base,dir_name_from_time(time))
        n = np.load(os.path.join(target,'n.npy'))
        u_r = np.load(os.path.join(target,'u_r.npy'))
        u_z = np.load(os.path.join(target,'u_z.npy'))


        cont = ax.contourf(rr, zz, n, levels=25)
        cbar = fig.colorbar(cont)
        cbar.set_label(r'$n_e$', rotation=0)

        u_r_plot = np.ma.masked_where(n == 0, u_r)
        u_z_plot = np.ma.masked_where(n == 0, u_z)

        quiv = ax.quiver(rr,zz,u_r_plot,u_z_plot)
        ax.set_title(f'Time: {time}')

    ani = animation.FuncAnimation(fig, ani_func, frames=times, fargs=(base, ax, fig, rr, zz), blit=False, interval=interval,repeat=False)
    if save:
        writergif = animation.PillowWriter(fps=30)
        ani.save(os.path.join(base,'animation.gif'),writer=writergif)
    plt.show()

def n_E_fld_plot(target,axes,fig):
    rr = np.load(os.path.join(target,'rr.npy'))[1:-2,0:25]
    zz = np.load(os.path.join(target,'zz.npy'))[1:-2,0:25]
    n = np.load(os.path.join(target,'n.npy'))[1:-2,0:25]
    u_r = np.load(os.path.join(target,'u_r.npy'))[1:-2,0:25]
    u_z = np.load(os.path.join(target,'u_z.npy'))[1:-2,0:25]
    E_r = np.load(os.path.join(target,'E_r.npy'))[1:-2,0:25]
    E_z = np.load(os.path.join(target,'E_z.npy'))[1:-2,0:25]

    n_contour_plot(fig, axes[0], rr, zz, n)
    axes[1].quiver(rr,zz,E_r,E_z)

def time_evol_plot(targets,base):
    fig = plt.figure(figsize=(12,12))
    outer_grid = fig.add_gridspec(4,2)

    for i in range(len(targets)):
        inner_grid = outer_grid[int(np.floor(i / 2)),i % 2].subgridspec(1,2,wspace=0,hspace=0)
        axes = inner_grid.subplots(sharey=True)

        t = os.path.join(base,targets[i])
        n_E_fld_plot(t,axes,fig)
