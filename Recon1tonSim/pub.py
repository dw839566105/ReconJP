import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot, argparse
from scipy.optimize import minimize
from scipy import interpolate
from numba import jit
from scipy import special
from scipy.linalg import norm
from scipy.stats import norm as normpdf
from scipy.spatial import distance
import warnings

class ReconData(tables.IsDescription):
    EventID = tables.Int64Col(pos=0)    # EventNo
    # inner recon
    E_sph_in = tables.Float16Col(pos=1)        # Energy
    x_sph_in = tables.Float16Col(pos=2)        # x position
    y_sph_in = tables.Float16Col(pos=3)        # y position
    z_sph_in = tables.Float16Col(pos=4)        # z position
    t0_in = tables.Float16Col(pos=5)          # time offset
    success_in = tables.Int64Col(pos=6)        # recon status   
    Likelihood_in = tables.Float16Col(pos=7)

    # outer recon
    E_sph_out = tables.Float16Col(pos=8)         # Energy
    x_sph_out = tables.Float16Col(pos=9)         # x position
    y_sph_out = tables.Float16Col(pos=10)        # y position
    z_sph_out = tables.Float16Col(pos=11)        # z position
    t0_out = tables.Float16Col(pos=12)          # time offset
    success_out = tables.Int64Col(pos=13)        # recon status 
    Likelihood_out = tables.Float16Col(pos=14)

    # truth info
    x_truth = tables.Float16Col(pos=15)        # x position
    y_truth = tables.Float16Col(pos=16)        # y position
    z_truth = tables.Float16Col(pos=17)        # z position
    E_truth = tables.Float16Col(pos=18)        # z position

@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x

def ReadTpl(filename='../MC/template.h5'):
    # Read MC grid recon result
    h = tables.open_file(filename)
    tpl = h.root.template[:]
    mesh = np.vstack((h.root.x[:], h.root.y[:], h.root.z[:])).T
    h.close()
    return mesh, tpl

def load_coeff(PEFile = '../calib/PE_coeff_1t_29_80.h5', TimeFile = '../calib/Time_coeff2_1t_0.1.h5'):
    # spherical harmonics coefficients for time and PEmake 
    h = tables.open_file(PEFile, 'r')
    coeff_pe = h.root.coeff_L[:]
    h.close()
    cut_pe, fitcut_pe = coeff_pe.shape

    h = tables.open_file(TimeFile,'r')
    coeff_time = h.root.coeff_L[:]
    h.close()
    cut_time, fitcut_time = coeff_time.shape
    return coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time

def r2c(c):
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    v = np.zeros(3)
    v[0] = norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    #v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    v[2] = np.arctan2(c[1],c[0])
    return v

class Likelihood_Truth:
    # print('Using method: do not fit energy, energy is estimated by normalized')
    def Likelihood(vertex, *args):
        '''
        vertex[1]: r
        vertex[2]: theta
        vertex[3]: phi
        '''
        coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe= args
        z, x = Likelihood_Truth.Calc_basis(vertex, PMT_pos, np.max((cut_time, cut_pe)))
        L1, E = Likelihood_Truth.Likelihood_PE(coeff_pe, z, x, pe_array, cut_pe)
        L2 = Likelihood_Truth.Likelihood_Time(coeff_time, z, x, vertex[3], fired_PMT, time_array, cut_time)
        return L1 + L2

    def Calc_basis(vertex, PMT_pos, cut): 
        # boundary
        v = r2c(vertex[:3])
        z = v[0]
        if z > 1-1e-3:
            z = 1-1e-3
        # calculate cos theta
        cos_theta = np.dot(v, PMT_pos.T) / (norm(v)*norm(PMT_pos,axis=1))
        ### Notice: Here may not continuous! ###
        cos_theta[np.isnan(cos_theta)] = 1 # for v in detector center    

        # Generate Legendre basis
        # x = legval(cos_theta, np.diag((np.ones(cut)))).T 
        x = legval(cos_theta, np.eye(cut).reshape((cut,cut,1))).T
        return z, x

    def Likelihood_PE(coeff_pe, z, x, pe_array, cut):
        # Recover coefficient
        k = legval(z, coeff_pe.T)
        # Recover expect
        expect = np.exp(np.dot(x,k))
        # Energy fit 
        nml = np.sum(expect)/np.sum(pe_array)
        expect = expect/nml
        k[0] = k[0] - np.log(nml) # 0-th

        # Poisson likelihood
        # p(q|lambda) = sum_n p(q|n)p(n|lambda)
        #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
        # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1
        a0 = expect ** pe_array
        a1 = np.exp(-expect)

        # -ln Likelihood
        L = - np.sum(np.log(a0) + np.log(a1))
        # avoid inf (very impossible vertex) 
        if(np.isinf(L) or L>1e20):
            L = 1e20
        return L, k[0]

    def Likelihood_Time(coeff_time, z, x, T0, fired_PMT, time_array, cut):
        x = x[fired_PMT][:,:cut]

        # Recover coefficient
        k = np.atleast_2d(legval(z, coeff_time.T)).T
        k[0,0] = T0

        # Recover expect
        T_i = np.dot(x, k)

        # Likelihood
        L = np.nansum(Likelihood_Truth.Likelihood_quantile(time_array, T_i[:,0], 0.1, 2.6))
        return L

    def Likelihood_quantile(y, T_i, tau, ts):
        # less = T_i[y<T_i] - y[y<T_i]
        # more = y[y>=T_i] - T_i[y>=T_i]    
        # R = (1-tau)*np.sum(less) + tau*np.sum(more)

        # since lucy ddm is not sparse, use PE as weight
        L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
        #nml = tau*(1-tau)/ts
        #L_norm = np.exp(-np.atleast_2d(L).T) * nml / ts
        #L = np.sum(np.log(L_norm), axis=1)
        L0 = L/ts
        return L0

class Likelihood_Raw:
    # print('Using method: do not fit energy, energy is estimated by normalized')
    def Likelihood(vertex, *args):
        '''
        vertex[1]: r
        vertex[2]: theta
        vertex[3]: phi
        '''
        coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_tpl, PE = args
        z, x = Calc_basis(vertex, PMT_pos, np.max((cut_time, cut_pe)))
        L1, E = Likelihood_PE(z, x, coeff_pe, pe_array, cut_pe, N, pdf_tpl)
        L2 = Likelihood_Time(z, x, vertex[4], coeff_time, fired_PMT, time_array, cut_time, PE)
        return L1 + L2

    def Calc_basis(vertex, PMT_pos, cut):
        # boundary
        v = r2c(vertex[1:4])
        z = norm(v)
        if z > 1-1e-3:
            z = 1-1e-3
        # calculate cos theta
        cos_theta = np.dot(v, PMT_pos.T) / (norm(v)*norm(PMT_pos,axis=1))
        ### Notice: Here may not continuous! ###
        cos_theta[np.isnan(cos_theta)] = 1 # for v in detector center    

        # Generate Legendre basis
        x = LG.legval(cos_theta, np.diag((np.ones(cut)))).T   
        return z, x
    
    def Likelihood_PE(z, x, coeff, pe_array, cut, N, pdf_tpl):
        # Recover coefficient
        k = LG.legval(z, coeff_pe.T)
        # Recover expect
        expect = np.exp(np.dot(x,k))
        # Energy fit 
        nml = np.sum(expect)/np.sum(pe_array)
        expect = expect/nml
        k[0] = k[0] - np.log(nml) # 0-th

        # Poisson likelihood
        # p(q|lambda) = sum_n p(q|n)p(n|lambda)
        #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
        # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1
        a0 = np.atleast_2d(expect).T ** N / (scipy.special.factorial(N))
        a1 = np.nansum(a0 * pdf_tpl, axis=1)
        a2 = np.exp(-expect)

        # -ln Likelihood
        L = - np.sum(np.sum(np.log(a1*a2)))
        # avoid inf (very impossible vertex) 
        if(np.isinf(L) or L>1e20):
            L = 1e20
        return L, k[0]
    
    def Likelihood_Time(z, x, T0, coeff, fired_PMT, time_array, cut, PE):
        x = x[fired_PMT][:,:cut]

        # Recover coefficient
        k = np.atleast_2d(LG.legval(z, coeff_time.T)).T
        k[0,0] = T0

        # Recover expect
        T_i = np.dot(x, k)

        # Likelihood
        L = - np.nansum(Likelihood_quantile(time_array, T_i[:,0], 0.1, 2.6, PE))
        return L

    def Likelihood_quantile(y, T_i, tau, ts, PE):
        # less = T_i[y<T_i] - y[y<T_i]
        # more = y[y>=T_i] - T_i[y>=T_i]    
        # R = (1-tau)*np.sum(less) + tau*np.sum(more)

        # since lucy ddm is not sparse, use PE as weight
        L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
        nml = tau*(1-tau)/ts**PE
        L_norm = np.exp(-np.atleast_2d(L).T * PE) * nml / ts
        L = np.log(np.sum(L_norm, axis=1))
        return L_norm

def construct(PMT_pos, coeff_PE, cut): 
    N = 30
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)
    
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False)
    mesh = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    tpl = np.zeros((len(mesh), 30))
    c = np.eye(cut).reshape((cut,cut,1))
    print(len(mesh))
    for i in np.arange(len(mesh)):
        if not i % 10000:
            print('processing:', i)
        vertex = mesh[i]
        k = legval(np.linalg.norm(vertex), coeff_PE.T)
        cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
        xx = legval(cos_theta, c).T
        tpl[i] = np.exp(np.dot(xx,k))
    index = np.linalg.norm(mesh, axis=1)<0.99
    return mesh[index], tpl[index]

def construct_outer(PMT_pos, coeff_PE, cut): 
    r = np.linspace(0.58, 0.62, 5)
    theta = np.arccos(np.linspace(-1, 1, 50))
    phi = np.linspace(0, 2*np.pi, 50)
    
    xx, yy, zz = np.meshgrid(r, theta, phi, sparse=False)
    mesh = np.vstack((xx.flatten()*np.sin(yy.flatten())*np.cos(zz.flatten()), \
                      xx.flatten()*np.sin(yy.flatten())*np.sin(zz.flatten()), \
                      xx.flatten()*np.cos(yy.flatten()))).T
    tpl = np.zeros((len(mesh), 30))
    c = np.eye(cut).reshape((cut,cut,1))
    print(len(mesh))
    for i in np.arange(len(mesh)):
        if not i % 10000:
            print('processing:', i)
        vertex = mesh[i]
        k = legval(np.linalg.norm(vertex), coeff_PE.T)
        cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
        xx = legval(cos_theta, c).T
        tpl[i] = np.exp(np.dot(xx,k))
    return mesh, tpl

class Initial:
    def ChargeWeighted(pe_array, PMT_pos, time_array):
        vertex = np.zeros(5)
        x_ini = 1.3 * np.sum(np.atleast_2d(pe_array).T*PMT_pos, axis=0)/np.sum(pe_array)
        E_ini = np.sum(pe_array)/60
        t_ini = np.quantile(time_array, 0.1)
        vertex[0] = E_ini
        vertex[1:4] = c2r(x_ini)
        vertex[1] /= 0.65
        vertex[-1] = t_ini
        return vertex
    
    def MCGrid(pe_array, mesh, tpl, time_array):
        vertex = np.zeros(5)
        rep = np.tile(pe_array,(len(tpl),1))
        scale = np.sum(tpl, axis=1)/np.sum(pe_array)
        tpl /= np.atleast_2d(scale).T
        L = -np.nansum(-tpl + np.log(tpl)*pe_array, axis=1)
        index = np.where(L == np.min(L))[0][0]
        
        x_ini = mesh[index]
        E_ini = np.sum(pe_array)/60
        t_ini = np.quantile(time_array, 0.1)
        vertex[0] = E_ini
        vertex[1:4] = c2r(x_ini)/1000
        vertex[1] /= 0.65
        vertex[-1] = t_ini
        return vertex
    
    def FitGrid(pe_array, mesh, tpl, time_array):
        vertex = np.zeros(5)
        rep = np.tile(pe_array,(len(tpl),1))
        scale = np.sum(tpl, axis=1)/np.sum(pe_array)
        tpl /= np.atleast_2d(scale).T
        L = -np.nansum(-tpl + np.log(tpl)*pe_array, axis=1)
        index = np.where(L == np.min(L))[0][0]
        
        x_ini = mesh[index]
        E_ini = np.sum(pe_array)/60
        t_ini = np.quantile(time_array, 0.1)
        vertex[0] = E_ini
        vertex[1:4] = c2r(x_ini)
        vertex[-1] = t_ini
        return vertex

    

    
        
