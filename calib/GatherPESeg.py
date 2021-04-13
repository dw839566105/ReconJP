from scipy.optimize import minimize
import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
bnd = 0.572/0.65
def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'PE_30_1t_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.56, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.65, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 30
rd1, coeff_pe1 = main_photon_sparse('coeff_point_10_photon_2MeV/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_point_10_photon_2MeV/',order)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 30
N_out = 80

coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd1, rd2, coeff_pe1_row, coeff_pe2_row, order = args
    y1 = np.polynomial.legendre.legval(rd1/0.65, theta[0:N_in])
    L1 = np.sum((coeff_pe1_row-y1)**2)
    y2 = np.polynomial.legendre.legval(rd2/0.65, theta[-N_out:])
    L2 = np.sum((coeff_pe2_row-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rd1/0.65, theta[0:N_in])
        L1 += np.sum((coeff_pe1_row-y1)**2)
        y2 = np.polynomial.legendre.legval(-rd2/0.65, theta[-N_out:])
        L2 += np.sum((coeff_pe2_row-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rd1/0.65, theta[0:N_in])
        L1 += np.sum((-coeff_pe1_row-y1)**2)
        y2 = np.polynomial.legendre.legval(-rd2/0.65, theta[-N_out:])
        L2 += np.sum((-coeff_pe2_row-y2)**2)
    return L1 + L2

eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(np.max(rd1)/0.65, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]), 
                                      #np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]) - \
                                      #   np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]),# 0-th
                                      np.sum((np.max(rd1)/0.65 * np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/((np.max(rd1)/0.65)**2-1)) - 
                                      np.sum((np.max(rd1)/0.65 * np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/((np.max(rd1)/0.65)**2-1))
                                     ]), # 1-st
            }

for i in np.arange(29):
    print(i)
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd1, rd2, coeff_pe1[i], coeff_pe2[i], i))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]
    plt.figure(dpi=300)
    rd = np.arange(0.01,0.65,0.001)
    plt.plot(rd1/0.65, coeff_pe1[i], 'r.', alpha = 0.5, label='inner data')
    plt.plot(rd2/0.65, coeff_pe2[i], 'r.', alpha = 0.5, label='outer data')
    plt.plot(rd1/0.65, np.polynomial.legendre.legval(rd1/0.65, result.x[0:N_in]), 'b-', label='inner fit')
    plt.plot(rd2/0.65, np.polynomial.legendre.legval(rd2/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit' % i)
    plt.savefig('Fit%d.pdf' % i)
    plt.figure(dpi=300)
    rd = np.arange(0.54,0.56,0.001)
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[0:N_in]), label='inner fit')
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[-N_out:]), label='outer fit')
    plt.axvline(np.max(rd1)/0.65, color='red', label='Discontinuity')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit local' % i)
    plt.savefig('Fit%d_local.pdf' % i)
    plt.close()
    exit()
with h5py.File('./PE_coeff_1t_seg.h5','w') as out:
    out.create_dataset('coeff_in', data = coeff_in)
    out.create_dataset('coeff_out', data = coeff_out)
    out.create_dataset('bd', data = bnd*0.65)
