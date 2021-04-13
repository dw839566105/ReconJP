# recon range: [-1,1], need * detector radius
import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse, textwrap
from argparse import RawTextHelpFormatter
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special
from scipy.linalg import norm
from scipy.stats import norm as normpdf
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)

# boundaries
shell = 0.65
Gain = 164
sigma = 40

import pub
def Recon(filename, output, mode, offset, types, initial, MC, method, verbose):

    '''
    reconstruction

    fid: root reference file convert to .h5
    fout: output file
    '''
    # Create the output file and the group
    print(filename) # filename
    # Create the output file and the group
    h5file = tables.open_file(output, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", pub.ReconData, "Recon")
    recondata = ReconTable.row
    
    # Loop for event
    f = uproot.open(filename)
    data = f['SimTriggerInfo']
    if types == 'Sim_root':
        PMTId = data['PEList.PMTId'].array()
        Time = data['PEList.HitPosInWindow'].array()
        Charge = data['PEList.Charge'].array()   
        SegmentId = ak.to_numpy(ak.flatten(data['truthList.SegmentId'].array()))
        VertexId = ak.to_numpy(ak.flatten(data['truthList.VertexId'].array()))
        x = ak.to_numpy(ak.flatten(data['truthList.x'].array()))
        y = ak.to_numpy(ak.flatten(data['truthList.y'].array()))
        z = ak.to_numpy(ak.flatten(data['truthList.z'].array()))
        E = ak.to_numpy(ak.flatten(data['truthList.EkMerged'].array()))
    
        for pmt, time_array, pe_array, sid, vid, xt, yt, zt, Et in zip(PMTId, Time, Charge, SegmentId, VertexId, x, y, z, E):
            recondata['x_truth'] = xt
            recondata['y_truth'] = yt
            recondata['z_truth'] = zt
            recondata['E_truth'] = Et
            recondata['EventID'] = sid
            fired_PMT = ak.to_numpy(pmt)
            time_array = ak.to_numpy(time_array)
     
            # PMT order: 0-29
            # PE /= Gain
            # pe_array, cid = np.histogram(pmt, bins=np.arange(31)-0.5, weights=PE)

            # For hit info
            pe_array, cid = np.histogram(fired_PMT, bins=np.arange(31)) 
            # For very rough estimate
            # pe_array = np.round(pe_array)

            if np.sum(pe_array)==0:
                continue

            if args.initial == 'WA':
                x0_in = pub.Initial.ChargeWeighted(pe_array, PMT_pos, time_array)
            elif args.initial == 'fit':
                x0_in = pub.Initial.FitGrid(pe_array, mesh, tpl, time_array)
            elif args.initial == 'MC':
                x0_in = pub.Initial.MCGrid(pe_array, mesh, tpl, time_array)

            x0_in = x0_in[1:]
            result_in = minimize(pub.Likelihood_Truth.Likelihood, x0_in, method='SLSQP', bounds=((0, 1), (None, None), (None, None), (None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe))
            z, x = pub.Likelihood_Truth.Calc_basis(result_in.x, PMT_pos, cut_pe)
            L, E_in = pub.Likelihood_Truth.Likelihood_PE(coeff_pe, z, x, pe_array, cut_pe)

            # xyz coordinate
            in2 = pub.r2c(result_in.x[:3])*shell
            recondata['x_sph_in'] = in2[0]
            recondata['y_sph_in'] = in2[1]
            recondata['z_sph_in'] = in2[2]
            recondata['success_in'] = result_in.success
            recondata['Likelihood_in'] = result_in.fun
            # outer recon
            if args.initial == 'WA':
                x0_out = result_in.copy()
                x0_out[0] = 0.92
            else:
                x0_out = pub.Initial.FitGrid(pe_array, mesh_out, tpl_out, time_array)
                x0_out = x0_out[1:]
            result_out = minimize(pub.Likelihood_Truth.Likelihood, x0_out, method='SLSQP',bounds=((0,1), (None, None), (None, None),(None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe))
            z, x = pub.Likelihood_Truth.Calc_basis(result_out.x, PMT_pos, cut_pe)
            L, E_out = pub.Likelihood_Truth.Likelihood_PE(coeff_pe, z, x, pe_array, cut_pe)
            
            out2 = pub.r2c(result_out.x[:3]) * shell
            recondata['x_sph_out'] = out2[0]
            recondata['y_sph_out'] = out2[1]
            recondata['z_sph_out'] = out2[2]
            recondata['success_out'] = result_out.success
            recondata['Likelihood_out'] = result_out.fun
            
            # 0-th order (Energy intercept)
            base_in = LG.legval(result_in.x[1], coeff_pe.T)
            base_out = LG.legval(result_out.x[1], coeff_pe.T)
            recondata['E_sph_in'] = np.exp(E_in - base_in[0] + np.log(2))
            recondata['E_sph_out'] = np.exp(E_out - base_out[0] + np.log(2))
            
            if (verbose):
                print('-'*60)
                print(f'inner: {np.exp(E_in - base_in[0] + np.log(2))}')
                print(f'outer: {np.exp(E_out - base_out[0] + np.log(2))}')


                print('inner')
                print(f'Template likelihood: {-np.max(L)}')
                print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (sid, in2[0], in2[1], in2[2], norm(in2), result_in.fun))
                print('outer')
                print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (sid, out2[0], out2[1], out2[2], norm(out2), result_out.fun))
            
        else:
            recondata['x_sph_in'] = 0
            recondata['y_sph_in'] = 0
            recondata['z_sph_in'] = 0
            recondata['E_sph_in'] = 0
            recondata['success_in'] = 0
            recondata['Likelihood_in'] = 0
            
            recondata['x_sph_out'] = 0
            recondata['y_sph_out'] = 0
            recondata['z_sph_out'] = 0
            recondata['E_sph_out'] = 0
            recondata['success_out'] = 0
            recondata['Likelihood_out'] = 0
            print('empty event!')
            print('-'*60)
        recondata.append()
    elif types == 'h5':
        pass
    
    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

parser = argparse.ArgumentParser(description='Process Reconstruction construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('-v', '--verbose', dest='verbose', type=int, default=1,
                    help='Output message')

parser.add_argument('--pe', dest='pe', metavar='PECoeff[*.h5]', type=str, 
                    default=r'../calib/h5/PE_coeff_1t_30_80.h5', 
                    help='The pe coefficients file [*.h5] to be loaded')

parser.add_argument('--time', dest='time', metavar='TimeCoeff[*.h5]', type=str,
                    default=r'../calib/h5/Time_coeff_1t_0.10.h5', 
                    help='The time coefficients file [*.h5] to be loaded')

parser.add_argument('--split', dest='split', choices=[True, False], default=True,
                    help='Whether the coefficients is segmented')

parser.add_argument('--mode', dest='mode', choices=['PE', 'Time', 'Combined'], default='Combined',
                    help='Whether use PE/Time info')

parser.add_argument('--TIME', dest='TIME', choices=[True, False], default=True,
                    help='Whether use Time info')

parser.add_argument('--offset', dest='offset', metavar='offset[*.h5]', type=str, default=False,
                    help='Whether use offset data, default is 0')

parser.add_argument('--type', dest='type', choices=['h5', 'Sim_root', 'wave'], default='Sim_root',
                    help = 'Load file type')

parser.add_argument('--initial', dest='initial', choices=['WA','MC','fit'], default='fit',
                    help = 'initial point method')

parser.add_argument('--MC', dest='MC', metavar='MCGrid[*.h5]', type=str, default=False,
                    help='The MC grid file [*.h5] to be loaded')

parser.add_argument('--PMT', dest='PMT', metavar='PMT[*.txt]', type=str, default=r'./PMT_1t.txt',
                    help='The PMT file [*.txt] to be loaded')

parser.add_argument('--method', dest='method', choices=['1','2','3'], default='2',
                    help=textwrap.dedent('''Method to be used in reconstruction.
                    '1' - Fit E,
                    '2' - normalized E,
                    '3' - dual annealing'''))

args = parser.parse_args()
print(args.filename)

PMT_pos = np.loadtxt(args.PMT)

coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time\
    = pub.load_coeff(PEFile = args.pe, TimeFile = args.time)

if args.initial == 'fit':
    mesh, tpl = pub.construct(PMT_pos, coeff_pe, cut_pe)
    mesh_out, tpl_out = pub.construct_outer(PMT_pos, coeff_pe, cut_pe)
elif args.initial == 'MC':
    mesh, tpl = pub.ReadTpl()
    mesh_out, tpl_out = pub.construct_outer(PMT_pos, coeff_pe, cut_pe)
Recon(args.filename, args.output, args.mode, args.offset, args.type, args.initial, args.MC, args.method, args.verbose)