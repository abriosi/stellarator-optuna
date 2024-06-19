import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import optuna
from simsopt.geo import SurfaceRZFourier

def remove_extra_files(vmec):
    try: 
        os.remove(vmec.output_file)
    except: 
        pass
    try: 
        os.remove(vmec.input_file + '_{:03d}_{:06d}'.format(vmec.mpi.group, vmec.iter))
    except: 
        pass

def make_plot(vmec, savefig=True, filename_suffix='optuna'):
    ntheta = 200
    nphi = 150 * vmec.indata.nfp
    theta = np.linspace(0, 1, num=ntheta)
    phi = np.linspace(0, 1, num=nphi)
    B = np.zeros((ntheta, nphi))

    vmec.run()
    vmec.write_input(f'{vmec.input_file}_{filename_suffix}')
    remove_extra_files(vmec)
    surf_opt = SurfaceRZFourier.from_vmec_input(f'{vmec.input_file}_{filename_suffix}', quadpoints_phi=phi, quadpoints_theta=theta)
    XYZ = surf_opt.gamma()

    phi2D, theta2D = np.meshgrid(2 * np.pi * phi, 2 * np.pi * theta)
    for imode in range(len(vmec.wout.xn_nyq)):
        angle = vmec.wout.xm_nyq[imode] * theta2D - vmec.wout.xn_nyq[imode] * phi2D
        B = B + vmec.wout.bmnc[imode, -1] * np.cos(angle)
    B_rescaled = (B - B.min()) / (B.max() - B.min())

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2], facecolors=cm.jet(np.transpose(B_rescaled)), rstride=1, cstride=1, antialiased=False)
    ax.auto_scale_xyz([0.7 * XYZ.min(), 0.7 * XYZ.max()], [0.7 * XYZ.min(), 0.7 * XYZ.max()], [0.7 * XYZ.min(), 0.7 * XYZ.max()])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    if savefig:
        plt.savefig(f'surf_{vmec.input_file}_{filename_suffix}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()

def create_sampler(sampler_name, seed=None):
    if sampler_name == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "TPESampler":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "CmaEsSampler":
        return optuna.samplers.CmaEsSampler(seed=seed, restart_strategy="ipop", inc_popsize=2)
    elif sampler_name == "GPSampler":
        return optuna.samplers.GPSampler(seed=seed)
    elif sampler_name == "NSGAIISampler":
        return optuna.samplers.NSGAIISampler(seed=seed)
    elif sampler_name == "QMCSampler":
        return optuna.samplers.QMCSampler(seed=seed)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")