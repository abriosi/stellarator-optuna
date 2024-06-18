import optuna
import argparse

import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.geo import SurfaceRZFourier

def remove_extra_files(vmec):
    try: os.remove(vmec.output_file)
    except: pass
    try: os.remove(vmec.input_file + '_{:03d}_{:06d}'.format(vmec.mpi.group, vmec.iter))
    except: pass

def make_plot(vmec, savefig=True, filename='surf_opt.png'):
    ntheta = 200
    nphi = 150*vmec.indata.nfp
    theta = np.linspace(0,1,num=ntheta)
    phi = np.linspace(0,1,num=nphi)
    B = np.zeros((ntheta,nphi))
    
    vmec.run();vmec.write_input('input.nfp4_QH_opt'); remove_extra_files(vmec)
    surf_opt = SurfaceRZFourier.from_vmec_input('input.nfp4_QH_opt', quadpoints_phi=phi, quadpoints_theta=theta)
    XYZ = surf_opt.gamma()
    
    phi2D,theta2D = np.meshgrid(2*np.pi*phi,2*np.pi*theta)
    for imode in range(len(vmec.wout.xn_nyq)):
        angle = vmec.wout.xm_nyq[imode]*theta2D - vmec.wout.xn_nyq[imode]*phi2D
        B = B + vmec.wout.bmnc[imode,-1]*np.cos(angle)
    B_rescaled = (B - B.min()) / (B.max() - B.min())
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2], facecolors=cm.jet(np.transpose(B_rescaled)), rstride=1, cstride=1, antialiased=False)
    ax.auto_scale_xyz([0.6*XYZ.min(), 0.6*XYZ.max()], [0.6*XYZ.min(), 0.6*XYZ.max()], [0.6*XYZ.min(), 0.6*XYZ.max()])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    if savefig: plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    else: plt.show()

# Define the objective function
def objective(trial, vmec, qs):
    dofs = [trial.suggest_float(f'{i}', -0.15, 0.15) for i in range(len(vmec.x))]
    vmec.x = dofs
    try: loss = np.log(qs.total()) + (vmec.aspect()-7)**2
    except: loss = 1e6
    remove_extra_files(vmec)
    return loss

def create_sampler(sampler_name, search_space=None, seed=None):
    if sampler_name == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "TPESampler":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "CmaEsSampler":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler_name == "GPSampler":
        return optuna.samplers.GPSampler(seed=seed)
    elif sampler_name == "PartialFixedSampler":
        # Example: {'x': 0} fixed_params and optuna.samplers.RandomSampler() as base_sampler
        fixed_params = {'x': 0}
        base_sampler = optuna.samplers.RandomSampler(seed=seed)
        return optuna.samplers.PartialFixedSampler(fixed_params, base_sampler)
    elif sampler_name == "NSGAIISampler":
        return optuna.samplers.NSGAIISampler(seed=seed)
    elif sampler_name == "QMCSampler":
        return optuna.samplers.QMCSampler(seed=seed)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

def main():
    parser = argparse.ArgumentParser(description="Optuna CLI for hyperparameter optimization.")
    parser.add_argument("--sampler", type=str, required=True, help="Sampler to use for optimization.")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials for optimization.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument("--storage", type=str, default=None, help="Database URL for Optuna storage.")
    parser.add_argument("--study-name", type=str, required=True, help="Name of the study.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout for the optimization in seconds.")
    args = parser.parse_args()

    sampler = create_sampler(args.sampler, args.seed)

    if args.storage:
        storage = optuna.storages.RDBStorage(args.storage)
    else:
        storage = None
        
    vmec = Vmec('input.nfp4_QH', verbose=False)
    surf = vmec.boundary;surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False);surf.fix("rc(0,0)")
    qs = QuasisymmetryRatioResidual(vmec, np.linspace(0,1,5,endpoint=True), helicity_n=-1, helicity_m=1)

    print(f'Initial aspect ratio: {vmec.aspect()}')
    print(f'Initial quasisymmetry residual: {qs.total()}')
    make_plot(vmec, savefig=True, filename='surf_init.png')

    study = optuna.create_study(study_name=args.study_name, direction="minimize", sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, vmec, qs), n_trials=args.trials, timeout=args.timeout)

    print("Best value (loss): ", study.best_value)
    print("Best parameters: ", study.best_params)
    
    vmec.x = [study.best_params[f'{i}'] for i in range(len(vmec.x))]
    print(f'Optimized aspect ratio: {vmec.aspect()}')
    print(f'Optimized quasisymmetry residual: {qs.total()}')
    make_plot(vmec, savefig=True, filename='surf_opt.png')

if __name__ == "__main__":
    main()
