import optuna
import argparse

import os
import traceback
import numpy as np
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.geo import SurfaceRZFourier
from qi_functions import QuasiIsodynamicResidual, MirrorRatioPen, MaxElongationPen

def remove_extra_files(vmec):
    try: os.remove(vmec.output_file)
    except: pass
    try: os.remove(vmec.input_file + '_{:03d}_{:06d}'.format(vmec.mpi.group, vmec.iter))
    except: pass

def make_plot(vmec, savefig=True, filename_suffix='optuna'):
    ntheta = 200
    nphi = 150*vmec.indata.nfp
    theta = np.linspace(0,1,num=ntheta)
    phi = np.linspace(0,1,num=nphi)
    B = np.zeros((ntheta,nphi))
    
    vmec.run();vmec.write_input(f'{vmec.input_file}_{filename_suffix}'); remove_extra_files(vmec)
    surf_opt = SurfaceRZFourier.from_vmec_input(f'{vmec.input_file}_{filename_suffix}', quadpoints_phi=phi, quadpoints_theta=theta)
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
    if savefig: plt.savefig(f'surf_{vmec.input_file}_{filename_suffix}.png', bbox_inches = 'tight', pad_inches = 0, dpi=300)
    else: plt.show()

# Define the objective function
def objective(trial, vmec, qs, aspect_target, min_iota, qi, elongation, mirror, quasisymmetry, max_bounds=0.15):
    dofs = [trial.suggest_float(f'{i}', -max_bounds, max_bounds) for i in range(len(vmec.x))]
    vmec.x = dofs
    try:
        if quasisymmetry:
            loss_confinement = qs.total()
        else:
            loss_confinement = np.abs(qi(vmec)) + np.abs(elongation(vmec)) + np.abs(mirror(vmec))
        loss_aspect = (vmec.aspect()-aspect_target)**2
        loss_iota = 50*np.min((np.min(np.abs(vmec.wout.iotaf))-min_iota,0))**2
    except Exception as e:
        # print(f"Trial {trial.number} failed with error: {e}")
        # traceback.print_exc()
        remove_extra_files(vmec)
        return None
    remove_extra_files(vmec)
    return loss_confinement + loss_aspect + loss_iota

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
    parser.add_argument("--QA_QH_QI", type=str, default='QH', help="Select to optimize QA, QH or QI.")
    parser.add_argument("--min_iota", type=float, default=0.41, help="Minimum rotational transform of the stellarator.")
    parser.add_argument("--max_mode", type=int, default=1, help="Plasma boundary maximum mode.")
    parser.add_argument("--max_nfev", type=int, default=10, help="Number of iterations for the least squares.")
    parser.add_argument("--aspect", type=float, default=6, help="Plasma boundary aspect ratio.")
    parser.add_argument("--sampler", type=str, required=True, help="Sampler to use for optimization.")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials for optimization.")
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
    
    if args.QA_QH_QI == 'QH':
        quasisymmetry = True
        helicity_n = -1
        vmec_input = 'input.nfp4_QH'
        max_bounds = 0.2
    elif args.QA_QH_QI == 'QA':
        quasisymmetry = True
        helicity_n = 0
        vmec_input = 'input.nfp2_QA'
        max_bounds = 0.2
    elif args.QA_QH_QI == 'QI':
        helicity_n = 1
        quasisymmetry = False
        vmec_input = 'input.nfp3_QI'
        max_bounds = 0.4
    vmec = Vmec(vmec_input, verbose=False)
    surf = vmec.boundary;surf.fix_all();vmec.run()
    surf.fixed_range(mmin=0, mmax=args.max_mode, nmin=-args.max_mode, nmax=args.max_mode, fixed=False);surf.fix("rc(0,0)")
    qs         = QuasisymmetryRatioResidual(vmec, np.linspace(0,1,5,endpoint=True), helicity_n=helicity_n, helicity_m=1)
    qi         = partial(QuasiIsodynamicResidual,snorms=[1/16, 5/16, 9/16, 13/16], nphi=151, nalpha=51, nBj=71, mpol=21, ntor=21, nphi_out=200, arr_out=True)
    elongation = partial(MaxElongationPen,t=6)
    mirror     = partial(MirrorRatioPen,t=0.20)

    print(f'Initial parameters: {vmec.x}')
    print(f'Initial min_iota: {np.min(np.abs(vmec.wout.iotaf))}')
    print(f'Initial aspect ratio: {vmec.aspect()}')
    if quasisymmetry:
        print(f'Initial quasisymmetry residual: {qs.total()}')
    else:
        print(f'Initial quasiisodynamic residual: {qi(vmec)}')
        print(f'Initial elongation penalty: {elongation(vmec)}')
        print(f'Initial mirror penalty: {mirror(vmec)}')
    make_plot(vmec, savefig=True, filename_suffix='init')
    
    # comparison with a least squares model
    from scipy.optimize import least_squares
    def fun(x):
        vmec.x = x
        try:
            loss_aspect = (vmec.aspect()-args.aspect)
            loss_iota = np.sqrt(50)*np.min((np.min(np.abs(vmec.wout.iotaf))-args.min_iota,0))
            if quasisymmetry:
                loss_confinement = np.sqrt(qs.total())
            else:
                loss_confinement = np.sqrt(np.sum(np.abs(qi(vmec))) + np.abs(elongation(vmec)) + np.abs(mirror(vmec)))
        except Exception as e:
            remove_extra_files(vmec)
            return np.inf
        remove_extra_files(vmec)
        return loss_confinement, loss_aspect, loss_iota
    res = least_squares(fun, vmec.x, bounds=([-max_bounds]*len(vmec.x), [max_bounds]*len(vmec.x)), verbose=2, max_nfev=args.max_nfev)
    
    study = optuna.create_study(study_name=args.study_name, direction="minimize", sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, vmec, qs, args.aspect, args.min_iota, qi, elongation, mirror, quasisymmetry, max_bounds), n_trials=args.trials, timeout=args.timeout)

    # print("Best value (loss): ", study.best_value)
    # print("Best parameters: ", study.best_params)
    
    # Least Squares Result
    vmec.x = res.x
    print(f'Least squares parameters: {res.x}')
    print(f'Least squares min_iota: {np.min(np.abs(vmec.wout.iotaf))}')
    print(f'Least squares aspect ratio: {vmec.aspect()}')
    if quasisymmetry:
        print(f'Least squares quasisymmetry residual: {qs.total()}')
    else:
        print(f'Least squares quasiisodynamic residual: {qi(vmec)}')
        print(f'Least squares elongation penalty: {elongation(vmec)}')
        print(f'Least squares mirror penalty: {mirror(vmec)}')
    make_plot(vmec, savefig=True, filename_suffix='least_squares')
    
    # Optuna Result
    vmec.x = [study.best_params[f'{i}'] for i in range(len(vmec.x))]
    print(f'Optuna optimization parameters: {vmec.x}')
    print(f'Optuna optimization min_iota: {np.min(np.abs(vmec.wout.iotaf))}')
    print(f'Optuna optimization aspect ratio: {vmec.aspect()}')
    if quasisymmetry:
        print(f'Optuna optimization quasisymmetry residual: {qs.total()}')
    else:
        print(f'Optuna optimization quasiisodynamic residual: {qi(vmec)}')
        print(f'Optuna optimization elongation penalty: {elongation(vmec)}')
        print(f'Optuna optimization mirror penalty: {mirror(vmec)}')
    make_plot(vmec, savefig=True, filename_suffix='optuna')
    

if __name__ == "__main__":
    main()
