import optuna
import argparse

import numpy as np
from modules.create_vmec import setup_vmec
from modules.optimization import objective
from modules.utils import make_plot, create_sampler

def main():
    parser = argparse.ArgumentParser(description="Optuna CLI for hyperparameter optimization.")
    parser.add_argument("--QA_QH_QI", type=str, default='QH', help="Select to optimize QA, QH or QI.")
    parser.add_argument("--min_iota", type=float, default=0.41, help="Minimum rotational transform of the stellarator.")
    parser.add_argument("--max_mode", type=int, default=1, help="Plasma boundary maximum mode.")
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
        max_bounds = 0.6
        
    vmec, qs, qi, elongation, mirror = setup_vmec(vmec_input, args.max_mode, helicity_n)

    study = optuna.create_study(study_name=args.study_name, direction="minimize", sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, vmec_input, args.max_mode, helicity_n, quasisymmetry, args.aspect, args.min_iota, max_bounds),
                   n_trials=args.trials, timeout=args.timeout)

    vmec, qs, qi, elongation, mirror = setup_vmec(vmec_input, args.max_mode, helicity_n)
    
    vmec.run()
    print(f'Initial parameters: {vmec.x}')
    print(f'Initial min_iota: {np.min(np.abs(vmec.wout.iotaf))}')
    print(f'Initial aspect ratio: {vmec.aspect()}')
    if quasisymmetry:
        print(f'Initial quasisymmetry residual: {qs.total()}')
    else:
        print(f'Initial quasiisodynamic residual: {np.sum(np.abs(qi(vmec)))}')
        print(f'Initial elongation penalty: {elongation(vmec)}')
        print(f'Initial mirror penalty: {mirror(vmec)}')
    make_plot(vmec, savefig=True, filename_suffix='init')

    vmec.x = [study.best_params[f'{i}'] for i in range(len(vmec.x))]
    vmec.run()
    print(f'Optuna optimization parameters: {vmec.x}')
    print(f'Optuna optimization min_iota: {np.min(np.abs(vmec.wout.iotaf))}')
    print(f'Optuna optimization aspect ratio: {vmec.aspect()}')
    if quasisymmetry:
        print(f'Optuna optimization quasisymmetry residual: {qs.total()}')
    else:
        print(f'Optuna optimization quasiisodynamic residual: {np.sum(np.abs(qi(vmec)))}')
        print(f'Optuna optimization elongation penalty: {elongation(vmec)}')
        print(f'Optuna optimization mirror penalty: {mirror(vmec)}')
    make_plot(vmec, savefig=True, filename_suffix='optuna')

if __name__ == "__main__":
    main()