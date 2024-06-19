import optuna
import argparse

import numpy as np
from modules.create_vmec import setup_vmec
from modules.optimization import objective
from modules.utils import make_plot
from scipy.optimize import least_squares

def main():
    parser = argparse.ArgumentParser(description="Optuna CLI for hyperparameter optimization.")
    parser.add_argument("--QA_QH_QI", type=str, default='QH', help="Select to optimize QA, QH or QI.")
    parser.add_argument("--min_iota", type=float, default=0.41, help="Minimum rotational transform of the stellarator.")
    parser.add_argument("--max_mode", type=int, default=1, help="Plasma boundary maximum mode.")
    parser.add_argument("--aspect", type=float, default=6, help="Plasma boundary aspect ratio.")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials for optimization.")
    args = parser.parse_args()

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

    vmec, qs, qi, elongation, mirror = setup_vmec(vmec_input, args.max_mode, helicity_n)
    
    res = least_squares(objective, vmec.x, args=(vmec, qs, qi, elongation, mirror, quasisymmetry, args.aspect, args.min_iota, max_bounds),
                        bounds=([-max_bounds]*len(vmec.x), [max_bounds]*len(vmec.x)), verbose=2, max_nfev=args.trials)

    vmec, qs, qi, elongation, mirror = setup_vmec(vmec_input, args.max_mode, helicity_n)
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

    vmec.x = res.x
    vmec.run()
    print(f'Least squares optimization parameters: {vmec.x}')
    print(f'Least squares optimization min_iota: {np.min(np.abs(vmec.wout.iotaf))}')
    print(f'Least squares optimization aspect ratio: {vmec.aspect()}')
    if quasisymmetry:
        print(f'Least squares optimization quasisymmetry residual: {qs.total()}')
    else:
        print(f'Least squares optimization quasiisodynamic residual: {np.sum(np.abs(qi(vmec)))}')
        print(f'Least squares optimization elongation penalty: {elongation(vmec)}')
        print(f'Least squares optimization mirror penalty: {mirror(vmec)}')
    make_plot(vmec, savefig=True, filename_suffix='leastsquares')

if __name__ == "__main__":
    main()