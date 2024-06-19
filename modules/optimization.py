import numpy as np
import traceback
from .utils import remove_extra_files
from .create_vmec import setup_vmec

def objective(trial, vmec, qs, qi, elongation, mirror, quasisymmetry, aspect_target, min_iota, max_bounds=0.15):

    # vmec, qs, qi, elongation, mirror = setup_vmec(vmec_input, max_mode, helicity_n)

    # Check if optimizing with least squares (array input) or with optuna (trial input)
    if isinstance(trial, np.ndarray):
        dofs = trial
    else:        
        dofs = [trial.suggest_float(f'{i}', -max_bounds, max_bounds) for i in range(len(vmec.x))]
    
    vmec.x = dofs

    try:
        if quasisymmetry:
            loss_confinement = qs.total()
        else:
            loss_confinement = np.sum(np.abs(qi(vmec))) + np.abs(elongation(vmec)) + np.abs(mirror(vmec))
        loss_aspect = (vmec.aspect() - aspect_target) ** 2
        loss_iota = 50 * np.min((np.min(np.abs(vmec.wout.iotaf)) - min_iota, 0)) ** 2
    except Exception as e:
        # print(f"Trial {trial.number} failed with error: {e}")
        # traceback.print_exc()
        if isinstance(trial, np.ndarray):
            return np.inf
        else:
            return None
    finally:
        remove_extra_files(vmec)

    if isinstance(trial, np.ndarray):
        return np.array([loss_confinement, loss_aspect, loss_iota])
    else:
        return np.log(loss_confinement + loss_aspect + loss_iota)