import numpy as np
from functools import partial
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from .qi_functions import QuasiIsodynamicResidual, MirrorRatioPen, MaxElongationPen

def setup_vmec(vmec_input, max_mode, helicity_n):
    vmec = Vmec(vmec_input, verbose=False)
    surf = vmec.boundary
    surf.fix_all()
    vmec.run()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    
    qs = QuasisymmetryRatioResidual(vmec, np.linspace(0, 1, 5, endpoint=True), helicity_n=helicity_n, helicity_m=1)
    qi = partial(QuasiIsodynamicResidual, snorms=[1/16, 5/16, 9/16, 13/16], nphi=151, nalpha=51, nBj=71, mpol=21, ntor=21, nphi_out=200, arr_out=True)
    elongation = partial(MaxElongationPen, t=6)
    mirror = partial(MirrorRatioPen, t=0.20)

    return vmec, qs, qi, elongation, mirror