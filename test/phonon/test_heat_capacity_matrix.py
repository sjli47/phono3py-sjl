"""Test for heat_capacity_matrix.py."""
import numpy as np
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.units import THzToEv
from phonopy.phonon.qpoints import QpointsPhonon
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix
from phono3py import Phono3py


cvm_nacl_ref = [
    [8.6047e-05, 8.6047e-05, 9.1067e-05, 1.6353e-04, 1.6353e-04, 2.3282e-04],
    [8.6047e-05, 8.6047e-05, 9.1067e-05, 1.6353e-04, 1.6353e-04, 2.3282e-04],
    [9.1067e-05, 9.1067e-05, 8.5842e-05, 1.2063e-04, 1.2063e-04, 1.6114e-04],
    [1.6353e-04, 1.6353e-04, 1.2063e-04, 8.2488e-05, 8.2488e-05, 8.5038e-05],
    [1.6353e-04, 1.6353e-04, 1.2063e-04, 8.2488e-05, 8.2488e-05, 8.5038e-05],
    [2.3282e-04, 2.3282e-04, 1.6114e-04, 8.5038e-05, 8.5038e-05, 7.6949e-05],
]


def test_cvm_nacl(nacl_pbe: Phono3py):
    """Test diagonal elements of group velocity matrix.

    This test only requires phonopy `DynamicalMatrix` instance.
    To get it, set `Phono3py.mesh_numbers = [1, 1, 1]` as a dummy parameter
    to enable to run `Phono3py.init_phph_interaction()` that is necessary
    to instanciate the `DynamicalMatrix` class.

    """
    qpoints = [[0.1, 0.1, 0.1]]
    nacl_pbe.mesh_numbers = [4, 4, 4]  # dummy
    nacl_pbe.init_phph_interaction()
    qph = QpointsPhonon(qpoints, nacl_pbe.dynamical_matrix)
    freqs = qph.frequencies[0] * THzToEv

    cv = mode_cv(300, freqs)
    cvm = mode_cv_matrix(300, freqs)
    np.testing.assert_allclose(np.diagonal(cvm), cv, rtol=0, atol=1e-8)
    np.testing.assert_allclose((cvm), cvm_nacl_ref, rtol=0, atol=1e-3)

    # for vec in cvm:
    #     print("[ " + ", ".join(["%.4e" % v for v in vec]) + "],")
