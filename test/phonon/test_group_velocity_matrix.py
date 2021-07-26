"""Test for group_velocity_matrix.py."""
import numpy as np
from phonopy.phonon.group_velocity import GroupVelocity
from phono3py.phonon.group_velocity_matrix import GroupVelocityMatrix
from phono3py import Phono3py


def test_gvm_nacl(nacl_pbe: Phono3py):
    """Test diagonal elements of group velocity matrix.

    This test only requires phonopy `DynamicalMatrix` instance.
    To get it, set `Phono3py.mesh_numbers = [1, 1, 1]` as a dummy parameter
    to enable to run `Phono3py.init_phph_interaction()` that is necessary
    to instanciate the `DynamicalMatrix` class.

    """
    qpoints = [[0.1, 0.1, 0.1]]
    nacl_pbe.mesh_numbers = [1, 1, 1]  # dummy
    nacl_pbe.init_phph_interaction()
    gv = GroupVelocity(nacl_pbe.dynamical_matrix,
                       symmetry=nacl_pbe.primitive_symmetry)
    gv.run(qpoints)
    gvm = GroupVelocityMatrix(nacl_pbe.dynamical_matrix,
                              symmetry=nacl_pbe.primitive_symmetry)
    gvm.run(qpoints)
    gvs = []
    for mat in gvm.group_velocity_matrices[0]:
        gvs.append(np.diagonal(mat.real))
    np.testing.assert_allclose(gvs, gv.group_velocities[0].T, atol=1e-5)
