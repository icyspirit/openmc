import os

import numpy as np
import openmc
import pytest

from tests.regression_tests import config


def test_ww_generator(run_in_tmpdir):
    # create a simple spherical shield model
    model = openmc.Model()

    water = openmc.Material()
    water.set_density('g/cc', 1.0)
    water.add_nuclide('H1', 0.66)
    water.add_nuclide('O16', 0.34)

    s = openmc.Sphere(r=50, boundary_type='vacuum')
    c = openmc.Cell(fill=water, region=-s)

    model.geometry = openmc.Geometry([c])

    model.settings.particles = 500
    model.settings.batches = 5
    model.settings.run_mode = 'fixed source'
    model.settings.max_history_splits = 100

    mesh = openmc.RegularMesh.from_domain(model.geometry.root_universe)
    energy_bounds = np.linspace(0.0, 1e6, 70)
    particle = 'neutron'

    # include another tally to make sure user-specified tallies and those automaticaly
    # created by weight window generators can coexist
    tally = openmc.Tally()
    ef = openmc.EnergyFilter(energy_bounds)
    tally.filters = [ef]
    tally.scores = ['flux']
    model.tallies = [tally]

    wwg = openmc.WeightWindowGenerator(mesh, energy_bounds, particle)
    wwg.update_parameters = {'ratio': 5.0, 'threshold': 0.8, 'value': 'mean'}

    model.settings.weight_window_generators = wwg
    model.run()

    # we test the effectiveness of the update method elsewhere, so
    # just test that the generation happens successfully here
    assert os.path.exists('weight_windows.h5')

    wws_mean = openmc.WeightWindowsList.from_hdf5()
    assert len(wws_mean) == 1

    # check that generation using the relative error works too
    wwg.update_parameters['value'] = 'rel_err'
    model.run()

    wws_rel_err = openmc.WeightWindowsList.from_hdf5()
    assert len(wws_rel_err) == 1

    # we should not get the same set of weight windows when switching to use of
    # rel. err.
    assert (wws_mean[0].lower_ww_bounds != wws_rel_err[0].lower_ww_bounds).any()


@pytest.mark.skipif(not config['mpi'], reason='requires MPI')
def test_ww_generator_mpi(run_in_tmpdir):
    # Windows generated on the fly must not depend on the number of MPI
    # processes: when tallies are reduced, every process has to transport with
    # the master's windows, not with the invalid ones it would compute from its
    # own zeroed tally results. A multigroup slab keeps the test free of
    # nuclear data.
    groups = openmc.mgxs.EnergyGroups([0.0, 2.0e7])
    xs = openmc.XSdata('slab', groups)
    xs.order = 0
    xs.set_total([1.0])
    xs.set_absorption([0.2])
    xs.set_scatter_matrix(np.array([[[0.8]]]))
    library = openmc.MGXSLibrary(groups)
    library.add_xsdata(xs)
    library.export_to_hdf5('mgxs.h5')

    material = openmc.Material()
    material.set_density('macro', 1.0)
    material.add_macroscopic('slab')

    model = openmc.Model()
    model.materials = openmc.Materials([material])
    model.materials.cross_sections = 'mgxs.h5'
    region = openmc.model.RectangularParallelepiped(
        0.0, 100.0, -10.0, 10.0, -10.0, 10.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=material, region=-region)
    model.geometry = openmc.Geometry([cell])

    model.settings.run_mode = 'fixed source'
    model.settings.energy_mode = 'multi-group'
    model.settings.particles = 2000
    model.settings.batches = 5
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.01, 0.0, 0.0)),
        angle=openmc.stats.Monodirectional((1.0, 0.0, 0.0)),
        energy=openmc.stats.Discrete([1.0e6], [1.0]))

    mesh = openmc.RegularMesh()
    mesh.lower_left = (0.0, -10.0, -10.0)
    mesh.upper_right = (100.0, 10.0, 10.0)
    mesh.dimension = (20, 1, 1)
    wwg = openmc.WeightWindowGenerator(
        mesh, max_realizations=model.settings.batches)
    # A bin scored in a single batch has a relative error of exactly 1, so a
    # threshold away from 1 keeps rounding from deciding whether it is valid
    wwg.update_parameters = {'ratio': 5.0, 'threshold': 0.9, 'value': 'mean'}
    model.settings.weight_window_generators = wwg

    model.run(openmc_exec=config['exe'])
    serial = openmc.WeightWindowsList.from_hdf5()[0].lower_ww_bounds

    mpi_args = [config['mpiexec'], '-n', config['mpi_np']]
    model.run(openmc_exec=config['exe'], mpi_args=mpi_args)
    parallel = openmc.WeightWindowsList.from_hdf5()[0].lower_ww_bounds

    assert (serial > 0.0).sum() > 1
    np.testing.assert_array_equal(serial > 0.0, parallel > 0.0)
    np.testing.assert_allclose(parallel, serial, rtol=1e-10)
