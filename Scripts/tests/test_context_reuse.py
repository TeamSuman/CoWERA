"""Integration test for the cached-Context path (Phase 2.3).

Validates that OpenMMRunner(reuse_context=True) builds the OpenMM Context ONCE
and reuses it across many run_segment calls (setState -> step -> getState),
using a tiny system on the OpenMM *Reference* platform -- so it runs on any host
with OpenMM installed, no GPU required.

Marked ``integration`` (deselected by default); run with -m integration on a host
with the MD stack.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _build_runner(tmp_path, reuse_context):
    omm = pytest.importorskip("simtk.openmm")
    omma = pytest.importorskip("simtk.openmm.app")
    unit = pytest.importorskip("simtk.unit")
    from wepy.runners.openmm import OpenMMRunner

    # Minimal 3-particle system on the Reference platform.
    system = omm.System()
    for _ in range(3):
        system.addParticle(12.0 * unit.amu)
    force = omm.HarmonicBondForce()
    force.addBond(0, 1, 0.15 * unit.nanometer, 1000.0)
    force.addBond(1, 2, 0.15 * unit.nanometer, 1000.0)
    system.addForce(force)

    integrator = omm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)

    topology = omma.Topology()
    chain = topology.addChain()
    res = topology.addResidue("X", chain)
    element = omma.Element.getBySymbol("C")
    for i in range(3):
        topology.addAtom(f"C{i}", element, res)

    dcd_folder = str(tmp_path) + os.sep
    runner = OpenMMRunner(system, topology, integrator, platform="Reference",
                          dcd_folder=dcd_folder, save_freq=1,
                          reuse_context=reuse_context)
    return runner, system, integrator


def _make_walker(system, integrator):
    unit = pytest.importorskip("simtk.unit")
    from wepy.runners.openmm import OpenMMState, OpenMMWalker, gen_sim_state

    positions = np.array([[0.0, 0.0, 0.0],
                          [0.15, 0.0, 0.0],
                          [0.30, 0.0, 0.0]]) * unit.nanometer
    sim_state = gen_sim_state(positions, system, integrator)
    return OpenMMWalker(OpenMMState(sim_state), 1.0)


def test_reuse_context_builds_once(tmp_path):
    runner, system, integrator = _build_runner(tmp_path, reuse_context=True)
    walker = _make_walker(system, integrator)

    for cycle in range(4):
        walker = runner.run_segment(walker, 5, walker_idx=0, cycle_idx=cycle)

    # exactly one cached Simulation/Context despite four segments
    assert len(runner._sim_cache) == 1


def test_default_path_does_not_cache(tmp_path):
    runner, system, integrator = _build_runner(tmp_path, reuse_context=False)
    walker = _make_walker(system, integrator)
    walker = runner.run_segment(walker, 5, walker_idx=0, cycle_idx=0)
    # default path builds fresh each segment -> cache stays empty
    assert len(runner._sim_cache) == 0
