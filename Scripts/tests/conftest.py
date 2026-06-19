"""Pytest configuration for the CoWERA test suite.

Adds the ``Scripts`` directory to ``sys.path`` so that the ``cowera`` package
can be imported directly, mirroring how ``run_cowera.py`` configures the path.
"""
import os
import sys

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
