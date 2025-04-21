#!/usr/bin/env python3
import numpy as np
import pytest
# https://realpython.com/pytest-python-testing/
from pymavlink import mavutil


@pytest.fixture
def dummy_test():
    return np.array([1, 2, 3])
