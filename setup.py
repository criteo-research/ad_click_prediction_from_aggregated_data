#!/usr/bin/env python3

import os

import setuptools


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]


_REQUIREMENTS_TXT = _read_reqs("requirements.txt")
_INSTALL_REQUIRES = [req for req in _REQUIREMENTS_TXT if "://" not in req]

setuptools.setup(
    name="aggregated-models",
    version="0.0.1",
    install_requires=_INSTALL_REQUIRES,
    tests_require=_read_reqs("tests-requirements.txt"),
    dependency_links=[],
    data_files=[(".", ["requirements.txt", "tests-requirements.txt"])],
    packages=setuptools.find_packages(),
)