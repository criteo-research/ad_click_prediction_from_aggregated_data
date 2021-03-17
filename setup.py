#!/usr/bin/env python3

import setuptools
import os


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines()
                if (s.strip() and not s.startswith("#"))]


_REQUIREMENTS_TXT = _read_reqs("requirements.txt")
_INSTALL_REQUIRES = [l for l in _REQUIREMENTS_TXT if "://" not in l]

setuptools.setup(
    name='aggregated_models',
    version='0.1',
    install_requires=_INSTALL_REQUIRES,
    tests_require=_read_reqs("tests-requirements.txt"),
    dependency_links=[],
    entry_points={
        'console_scripts': [
            'run-app = app.main:main',
        ],
    },
    data_files=[('.', ['requirements.txt', 'tests-requirements.txt'])],
    packages=setuptools.find_packages())
