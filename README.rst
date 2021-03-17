Requirements
============

- Install maven build system
- Install Python3.6
- Install virtualenv(3)
- Install python pip package manager

Things to know
==============

- This template project targets python3
- Inside the jmoab it will generate a single package that is auto-exectuable (up to the version of the interpreter)
- It will be compliant deployment to marathon

Jargon
------

- Virtualenv: A kind of sandbox created in order to have reproductible build
              and avoid packages conflicts between your system/other projects

- Tox: a Virtualenv manager to target multiple virtualenvs

- Pylama: A linter that combines several other linters

- Pex: A file format to allow python program to ship as single binary (up to the version of the interpreter)

- Setup.py: File used to describe your project metadata (name, dependencies, entrypoints, ...)
            WARNING: You will have to keep in sync the Setup.py and requirements.txt

- requirements.txt: File used by pip (python pacakge manager) to list dependencies of the project
                    WARNING: You will have to keep in sync the Setup.py and requirements.txt

- mvn: It is the maven build system used by the JAVA MOAB to build and deploy projects
       As python projects are for now in the Java MOAB, we use it to launch commands needed to build this project


How to build
============

During developpement phase
--------------------------

  1) Create a virtualenv to avoid that local packages clash with your system
    a) virtualenv3 .venv
    b) source .venv/bin/activate.sh

  2) Once in your venv, install all the dependencies
    a) pip install -r requirements.txt
    b) pip install -r tests-requirements.txt

  3) Build and install your project
   a) pip install -e .

  4) Run your program
   b) python app/main.py
      You should see "Hello Anonymous"


During release phase
--------------------

  1) Get out of your virtualenv by running in your shell
   a) deactivate

  2) Use maven to build, test, release your project
   a) mvn clean verify

  3) You will find your executable in dist/run-app
    b) dist/run-app.pex should print
       "Hello Anonymous"

How to test
===========

  1) In your virtualenv, run the command `tox`
     It will run tests, code coverage, linter for python3 and pypy

