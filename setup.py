import os
from setuptools import setup

requirements_files = ['requirements.txt']

install_requires = []
for filename in requirements_files:
    f = open(os.path.join(os.path.dirname(__file__), filename))
    requirements = f.read().splitlines()
    install_requires.extend(requirements)

setup(
    name="simrl",
    version="0.1",
    packages=['simrl'],
    install_requires=install_requires
)
