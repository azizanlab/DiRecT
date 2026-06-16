from setuptools import setup, find_packages
from codecs import open
from os import path

import yaml

from mmd import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))

# environment.yml is the single source of truth for dependencies. Extract the
# pip: section and drop pip flags (--extra-index-url, --find-links) which are
# install-time options, not requirement specifiers.
with open(path.join(here, 'environment.yml'), encoding='utf-8') as f:
    env = yaml.safe_load(f)

requires_list = []
for dep in env.get('dependencies', []):
    if isinstance(dep, dict) and 'pip' in dep:
        for item in dep['pip']:
            if isinstance(item, str) and not item.startswith('--'):
                requires_list.append(item)


setup(name='mmd',
    version=__version__,
    description='Projected Coupled Diffusion (PCD) implementation based on MMD',
    author='Hao Luan',
    author_email='haoluan@comp.nus.edu.sg',
    packages=find_packages(where=''),
    install_requires=requires_list,
)
