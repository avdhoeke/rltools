"""Installation with setuptools or pip."""
from setuptools import setup, find_packages
import os
import ast


def get_version_from_init():
    """Obtain library version from main init."""
    init_file = os.path.join(
        os.path.dirname(__file__), 'rltools', '__init__.py'
    )
    with open(init_file) as fd:
        for line in fd:
            if line.startswith('__version__'):
                return ast.literal_eval(line.split('=', 1)[1].strip())


with open('README.md') as f:
    readme = f.read()

with open('COPYING') as f:
    lic = f.read()


setup(
    name='rltools',
    version=get_version_from_init(),
    description='rltools: A reinforcement learning toolbox gathering state of the art algorithms',
    long_description=readme,
    author='Arthur Vandenhoeke',
    author_email='arthur.vandenhoeke@gmail.com',
    url='',
    license=lic,
    packages=find_packages(exclude=('tests', 'docs', 'example')),
    install_requires=[
        'numpy',
    ],
    package_data={'rltools': []},
)