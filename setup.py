from setuptools import find_packages, setup

# https://setuptools.pypa.io/en/latest/references/keywords.html

github_url = 'https://github.com/aliefink/NeuroCluster'

authors = ['Alexandra Fink','Christina Maher','Salman Qasim','Ignacio Saez']

# Get requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='NeuroCluster',
    version='1.0.0',
    description='Python package for non-parametric cluster-based permutation testing for time-frequency resolved neurophysiological signals related to complex behaviors.',
    long_description=open('README.md').read(),
    url=github_url,
    author=', '.join(authors), 
    packages=find_packages(),   
    package_data={'': ['data/*']},
    include_package_data=True,
    install_requires=required,
)


