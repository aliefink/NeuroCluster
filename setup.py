from setuptools import find_packages, setup
import requests

from setuptools import find_packages, setup

setup(
    name='NeuroCluster',
    version='1.0.0',
    description='Python package for non-parametric cluster-based permutation testing for time-frequency resolved neurophysiological signals related to complex behaviors.'
    url=github_url,
    author=', '.join(authors), 
    packages=find_packages(),   
    package_data={'': ['data/*']},
    include_package_data=True,
    install_requires=required,
)

setup(
    name='src',
    packages=find_packages(),
)

# Get the repository owner and name from the GitHub URL
github_url = 'https://github.com/seqasim/LFPAnalysis'
owner, repo = github_url.split('/')[-2:]

# Get the list of contributors from the GitHub API
response = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contributors')
contributors = response.json()

# Create a list of author strings in the format "Name <email>"
authors = [f"{c['login']}" for c in contributors]

# Get long description
with open("README.md", "r") as fh:
    __long_description__ = fh.read()

# Get requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='LFPAnalysis',
    version='1.0.0',
    description='Package to process LFP data',
    url=github_url,
    author=', '.join(authors), 
    packages=find_packages(),   
    package_data={'': ['data/*']},
    include_package_data=True,
    install_requires=required,
)