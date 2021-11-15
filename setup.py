try:
    from setuptools import setup, find_packages
except ImportError:
    from disutils.core import setup, find_packages

import convnmf

config = {
    'name': 'convnmf',
    'packages': find_packages(exclude=['doc']),
    'description': 'Tools for Convolutive Matrix Factorization',
    'author': 'Alex Williams',
    'author_email': 'alex.h.willia@gmail.com',
    'url': 'https://github.com/ahwillia/torch_convnmf',
}

setup(**config)
