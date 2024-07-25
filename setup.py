from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.call(["python", "post_install.py"])

setup(
    name='anndict',
    version='0.1',
    packages=find_packages(),
    description='Conveniently process a dictionary of anndatas (adata_dict)',
    author='ggit12',
    # author_email='your.email@example.com',
    license='BSD-3-Clause',
    install_requires=[
        # list of packages this package depends on
        'numpy', 
        'pandas',
        'scikit-learn',
        'scanpy',
        'anndata',
        'IPython',
        'scipy',
        'seaborn',
        'matplotlib',
        'squidpy',
        'harmonypy',
        'openai'
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)
