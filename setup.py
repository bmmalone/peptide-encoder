from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

import importlib
import logging
import shutil

def _safe_read_lines(f):        
    with open(f) as in_f:
        r = in_f.readlines()
    r = [l.strip() for l in r]
    return r

console_scripts = [
    'train-pepenc-models=pepenc.models.train_pepenc_models:main',
]

install_requires = [
    "joblib",
    "numpy",
    "lifesci",
    "pyllars",
    "tqdm",
    "pandas",
    "pyyaml",
    "ray[tune]",
    "torch",
    "gym",
    "dm-tree",
    "opencv-python",
]

tests_require = [
    'pytest',
    'coverage',
    'pytest-cov',
    'coveralls',
    'pytest-runner',
]

gpu_requires = []

docs_require = [
    'sphinx',
    'sphinx_rtd_theme'
]

all_requires = (
    tests_require +
    gpu_requires +
    docs_require
)

extras = {
    'test': tests_require,
    'gpu': gpu_requires,
    'docs': docs_require,
    'all': all_requires
}

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

def _post_install(self):
    import site
    importlib.reload(site)


class my_install(_install):
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _install.run(self)
        _post_install(self)

class my_develop(_develop):  
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _develop.run(self)
        _post_install(self)

def readme():
    with open('README.md') as f:
        return f.read()

def description():
    description = ("An encoder for peptides (short amino acid sequences) based on BLOSUM similarity.")
    return description

setup(
    name='peptide_encoder',
    version='0.2.2',
    description=description(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    keywords="peptide encoding blossum",
    url="https://github.com/bmmalone/peptide-encoder",
    author="Brandon Malone",
    author_email="bmmalone@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={'install': my_install,  # override install
                'develop': my_develop   # develop is used for pip install -e .
    },
    include_package_data=True,
    tests_require=tests_require,
    extras_require=extras,
    entry_points = {
        'console_scripts': console_scripts
    },
    zip_safe=False,
    classifiers=classifiers,    
)
