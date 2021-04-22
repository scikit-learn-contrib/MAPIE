import codecs
from setuptools import find_packages, setup


DISTNAME = 'MAPIE'
VERSION = "0.0.0"
DESCRIPTION = 'A scikit-learn-compatible module for estimating prediction intervals.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
URL = 'https://github.com/simai-ml/MAPIE'
DOWNLOAD_URL = 'https://github.com/simai-ml/MAPIE'
LICENSE = 'new BSD'
MAINTAINER = 'V. Taquet, G. Martinon'
MAINTAINER_EMAIL = 'vtaquet@quantmetry.com, gmartinon@quantmetry.com'
PYTHON_REQUIRES = ">=3.7"
PACKAGES = find_packages()
INSTALL_REQUIRES = ['scikit-learn']
EXTRAS_REQUIRE = {
    'tests': ['flake8', 'mypy', 'pytest', 'pytest-cov'],
    'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'numpydoc', 'matplotlib']
}
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
]

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    zip_safe=False  # the package can run out of an .egg file
)
