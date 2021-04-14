# Copyright 2021 Intel Corporation

from setuptools import setup

CONSOLE_SCRIPTS = [
    'plaidml-setup = plaidml.plaidml_setup:main',
]

REQUIRED_PACKAGES = [
    'cffi',
    'enum34>=1.1.6',
    'numpy',
    'six',
]


def main():
    setup(
        name='plaidml',
        version='@PLAIDML_VERSION@',
        author='Intel Corporation',
        author_email='plaidml-dev@googlegroups.com',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: C++',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        description='PlaidML machine learning accelerator',
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        install_requires=REQUIRED_PACKAGES,
        keywords='plaidml ml machine learning tensor compiler',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        long_description='PlaidML is a framework for making machine learning work everywhere.',
        package_data={'plaidml': ['@_PLAIDML_BINARY@']},
        package_dir={'': '@PROJECT_BINARY_DIR@'},
        packages=[
            'plaidml',
            'plaidml.core',
            'plaidml.edsl',
            'plaidml.exec',
            'plaidml.op',
        ],
        url='https://www.intel.ai/plaidml',
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
