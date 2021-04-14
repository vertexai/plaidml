# Copyright 2019 Intel Corporation

from setuptools import setup

CONSOLE_SCRIPTS = [
    'plaidbench = plaidbench.cli:plaidbench',
]

INSTALL_REQUIRES = [
    'click>=6.0.0',
    'colorama',
    'enum34>=1.1.6',
    'h5py==2.10',
    'numpy',
    'plaidml',
    'six',
]


def main():
    setup(
        name='plaidbench',
        version='@PLAIDML_VERSION@',
        author='Intel Corporation',
        author_email='plaidml-dev@googlegroups.com',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: C++',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
        ],
        description='PlaidML benchmarking tool',
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        install_requires=INSTALL_REQUIRES,
        keywords='plaidml ml machine learning tensor compiler',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        long_description='Benchmarks for machine-learning implementations.',
        package_data={'plaidbench': [
            '*.npy',
            'golden/*/*.npy',
            'networks/keras/*.h5',
        ]},
        packages=[
            'plaidbench',
            'plaidbench.networks',
            'plaidbench.networks.keras',
            'plaidbench.networks.ops',
        ],
        url='https://www.intel.ai/plaidml',
    )


if __name__ == "__main__":
    main()
