import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VBIndex",
    version="0.0.2",
    author="Lucas da Costa Campos",
    author_email="lqccampos@gmail.com",
    description="Library and command-line tool to calculate the Vogt-Bailey index of a dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VBIndex/py_vbindex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "nibabel",
        "multiprocess"
    ],
    entry_points={
        'console_scripts':[
            'vb_index = vb_toolbox.app:main',
        ]
    }
)

