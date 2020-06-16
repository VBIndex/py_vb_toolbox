import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vb_toolbox",
    version="1.1.0",
    author="Lucas da Costa Campos",
    author_email="lqccampos@gmail.com",
    description="Library and command-line tool to calculate the Vogt-Bailey index of a dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VBIndex/py_vb_toolbox",
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
        "multiprocess",
        "pybids",
    ],
    entry_points={
        'console_scripts':[
            'vb_tool = vb_toolbox.app:main',
        ]
    }
)

