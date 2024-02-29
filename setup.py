import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vb_toolbox",
    version="2.2.0",
    author="The BOB Lab",
    author_email="team@boblab.info",
    description="Library and command-line tool to calculate the Vogt-Bailey index of a dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VBIndex/py_vb_toolbox",
    include_package_data=True,
    packages=["vb_toolbox"],
    package_data={"vb_toolbox": ["vb_gui_icon.png"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires='>=3.6',
    install_requires=[
        "argparse",
        "numpy",
        "scipy",
        "nibabel",
        "multiprocess",
        "textwrap",
        "sys",
        "warnings",
        "traceback",
        "glob",
        "os",
        "signal",
        "shutil"        
    ],
    entry_points={
        'console_scripts':[
            'vb_tool = vb_toolbox.app:main',
        ]
    }
)
