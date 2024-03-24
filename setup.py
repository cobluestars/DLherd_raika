from setuptools import setup, find_packages

setup(
    name="dlherd_raika",
    version="0.1.0",
    author="cobluestars",
    auther_email="cobaltbluestars@gmail.com",
    description="DLherd-Raika is a Python library building on Dataherd-Raika to simulate large-scale user behavior datasets using deep learning. By employing Q-Learning and Model-based Learning(MCTS), it refines probability distributions and variables, enhancing dataset simulation accuracy.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cobluestars/DLherd_raika",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)