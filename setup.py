import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "afqsrungekutta",
    version = "0.0.1",
    author = "Alejandro Francisco Queiruga",
    description = "",
    license = "none",
    keywords = "",
    packages = find_packages(exclude=['test']),
    test_suite='test',
    long_description=read('README.md'),
    classifiers=[],
)
