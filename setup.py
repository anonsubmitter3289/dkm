from setuptools import setup

setup(
    name="dkm",
    packages=["dkm"],
    version="0.0.1",
    author="Anon Submitter 3289",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
