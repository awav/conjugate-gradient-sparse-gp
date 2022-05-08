from setuptools import setup, find_packages

pkgs = find_packages()

# req = ["numpy", "scipy", "tensorflow"]
req = ["numpy", "scipy"]

setup(
    name="cggp",
    version="0.0.1",
    packages=pkgs,
    install_requires=req,
    dependency_links=[],
)