from setuptools import setup, find_packages

def requirements_from_file(file_name):
    return open(file_name).read().splitlines()

setup(
    name='blindpy',
    version='0.0.0',
    packages=find_packages(),
    install_requires=requirements_from_file('requirements.txt'),
    entry_points={
        "console_scripts": [
            "blindpy=blindpy.main:entry_point",
            ]
        },
)
