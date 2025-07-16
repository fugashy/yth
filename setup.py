from setuptools import setup, find_packages

def requirements_from_file(file_name):
    return open(file_name).read().splitlines()

setup(
    name='yth',
    version='0.0.0',
    packages=find_packages(),
    install_requires=requirements_from_file('requirements.txt'),
    package_data={
        "yth": ["templates/*.html"]
    },
    entry_points={
        "console_scripts": [
            "yth=yth.main:entry_point",
            ]
        },
)
