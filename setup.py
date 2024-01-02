from setuptools import setup, find_packages


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


package_list = find_packages(exclude=["*.tests*"])
long_description = read_file("README.md")
version = 2.0
requirements = read_requirements("requirements.txt")

setup(
    name="datadayessentials",
    version=version,
    author="Data Science",
    url="https://github.com/data-day-science/datadayessentials",
    description="Common functionality to other datascience tools",
    package_dir={
        "datadayessentials": "datadayessentials",
        "datadayessentials.authentications": "datadayessentials/authentications",
        "datadayessentials.model_inference": "datadayessentials/model_inference",
        "datadayessentials.data_retrieval": "datadayessentials/data_retrieval",
        "datadayessentials.data_transformation": "datadayessentials/data_transformation",
        "datadayessentials.modelling": "datadayessentials/modelling",
    },
    packages=package_list,  # Don't include test directory in binary distribution
    package_data={
        "datadayessentials.data_retrieval": [
            "schemas/*.json",
            "sql_queries/*.sql",
            "*.yml",
        ],
        "datadayessentials": ["*.yml"],
        "datadayessentials.authentications": ["*.yml"],
        "datadayessentials.model_inference": ["*.yml"],
        "datadayessentials.data_transformation": ["*.yml"],
        "": ["*.yml"],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
