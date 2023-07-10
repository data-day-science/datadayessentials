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
    name="datascience_core",
    version=version,
    author="Data Science",
    url="https://github.com/Carfinance247/datascience_core",
    description="Common functionality to other datascience tools",
    package_dir={
        "datascience_core": "datascience_core",
        "datascience_core.authentications": "datascience_core/authentications",
        "datascience_core.model_inference": "datascience_core/model_inference",
        "datascience_core.data_retrieval": "datascience_core/data_retrieval",
        "datascience_core.data_transformation": "datascience_core/data_transformation",
        "datascience_core.modelling": "datascience_core/modelling",
    },
    packages=package_list,  # Don't include test directory in binary distribution
    package_data={
        "datascience_core.data_retrieval": [
            "schemas/*.json",
            "sql_queries/*.sql",
            "*.yml",
        ],
        "datascience_core": ["*.yml"],
        "datascience_core.authentications": ["*.yml"],
        "datascience_core.model_inference": ["*.yml"],
        "datascience_core.data_transformation": ["*.yml"],
        "": ["*.yml"],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
