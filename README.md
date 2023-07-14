# Data-Day-Essentials [![Coverage Status](./.reports/coverage/coverage-badge.svg)](./.reports/coverage/index.html)



## Overview Data-Day-Essentials
Data-Day-Essentials is a collection of packages that features tools that are common to any data science projects:
- Loading and saving to azure
- Loading data from the data warehouse
- Transforming pandas dataframes
- Registering datasets with MLStudio
- Model inference
- Jupyter notebook tools
- Evaluating models (currently just classifiers)

For a user guide please see our [COMING SOON]

For API documentation please also see the relevant link above.

## Developer Guidance
### Updating coverage report

Please run `sh coverage_report.sh` and commit the generated files to update the coverage report.

To view the coverage report go to [here](./.reports/coverage/index.html).

### Updating documentation

The API documentation is updated automatically on every push/PR into main. Please ensure that all classes/functions have docstrings and type annotations and these will be included in the documentation automatically. 

### Using this repo in requirments files

To use this repo in a CI script or remotely you may need to use the private SSH key found here:

https://example

Once this is done you should be able to install this package by including the following inside a requirements.txt file:

```
wheel
git+https://github.com/data-day-science/datadayessentials@main#egg=datascience-core
```

Note: The above line uses the main branch to install from. Please adjust if you require a specific commit or version.
