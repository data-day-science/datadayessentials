# Datascience Core [![Coverage Status](./.reports/coverage/coverage-badge.svg)](./.reports/coverage/index.html)

[__API Documentation__](https://ds-core-docs.azurewebsites.net/datascience_core.html) | [__Confluence User Guide__](https://247group.atlassian.net/wiki/spaces/247PROD/pages/2818539589/Data+Science+Core)

## Overview Datascience Core
Datascience Core is a collection of packages that features tools that are common to any data science projects:
- Loading and saving to azure
- Loading data from the data warehouse
- Transforming pandas dataframes
- Registering datasets with MLStudio
- Model inference
- Jupyter notebook tools
- Evaluating models (currently just classifiers)

For a user guide please see our confluence page (link above).

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
git+https://github.com/data-day-science/datascience_core@main#egg=datascience-core
```

Note: The above line uses the main branch to install from. Please adjust if you require a specific commit or version.
