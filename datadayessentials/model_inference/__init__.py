"""
Module for hitting Azure Batch Endpoints set up. Currently available are:

- Affordability Batch Endpoint (AffordabilityServiceHitter)
- Scorecard Batch endpont (ScorecardServiceHitter)

Both of the above have the ability to score multiple models at the same time, they also both have caching - the caching uses payload hashes.
"""


# from ._base import MLModel
from ._model_inference import GenericServiceHitter 
from ._base import IServiceHitter, ServiceHitterCacher


__all__ = [
    'IServiceHitter',
    'GenericServiceHitter',
    'ServiceHitterCacher'
]
