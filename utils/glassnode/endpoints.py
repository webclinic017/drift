from .utils import fetch


def create_endpoints_dict(endpoints):
    return {
        endpoint['path']: {
            'assets': {asset['symbol']: asset['tags'] for asset in endpoint['assets']},
            'currencies': endpoint['currencies'],
            'resolutions': endpoint['resolutions'],
            'formats': endpoint['formats']
        }
        for endpoint in endpoints
    }


"""
@pattern Singleton (GoF:127)
"""


class MetaEndpoints(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Endpoints(metaclass=MetaEndpoints):
    _endpoints = None

    @property
    def endpoints(self):
        return self._endpoints

    @endpoints.setter
    def endpoints(self, api_key):
        self._endpoints = create_endpoints_dict(fetch('/v2/metrics/endpoints', {'api_key': api_key}))

    def query(self, path):
        return self._endpoints[path]
