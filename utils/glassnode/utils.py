import requests
import calendar
import pandas as pd

from datetime import datetime


def unix_timestamp(date_str):
    """
    Returns a unix timestamp to a given date string.

    :param date_str: Date in string format (ex. '2021-01-01').
    :return: Int Unix-timestamp.
    """
    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return calendar.timegm(dt_obj.utctimetuple())


def is_supported_by_endpoint(glassnode_client, url):
    path = glassnode_client.endpoints.query(url)
    if glassnode_client.asset not in path["assets"]:
        print(f"{url} metric is not available for {glassnode_client.asset}")
        return False
    if glassnode_client.resolution not in path["resolutions"]:
        print(f"{url} metric is not available for {glassnode_client.resolution}")
        return False
    return True


def response_to_dataframe(response):
    """
    Returns DataFrame from a response objects (ex. {"t":1604361600,"v":0.002}).

    :param response: Response from API.
    :return: DataFrame.
    """
    try:
        df = pd.DataFrame(response)
        df.set_index("t", inplace=True)
        df.index = pd.to_datetime(df.index, unit="s")
        df.index.name = None
        df.sort_index(ascending=False, inplace=True)
        return df
    except Exception as e:
        print(e)


def dataframe_with_inner_object(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return pd.concat([df.drop(["o"], axis=1), df["o"].apply(pd.Series)], axis=1)

    return wrapper


def fetch(endpoint, params=None):
    """
    Returns an object of time, value pairs for a metric from the Glassnode API.

    :param params:
    :param endpoint: Endpoint url corresponding to some metric (ex. '/v1/metrics/market/price_usd')
    :return: DataFrame of {'t' : datetime, 'v' : 'metric-value'} pairs
    """
    r = requests.get(f"https://api.glassnode.com{endpoint}", params=params, stream=True)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(e.response.text)

    try:
        return r.json()
    except Exception as e:
        print(e)
