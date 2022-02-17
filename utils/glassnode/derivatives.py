from .utils import *
from .client import GlassnodeClient


class Derivatives:
    """
    Derivatives class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Derivatives object.
    futures_perpetual_funding_rate([exchange]):
        Returns the average funding rate (in %) set by exchanges for perpetual futures contracts.
    futures_perpetual_funding_rate_all():
        Returns the average funding rate (in %) set by exchanges for perpetual futures contracts.
    futures_volume([exchange]):
        Returns the total volume traded in futures contracts in the last 24 hours.
    futures_volume_latest_24h():
        Returns the total volume traded in futures contracts per exchange over the last 24 hours.
    futures_volume_stacked():
        Returns the total volume traded in futures contracts in the last 24 hours.
    futures_volume_perpetual([exchange]):
        Returns The total volume traded in perpetual futures contracts in the last 24 hours.
    futures_volume_perpetual_stacked():
        Returns the total volume traded in perpetual futures contracts in the last 24 hours.
    futures_open_interest([exchange]):
        Returns the total amount of funds allocated in open futures contracts.
    futures_open_interest_current():
        Returns the current amount of allocated funds in futures contracts per exchange.
    futures_open_interest_perpetual([exchange]):
        Returns the total amount of funds allocated in open perpetual futures contracts.
    futures_open_interest_perpetual_stacked():
        Returns the total amount of funds allocated in open perpetual futures contracts.
    futures_open_interest_stacked():
        Returns the total amount of funds allocated in open futures contracts.
    futures_long_liquidations([exchange]):
        Returns the sum liquidated volume from long positions in futures contracts.
    futures_long_liquidations_mean([exchange]):
        Returns the mean liquidated volume from long positions in futures contracts.
    futures_short_liquidations([exchange]):
        Returns the sum liquidated volume from short positions in futures contracts.
    futures_short_liquidations_mean([exchange]):
        Returns the mean liquidated volume from short positions in futures contracts.
    """

    def __init__(self, glassnode_client: GlassnodeClient):
        self._gc = glassnode_client

    def futures_perpetual_funding_rate(self, exchange: str = None) -> pd.DataFrame:
        """
        The average funding rate (in %) set by exchanges for perpetual futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesFundingRatePerpetual>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_funding_rate_perpetual"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    @dataframe_with_inner_object
    def futures_perpetual_funding_rate_all(self) -> pd.DataFrame:
        """
        The average funding rate (in %) set by exchanges for perpetual futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesFundingRatePerpetualAll>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_funding_rate_perpetual_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def futures_volume(self, exchange: str = None) -> pd.DataFrame:
        """
        The total volume traded in futures contracts in the last 24 hours.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesVolumeDailySum>`_
        """
        url = "/v1/metrics/derivatives/futures_volume_daily_sum"
        if not is_supported_by_endpoint(self._gc, url):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(url, {"e": exchange}))

    # TODO: Unpack inner object from response
    def futures_volume_latest_24h(self) -> pd.DataFrame:
        """
        The total volume traded in futures contracts per exchange over the last 24 hours.
        Values are updated every 10 min.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesVolumeDailyLatest>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_volume_daily_latest"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def futures_volume_stacked(self) -> pd.DataFrame:
        """
        The total volume traded in futures contracts in the last 24 hours.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesVolumeDailySumAll>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_volume_daily_sum_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def futures_volume_perpetual(self, exchange: str = None) -> pd.DataFrame:
        """
        The total volume traded in perpetual (non-expiring) futures contracts in the last 24 hours.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesVolumeDailyPerpetualSum>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_volume_daily_perpetual_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    @dataframe_with_inner_object
    def futures_volume_perpetual_stacked(self) -> pd.DataFrame:
        """
        The total volume traded in perpetual (non-expiring) futures contracts in the last 24 hours.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesVolumeDailyPerpetualSumAll>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_volume_daily_perpetual_sum_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def futures_open_interest(self, exchange: str = None) -> pd.DataFrame:
        """
        The total amount of funds allocated in open futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesOpenInterestSum>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_open_interest_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    # TODO: Unpack inner object from response
    def futures_open_interest_current(self) -> pd.DataFrame:
        """
        The current amount of allocated funds in futures contracts per exchange.Values are updated every 10 min.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesOpenInterestLatest>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_open_interest_latest"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def futures_open_interest_perpetual(self, exchange: str = None) -> pd.DataFrame:
        """
        The total amount of funds allocated in open perpetual (non-expiring) futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesOpenInterestPerpetualSum>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_open_interest_perpetual_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    @dataframe_with_inner_object
    def futures_open_interest_perpetual_stacked(self) -> pd.DataFrame:
        """
        The total amount of funds allocated in open perpetual (non-expiring) futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesOpenInterestPerpetualSumAll>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_open_interest_perpetual_sum_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def futures_open_interest_stacked(self) -> pd.DataFrame:
        """
        The total amount of funds allocated in open futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesOpenInterestSumAll>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_open_interest_sum_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def futures_long_liquidations(self, exchange: str = None) -> pd.DataFrame:
        """
        The sum liquidated volume from long positions in futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesLiquidatedVolumeLongSum>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_liquidated_volume_long_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    def futures_long_liquidations_mean(self, exchange: str = None) -> pd.DataFrame:
        """
        The mean liquidated volume from long positions in futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesLiquidatedVolumeLongMean>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_liquidated_volume_long_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    def futures_short_liquidations(self, exchange: str = None) -> pd.DataFrame:
        """
        The sum liquidated volume from short positions in futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesLiquidatedVolumeShortSum>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_liquidated_volume_short_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    def futures_short_liquidations_mean(self, exchange: str = None) -> pd.DataFrame:
        """
        The mean liquidated volume from short positions in futures contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FuturesLiquidatedVolumeShortMean>`_
        """
        endpoint = "/v1/metrics/derivatives/futures_liquidated_volume_short_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    def futures_estimated_leverage_ratio(self, exchange: str = None) -> pd.DataFrame:

        endpoint = "/v1/metrics/derivatives/futures_estimated_leverage_ratio"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))
