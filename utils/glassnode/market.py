from .utils import *


class Market:
    """
    Market class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Market object.
    price():
        Returns the asset's price in USD.
    price_ohlc():
        Returns OHLC candlestick data.
    price_drawdown_from_ath():
        Returns the percent drawdown from previous all-time high.
    marketcap():
        Returns the market capitalization of the asset.
    mvrv_ratio():
        Returns MVRV ratio.
    realized_cap():
        Returns realized cap data.
    mvrv_z_score():
        Returns MVRV Z-Score.
    sth_mvrv():
        Returns Short Term Holder MVRV data.
    lth_mvrv():
        Returns Long Term Holder MVRV data.
    realized_price():
        Returns realized price data.
    """

    def __init__(self, glassnode_client):
        self._gc = glassnode_client

    def price(self) -> pd.DataFrame:
        """
        The asset's price in USD.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.PriceUsd>`_

        :return: A DataFrame containing the asset's price data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/market/price_usd"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def price_ohlc(self) -> pd.DataFrame:
        """
        OHLC candlestick chart of the asset's price in USD.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.PriceUsdOhlc>`_

        :return: A DataFrame containing OHLC candlestick data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/market/price_usd_ohlc"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def price_drawdown_from_ath(self) -> pd.DataFrame:
        """
        The percent drawdown of the asset's price from the previous all-time high.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.PriceDrawdownRelative>`_

        :return: A DataFrame containing the percent drawdown data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/market/price_drawdown_relative"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def marketcap(self) -> pd.DataFrame:
        """
        The market capitalization (or network value) is defined as
        the product of the current supply by the current USD price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.MarketcapUsd>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/marketcap_usd"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def mvrv_ratio(self) -> pd.DataFrame:
        """
        MVRV is the ratio between market cap and realised cap.
        It gives an indication of when the traded price is below a “fair value”.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.Mvrv>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/mvrv"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def realized_cap(self) -> pd.DataFrame:
        """
        Realized Cap values different part of the supplies at different prices (instead of using current daily close).
        Specifically, it is computed by valuing each UTXO by the price when it was last moved.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.MarketcapRealizedUsd>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/marketcap_realized_usd"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def mvrv_z_score(self) -> pd.DataFrame:
        """
        The MVRV Z-Score is used to assess when Bitcoin is over/undervalued relative to its "fair value".
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.MvrvZScore>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/mvrv_z_score"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def sth_mvrv(self) -> pd.DataFrame:
        """
        Short Term Holder MVRV (STH-MVRV) is MVRV that takes into account only UTXOs younger than 155 days and
        serves as an indicator to assess the behaviour of short term investors.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.MvrvLess155>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/mvrv_less_155"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def lth_mvrv(self) -> pd.DataFrame:
        """
        Long Term Holder MVRV (LTH-MVRV) is MVRV that takes into account only UTXOs with a lifespan of at least 155 days
        and serves as an indicator to assess the behaviour of long term investors
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.MvrvMore155>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/mvrv_more_155"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def realized_price(self) -> pd.DataFrame:
        """
        Realized Price is the Realized Cap divided by the current supply.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=market.PriceRealizedUsd>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/market/price_realized_usd"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
