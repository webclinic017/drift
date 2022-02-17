from .utils import *


class Protocols:
    """
    Protocols class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Protocols object.
    uniswap_transactions():
        Returns the total number of transactions
        that contains an interaction within Uniswap contracts.
    uniswap_liquidity():
        Returns the current liquidity on Uniswap.
    uniswap_volume():
        Returns the total volume traded on Uniswap.
    """

    def __init__(self, glassnode_client):
        self._gc = glassnode_client
        self._endpoints = self._gc.endpoints

    def uniswap_transactions(self) -> pd.DataFrame:
        """
        The total number of transactions that contains an interaction within Uniswap contracts.
        Includes Mint, Burn, and Swap events on the Uniswap core contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=protocols.UniswapTransactionCount>`_

        :return: A DataFrame containing Uniswap transactions data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/protocols/uniswap_transaction_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def uniswap_liquidity(self) -> pd.DataFrame:
        """
        The current liquidity on Uniswap.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=protocols.UniswapLiquidityLatest>`_

        :return: A DataFrame containing Uniswap liquidity data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/protocols/uniswap_liquidity_latest"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def uniswap_volume(self) -> pd.DataFrame:
        """
        The total volume traded on Uniswap.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=protocols.UniswapVolumeSum>`_

        :return: A DataFrame containing Uniswap volume data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/protocols/uniswap_volume_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
