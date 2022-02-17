from .utils import *


class Mining:
    """
    Mining class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Mining object.
    difficulty():
        Returns difficulty to mine a block.
    hash_rate():
        Returns hash rate.
    miner_revenue_total():
        Returns the total miner revenue.
    miner_revenue_fees():
        Returns the percentage of miner revenue derived from fees.
    miner_revenue_block_rewards():
        Returns the total amount of newly minted coins.
    miner_outflow_multiple():
        Returns the miner outflow multiple.
    thermocap():
        Returns Thermocap data.
    market_cap_to_thermocap_ratio():
        Returns the Marketcap to Thermocap Ratio.
    miner_unspent_supply():
        Returns unspent miner supply.
    miner_names():
        Returns miner names for a mining endpoint.
    """

    def __init__(self, glassnode_client):
        self._gc = glassnode_client

    def difficulty(self) -> pd.DataFrame:
        """
        The current estimated number of hashes required to mine a block. Values are denoted in raw hashes.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.DifficultyLatest>`_

        :return: A DataFrame with the latest difficulty data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/difficulty_latest"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def hash_rate(self) -> pd.DataFrame:
        """
        The average estimated number of hashes per second produced by the miners in the network.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.HashRateMean>`_

        :return: A DataFrame with hash rate data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/hash_rate_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def miner_revenue_total(self, miner=None) -> pd.DataFrame:
        """
        The total miner revenue, i.e. fees plus newly minted coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.RevenueSum>`_

        :return: A DataFrame with total revenue data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/revenue_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"m": miner}))

    def miner_revenue_fees(self) -> pd.DataFrame:
        """
        The percentage of miner revenue derived from fees, i.e. fees divided by fees plus minted coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.RevenueFromFees>`_

        :return: A DataFrame with revenue fees data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/revenue_from_fees"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def miner_revenue_block_rewards(self, miner=None) -> pd.DataFrame:
        """
        The total amount of newly minted coins, i.e. block rewards.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.VolumeMinedSum>`_

        :return: A DataFrame with revenue block rewards data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/volume_mined_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"m": miner}))

    def miner_outflow_multiple(self, miner=None) -> pd.DataFrame:
        """
        The Miner Outflow Multiple indicates periods where the amount of bitcoins flowing out of
        miner addresses is high with respect to its historical average.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.MinersOutflowMultiple>`_

        :return: A DataFrame MOM data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/miners_outflow_multiple"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"m": miner}))

    def thermocap(self) -> pd.DataFrame:
        """
        "Thermocap" is the aggregated amount of coins paid to miners and serves as a proxy to mining resources spent.
        It serves a measure of the true capital flow into Bitcoin.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.Thermocap>`_

        :return: A DataFrame with thermocap data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/thermocap"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def market_cap_to_thermocap_ratio(self) -> pd.DataFrame:
        """
        The Marketcap to Thermocap Ratio can be used to assess if the asset's price is currently trading
        at a premium with respect to total security spend by miners.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.MarketcapThermocapRatio>`_

        :return: A DataFrame with M/T ratio data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/marketcap_thermocap_ratio"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def miner_unspent_supply(self) -> pd.DataFrame:
        """
        The total mount of coins in coinbase transactions that have never been moved.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=mining.MinersUnspentSupply>`_

        :return: A DataFrame with unspent miner supply data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/mining/miners_unspent_supply"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def miner_names(self, endpoint="revenue_sum") -> list:
        """
        Returns a list of miner names for a mining endpoint.

        :param endpoint: Available endpoints: revenue_sum, volume_mined_sum, miners_outflow_multiple
        :return: A List with miner names.
        :rtype: List
        """
        miners = self._gc.get(f"/v1/metrics/mining/{endpoint}/miners")
        return miners[self._gc.asset]
