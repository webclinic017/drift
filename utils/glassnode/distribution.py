from .utils import *


class Distribution:
    """
    Distribution class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Distribution object.
    exchange_balance_total(exchange):
        Returns the total amount of coins held on exchange addresses.
    exchange_balance_percent(exchange):
        Returns the percent supply held on exchange addresses.
    exchange_balance_stacked():
        Returns the total amount of coins held on exchange addresses.
    miner_balance():
        Returns the total supply held in miner addresses.
    miner_balance_stacked():
        Returns the total supply held in miner addresses.
    balance_miners_change():
        Returns 30d change of the supply held in miner addresses.
    supply_top_one_pct_addresses():
        Returns the percentage of supply held by the top 1% addresses.
    gini_coefficient():
        Returns gini coefficient data.
    herfindahl_index():
        Returns herfindahl index data.
    supply_in_smart_contracts():
        Returns percent of total supply that is held in smart contracts.
    """

    def __init__(self, glassnode_client):
        self._gc = glassnode_client

    def exchange_balance_total(self, exchange=None) -> pd.DataFrame:
        """
        The total amount of coins held on exchange addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=distribution.BalanceExchanges>`_

        :return: A DataFrame with exchange balance data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_exchanges"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    def exchange_balance_percent(self, exchange=None) -> pd.DataFrame:
        """
        The percent supply held on exchange addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=distribution.BalanceExchangesRelative>`_

        :return: A DataFrame with exchange balance data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_exchanges_relative"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint, {"e": exchange}))

    def exchange_balance_stacked(self) -> pd.DataFrame:
        """
        The total amount of coins held on exchange addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=distribution.BalanceExchangesAll>`_

        :return: A DataFrame with stacked exchange balance data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_exchanges_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def miner_balance(self) -> pd.DataFrame:
        """
        The total supply held in miner addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=distribution.BalanceMinersSum>`_

        :return: A DataFrame miner balance data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_miners_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def miner_balance_stacked(self) -> pd.DataFrame:
        """
        The total supply held in miner addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=distribution.BalanceMinersAll>`_

        :return: A DataFrame with stacked miner balance data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_miners_all"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def balance_miners_change(self) -> pd.DataFrame:
        """
        The 30d change of the supply held in miner addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=distribution.BalanceMinersChange>`_

        :return: A DataFrame with 30d change of the supply held in miner addresses.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_miners_change"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_top_one_pct_addresses(self) -> pd.DataFrame:
        """
        The percentage of supply held by the top 1% addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=distribution.Balance1PctHolders>`_

        :return: A DataFrame with top 1% supply data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/balance_1pct_holders"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def gini_coefficient(self) -> pd.DataFrame:
        """
        The gini coefficient for the distribution of coins over addresses.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=distribution.Gini>`_

        :return: A DataFrame Gini Coefficient data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/gini"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def herfindahl_index(self) -> pd.DataFrame:
        """
        A metric for decentralization.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=distribution.Herfindahl>`_

        :return: A DataFrame Herfindahl index data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/herfindahl"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_in_smart_contracts(self) -> pd.DataFrame:
        """
        The percent of total supply of the token that is held in smart contracts.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=distribution.SupplyContracts>`_

        :return: A DataFrame smart contracts supply data.
        :rtype: DataFrame
        """
        endpoint = "/v1/metrics/distribution/supply_contracts"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
