from .utils import *
from .client import GlassnodeClient


class Entities:
    """
    Entities class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Entities object.
    sending_entities():
        Returns the number of unique entities that were active as a sender.
    receiving_entities():
        Returns the number of unique entities that were active as a receiver.
    active_entities():
        Returns the number of unique entities that were active either as a sender or receiver.
    new_entities():
        Returns The number of unique entities that appeared for the first time in a transaction.
    entities_net_growth():
        Returns the net growth of unique entities in the network.
    number_of_whales():
        Returns the number of unique entities holding at least 1k coins.
    supply_balance_less_0001():
        Returns the total circulating supply held by entities with a balance lower than 0.001 coins.
    supply_balance_0001_001():
        Returns the total circulating supply held by entities with a balance between 0.001 and 0.01 coins.
    supply_balance_001_01():
        Returns the total circulating supply held by entities with a balance between 0.01 and 0.1 coins.
    supply_balance_01_1():
        Returns the total circulating supply held by entities with a balance between 0.1 and 1 coins.
    supply_balance_1_10():
        Returns the total circulating supply held by entities with a balance between 1 and 10 coins.
    supply_balance_10_100():
        Returns the total circulating supply held by entities with a balance between 10 and 100 coins.
    supply_balance_100_1k():
        Returns the total circulating supply held by entities with a balance between 100 and 1,000 coins.
    supply_balance_1k_10k():
        Returns the total circulating supply held by entities with a balance between 1,000 and 10,000 coins.
    supply_balance_10k_100k():
        Returns the total circulating supply held by entities with a balance between 10,000 and 100,000 coins.
    supply_balance_more_100k():
        Returns the total circulating supply held by entities with a balance of at least 100,000 coins.
    entities_supply_distribution():
        Returns relative distribution of the circulating supply held by entities with specific balance bands.
    percent_entities_in_profit():
        Returns the percentage of entities in the network that are currently in profit.
    """

    def __init__(self, glassnode_client: GlassnodeClient):
        self._gc = glassnode_client

    def sending_entities(self) -> pd.DataFrame:
        """
        The number of unique entities that were active as a sender.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SendingCount>`_
        """
        endpoint = "/v1/metrics/entities/sending_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def receiving_entities(self) -> pd.DataFrame:
        """
        The number of unique entities that were active as a receiver.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.ReceivingCount>`_
        """
        endpoint = "/v1/metrics/entities/receiving_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def active_entities(self) -> pd.DataFrame:
        """
        The number of unique entities that were active either as a sender or receiver.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.ActiveCount>`_
        """
        endpoint = "/v1/metrics/entities/active_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def new_entities(self) -> pd.DataFrame:
        """
        The number of unique entities that appeared for the first time
        in a transaction of the native coin in the network.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.NewCount>`_
        """
        endpoint = "/v1/metrics/entities/new_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def entities_net_growth(self) -> pd.DataFrame:
        """
        The net growth of unique entities in the network.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.NetGrowthCount>`_
        """
        endpoint = "/v1/metrics/entities/net_growth_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def number_of_whales(self) -> pd.DataFrame:
        """
        The number of unique entities holding at least 1k coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.Min1KCount>`_
        """
        endpoint = "/v1/metrics/entities/min_1k_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_less_0001(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance lower than 0.001 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalanceLess0001>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_less_0001"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_0001_001(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 0.001 and 0.01 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance0001001>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_0001_001"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_001_01(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 0.01 and 0.1 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance00101>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_001_01"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_01_1(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 0.1 and 1 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance011>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_01_1"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_1_10(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 1 and 10 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance110>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_1_10"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_10_100(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 10 and 100 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance10100>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_10_100"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_100_1k(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 100 and 1,000 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance1001K>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_100_1k"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_1k_10k(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 1,000 and 10,000 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance1K10K>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_1k_10k"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_10k_100k(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance between 10,000 and 100,000 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalance10K100K>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_10k_100k"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_balance_more_100k(self) -> pd.DataFrame:
        """
        The total circulating supply held by entities with a balance of at least 100,000 coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyBalanceMore100K>`_
        """
        endpoint = "/v1/metrics/entities/supply_balance_more_100k"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def entities_supply_distribution(self) -> pd.DataFrame:
        """
        Relative distribution of the circulating supply held by entities with specific balance bands.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.SupplyDistributionRelative>`_
        """
        endpoint = "/v1/metrics/entities/supply_distribution_relative"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def percent_entities_in_profit(self) -> pd.DataFrame:
        """
        The percentage of entities in the network that are currently in profit.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=entities.ProfitRelative>`_
        """
        endpoint = "/v1/metrics/entities/profit_relative"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
