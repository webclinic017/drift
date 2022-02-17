from .utils import *
from .client import GlassnodeClient


class Blockchain:
    """
    Blockchain class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Blockchain object.
    utxos_total():
        Returns the total number of UTXOs in the network.
    utxos_created():
        Returns the number of created unspent transaction outputs.
    utxos_spent():
        Returns the number of spent transaction outputs.
    utxo_value_created_total():
        Returns the total amount of coins in newly created UTXOs.
    utxo_value_spent_total():
        Returns the total amount of coins in spent transaction outputs.
    utxo_value_created_mean():
        Returns the mean amount of coins in newly created UTXOs.
    utxo_value_spent_mean():
        Returns the mean amount of coins in spent transaction outputs.
    utxo_value_created_median():
        Returns the median amount of coins in newly created UTXOs.
    utxo_value_spent_median():
        Returns the median amount of coins in spent transaction outputs.
    utxos_in_profit():
        Returns the number of unspent transaction outputs in profit.
    utxos_in_loss():
        Returns the number of unspent transaction outputs in loss.
    percent_utxos_in_profit():
        Returns the percentage of unspent transaction outputs in profit.
    block_heights():
        Returns the block height.
    blocks_mined():
        Returns the number of blocks mined.
    block_interval_mean():
        Returns the mean time (in seconds) between mined blocks.
    block_interval_median():
        Returns the median time (in seconds) between mined blocks.
    block_size_mean():
        Returns the mean size of all blocks created within the time period (in bytes).
    block_size_total():
        Returns the total size of all blocks created within the time period (in bytes).
    """

    def __init__(self, glassnode_client: GlassnodeClient):
        self._gc = glassnode_client

    def utxos_total(self) -> pd.DataFrame:
        """
        The total number of UTXOs in the network.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoCount>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxos_created(self) -> pd.DataFrame:
        """
        The number of created unspent transaction outputs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoCreatedCount>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_created_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxos_spent(self) -> pd.DataFrame:
        """
        The number of spent transaction outputs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoSpentCount>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_spent_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxo_value_created_total(self) -> pd.DataFrame:
        """
        The total amount of coins in newly created UTXOs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoCreatedValueSum>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_created_value_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxo_value_spent_total(self) -> pd.DataFrame:
        """
        The total amount of coins in spent transaction outputs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoSpentValueSum>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_spent_value_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxo_value_created_mean(self) -> pd.DataFrame:
        """
        The mean amount of coins in newly created UTXOs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoCreatedValueMean>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_created_value_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxo_value_spent_mean(self) -> pd.DataFrame:
        """
        The mean amount of coins in spent transaction outputs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoSpentValueMean>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_spent_value_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxo_value_created_median(self) -> pd.DataFrame:
        """
        The median amount of coins in newly created UTXOs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoCreatedValueMedian>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_created_value_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxo_value_spent_median(self) -> pd.DataFrame:
        """
        The median amount of coins in spent transaction outputs.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoSpentValueMedian>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_spent_value_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxos_in_profit(self) -> pd.DataFrame:
        """
        The number of unspent transaction outputs whose price at creation time was lower than the current price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoProfitCount>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_profit_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def utxos_in_loss(self) -> pd.DataFrame:
        """
        The number of unspent transaction outputs whose price at creation time was higher than the current price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoLossCount>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_loss_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def percent_utxos_in_profit(self) -> pd.DataFrame:
        """
        The percentage of unspent transaction outputs whose price at creation time was lower than the current price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.UtxoProfitRelative>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/utxo_profit_relative"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def block_heights(self) -> pd.DataFrame:
        """
        The block height, i.e. the total number of blocks ever created and included in the main blockchain.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.BlockHeight>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/block_height"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def blocks_mined(self) -> pd.DataFrame:
        """
        The number of blocks created and included in the main blockchain in that time period.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.BlockCount>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/block_count"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def block_interval_mean(self) -> pd.DataFrame:
        """
        The mean time (in seconds) between mined blocks.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.BlockIntervalMean>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/block_interval_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def block_interval_median(self) -> pd.DataFrame:
        """
        The median time (in seconds) between mined blocks.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.BlockIntervalMedian>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/block_interval_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def block_size_mean(self) -> pd.DataFrame:
        """
        The mean size of all blocks created within the time period (in bytes).
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.BlockSizeMean>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/block_size_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def block_size_total(self) -> pd.DataFrame:
        """
        The total size of all blocks created within the time period (in bytes).
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=blockchain.BlockSizeSum>`_

        :return: DataFrame
        """
        endpoint = "/v1/metrics/blockchain/block_size_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
