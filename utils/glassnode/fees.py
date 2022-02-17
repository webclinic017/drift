from .utils import *
from .client import GlassnodeClient


class Fees:
    """
    Fees class.

    Methods
    -------
    __init__(glassnode_client):
        Constructs a Mining object.
    fee_ratio_multiple():
        Returns the Fee Ratio Multiple (FRM).
    fees_total():
        Returns the total amount of fees paid to miners.
    fees_mean():
        Returns the mean fee per transaction.
    fees_median():
        Returns the median fee per transaction.
    gas_used_total():
        Returns the total amount of gas used in all transactions.
    gas_used_mean():
        Returns the mean amount of gas used per transaction.
    gas_used_median():
        Returns the median amount of gas used per transaction.
    gas_price_mean():
        Returns the mean gas price paid per transaction.
    gas_price_median():
        Returns the median gas price paid per transaction.
    transaction_gas_limit_mean():
        Returns the mean gas limit per transaction.
    transaction_gas_limit_median():
        Returns the median gas limit per transaction.
    exchange_fees_total():
        Returns the total amount of fees paid in transactions related to on-chain exchange activity.
    exchange_fees_mean():
        Returns the mean amount of fees paid in transactions related to on-chain exchange activity.
    exchange_fees_dominance():
        Returns the exchange fee dominance.
    """

    def __init__(self, glassnode_client: GlassnodeClient):
        self._gc = glassnode_client

    def fee_ratio_multiple(self) -> pd.DataFrame:
        """
        The Fee Ratio Multiple (FRM) is a measure of a blockchain's security
        and gives an assessment how secure a chain is once block rewards disappear.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.FeeRatioMultiple>`_
        """
        endpoint = "/v1/metrics/fees/fee_ratio_multiple"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def fees_total(self) -> pd.DataFrame:
        """
        The total amount of fees paid to miners. Issued (minted) coins are not included.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.VolumeSum>`_
        """
        endpoint = "/v1/metrics/fees/volume_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def fees_mean(self) -> pd.DataFrame:
        """
        The mean fee per transaction. Issued (minted) coins are not included.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.VolumeMean>`_
        """
        endpoint = "/v1/metrics/fees/volume_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def fees_median(self) -> pd.DataFrame:
        """
        The median fee per transaction. Issued (minted) coins are not included.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.VolumeMedian>`_
        """
        endpoint = "/v1/metrics/fees/volume_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def gas_used_total(self) -> pd.DataFrame:
        """
        The total amount of gas used in all transactions.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasUsedSum>`_
        """
        endpoint = "/v1/metrics/fees/gas_used_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def gas_used_mean(self) -> pd.DataFrame:
        """
        The mean amount of gas used per transaction.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasUsedMean>`_
        """
        endpoint = "/v1/metrics/fees/gas_used_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def gas_used_median(self) -> pd.DataFrame:
        """
        The median amount of gas used per transaction.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasUsedMedian>`_
        """
        endpoint = "/v1/metrics/fees/gas_used_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def gas_price_mean(self) -> pd.DataFrame:
        """
        The mean gas price paid per transaction.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasPriceMean>`_
        """
        endpoint = "/v1/metrics/fees/gas_price_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def gas_price_median(self) -> pd.DataFrame:
        """
        The median gas price paid per transaction.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasPriceMedian>`_
        """
        endpoint = "/v1/metrics/fees/gas_price_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def transaction_gas_limit_mean(self) -> pd.DataFrame:
        """
        The mean gas limit per transaction.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasLimitTxMean>`_
        """
        endpoint = "/v1/metrics/fees/gas_limit_tx_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def transaction_gas_limit_median(self) -> pd.DataFrame:
        """
        The median gas limit per transaction.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=fees.GasLimitTxMedian>`_
        """
        endpoint = "/v1/metrics/fees/gas_limit_tx_median"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def exchange_fees_total(self) -> pd.DataFrame:
        """
        The total amount of fees paid in transactions related to on-chain exchange activity.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.ExchangesSum>`_
        """
        endpoint = "/v1/metrics/fees/exchanges_sum"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def exchange_fees_mean(self) -> pd.DataFrame:
        """
        The mean amount of fees paid in transactions related to on-chain exchange activity.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.ExchangesMean>`_
        """
        endpoint = "/v1/metrics/fees/exchanges_mean"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def exchange_fees_dominance(self) -> pd.DataFrame:
        """
        The Exchange Fee Dominance metric is defined as the percent amount of total fees
        paid in transactions related to on-chain exchange activity.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=fees.ExchangesRelative>`_
        """
        endpoint = "/v1/metrics/fees/exchanges_relative"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
