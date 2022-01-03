from .utils import *
from .client import GlassnodeClient


class Supply:
    """
        Supply class.

        Methods
        -------
        __init__(glassnode_client):
            Constructs a Supply object.
        liquid_illiquid_supply():
            Returns the total supply held by illiquid, liquid, and highly liquid entities.
        liquid_supply_change():
            Returns the monthly (30d) net change of supply held by liquid and highly liquid entities.
        illiquid_supply_change():
            Returns the monthly (30d) net change of supply held by illiquid entities.
        circulating_supply():
            Returns the total amount of all coins ever created/issued.
        issuance():
            Returns the total amount of new coins added to the current supply.
        inflation_rate():
            Returns the yearly inflation rate.
        supply_last_active_less_24h():
            Returns the amount of circulating supply last moved in the last 24 hours.
        supply_last_active_1d_1w():
            Returns the amount of circulating supply last moved between 1 day and 1 week ago.
        supply_last_active_1w_1m():
            Returns the amount of circulating supply last moved between 1 week and 1 month ago.
        supply_last_active_1m_3m():
            Returns the amount of circulating supply last moved between 1 month and 3 months ago.
        supply_last_active_3m_6m():
            Returns the amount of circulating supply last moved between 3 months and 6 months ago.
        supply_last_active_6m_12m():
            Returns the amount of circulating supply last moved between 6 months and 12 months ago.
        supply_last_active_1y_2y():
            Returns the amount of circulating supply last moved between 1 year and 6 years ago.
        supply_last_active_2y_3y():
            Returns the amount of circulating supply last moved between 2 years and 3 years ago.
        supply_last_active_3y_5y():
            Returns the amount of circulating supply last moved between 3 years and 5 years ago.
        supply_last_active_5y_7y():
            Returns the amount of circulating supply last moved between 5 years and 7 years ago.
        supply_last_active_7y_10y():
            Returns the amount of circulating supply last moved between 7 years and 10 years ago.
        supply_last_active_more_10y():
            Returns the amount of circulating supply last moved more than 10 years ago.
        hodl_waves():
            Returns a bundle of all active supply age bands, aka HODL waves.
        supply_last_active_more_1y_ago():
            Returns the percent of circulating supply that has not moved in at least 1 year.
        supply_last_active_more_2y_ago():
            Returns the percent of circulating supply that has not moved in at least 2 years.
        supply_last_active_more_3y_ago():
            Returns the percent of circulating supply that has not moved in at least 3 years.
        supply_last_active_more_5y_ago():
            Returns the percent of circulating supply that has not moved in at least 5 years.
        realized_cap_hodl_waves():
            Returns HODL waves weighted by Realized Price.
        adjusted_supply():
            Returns the circulating supply adjusted by accounting for lost coins.
        supply_in_profit():
            Returns the circulating supply in profit.
        supply_in_loss():
            Returns the circulating supply in loss.
        supply_in_profit_relative():
            Returns the percentage of circulating supply in profit.
        short_term_holder_supply():
            Returns the total amount of circulating supply held by short-term holders.
        long_term_holder_supply():
            Returns the total amount of circulating supply held by long-term holders.
        short_term_holder_supply_in_loss():
            Returns the total amount of circulating supply that is currently at loss and held by short-term holders.
        long_term_holder_supply_in_loss():
            Returns the total amount of circulating supply that is currently at loss and held by long-term holders.
        short_term_holder_supply_in_profit():
            Returns the total amount of circulating supply that is currently in profit and held by short-term holders.
        long_term_holder_supply_in_profit():
            Returns the total amount of circulating supply that is currently in profit and held by long-term holders.
        relative_long_short_term_holder_supply():
            Returns the relative amount of circulating supply of held by long- and short-term holders in profit/loss.
        long_term_holder_position_change():
            Returns the monthly net position change of long-term holders.
    """
    def __init__(self, glassnode_client: GlassnodeClient):
        self._gc = glassnode_client

    def liquid_illiquid_supply(self) -> pd.DataFrame:
        """
        The total supply held by illiquid, liquid, and highly liquid entities.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LiquidIlliquidSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/liquid_illiquid_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def liquid_supply_change(self) -> pd.DataFrame:
        """
        The monthly (30d) net change of supply held by liquid and highly liquid entities.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LiquidChange>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/liquid_change'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def illiquid_supply_change(self) -> pd.DataFrame:
        """
        The monthly (30d) net change of supply held by illiquid entities.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.IlliquidChange>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/illiquid_change'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def circulating_supply(self) -> pd.DataFrame:
        """
        The total amount of all coins ever created/issued, i.e. the circulating supply.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Current>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/current'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def issuance(self) -> pd.DataFrame:
        """
        The total amount of new coins added to the current supply,
        i.e. minted coins or new coins released to the network.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Issued>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/issued'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def inflation_rate(self) -> pd.DataFrame:
        """
        The yearly inflation rate, i.e. the percentage of new coins issued,
        divided by the current supply (annualized).
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.InflationRate>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/inflation_rate'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_less_24h(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved in the last 24 hours.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active24H>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_24h'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_1d_1w(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 1 day and 1 week ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active1D1W>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_1d_1w'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_1w_1m(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 1 week and 1 month ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active1W1M>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_1w_1m'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_1m_3m(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 1 month and 3 months ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active1M3M>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_1m_3m'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_3m_6m(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 3 months and 6 months ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active3M6M>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_3m_6m'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_6m_12m(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 6 months and 12 months ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active6M12M>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_6m_12m'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_1y_2y(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 1 year and 2 years ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active1Y2Y>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_1y_2y'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_2y_3y(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 2 years and 3 years ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active1Y2Y>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_2y_3y'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_3y_5y(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 3 years and 5 years ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active3Y5Y>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_3y_5y'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_5y_7y(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 5 years and 7 years ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active5Y7Y>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_5y_7y'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_7y_10y(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved between 7 years and 10 years ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.Active7Y10Y>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_7y_10y'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_more_10y(self) -> pd.DataFrame:
        """
        The amount of circulating supply last moved more than 10 years ago.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ActiveMore10Y>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_more_10y'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def hodl_waves(self) -> pd.DataFrame:
        """
        Bundle of all active supply age bands, aka HODL waves.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.HodlWaves>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/hodl_waves'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_more_1y_ago(self) -> pd.DataFrame:
        """
        The percent of circulating supply that has not moved in at least 1 year.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ActiveMore1YPercent>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_more_1y_percent'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_more_2y_ago(self) -> pd.DataFrame:
        """
        The percent of circulating supply that has not moved in at least 2 years.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ActiveMore2YPercent>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_more_2y_percent'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_more_3y_ago(self) -> pd.DataFrame:
        """
        The percent of circulating supply that has not moved in at least 3 years.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ActiveMore3YPercent>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_more_3y_percent'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_last_active_more_5y_ago(self) -> pd.DataFrame:
        """
        The percent of circulating supply that has not moved in at least 5 years.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ActiveMore5YPercent>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/active_more_5y_percent'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def realized_cap_hodl_waves(self) -> pd.DataFrame:
        """
        HODL Waves weighted by Realized Price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.RcapHodlWaves>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/rcap_hodl_waves'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def adjusted_supply(self) -> pd.DataFrame:
        """
        The circulating supply adjusted by accounting for lost coins.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.CurrentAdjusted>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/current_adjusted'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_in_profit(self) -> pd.DataFrame:
        """
        The circulating supply in profit,
        i.e. the amount of coins whose price at the time they last moved was lower than the current price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ProfitSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/profit_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_in_loss(self) -> pd.DataFrame:
        """
        The circulating supply in loss,
        i.e. the amount of coins whose price at the time they last moved was higher than the current price.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LossSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/loss_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_in_profit_relative(self) -> pd.DataFrame:
        """
        The percentage of circulating supply in profit.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.ProfitRelative>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/profit_relative'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def short_term_holder_supply(self) -> pd.DataFrame:
        """
        The total amount of circulating supply held by short-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.SthSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/sth_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def long_term_holder_supply(self) -> pd.DataFrame:
        """
        The total amount of circulating supply held by long-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LthSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/lth_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def short_term_holder_supply_in_loss(self) -> pd.DataFrame:
        """
        The total amount of circulating supply that is currently at loss and held by short-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.SthLossSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/sth_loss_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def long_term_holder_supply_in_loss(self) -> pd.DataFrame:
        """
        The total amount of circulating supply that is currently at loss and held by long-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LthLossSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/lth_loss_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def short_term_holder_supply_in_profit(self) -> pd.DataFrame:
        """
        The total amount of circulating supply that is currently in profit and held by short-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.SthProfitSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/sth_profit_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def long_term_holder_supply_in_profit(self) -> pd.DataFrame:
        """
        The total amount of circulating supply that is currently in profit and held by long-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LthProfitSum>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/lth_profit_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def relative_long_short_term_holder_supply(self) -> pd.DataFrame:
        """
        The relative amount of circulating supply of held by long- and short-term holders in profit/loss.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LthSthProfitLossRelative>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/lth_sth_profit_loss_relative'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def long_term_holder_position_change(self) -> pd.DataFrame:
        """
        The monthly net position change of long-term holders,
        i.e. the 30 day change in supply held by long-term holders.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=supply.LthNetChange>`_

        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/supply/lth_net_change'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
