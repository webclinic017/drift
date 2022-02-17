from .utils import *
from .client import GlassnodeClient


class Indicators:
    def __init__(self, glassnode_client: GlassnodeClient):
        self._gc = glassnode_client

    def rhodl_ratio(self) -> pd.DataFrame:
        """
        The Realized HODL Ratio is a market indicator that uses a ratio of the Realized Cap HODL Waves.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.RhodlRatio>`_
        """
        endpoint = "/v1/metrics/indicators/rhodl_ratio"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def cvdd(self) -> pd.DataFrame:
        """
        Cumulative Value-Days Destroyed (CVDD) is the ratio of the cumulative USD value of Coin Days Destroyed and
        the market age (in days). Historically, CVDD has been an accurate indicator for global Bitcoin market bottoms.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Cvdd>`_
        """
        endpoint = "/v1/metrics/indicators/cvdd"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def hash_ribbon(self) -> pd.DataFrame:
        """
        The Hash Ribbon is a market indicator that assumes that Bitcoin tends to reach a bottom when miners capitulate,
        i.e. when Bitcoin becomes too expensive to mine relative to the cost of mining. The Hash Ribbon indicates that
        the worst of the miner capitulation is over when the 30d MA of the hash rate crosses above the 60d MA.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.HashRibbon>`_
        """
        endpoint = "/v1/metrics/indicators/hash_ribbon"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def difficulty_ribbon(self) -> pd.DataFrame:
        """
        The Difficulty Ribbon is an indicator that uses simple moving averages
         of the Bitcoin mining difficulty to create the ribbon.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.DifficultyRibbon>`_
        """
        endpoint = "/v1/metrics/indicators/difficulty_ribbon"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def difficulty_ribbon_compression(self) -> pd.DataFrame:
        """
        Difficulty Ribbon Compression is a market indicator that uses a normalized
        standard deviation to quantify compression of the Difficulty Ribbon.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.DifficultyRibbonCompression>`_
        """
        endpoint = "/v1/metrics/indicators/difficulty_ribbon_compression"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def nvt_ratio(self) -> pd.DataFrame:
        """
        The Network Value to Transactions (NVT) Ratio is computed by dividing
        the market cap by the transferred on-chain volume measured in USD.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Nvt>`_
        """
        endpoint = "/v1/metrics/indicators/nvt"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def nvt_signal(self) -> pd.DataFrame:
        """
        The NVT Signal (NVTS) is a modified version of the original NVT Ratio.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Nvts>`_
        """
        endpoint = "/v1/metrics/indicators/nvts"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def velocity(self) -> pd.DataFrame:
        """
        Velocity is a measure of how quickly units are circulating in the network and is calculated
        by dividing the on-chain transaction volume (in USD) by the market cap, i.e. the inverse of the NVT ratio.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Velocity>`_
        """
        endpoint = "/v1/metrics/indicators/velocity"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_adjusted_cdd(self) -> pd.DataFrame:
        """
        Adjusted Coin Days Destroyed simply divides CDD by the circulating supply.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.CddSupplyAdjusted>`_
        """
        endpoint = "/v1/metrics/indicators/cdd_supply_adjusted"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def binary_cdd(self) -> pd.DataFrame:
        """
        Binary Coin Days Destroyed is computed by thresholding Adjusted CDD by its average over time.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.CddSupplyAdjustedBinary>`_
        """
        endpoint = "/v1/metrics/indicators/cdd_supply_adjusted_binary"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def supply_adjusted_dormancy(self) -> pd.DataFrame:
        """
        Dormancy is the average number of days destroyed per coin transacted,
        and is defined as the ratio of coin days destroyed and total transfer volume.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.AverageDormancySupplyAdjusted>`_
        """
        endpoint = "/v1/metrics/indicators/average_dormancy_supply_adjusted"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def sopd_ath_partitioned(self) -> pd.DataFrame:
        """
        UTXO Realized Price Distribution (URPD) shows at which prices UTXOs were spent that day.
        ATH-partitioned means that the price buckets are defined by dividing the range between
        0 and the current ATH in 100 equally-spaced partitions.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.SpentOutputPriceDistributionAth>`_
        """
        endpoint = "/v1/metrics/indicators/spent_output_price_distribution_ath"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def sopd_percent_partitioned(self) -> pd.DataFrame:
        """
        UTXO Realized Price Distribution (URPD) shows at which prices UTXOs were spent that day.
        %-partitioned means that the price buckets are defined by taking the day's closing price
        and creating 50 equally-spaced bucket each above and below the current price in steps of +/- 2%.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.SpentOutputPriceDistributionPercent>`_
        """
        endpoint = "/v1/metrics/indicators/spent_output_price_distribution_percent"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def puell_multiple(self) -> pd.DataFrame:
        """
        The Puell Multiple is calculated by dividing the daily issuance value of bitcoins (in USD)
        by the 365-day moving average of daily issuance value.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.PuellMultiple>`_
        """
        endpoint = "/v1/metrics/indicators/puell_multiple"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def asopr(self) -> pd.DataFrame:
        """
        Adjusted SOPR is SOPR ignoring all outputs with a lifespan of less than 1 hour.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.SoprAdjusted>`_
        """
        endpoint = "/v1/metrics/indicators/sopr_adjusted"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def reserve_risk(self) -> pd.DataFrame:
        """
        When confidence is high and price is low, there is an attractive risk/reward to invest (Reserve Risk is low).
        When confidence is low and price is high then risk/reward is unattractive at that time (Reserve Risk is high).
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.ReserveRisk>`_
        """
        endpoint = "/v1/metrics/indicators/reserve_risk"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def sth_sopr(self) -> pd.DataFrame:
        """
        Short Term Holder SOPR (STH-SOPR) is SOPR that takes into account only spent outputs
        younger than 155 days and serves as an indicator to assess the behaviour of short term investors.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.SoprLess155>`_
        """
        endpoint = "/v1/metrics/indicators/sopr_less_155"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def lth_sopr(self) -> pd.DataFrame:
        """
        Long Term Holder SOPR (LTH-SOPR) is SOPR that takes into account only spent outputs with a lifespan
        of at least 155 days and serves as an indicator to assess the behaviour of long term investors.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.SoprMore155>`_
        """
        endpoint = "/v1/metrics/indicators/sopr_more_155"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def hodler_net_position_change(self) -> pd.DataFrame:
        """
        HODLer Net Position Change shows the monthly position change of long term investors.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.HodlerNetPositionChange>`_
        """
        endpoint = "/v1/metrics/indicators/hodler_net_position_change"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def hodled_or_lost_coins(self) -> pd.DataFrame:
        """
        Lost or HODLed Bitcoins indicates moves of large and old stashes.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.HodledLostCoins>`_
        """
        endpoint = "/v1/metrics/indicators/hodled_lost_coins"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def sopr(self) -> pd.DataFrame:
        """
        The Spent Output Profit Ratio (SOPR) is computed by dividing the realized value (in USD)
        divided by the value at creation (USD) of a spent output. Or simply: price sold / price paid.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Sopr>`_
        """
        endpoint = "/v1/metrics/indicators/sopr"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def cdd(self) -> pd.DataFrame:
        """
        Coin Days Destroyed (CDD) for any given transaction is calculated by taking the number of coins
        in a transaction and multiplying it by the number of days it has been since those coins were last spent.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Cdd>`_
        """
        endpoint = "/v1/metrics/indicators/cdd"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def asol(self) -> pd.DataFrame:
        """
        Average Spent Output Lifespan (ASOL) is the average age (in days) of spent transaction outputs.
        Outputs with a lifespan of less than 1h are discarded.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Asol>`_
        """
        endpoint = "/v1/metrics/indicators/asol"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def msol(self) -> pd.DataFrame:
        """
        Median Spent Output Lifespan (MSOL) is the median age (in days) of spent transaction outputs.
        Outputs with a lifespan of less than 1h are discarded.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Msol>`_
        """
        endpoint = "/v1/metrics/indicators/msol"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def dormancy(self) -> pd.DataFrame:
        """
        Dormancy is the average number of days destroyed per coin transacted,
        and is defined as the ratio of coin days destroyed and total transfer volume.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.AverageDormancy>`_
        """
        endpoint = "/v1/metrics/indicators/average_dormancy"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def liveliness(self) -> pd.DataFrame:
        """
        Liveliness is defined as the ratio of the sum of Coin Days Destroyed and the sum of all coin days ever created.
        Liveliness increases as long term holder liquidate positions and decreases while they accumulate to HODL.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Liveliness>`_
        """
        endpoint = "/v1/metrics/indicators/liveliness"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def relative_unrealized_profit(self) -> pd.DataFrame:
        """
        Relative Unrealized Profit is defined as the total profit in USD of all coins in existence
        whose price at realisation time was lower than the current price normalised by the market cap.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.UnrealizedProfit>`_
        """
        endpoint = "/v1/metrics/indicators/unrealized_profit"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def relative_unrealized_loss(self) -> pd.DataFrame:
        """
        Relative Unrealized Loss is defined as the total loss in USD of all coins in existence
        whose price at realisation time was higher than the current price normalised by the market cap.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.UnrealizedLoss>`_
        """
        endpoint = "/v1/metrics/indicators/unrealized_loss"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def nupl(self) -> pd.DataFrame:
        """
        Net Unrealized Profit/Loss (NUPL) is the difference between Relative Unrealized Profit/Loss.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.NetUnrealizedProfitLoss>`_
        """
        endpoint = "/v1/metrics/indicators/net_unrealized_profit_loss"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def sth_nupl(self) -> pd.DataFrame:
        """
        Short Term Holder NUPL (STH-NUPL) is Net Unrealized Profit/Loss that takes into account only UTXOs
        younger than 155 days and serves as an indicator to assess the behaviour of short term investors.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.NuplLess155>`_
        """
        endpoint = "/v1/metrics/indicators/nupl_less_155"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def lth_nupl(self) -> pd.DataFrame:
        """
        Long Term Holder NUPL (LTH-NUPL) is Net Unrealized Profit/Loss that takes into account only UTXOs
        with a lifespan of at least 155 days and serves as an indicator to assess the behaviour of long term investors.
        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.NuplMore155>`_
        """
        endpoint = "/v1/metrics/indicators/nupl_more_155"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    @dataframe_with_inner_object
    def ssr(self) -> pd.DataFrame:
        """
        The Stablecoin Supply Ratio (SSR) is the ratio between Bitcoin supply and the supply of stablecoins denoted
        in BTC, or: Bitcoin Marketcap / Stablecoin Marketcap. We use the following stablecoins for the supply: USDT,
        TUSD, USDC, PAX, GUSD, DAI, SAI, and BUSD. When the SSR is low, the current stablecoin supply has more "buying power"
        to purchase BTC. It serves as a proxy for the supply/demand mechanics between BTC and USD. For more information see
        this article (https://medium.com/@glassnode/stablecoins-buying-power-over-bitcoin-3475c0d8779d).

        `View in Studio <https://studio.glassnode.com/metrics?a=BTC&m=indicators.Ssr>`_
        """
        endpoint = "/v1/metrics/indicators/ssr"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def bvin(self):
        """
        The Bitcoin Volatility Index (BVIN) is an implied volatility index that also represents the fair value of a bitcoin variance swap. The index is calculated by CryptoCompare using options data from Deribit and has been developed in collaboration with Carol Alexander and Arben Imeraj at the University of Sussex Business School. The index is suitable for use as a settlement price for bitcoin volatility futures. For more information on the methodology please see Alexander and Imeraj (2020).
        """
        endpoint = "/v1/metrics/indicators/bvin"
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
