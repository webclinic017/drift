from .utils import *


class ETH2:
    """
        ETH 2.0 class.

        Methods
        -------
        __init__(glassnode_client):
            Constructs an ETH2 object.
        new_deposits():
            Returns the number transactions depositing 32 ETH to the ETH2 deposit contract.
        new_value_staked():
            Returns the amount of ETH transferred to the ETH2 deposit contract.
        new_validators():
            Returns the number of new validators depositing 32 ETH to the ETH2 deposit contract.
        total_number_of_deposits():
            Returns the total number of transactions to the ETH2 deposit contract.
        total_value_staked():
            Returns the amount of ETH deposited to the ETH2 deposit contract.
        total_number_of_validators():
            Returns the total number of unique validators.
        phase_zero_staking_goal():
            Returns the percentage of the Phase 0 staking goal.
    """
    def __init__(self, glassnode_client):
        self._gc = glassnode_client

    def new_deposits(self) -> pd.DataFrame:
        """
        The number transactions depositing 32 ETH to the ETH2 deposit contract.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingDepositsCount>`_

        :return: A DataFrame with ETH2 deposit data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_deposits_count'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def new_value_staked(self) -> pd.DataFrame:
        """
        The amount of ETH transferred to the ETH2 deposit contract.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingVolumeSum>`_

        :return: A DataFrame with staked value data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_volume_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def new_validators(self) -> pd.DataFrame:
        """
        The number of new validators (accounts) depositing 32 ETH to the ETH2 deposit contract.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingValidatorsCount>`_

        :return: A DataFrame with new validators data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_validators_count'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def total_number_of_deposits(self) -> pd.DataFrame:
        """
        The total number of transactions to the ETH2 deposit contract.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingTotalDepositsCount>`_

        :return: A DataFrame with ETH2 deposit data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_total_deposits_count'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def total_value_staked(self) -> pd.DataFrame:
        """
        The amount of ETH that has been deposited to the ETH2 deposit contract,
        the current ETH balance on the ETH2 deposit contract.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingTotalVolumeSum>`_

        :return: A DataFrame with ETH2 deposit data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_total_volume_sum'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def total_number_of_validators(self) -> pd.DataFrame:
        """
        The total number of unique validators (accounts) that have deposited 32 ETH to the ETH2 deposit contract.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingTotalValidatorsCount>`_

        :return: A DataFrame with ETH2 deposit data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_total_validators_count'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))

    def phase_zero_staking_goal(self) -> pd.DataFrame:
        """
        The percentage of the Phase 0 staking goal.
        `View in Studio <https://studio.glassnode.com/metrics?a=ETH&m=eth2.StakingPhase0GoalPercent>`_

        :return: A DataFrame with staking goal data.
        :rtype: DataFrame
        """
        endpoint = '/v1/metrics/eth2/staking_phase_0_goal_percent'
        if not is_supported_by_endpoint(self._gc, endpoint):
            return pd.DataFrame()

        return response_to_dataframe(self._gc.get(endpoint))
