from .labellers.fixed_time_three_class_balanced import (
    FixedTimeHorionThreeClassBalancedEventLabeller,
)
from .labellers.fixed_time_three_class_imbalanced import (
    FixedTimeHorionThreeClassImbalancedEventLabeller,
)
from .labellers.fixed_time_two_class import FixedTimeHorionTwoClassEventLabeller

labellers_map = dict(
    two_class=FixedTimeHorionTwoClassEventLabeller,
    three_class_balanced=FixedTimeHorionThreeClassBalancedEventLabeller,
    three_class_imbalanced=FixedTimeHorionThreeClassImbalancedEventLabeller,
)
