__version__ = '1.4.0'

# Import main calculator
from reformpy.calculator import Reform_Calculator

# Import entropy wrapper for maximum entropy optimization
from reformpy.entropy_calculator import (
    EntropyMaximizingCalculator,
    wrap_calculator_with_entropy
)

__all__ = [
    'Reform_Calculator',
    'EntropyMaximizingCalculator',
    'wrap_calculator_with_entropy',
]
