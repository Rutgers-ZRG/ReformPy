__version__ = '2.0.0'

# Import main calculator
from reformpy.calculator import Reform_Calculator

# Import entropy wrapper for maximum entropy optimization
from reformpy.entropy_calculator import EntropyMaximizingCalculator

__all__ = [
    'Reform_Calculator',
    'EntropyMaximizingCalculator',
]
