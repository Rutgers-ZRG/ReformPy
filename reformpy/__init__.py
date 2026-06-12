__version__ = '2.0.0'

__all__ = []

# The core calculators pull in heavy runtime dependencies (mpi4py, libfp).
# Import them defensively so that optional, self-contained components -- in
# particular the vendored ``reformpy.torch_fplib`` subpackage installed via
# ``pip install reformpy[torch]`` -- stay importable even when the core stack
# is not present. This mirrors the libfp -> libfppy fallback in calculator.py.
try:
    from reformpy.calculator import Reform_Calculator
    from reformpy.entropy_calculator import EntropyMaximizingCalculator
    __all__ += ['Reform_Calculator', 'EntropyMaximizingCalculator']
except ImportError as _exc:
    import warnings as _warnings
    _warnings.warn(
        "reformpy: core calculators (Reform_Calculator, "
        "EntropyMaximizingCalculator) are unavailable: {}. Install the core "
        "dependencies (e.g. mpi4py, libfp) to use them.".format(_exc),
        stacklevel=2,
    )
