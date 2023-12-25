#from jax.config import config

PRECISION = 'float'  # 'double'

DO_JIT = False

DEBUG = False

POL_CONV = 1.0
MAX_N_POL = 30

#def update_jax_precision(precision):
#    if precision == 'double':
#        config.update("jax_enable_x64", True)
#    else:
#        config.update("jax_enable_x64", False)


#update_jax_precision(PRECISION)

__all__ = ['PRECISION', 'DO_JIT', 'DEBUG', "update_jax_precision"]
