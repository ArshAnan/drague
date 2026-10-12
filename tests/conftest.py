"""Run every test once with NumPy and once with jax.numpy (when JAX is installed)."""

import numpy as np
import pytest

BACKENDS = [pytest.param(np, id="numpy")]
try:
    import jax

    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    BACKENDS.append(pytest.param(jnp, id="jax"))
except ImportError:
    pass


@pytest.fixture(params=BACKENDS)
def xp(request):
    return request.param
