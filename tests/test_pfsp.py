"""PFSP puts more weight on opponents the learner has not beaten."""

import numpy as np

from drague.algos.pfsp import prioritized_weights


def test_never_beaten_is_more_likely_than_always_beaten():
    weights = prioritized_weights([0.0, 1.0])
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert weights[0] > weights[1]


def test_empty_pool_has_no_weights():
    assert prioritized_weights([]).size == 0
    weights = prioritized_weights(np.ones(4))
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert np.all(weights > 0)
