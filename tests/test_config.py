"""GameConfig rejects settings whose spawn could start an episode already crashed or tracking."""

import pytest

from drague.config import GameConfig


def test_defaults_valid_for_all_team_sizes():
    for P in range(1, 9):
        GameConfig(n_pursuers=P)


@pytest.mark.parametrize("kwargs", [
    {"n_pursuers": 0},
    {"spawn_ring_radius": 3.8},                   # 3.8 + 0.2*sqrt(2) pokes out of a 4 m arena
    {"spawn_height": 0.3},                        # 0.3 - 0.2 jitter is below z_min
    {"spawn_height": 2.4},                        # 2.4 + 0.2 jitter is above z_max
    {"spawn_ring_radius": 1.2},                   # could start inside the tracking radius
    {"n_pursuers": 32},                           # ring slots too close together
])
def test_rejects_unsafe_spawn(kwargs):
    with pytest.raises(ValueError):
        GameConfig(**kwargs)
