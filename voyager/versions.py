"""Stable version identifiers for Voyager environments and benchmark artifacts."""

ENVIRONMENT_VERSION = "stage5.5"
SCENARIO_VERSION = "stage5_5_standard_300_v1"
DENSE_REWARD_VERSION = "stage5.5_economy_group_v1"
OBSERVATION_VERSION = "structured_210_v1"
ACTION_VERSION = "discrete_13_v1"
ACHIEVEMENT_VERSION = "stage5.6_16_v1"
BENCHMARK_SCHEMA_VERSION = "1.0.0"

DEFAULT_SCENARIO_CONFIG: dict[str, int | float] = {
    "num_agents": 10,
    "map_size": 32,
    "max_steps": 300,
    "local_view_size": 7,
    "inventory_capacity": 10,
    "storm_start_step": 200,
    "storm_interval": 200,
    "storm_duration": 25,
    "storm_damage": 1.0,
    "food_regen_interval": 50,
    "food_spawn_rate": 0.04,
}
