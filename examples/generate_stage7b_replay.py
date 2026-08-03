"""Generate the committed Stage 7B deterministic-core replay."""

from voyager.replay.civilization_v2 import record_civilization_deterministic_core

if __name__ == "__main__":
    replay = record_civilization_deterministic_core(overwrite=True)
    print(replay)
