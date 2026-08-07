"""Generate the canonical VoyagerIsland-v1 Replay 2.3 demonstration."""

from pathlib import Path

from voyager.replay.island import record_island_oracle_replay

if __name__ == "__main__":
    output = record_island_oracle_replay(Path("runs/replays"), overwrite=True)
    print(output)
