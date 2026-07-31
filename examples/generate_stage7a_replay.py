"""Generate the committed Stage 7A five-minute vertical-slice replay."""

from voyager.replay.civilization import record_civilization_vertical_slice

if __name__ == "__main__":
    replay = record_civilization_vertical_slice(overwrite=True)
    print(replay)
