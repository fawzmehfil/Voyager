"""Export the frozen 25-second Stage 6A showcase replay."""

from voyager.replay import DEFAULT_OUTPUT_PATH, export_vertical_slice


def main() -> None:
    payload = export_vertical_slice()
    print(
        f"Exported {payload['replay_id']} to {DEFAULT_OUTPUT_PATH} "
        f"({payload['duration_seconds']:.0f}s)."
    )


if __name__ == "__main__":
    main()
