import { useEffect, useRef, useState } from "react";
import Phaser from "phaser";
import { ReplayScene } from "../replay/ReplayScene";
import type {
  ReplayArtifact,
  ReplayStatus,
  Stockpile,
} from "../replay/types";

const REPLAY_PATH = `${import.meta.env.BASE_URL}replays/stage6_vertical_slice_v1.json`;

const emptyStockpile: Stockpile = { food: 0, wood: 0, stone: 0 };

const initialStatus: ReplayStatus = {
  step: 0,
  playing: true,
  ended: false,
  storm: false,
  alive: 10,
  camp: {
    x: 0,
    y: 0,
    stockpile: emptyStockpile,
    shelter_progress: 0,
  },
};

export function ReplayPlayer() {
  const gameHost = useRef<HTMLDivElement>(null);
  const game = useRef<Phaser.Game | null>(null);
  const scene = useRef<ReplayScene | null>(null);
  const [replay, setReplay] = useState<ReplayArtifact | null>(null);
  const [status, setStatus] = useState(initialStatus);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(REPLAY_PATH)
      .then((response) => {
        if (!response.ok) throw new Error(`Replay returned ${response.status}.`);
        return response.json() as Promise<ReplayArtifact>;
      })
      .then((artifact) => {
        if (!cancelled) {
          setReplay(artifact);
          setStatus((current) => ({
            ...current,
            alive: artifact.world.initial.agents.length,
            camp: artifact.world.initial.camp,
          }));
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Replay failed to load.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!replay || !gameHost.current || game.current) return;
    const replayScene = new ReplayScene(replay, { onStatus: setStatus });
    scene.current = replayScene;
    game.current = new Phaser.Game({
      type: Phaser.AUTO,
      parent: gameHost.current,
      width: 1280,
      height: 720,
      backgroundColor: "#1f6f78",
      pixelArt: true,
      antialias: false,
      roundPixels: true,
      render: {
        antialias: false,
        antialiasGL: false,
        pixelArt: true,
        roundPixels: true,
        powerPreference: "high-performance",
      },
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width: 1280,
        height: 720,
      },
      scene: [replayScene],
    });

    return () => {
      game.current?.destroy(true);
      game.current = null;
      scene.current = null;
    };
  }, [replay]);

  const maxSteps = replay?.summary.world_steps ?? 300;
  const progress = (status.step / maxSteps) * 100;
  const shelterPercent = Math.round(status.camp.shelter_progress * 100);

  const togglePlaying = () => {
    scene.current?.setPlaying(!status.playing);
  };

  const restart = () => {
    scene.current?.restartReplay();
  };

  const toggleSound = () => {
    const next = !soundEnabled;
    setSoundEnabled(next);
    scene.current?.setSoundEnabled(next);
  };

  return (
    <section className="replay-shell" aria-label="Voyager showcase replay">
      <div className="replay-frame">
        <div
          className="game-host"
          ref={gameHost}
          aria-label="Animated top-down replay of ten agents surviving on an island"
        >
          {!replay && !loadError && (
            <div className="loading-state">
              <span className="loading-coconut" aria-hidden="true" />
              <p>GENERATING ISLAND...</p>
            </div>
          )}
          {loadError && (
            <div className="loading-state error-state">
              <p>REPLAY COULD NOT LOAD</p>
              <small>{loadError}</small>
            </div>
          )}
        </div>

        <div className="hud-top" aria-live="polite">
          <div className="identity-panel">
            <span className="voyager-mark" aria-hidden="true">V</span>
            <div>
              <strong>VOYAGER</strong>
              <span>PPO · SEED 10,000,010</span>
            </div>
          </div>
          <div className="run-panel">
            <span>STEP {String(status.step).padStart(3, "0")} / 300</span>
            <span className={status.storm ? "storm-live" : ""}>
              {status.storm ? "STORM ACTIVE" : "CLEAR SKIES"}
            </span>
            <strong>{status.alive} ALIVE</strong>
          </div>
        </div>

        <div className="camp-hud">
          <div className="resource-readout food-readout">
            <span className="resource-pixel" aria-hidden="true" />
            <span>FOOD</span>
            <strong>{status.camp.stockpile.food}</strong>
          </div>
          <div className="resource-readout wood-readout">
            <span className="resource-pixel" aria-hidden="true" />
            <span>WOOD</span>
            <strong>{status.camp.stockpile.wood}</strong>
          </div>
          <div className="resource-readout stone-readout">
            <span className="resource-pixel" aria-hidden="true" />
            <span>STONE</span>
            <strong>{status.camp.stockpile.stone}</strong>
          </div>
          <div className="shelter-readout">
            <span>SHELTER</span>
            <div className="shelter-track" aria-hidden="true">
              <span style={{ width: `${shelterPercent}%` }} />
            </div>
            <strong>{shelterPercent}%</strong>
          </div>
        </div>

        <div className="transport">
          <button type="button" onClick={togglePlaying} disabled={!replay}>
            <span aria-hidden="true">{status.playing ? "Ⅱ" : "▶"}</span>
            {status.playing ? "PAUSE" : status.ended ? "PLAY AGAIN" : "PLAY"}
          </button>
          <button type="button" onClick={restart} disabled={!replay}>
            <span aria-hidden="true">↺</span>
            RESTART
          </button>
          <button
            type="button"
            className={soundEnabled ? "sound-on" : ""}
            onClick={toggleSound}
            disabled={!replay}
            aria-pressed={soundEnabled}
          >
            <span aria-hidden="true">{soundEnabled ? "♪" : "×"}</span>
            SOUND {soundEnabled ? "ON" : "OFF"}
          </button>
        </div>

        <div className="cinematic-progress" aria-hidden="true">
          <span style={{ width: `${progress}%` }} />
        </div>
      </div>
    </section>
  );
}
