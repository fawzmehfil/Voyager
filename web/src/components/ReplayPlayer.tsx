import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import Phaser from "phaser";
import { ReplayScene } from "../replay/ReplayScene";
import {
  ApiReplaySource,
  LegacyReplaySource,
  type ReplaySource,
} from "../replay/ReplaySource";
import type {
  LoadedReplay,
  ReplayEvent,
  ReplayManifestV2,
} from "../replay/platformTypes";
import type {
  ReplayAgent,
  ReplayArtifact,
  ReplayStatus,
  Stockpile,
} from "../replay/types";

const emptyStockpile: Stockpile = { food: 0, wood: 0, stone: 0 };
const initialStatus: ReplayStatus = {
  step: 0,
  playing: true,
  ended: false,
  storm: false,
  alive: 10,
  camp: { x: 0, y: 0, stockpile: emptyStockpile, shelter_progress: 0 },
};
const speeds = [0.25, 0.5, 1, 2, 4];
const tabs = [
  "agents",
  "events",
  "achievements",
  "metrics",
  "rewards",
  "diagnostics",
  "replay info",
];

export interface ReplayPlayerHandle {
  setPlaying: (playing: boolean) => void;
  seekTo: (tick: number) => void;
  stepBy: (delta: number) => void;
  restart: () => void;
  setSpeed: (speed: number) => void;
}

interface ReplayPlayerProps {
  replayId?: string;
  presentation?: boolean;
  compact?: boolean;
  hideControls?: boolean;
  onStatus?: (status: ReplayStatus) => void;
  onLoaded?: (loaded: LoadedReplay) => void;
}

export const ReplayPlayer = forwardRef<ReplayPlayerHandle, ReplayPlayerProps>(
  function ReplayPlayer(
    {
      replayId,
      presentation = false,
      compact = false,
      hideControls = false,
      onStatus,
      onLoaded,
    },
    ref,
  ) {
    const initialView = useRef(readViewState(compact));
    const gameHost = useRef<HTMLDivElement>(null);
    const game = useRef<Phaser.Game | null>(null);
    const scene = useRef<ReplayScene | null>(null);
    const onStatusRef = useRef(onStatus);
    const onLoadedRef = useRef(onLoaded);
    const [loaded, setLoaded] = useState<LoadedReplay | null>(null);
    const [status, setStatus] = useState(initialStatus);
    const [soundEnabled, setSoundEnabled] = useState(!initialView.current.muted);
    const [speed, setSpeedState] = useState(initialView.current.speed);
    const [automaticCamera, setAutomaticCamera] = useState(
      initialView.current.automaticCamera,
    );
    const [selectedAgent, setSelectedAgent] = useState<string | null>(
      initialView.current.selectedAgent,
    );
    const [activePanel, setActivePanel] = useState<string | null>(
      initialView.current.panel,
    );
    const [loadError, setLoadError] = useState("");

    const replay = loaded?.artifact ?? null;
    const manifest = loaded?.manifest ?? null;
    onStatusRef.current = onStatus;
    onLoadedRef.current = onLoaded;

    useEffect(() => {
      let cancelled = false;
      setLoaded(null);
      setLoadError("");
      const source: ReplaySource = replayId
        ? new ApiReplaySource(replayId)
        : new LegacyReplaySource();
      source
        .load()
        .catch(async (error: unknown) => {
          if (replayId === "ppo_seed0_deterministic") {
            return new LegacyReplaySource().load();
          }
          throw error;
        })
        .then((value) => {
          if (cancelled) return;
          setLoaded(value);
          onLoadedRef.current?.(value);
          setStatus((current) => ({
            ...current,
            alive: value.artifact.world.initial.agents.length,
            camp: value.artifact.world.initial.camp,
          }));
        })
        .catch((error: unknown) => {
          if (!cancelled) {
            setLoadError(
              error instanceof Error ? error.message : "Replay failed to load.",
            );
          }
        });
      return () => {
        cancelled = true;
        source.cancel?.();
      };
    }, [replayId]);

    useEffect(() => {
      if (!replay || !gameHost.current || game.current) return;
      const replayScene = new ReplayScene(
        replay,
        {
          onStatus: (next) => {
            setStatus(next);
            onStatusRef.current?.(next);
          },
          onSelectAgent: (agentId) => {
            setSelectedAgent(agentId);
            setActivePanel("agents");
            persistViewState({ selectedAgent: agentId, panel: "agents" }, true);
          },
          onManualCamera: () => {
            setAutomaticCamera(false);
            persistViewState({ automaticCamera: false }, false);
          },
        },
        loaded?.camera.cues ?? [],
      );
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
      const restoreTimer = window.setTimeout(() => {
        replayScene.setPlaybackRate(initialView.current.speed);
        replayScene.setAutomaticCamera(initialView.current.automaticCamera);
        if (initialView.current.selectedAgent) {
          replayScene.followAgent(initialView.current.selectedAgent);
        }
        if (initialView.current.tick > 0) {
          replayScene.seekTo(initialView.current.tick);
        }
        replayScene.setPlaying(initialView.current.playing);
      }, 80);
      return () => {
        window.clearTimeout(restoreTimer);
        game.current?.destroy(true);
        game.current = null;
        scene.current = null;
      };
    }, [replay]);

    useEffect(() => {
      if (compact || status.step % 4 !== 0) return;
      persistViewState({ tick: status.step }, false);
    }, [compact, status.step]);

    useEffect(() => {
      if (compact) return;
      const keydown = (event: KeyboardEvent) => {
        if (event.target instanceof HTMLInputElement) return;
        if (event.code === "Space") {
          event.preventDefault();
          const next = !status.playing;
          scene.current?.setPlaying(next);
          persistViewState({ playing: next }, false);
        } else if (event.key === "ArrowLeft") scene.current?.stepBy(-1);
        else if (event.key === "ArrowRight") scene.current?.stepBy(1);
        else if (event.key.toLowerCase() === "r") scene.current?.restartReplay();
        else if (event.key.toLowerCase() === "c") returnToStory();
        else if (event.key.toLowerCase() === "i") {
          setActivePanel((current) => (current ? null : "agents"));
        }
      };
      window.addEventListener("keydown", keydown);
      return () => window.removeEventListener("keydown", keydown);
    }, [compact, status.playing]);

    useImperativeHandle(ref, () => ({
      setPlaying: (playing) => scene.current?.setPlaying(playing),
      seekTo: (tick) => scene.current?.seekTo(tick),
      stepBy: (delta) => scene.current?.stepBy(delta),
      restart: () => scene.current?.restartReplay(),
      setSpeed: (next) => scene.current?.setPlaybackRate(next),
    }));

    const maxSteps = replay?.summary.world_steps ?? 300;
    const shelterPercent = Math.round(status.camp.shelter_progress * 100);
    const currentAgents = useMemo(
      () => agentsAt(replay, status.step),
      [replay, status.step],
    );
    const currentSelected =
      currentAgents.find((agent) => agent.id === selectedAgent) ?? null;
    const markers = useMemo(
      () => markerEvents(loaded?.events ?? [], maxSteps),
      [loaded?.events, maxSteps],
    );

    const seek = (tick: number, history = false) => {
      scene.current?.seekTo(tick);
      persistViewState({ tick }, history);
    };
    const selectSpeed = (next: number) => {
      setSpeedState(next);
      scene.current?.setPlaybackRate(next);
      persistViewState({ speed: next }, false);
    };
    const returnToStory = () => {
      setAutomaticCamera(true);
      scene.current?.setAutomaticCamera(true);
      persistViewState({ automaticCamera: true }, false);
    };
    const selectAgent = (agentId: string) => {
      setSelectedAgent(agentId);
      scene.current?.followAgent(agentId);
      persistViewState({ selectedAgent: agentId }, true);
    };

    return (
      <section
        className={`replay-shell ${presentation ? "presentation-player" : ""} ${compact ? "compact-player" : ""}`}
        aria-label="Voyager replay"
      >
        <div className="replay-layout">
          <div className="replay-frame">
            <div
              className="game-host"
              ref={gameHost}
              aria-label="Animated top-down replay of agents surviving on an island"
            >
              {!replay && !loadError && (
                <div className="loading-state">
                  <span className="loading-coconut" aria-hidden="true" />
                  <p>CHARTING ISLAND...</p>
                </div>
              )}
              {loadError && (
                <div className="loading-state error-state">
                  <p>REPLAY COULD NOT LOAD</p>
                  <small>{loadError}</small>
                </div>
              )}
            </div>

            {!presentation && (
              <div className="hud-top" aria-live="polite">
                <div className="identity-panel">
                  <span className="voyager-mark" aria-hidden="true">V</span>
                  <div>
                    <strong>VOYAGER</strong>
                    <span>
                      {(manifest?.source.policy_id ?? replay?.source.policy_id ?? "LOADING")
                        .replaceAll("_", " ")
                        .toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="run-panel">
                  <span>STEP {String(status.step).padStart(3, "0")} / {maxSteps}</span>
                  <span className={status.storm ? "storm-live" : ""}>
                    {status.storm ? "STORM ACTIVE" : "CLEAR SKIES"}
                  </span>
                  <strong>{status.alive} ALIVE</strong>
                </div>
              </div>
            )}

            <CampHud status={status} shelterPercent={shelterPercent} />

            {!hideControls && (
              <>
                <div className="transport">
                  <button
                    type="button"
                    onClick={() => scene.current?.stepBy(-1)}
                    disabled={!replay}
                    aria-label="Previous tick"
                  >‹</button>
                  <button
                    type="button"
                    onClick={() => {
                      const next = !status.playing;
                      scene.current?.setPlaying(next);
                      persistViewState({ playing: next }, false);
                    }}
                    disabled={!replay}
                  >
                    {status.playing ? "Ⅱ PAUSE" : "▶ PLAY"}
                  </button>
                  <button
                    type="button"
                    onClick={() => scene.current?.stepBy(1)}
                    disabled={!replay}
                    aria-label="Next tick"
                  >›</button>
                  <button
                    type="button"
                    onClick={() => scene.current?.restartReplay()}
                    disabled={!replay}
                  >↺</button>
                  {!compact && (
                    <>
                      <button
                        type="button"
                        className={automaticCamera ? "sound-on" : ""}
                        onClick={() => {
                          const next = !automaticCamera;
                          setAutomaticCamera(next);
                          scene.current?.setAutomaticCamera(next);
                          persistViewState({ automaticCamera: next }, false);
                        }}
                      >CAM {automaticCamera ? "AUTO" : "FREE"}</button>
                      {!automaticCamera && (
                        <button type="button" onClick={returnToStory}>RETURN TO STORY</button>
                      )}
                      <button
                        type="button"
                        className={soundEnabled ? "sound-on" : ""}
                        onClick={() => {
                          const next = !soundEnabled;
                          setSoundEnabled(next);
                          scene.current?.setSoundEnabled(next);
                          persistViewState({ muted: !next }, false);
                        }}
                      >{soundEnabled ? "♪ ON" : "× SOUND"}</button>
                      <button
                        type="button"
                        onClick={() =>
                          setActivePanel((current) => {
                            const next = current ? null : "agents";
                            persistViewState({ panel: next }, true);
                            return next;
                          })
                        }
                      >{activePanel ? "CLOSE" : "INSPECT"}</button>
                    </>
                  )}
                </div>

                <div className="timeline-control">
                  <input
                    type="range"
                    min={0}
                    max={maxSteps}
                    value={status.step}
                    onChange={(event) => seek(Number(event.target.value))}
                    aria-label="Replay tick"
                  />
                  {markers.map((marker) => (
                    <button
                      key={`${marker.tick}-${marker.type}`}
                      className={`timeline-marker ${marker.type}`}
                      style={{ left: `${(marker.tick / maxSteps) * 100}%` }}
                      onClick={() => seek(marker.tick, true)}
                      title={`${marker.label} · tick ${marker.tick}`}
                      aria-label={`Jump to ${marker.label} at tick ${marker.tick}`}
                    />
                  ))}
                </div>

                {!compact && (
                  <div className="speed-control" aria-label="Playback speed">
                    {speeds.map((value) => (
                      <button
                        type="button"
                        key={value}
                        className={speed === value ? "active" : ""}
                        onClick={() => selectSpeed(value)}
                      >{value}×</button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          {activePanel && !presentation && !compact && replay && loaded && (
            <InspectorDrawer
              activePanel={activePanel}
              setActivePanel={(panel) => {
                setActivePanel(panel);
                persistViewState({ panel }, true);
              }}
              loaded={loaded}
              agents={currentAgents}
              selectedAgent={currentSelected}
              selectAgent={selectAgent}
              seek={seek}
              tick={status.step}
            />
          )}
        </div>
      </section>
    );
  },
);

function CampHud({
  status,
  shelterPercent,
}: {
  status: ReplayStatus;
  shelterPercent: number;
}) {
  return (
    <div className="camp-hud">
      {(["food", "wood", "stone"] as const).map((resource) => (
        <div className={`resource-readout ${resource}-readout`} key={resource}>
          <span className="resource-pixel" aria-hidden="true" />
          <span>{resource.toUpperCase()}</span>
          <strong>{status.camp.stockpile[resource]}</strong>
        </div>
      ))}
      <div className="shelter-readout">
        <span>SHELTER</span>
        <div className="shelter-track" aria-hidden="true">
          <span style={{ width: `${shelterPercent}%` }} />
        </div>
        <strong>{shelterPercent}%</strong>
      </div>
    </div>
  );
}

interface InspectorProps {
  activePanel: string;
  setActivePanel: (panel: string | null) => void;
  loaded: LoadedReplay;
  agents: ReplayAgent[];
  selectedAgent: ReplayAgent | null;
  selectAgent: (id: string) => void;
  seek: (tick: number, history?: boolean) => void;
  tick: number;
}

function InspectorDrawer(props: InspectorProps) {
  return (
    <aside className="inspector" aria-label="Replay inspection">
      <div className="inspector-tabs">
        {tabs.map((tab) => (
          <button
            type="button"
            key={tab}
            className={props.activePanel === tab ? "active" : ""}
            onClick={() => props.setActivePanel(tab)}
          >{tab.toUpperCase()}</button>
        ))}
      </div>
      <button
        className="drawer-close"
        type="button"
        onClick={() => props.setActivePanel(null)}
        aria-label="Close inspector"
      >×</button>
      <div className="inspector-content">
        {props.activePanel === "agents" && <AgentsPanel {...props} />}
        {props.activePanel === "events" && <EventsPanel {...props} />}
        {props.activePanel === "achievements" && <AchievementsPanel {...props} />}
        {props.activePanel === "metrics" && <MetricsPanel {...props} />}
        {props.activePanel === "rewards" && <RewardsPanel {...props} />}
        {props.activePanel === "diagnostics" && <DiagnosticsPanel {...props} />}
        {props.activePanel === "replay info" && <ReplayInfoPanel {...props} />}
      </div>
    </aside>
  );
}

function AgentsPanel({ agents, selectedAgent, selectAgent, tick }: InspectorProps) {
  return (
    <div>
      <PanelTitle eyebrow={`TICK ${tick}`} title="CREW ROSTER" />
      <div className="agent-list">
        {agents.map((agent) => (
          <button
            type="button"
            key={agent.id}
            className={selectedAgent?.id === agent.id ? "selected" : ""}
            onClick={() => selectAgent(agent.id)}
          >
            <span className={`role-dot ${agent.role}`} />
            <span><strong>{agent.name}</strong><small>{agent.role} · {agent.action}</small></span>
            <span className={agent.alive ? "alive" : "dead"}>
              {agent.alive ? `${Math.round(agent.health)} HP` : "LOST"}
            </span>
          </button>
        ))}
      </div>
      {selectedAgent && (
        <div className="agent-card">
          <h3>{selectedAgent.name}</h3>
          <p>{selectedAgent.role.toUpperCase()} · {selectedAgent.action.replaceAll("_", " ")}</p>
          <Meter label="HEALTH" value={selectedAgent.health} color="#d9504d" />
          <Meter label="HUNGER" value={100 - selectedAgent.hunger} color="#e2b94f" />
          <Meter label="ENERGY" value={selectedAgent.energy} color="#4eaec0" />
          <p>
            PACK · F{selectedAgent.inventory.food} W{selectedAgent.inventory.wood} S
            {selectedAgent.inventory.stone}
          </p>
        </div>
      )}
    </div>
  );
}

function EventsPanel({ loaded, seek }: InspectorProps) {
  return (
    <div>
      <PanelTitle eyebrow={`${loaded.events.length} RECORDED`} title="WORLD EVENTS" />
      <div className="event-list">
        {loaded.events.slice().reverse().map((event, index) => (
          <button
            type="button"
            key={`${event.tick}-${event.type}-${index}`}
            onClick={() => seek(event.tick, true)}
          >
            <span>{String(event.tick).padStart(3, "0")}</span>
            <strong>{eventLabel(event)}</strong>
            <small>{event.actors.join(", ") || "WORLD"}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function AchievementsPanel({ loaded, tick }: InspectorProps) {
  const steps = loaded.artifact.summary.achievement_steps;
  const registry =
    loaded.manifest?.registries.achievements?.map((entry) => String(entry.id)) ??
    loaded.artifact.summary.achievements;
  return (
    <div>
      <PanelTitle
        eyebrow={`${Object.values(steps).filter((step) => step <= tick).length} / ${registry.length} UNLOCKED`}
        title="ACHIEVEMENTS"
      />
      <div className="achievement-grid">
        {registry.map((id) => {
          const unlock = steps[id];
          const active = unlock !== undefined && unlock <= tick;
          return (
            <div key={id} className={active ? "unlocked" : ""}>
              <span>{active ? "◆" : "◇"}</span>
              <strong>{id.replaceAll("_", " ")}</strong>
              <small>{unlock === undefined ? "LOCKED" : `TICK ${unlock}`}</small>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricsPanel({ loaded }: InspectorProps) {
  const global = (loaded.metrics.global ?? {}) as Record<string, number[]>;
  return (
    <div>
      <PanelTitle eyebrow="RECORDED SERIES" title="CIVILIZATION" />
      {["survivors", "camp_food", "shelter_progress", "mean_health"].map((name) => (
        <Sparkline key={name} label={name} values={global[name] ?? []} />
      ))}
    </div>
  );
}

function RewardsPanel({ loaded }: InspectorProps) {
  const totals = (loaded.metrics.reward_components ?? {}) as Record<string, number>;
  return (
    <div>
      <PanelTitle eyebrow="DENSE COMPONENTS" title="REWARDS" />
      <dl className="value-table">
        {Object.entries(totals).map(([name, value]) => (
          <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd>{value.toFixed(2)}</dd></div>
        ))}
      </dl>
    </div>
  );
}

function DiagnosticsPanel({ loaded, tick }: InspectorProps) {
  const current = loaded.actions.filter((action) => action.tick === tick);
  const invalid = current.reduce(
    (total, action) => total + action.invalid_probability_mass,
    0,
  );
  return (
    <div>
      <PanelTitle eyebrow="POLICY TRACE" title="DIAGNOSTICS" />
      <div className="diagnostic-hero">
        <strong>{invalid.toFixed(4)}</strong>
        <span>INVALID LOGIT MASS · TICK {tick}</span>
      </div>
      <div className="event-list">
        {current.map((action) => (
          <div className="diagnostic-row" key={action.agent_id}>
            <span>{action.agent_id}</span>
            <strong>{action.selected_action.replaceAll("_", " ")}</strong>
            <small>RAW {action.raw_action.replaceAll("_", " ")}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReplayInfoPanel({ loaded }: InspectorProps) {
  const manifest = loaded.manifest;
  return (
    <div>
      <PanelTitle eyebrow={loaded.legacy ? "LEGACY FIXTURE" : "VALIDATED V2"} title="REPLAY INFO" />
      <dl className="value-table">
        <Info label="Replay" value={loaded.artifact.replay_id} />
        <Info label="Policy" value={loaded.artifact.source.policy_id} />
        <Info label="Seed" value={String(loaded.artifact.source.evaluation_seed)} />
        <Info label="Scenario" value={loaded.artifact.source.scenario_id} />
        <Info label="Inference" value={loaded.artifact.source.inference_mode} />
        <Info label="Replay format" value={loaded.artifact.schema_version} />
        <Info label="Git" value={manifest?.source.git_revision?.slice(0, 12) ?? "legacy"} />
        <Info label="Checksum" value={manifest?.manifest_sha256.slice(0, 14) ?? "legacy fixture"} />
      </dl>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return <div><dt>{label}</dt><dd>{value}</dd></div>;
}

function PanelTitle({ eyebrow, title }: { eyebrow: string; title: string }) {
  return <header className="panel-title"><span>{eyebrow}</span><h2>{title}</h2></header>;
}

function Meter({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="agent-meter">
      <span>{label}</span>
      <i><b style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }} /></i>
      <strong>{Math.round(value)}</strong>
    </div>
  );
}

function Sparkline({ label, values }: { label: string; values: number[] }) {
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const points = values
    .filter((_, index) => index % Math.max(1, Math.floor(values.length / 80)) === 0)
    .map((value, index, sampled) => {
      const x = (index / Math.max(1, sampled.length - 1)) * 280;
      const y = 62 - ((value - min) / Math.max(1e-6, max - min)) * 54;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <div className="sparkline">
      <span>{label.replaceAll("_", " ").toUpperCase()}</span>
      <strong>{values.at(-1)?.toFixed(1) ?? "—"}</strong>
      <svg viewBox="0 0 280 66" role="img" aria-label={`${label} metric history`}>
        <polyline points={points} />
      </svg>
    </div>
  );
}

function agentsAt(replay: ReplayArtifact | null, tick: number): ReplayAgent[] {
  if (!replay) return [];
  if (tick <= 0) return replay.world.initial.agents;
  return replay.frames[Math.min(tick, replay.frames.length) - 1]?.agents ??
    replay.world.initial.agents;
}

function markerEvents(events: ReplayEvent[], maxSteps: number) {
  const important = events.filter(
    (event) => event.importance >= 68 || event.type.startsWith("storm"),
  );
  if (important.length) {
    return important.map((event) => ({
      tick: event.tick,
      type: event.type,
      label: eventLabel(event),
    }));
  }
  return [
    { tick: Math.min(115, maxSteps), type: "build", label: "Shelter complete" },
    { tick: Math.min(200, maxSteps), type: "storm_started", label: "Storm" },
    { tick: maxSteps, type: "finale", label: "Finale" },
  ];
}

function eventLabel(event: ReplayEvent): string {
  const achievement = event.payload.achievement_id;
  return String(achievement ?? event.type).replaceAll("_", " ").toUpperCase();
}

interface ViewState {
  tick: number;
  speed: number;
  selectedAgent: string | null;
  panel: string | null;
  automaticCamera: boolean;
  muted: boolean;
  playing: boolean;
}

function readViewState(compact: boolean): ViewState {
  if (compact) {
    return {
      tick: 0,
      speed: 1,
      selectedAgent: null,
      panel: null,
      automaticCamera: true,
      muted: true,
      playing: true,
    };
  }
  const params = new URLSearchParams(window.location.search);
  const speed = Number(params.get("speed") ?? 1);
  return {
    tick: Math.max(0, Number(params.get("t") ?? 0)),
    speed: speeds.includes(speed) ? speed : 1,
    selectedAgent: params.get("agent"),
    panel: params.get("panel"),
    automaticCamera: params.get("camera") !== "manual",
    muted: params.get("muted") !== "false",
    playing: params.get("autoplay") !== "false",
  };
}

function persistViewState(patch: Partial<ViewState>, pushHistory: boolean) {
  const url = new URL(window.location.href);
  if (patch.tick !== undefined) url.searchParams.set("t", String(patch.tick));
  if (patch.speed !== undefined) url.searchParams.set("speed", String(patch.speed));
  if (patch.selectedAgent !== undefined) {
    if (patch.selectedAgent) url.searchParams.set("agent", patch.selectedAgent);
    else url.searchParams.delete("agent");
  }
  if (patch.panel !== undefined) {
    if (patch.panel) url.searchParams.set("panel", patch.panel);
    else url.searchParams.delete("panel");
  }
  if (patch.automaticCamera !== undefined) {
    url.searchParams.set("camera", patch.automaticCamera ? "auto" : "manual");
  }
  if (patch.muted !== undefined) url.searchParams.set("muted", String(patch.muted));
  if (patch.playing !== undefined) {
    url.searchParams.set("autoplay", String(patch.playing));
  }
  window.history[pushHistory ? "pushState" : "replaceState"]({}, "", url);
}
