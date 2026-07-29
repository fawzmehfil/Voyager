import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { loadCatalog } from "./replay/ReplaySource";
import type { CatalogReplay } from "./replay/platformTypes";
import type { ReplayStatus } from "./replay/types";
import type { ReplayPlayerHandle } from "./components/ReplayPlayer";

const ReplayPlayer = lazy(() =>
  import("./components/ReplayPlayer").then((module) => ({
    default: module.ReplayPlayer,
  })),
);
const DEFAULT_REPLAY = "ppo_seed0_deterministic";

export function App() {
  const path = window.location.pathname;
  if (path === "/compare") return <ComparisonPage />;
  if (path.startsWith("/present/")) {
    return (
      <PresentationPage replayId={decodeURIComponent(path.slice("/present/".length))} />
    );
  }
  if (path.startsWith("/replays/")) {
    return <ReplayPage replayId={decodeURIComponent(path.slice("/replays/".length))} />;
  }
  return <ShowcasePage />;
}

function ShowcasePage() {
  const [catalogOpen, setCatalogOpen] = useState(false);
  return (
    <main className="page-shell">
      <header className="page-intro">
        <div>
          <p className="eyebrow">STAGE 6 · SAVED CIVILIZATIONS</p>
          <h1>VOYAGER</h1>
        </div>
        <div className="intro-actions">
          <p>
            TEN AGENTS. ONE ISLAND. <span>25 SECONDS TO BUILD A CIVILIZATION.</span>
          </p>
          <button type="button" onClick={() => setCatalogOpen((open) => !open)}>
            {catalogOpen ? "CLOSE RUNS" : "EXPLORE RUNS"}
          </button>
        </div>
      </header>
      {catalogOpen && <RunLibrary />}
      <PlayerFallback>
        <ReplayPlayer replayId={DEFAULT_REPLAY} />
      </PlayerFallback>
      <footer>
        <span>FROZEN STAGE 5.6 POLICY · RECORDED WORLD</span>
        <span>DRAG TO EXPLORE · PRESS I TO INSPECT</span>
      </footer>
    </main>
  );
}

function ReplayPage({ replayId }: { replayId: string }) {
  return (
    <main className="page-shell general-page">
      <CompactHeader replayId={replayId} />
      <PlayerFallback>
        <ReplayPlayer replayId={replayId} />
      </PlayerFallback>
      <footer>
        <a href="/">SHOWCASE</a>
        <a href={`/present/${encodeURIComponent(replayId)}`}>PRESENTATION MODE</a>
      </footer>
    </main>
  );
}

function PresentationPage({ replayId }: { replayId: string }) {
  return (
    <main className="presentation-page">
      <PlayerFallback>
        <ReplayPlayer replayId={replayId} presentation />
      </PlayerFallback>
    </main>
  );
}

function ComparisonPage() {
  const params = new URLSearchParams(window.location.search);
  const leftId = params.get("left") ?? "random";
  const rightId = params.get("right") ?? DEFAULT_REPLAY;
  const left = useRef<ReplayPlayerHandle>(null);
  const right = useRef<ReplayPlayerHandle>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [status, setStatus] = useState<Record<string, ReplayStatus>>({});
  const [compatible, setCompatible] = useState<string | null>(null);
  const syncing = useRef(false);

  useEffect(() => {
    fetch(`/api/v1/compare?left=${encodeURIComponent(leftId)}&right=${encodeURIComponent(rightId)}`)
      .then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json()) as {
            error?: { reasons?: string[]; message?: string };
            detail?: string;
          };
          throw new Error(
            payload.error?.reasons?.join(", ") ??
              payload.error?.message ??
              payload.detail ??
              "Runs are unavailable or incompatible.",
          );
        }
        setCompatible("");
      })
      .catch((error: unknown) =>
        setCompatible(error instanceof Error ? error.message : "Runs are incompatible."),
      );
  }, [leftId, rightId]);

  useEffect(() => {
    const leftTick = status.left?.step;
    const rightTick = status.right?.step;
    if (
      leftTick === undefined ||
      rightTick === undefined ||
      Math.abs(leftTick - rightTick) <= 1 ||
      syncing.current
    ) {
      return;
    }
    syncing.current = true;
    const target = Math.max(leftTick, rightTick);
    left.current?.setPlaying(false);
    right.current?.setPlaying(false);
    left.current?.seekTo(target);
    right.current?.seekTo(target);
    window.setTimeout(() => {
      left.current?.setPlaying(playing);
      right.current?.setPlaying(playing);
      syncing.current = false;
    }, 120);
  }, [playing, status]);

  const both = (action: (player: ReplayPlayerHandle) => void) => {
    if (left.current) action(left.current);
    if (right.current) action(right.current);
  };
  const leftStatus = status.left;
  const rightStatus = status.right;
  return (
    <main className="compare-page">
      <header className="compare-header">
        <a href="/">← VOYAGER</a>
        <div>
          <span>SEED-MATCHED POLICY COMPARISON</span>
          <h1>{leftId.replaceAll("_", " ")} <i>VS</i> {rightId.replaceAll("_", " ")}</h1>
        </div>
        <div className="compare-deltas">
          <strong>{(rightStatus?.alive ?? 0) - (leftStatus?.alive ?? 0) >= 0 ? "+" : ""}
            {(rightStatus?.alive ?? 0) - (leftStatus?.alive ?? 0)} SURVIVORS</strong>
          <span>{Math.round((rightStatus?.camp.shelter_progress ?? 0) * 100) -
            Math.round((leftStatus?.camp.shelter_progress ?? 0) * 100)}% SHELTER</span>
        </div>
      </header>
      {compatible ? (
        <div className="comparison-error">
          <strong>THESE WORLDS CANNOT BE SYNCHRONIZED</strong>
          <p>{compatible}</p>
          <a href="/">RETURN TO SHOWCASE</a>
        </div>
      ) : (
        <>
          <div className="comparison-grid">
            <div>
              <span className="side-label">A · {leftId.replaceAll("_", " ")}</span>
              <PlayerFallback>
                <ReplayPlayer
                  ref={left}
                  replayId={leftId}
                  compact
                  hideControls
                  onStatus={(value) =>
                    setStatus((current) => ({ ...current, left: value }))
                  }
                />
              </PlayerFallback>
            </div>
            <div>
              <span className="side-label">B · {rightId.replaceAll("_", " ")}</span>
              <PlayerFallback>
                <ReplayPlayer
                  ref={right}
                  replayId={rightId}
                  compact
                  hideControls
                  onStatus={(value) =>
                    setStatus((current) => ({ ...current, right: value }))
                  }
                />
              </PlayerFallback>
            </div>
          </div>
          <div className="comparison-controls">
            <button
              type="button"
              onClick={() => {
                const next = !playing;
                setPlaying(next);
                both((player) => player.setPlaying(next));
              }}
            >{playing ? "Ⅱ PAUSE BOTH" : "▶ PLAY BOTH"}</button>
            <button type="button" onClick={() => both((player) => player.stepBy(-1))}>‹ STEP</button>
            <button type="button" onClick={() => both((player) => player.stepBy(1))}>STEP ›</button>
            <button type="button" onClick={() => both((player) => player.restart())}>↺ RESTART</button>
            {[0.5, 1, 2, 4].map((value) => (
              <button
                type="button"
                key={value}
                className={speed === value ? "active" : ""}
                onClick={() => {
                  setSpeed(value);
                  both((player) => player.setSpeed(value));
                }}
              >{value}×</button>
            ))}
          </div>
        </>
      )}
    </main>
  );
}

function RunLibrary() {
  const [runs, setRuns] = useState<CatalogReplay[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    loadCatalog().then(setRuns).catch(() => setError("START THE VOYAGER SERVER TO BROWSE RUNS"));
  }, []);
  const visible = runs.filter((run) =>
    `${run.policy_id} ${run.seed} ${run.tags.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <section className="run-library">
      <div className="library-heading">
        <div><span>RECORDED ON THE SAME ISLAND</span><h2>CHOOSE A CIVILIZATION</h2></div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="SEARCH POLICY, SEED, TAG"
          aria-label="Search replays"
        />
      </div>
      {error && <p className="library-error">{error}</p>}
      <div className="run-cards">
        {visible.map((run) => (
          <a
            href={`/replays/${encodeURIComponent(run.replay_id)}`}
            key={run.replay_id}
            className={run.tags.includes("showcase") ? "showcase-card" : ""}
          >
            <IslandThumbnail replayId={run.replay_id} />
            <small>{run.policy_kind.toUpperCase()} · SEED {run.seed.toLocaleString()}</small>
            <strong>{run.policy_id.replaceAll("_", " ")}</strong>
            <div>
              <span>{run.terminal_summary.survivors} SURVIVED</span>
              <span>{run.terminal_summary.achievements.length} ACHIEVEMENTS</span>
            </div>
            <em>
              {run.tags.includes("showcase")
                ? "DEFAULT SHOWCASE"
                : run.tags.includes("failure-case")
                  ? "HONEST FAILURE"
                  : run.inference_mode?.toUpperCase() ?? "BASELINE"}
            </em>
          </a>
        ))}
      </div>
    </section>
  );
}

function IslandThumbnail({ replayId }: { replayId: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/v1/replays/${encodeURIComponent(replayId)}/initial`)
      .then((response) => response.json())
      .then((initial: { terrain: string[][]; camp: { x: number; y: number } }) => {
        if (cancelled || !canvas.current) return;
        const context = canvas.current.getContext("2d");
        if (!context) return;
        context.imageSmoothingEnabled = false;
        const colors: Record<string, string> = {
          water: "#287d86",
          beach: "#dba45e",
          grass: "#4a9346",
          forest: "#286a39",
          quarry: "#68736a",
        };
        initial.terrain.forEach((row, y) =>
          row.forEach((terrain, x) => {
            context.fillStyle = colors[terrain] ?? "#8a5ca0";
            context.fillRect(x, y, 1, 1);
          }),
        );
        context.fillStyle = "#f0c65d";
        context.fillRect(initial.camp.x - 1, initial.camp.y - 1, 3, 3);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [replayId]);
  return <canvas className="island-thumbnail" width={32} height={32} aria-hidden="true" />;
}

function CompactHeader({ replayId }: { replayId: string }) {
  return (
    <header className="compact-header">
      <a href="/"><span className="voyager-mark">V</span> VOYAGER</a>
      <div><span>SAVED REPLAY</span><h1>{replayId.replaceAll("_", " ")}</h1></div>
      <nav><a href="/">RUNS</a><a href={`/compare?left=random&right=${encodeURIComponent(replayId)}`}>COMPARE</a></nav>
    </header>
  );
}

function PlayerFallback({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="replay-frame app-loading" aria-label="Loading replay player">
          <span>LOADING REPLAY...</span>
        </div>
      }
    >{children}</Suspense>
  );
}
