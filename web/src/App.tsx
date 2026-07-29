import { lazy, Suspense } from "react";

const ReplayPlayer = lazy(() =>
  import("./components/ReplayPlayer").then((module) => ({
    default: module.ReplayPlayer,
  })),
);

export function App() {
  return (
    <main className="page-shell">
      <header className="page-intro">
        <div>
          <p className="eyebrow">STAGE 6A · SHOWCASE REPLAY</p>
          <h1>VOYAGER</h1>
        </div>
        <p>
          TEN AGENTS. ONE ISLAND. <span>25 SECONDS TO BUILD A CIVILIZATION.</span>
        </p>
      </header>
      <Suspense
        fallback={
          <div className="replay-frame app-loading" aria-label="Loading replay player">
            <span>LOADING REPLAY...</span>
          </div>
        }
      >
        <ReplayPlayer />
      </Suspense>
      <footer>
        <span>FROZEN STAGE 5.6 POLICY</span>
        <span>10 / 10 SURVIVORS · 16 / 16 ACHIEVEMENTS</span>
      </footer>
    </main>
  );
}
