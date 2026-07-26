const plannedRuns = [
  {
    name: "Random",
    detail: "Baseline rollout once Stage 1 exposes a working environment.",
  },
  {
    name: "Scripted",
    detail: "Greedy and cooperative policies planned for Stage 4.",
  },
  {
    name: "PPO",
    detail: "TensorFlow PPO training planned for Stage 5.",
  },
];

export function App() {
  return (
    <main className="shell">
      <section className="intro">
        <p className="eyebrow">Stage 0</p>
        <h1>Voyager</h1>
        <p className="subtitle">Multi-agent RL survival economy environment</p>
      </section>

      <section className="status-grid" aria-label="Project status">
        <article>
          <h2>Stage 0: Project Skeleton</h2>
          <p>
            The repository now has the Python package, documentation, examples, run
            conventions, and a static web placeholder.
          </p>
        </article>
        <article>
          <h2>Next: Single-agent Crafter-style prototype</h2>
          <p>
            Stage 1 will replace the placeholder environment with a deterministic
            island, basic survival state, resource gathering, and a working Gymnasium
            loop.
          </p>
        </article>
      </section>

      <section className="runs" aria-labelledby="runs-title">
        <div className="section-heading">
          <p className="eyebrow">Planned comparisons</p>
          <h2 id="runs-title">Run Slots</h2>
        </div>
        <div className="run-grid">
          {plannedRuns.map((run) => (
            <article className="run-card" key={run.name}>
              <h3>{run.name}</h3>
              <p>{run.detail}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
