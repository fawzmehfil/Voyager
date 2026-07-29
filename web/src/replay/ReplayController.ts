export interface ReplayViewState {
  tick: number;
  playing: boolean;
  speed: number;
  selectedAgent: string | null;
  panel: string | null;
  automaticCamera: boolean;
  muted: boolean;
}

export class ReplayController {
  private state: ReplayViewState;
  private listeners = new Set<(state: ReplayViewState) => void>();
  private urlTimer = 0;

  constructor(private readonly maxTick: number, initial?: Partial<ReplayViewState>) {
    this.state = {
      tick: 0,
      playing: true,
      speed: 1,
      selectedAgent: null,
      panel: null,
      automaticCamera: true,
      muted: true,
      ...initial,
    };
  }

  get value(): ReplayViewState {
    return this.state;
  }

  subscribe(listener: (state: ReplayViewState) => void): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  update(patch: Partial<ReplayViewState>, pushHistory = false): void {
    this.state = {
      ...this.state,
      ...patch,
      tick: Math.max(0, Math.min(this.maxTick, patch.tick ?? this.state.tick)),
    };
    this.listeners.forEach((listener) => listener(this.state));
    window.clearTimeout(this.urlTimer);
    this.urlTimer = window.setTimeout(() => this.persistUrl(pushHistory), 100);
  }

  private persistUrl(push: boolean): void {
    const url = new URL(window.location.href);
    url.searchParams.set("t", String(this.state.tick));
    url.searchParams.set("speed", String(this.state.speed));
    url.searchParams.set("camera", this.state.automaticCamera ? "auto" : "manual");
    url.searchParams.set("muted", String(this.state.muted));
    if (this.state.selectedAgent) url.searchParams.set("agent", this.state.selectedAgent);
    else url.searchParams.delete("agent");
    if (this.state.panel) url.searchParams.set("panel", this.state.panel);
    else url.searchParams.delete("panel");
    window.history[push ? "pushState" : "replaceState"]({}, "", url);
  }

  static fromUrl(maxTick: number): ReplayController {
    const params = new URLSearchParams(window.location.search);
    return new ReplayController(maxTick, {
      tick: Number(params.get("t") ?? 0),
      speed: Number(params.get("speed") ?? 1),
      selectedAgent: params.get("agent"),
      panel: params.get("panel"),
      automaticCamera: params.get("camera") !== "manual",
      muted: params.get("muted") !== "false",
      playing: params.get("autoplay") !== "false",
    });
  }
}
