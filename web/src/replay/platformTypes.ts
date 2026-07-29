import type { ReplayArtifact } from "./types";

export interface CatalogReplay {
  replay_id: string;
  source: string;
  policy_id: string;
  policy_kind: string;
  inference_mode: string | null;
  seed: number;
  scenario: string;
  status: string;
  tags: string[];
  world_steps: number;
  tick_rate: number;
  terminal_summary: {
    survivors: number;
    deaths: number;
    dense_return: number;
    shelter_progress: number;
    achievements: string[];
    camp_stockpile: { food: number; wood: number; stone: number };
  };
}

export interface ReplayManifestV2 {
  replay_id: string;
  versions: Record<string, string>;
  source: {
    policy_kind: string;
    policy_id: string;
    inference_mode: string | null;
    evaluation_seed: number;
    training_seed: number | null;
    checkpoint_sha256: string | null;
    git_revision: string | null;
    dependency_versions: Record<string, string>;
  };
  environment_config: Record<string, number | string | boolean>;
  tick_rate: number;
  world_steps: number;
  agent_steps: number;
  tags: string[];
  terminal_summary: CatalogReplay["terminal_summary"] & {
    achievement_steps: Record<string, number>;
    shelter_completion_step: number | null;
    resource_flow: Record<string, unknown>;
  };
  registries: Record<string, Array<Record<string, unknown>>>;
  manifest_sha256: string;
  catalog_source: string;
}

export interface ReplayEvent {
  tick: number;
  type: string;
  importance: number;
  actors: string[];
  targets: string[];
  position?: { x: number; y: number };
  tags: string[];
  payload: Record<string, unknown>;
}

export interface ReplayAction {
  tick?: number;
  agent_id: string;
  role: string;
  selected_action: string;
  raw_action: string;
  action_mask: number[];
  invalid_probability_mass: number;
  reward: number;
  dense_reward_components: Record<string, number>;
}

export interface CameraCue {
  start_tick: number;
  end_tick: number;
  kind: string;
  target: { x: number; y: number };
  zoom: number;
  label: string;
}

export interface LoadedReplay {
  artifact: ReplayArtifact;
  manifest: ReplayManifestV2 | null;
  events: ReplayEvent[];
  actions: ReplayAction[];
  metrics: Record<string, unknown>;
  camera: { version: string; mode: string; cues: CameraCue[] };
  legacy: boolean;
}
