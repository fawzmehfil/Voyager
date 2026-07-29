export type TerrainType = "water" | "beach" | "grass" | "forest" | "quarry";
export type ResourceType = "none" | "food" | "wood" | "stone";
export type AgentRole = "forager" | "woodcutter" | "builder";

export interface Stockpile {
  food: number;
  wood: number;
  stone: number;
}

export interface ReplayCamp {
  x: number;
  y: number;
  stockpile: Stockpile;
  shelter_progress: number;
}

export interface ReplayResource {
  x: number;
  y: number;
  type: ResourceType;
  quantity: number;
}

export interface ReplayAppearance {
  skin: string;
  hair: string;
  accent: string;
  accessory: string;
  variant: number;
}

export interface ReplayAgent {
  id: string;
  name: string;
  role: AgentRole;
  appearance: ReplayAppearance;
  x: number;
  y: number;
  health: number;
  hunger: number;
  energy: number;
  alive: boolean;
  inventory: Stockpile;
  action: string;
  event: string;
}

export interface ReplayFrame {
  step: number;
  storm: boolean;
  camp: ReplayCamp;
  agents: ReplayAgent[];
  resource_changes: ReplayResource[];
  new_achievements: string[];
}

export interface ReplayArtifact {
  schema_version: string;
  replay_id: string;
  tick_rate: number;
  duration_seconds: number;
  source: {
    benchmark_id: string;
    scenario_id: string;
    policy_id: string;
    policy_kind: string;
    inference_mode: string;
    training_seed: number;
    evaluation_seed: number;
    checkpoint: string;
    checkpoint_sha256: string;
    manifest_sha256: string;
  };
  world: {
    width: number;
    height: number;
    initial: {
      terrain: TerrainType[][];
      resources: ReplayResource[];
      camp: ReplayCamp;
      agents: ReplayAgent[];
    };
  };
  frames: ReplayFrame[];
  summary: {
    world_steps: number;
    agent_steps: number;
    dense_return: number;
    survivors: number;
    deaths: number;
    shelter_completion_step: number;
    camp_stockpile: Stockpile;
    achievements: string[];
    achievement_steps: Record<string, number>;
  };
}

export interface ReplayStatus {
  step: number;
  playing: boolean;
  ended: boolean;
  storm: boolean;
  alive: number;
  camp: ReplayCamp;
}
