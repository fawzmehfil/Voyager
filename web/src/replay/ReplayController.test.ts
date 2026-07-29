// @vitest-environment jsdom

import { describe, expect, it, vi } from "vitest";
import { ReplayController } from "./ReplayController";
import { adaptV2ToRenderer, validateManifestV2 } from "./ReplaySource";

describe("ReplayController", () => {
  it("clamps exact seeks and exposes all playback state", () => {
    const controller = new ReplayController(300);
    const listener = vi.fn();
    controller.subscribe(listener);
    controller.update({
      tick: 450,
      speed: 4,
      playing: false,
      selectedAgent: "agent_7",
      panel: "agents",
      automaticCamera: false,
    });
    expect(controller.value).toMatchObject({
      tick: 300,
      speed: 4,
      playing: false,
      selectedAgent: "agent_7",
      panel: "agents",
      automaticCamera: false,
    });
    expect(listener).toHaveBeenCalledTimes(2);
  });
});

describe("replay runtime validation", () => {
  it("rejects unsupported major replay data", () => {
    expect(() =>
      validateManifestV2({
        replay_id: "future",
        world_steps: 10,
        tick_rate: 12,
        versions: { replay: "stage6_replay_3.0.0" },
      }),
    ).toThrow(/unsupported or malformed/i);
  });

  it("accepts extension-bearing v2 manifests", () => {
    const manifest = validateManifestV2({
      replay_id: "stage7-fixture",
      world_steps: 10000,
      tick_rate: 12,
      versions: { replay: "stage6_replay_2.1.0" },
      source: {},
      terminal_summary: {},
      registries: {},
      extensions: {
        "stage7.synthetic": {
          day_phase: "night",
          new_structure: "signal_tower",
          new_resource: "fiber",
          agent_count: 24,
        },
      },
    });
    expect(manifest.replay_id).toBe("stage7-fixture");
  });

  it("renders coherent fallbacks for future entities and more than ten agents", () => {
    const agents = Array.from({ length: 14 }, (_, index) => ({
      id: `agent_${index}`,
      name: `Explorer ${index}`,
      role: index === 13 ? "navigator" : "forager",
      appearance: {
        skin: "umber",
        hair: "crop",
        accent: "fern",
        accessory: "satchel",
        variant: index,
      },
      x: index % 4,
      y: Math.floor(index / 4),
      health: 100,
      hunger: 10,
      energy: 100,
      alive: true,
      inventory: { food: 0, wood: 0, stone: 0 },
    }));
    const manifest = validateManifestV2({
      replay_id: "future-visuals",
      world_steps: 1,
      agent_steps: 14,
      tick_rate: 12,
      versions: { replay: "stage6_replay_2.1.0", scenario: "stage7" },
      source: {
        policy_id: "future",
        policy_kind: "ppo",
        inference_mode: "deterministic",
        evaluation_seed: 7,
      },
      terminal_summary: {
        dense_return: 0,
        survivors: 14,
        deaths: 0,
        shelter_completion_step: null,
        camp_stockpile: { food: 0, wood: 0, stone: 0 },
        achievements: [],
        achievement_steps: {},
      },
      registries: {},
      manifest_sha256: "fixture",
    });
    const initial = {
      width: 2,
      height: 2,
      terrain: [["crystal_cave", "grass"], ["water", "night_ground"]],
      resources: [
        { id: "fiber", x: 0, y: 0, type: "fiber", quantity: 2 },
      ],
      structures: [{ id: "tower", type: "signal_tower", x: 1, y: 1 }],
      camp: {
        x: 1,
        y: 1,
        stockpile: { food: 0, wood: 0, stone: 0 },
        shelter_progress: 0,
      },
      agents,
    };
    const artifact = adaptV2ToRenderer(manifest, initial as never, []);
    expect(artifact.world.initial.agents).toHaveLength(14);
    expect(artifact.world.initial.agents[13].role).toBe("builder");
    expect(artifact.world.initial.terrain[0][0]).toBe("quarry");
    expect(artifact.world.initial.resources[0].type).toBe("stone");
  });
});
