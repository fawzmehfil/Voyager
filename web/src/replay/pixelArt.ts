import Phaser from "phaser";
import type { AgentRole, ReplayAgent, TerrainType } from "./types";

type PixelContext = CanvasRenderingContext2D;

const PALETTE = {
  ink: "#17221d",
  deepInk: "#0b1616",
  waterDark: "#1f6f78",
  water: "#2d96a0",
  waterLight: "#65c3bd",
  foam: "#d9ead0",
  sandDark: "#b97d45",
  sand: "#dda85e",
  sandLight: "#f2ce7b",
  grassDark: "#2e6e3b",
  grass: "#4c9646",
  grassLight: "#79b84a",
  leafDark: "#174b2a",
  leaf: "#26703a",
  leafLight: "#55a947",
  stoneDark: "#414d4d",
  stone: "#67716a",
  stoneLight: "#9a9a7f",
  woodDark: "#5a3527",
  wood: "#8f5733",
  woodLight: "#c17a48",
  rope: "#d8b46b",
  fire: "#f4772d",
  fireLight: "#ffd15c",
  white: "#f6ebcf",
};

const SKINS: Record<string, [string, string, string]> = {
  deep: ["#5a2f2b", "#7d4437", "#ad6a4e"],
  umber: ["#70402f", "#985c43", "#ca805a"],
  copper: ["#844b35", "#b16b4a", "#df9569"],
  golden: ["#9a6242", "#ca895e", "#efb87e"],
};

const ACCENTS: Record<string, string> = {
  fern: "#5daa45",
  coral: "#e96f5f",
  lagoon: "#37aeb0",
  mango: "#f2a43a",
  sky: "#62a9dc",
  hibiscus: "#d95983",
  lime: "#99c94d",
  violet: "#8b70c7",
  sun: "#eccf4f",
  mint: "#72c99e",
};

const HAIR = ["#2c211e", "#3b2520", "#4b2d21", "#201d22"];

function canvasTexture(
  scene: Phaser.Scene,
  key: string,
  width: number,
  height: number,
  draw: (context: PixelContext) => void,
) {
  if (scene.textures.exists(key)) {
    return;
  }
  const texture = scene.textures.createCanvas(key, width, height);
  if (!texture) {
    throw new Error(`Could not create texture ${key}.`);
  }
  const context = texture.context;
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, width, height);
  draw(context);
  texture.refresh();
}

function fill(
  context: PixelContext,
  color: string,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  context.fillStyle = color;
  context.fillRect(x, y, width, height);
}

function hash(x: number, y: number, salt = 0) {
  let value = (x * 374761393 + y * 668265263 + salt * 982451653) >>> 0;
  value = (value ^ (value >>> 13)) * 1274126177;
  return (value ^ (value >>> 16)) >>> 0;
}

function drawTerrain(context: PixelContext, terrain: TerrainType, variant: number) {
  if (terrain === "water") {
    fill(context, PALETTE.waterDark, 0, 0, 16, 16);
    fill(context, PALETTE.water, 0, 2 + variant, 16, 10);
    fill(context, PALETTE.waterLight, 2, 4 + variant, 5, 1);
    fill(context, PALETTE.waterLight, 10, 10 - variant, 4, 1);
    fill(context, PALETTE.waterDark, 5, 14, 6, 1);
    return;
  }

  const base = terrain === "beach" ? PALETTE.sand : PALETTE.grass;
  fill(context, base, 0, 0, 16, 16);
  if (terrain === "beach") {
    fill(context, PALETTE.sandLight, 2 + variant, 3, 1, 1);
    fill(context, PALETTE.sandDark, 11, 8 + variant, 1, 1);
    fill(context, PALETTE.sandLight, 6, 13, 2, 1);
    return;
  }

  fill(context, PALETTE.grassLight, 3 + variant, 4, 1, 2);
  fill(context, PALETTE.grassDark, 12, 11 - variant, 1, 2);
  fill(context, PALETTE.grassLight, 8, 14, 2, 1);
  if (terrain === "forest") {
    fill(context, PALETTE.leafDark, 1, 1, 4, 3);
    fill(context, PALETTE.leaf, 12, 3, 3, 4);
  }
  if (terrain === "quarry") {
    fill(context, PALETTE.stoneDark, 2, 10, 4, 3);
    fill(context, PALETTE.stone, 3, 9, 4, 3);
    fill(context, PALETTE.stoneLight, 4, 9, 2, 1);
  }
}

function drawFoodNode(context: PixelContext, quantity: number) {
  fill(context, PALETTE.deepInk, 4, 12, 10, 4);
  fill(context, PALETTE.leafDark, 2, 9, 14, 5);
  fill(context, PALETTE.leafDark, 5, 6, 8, 9);
  fill(context, PALETTE.leaf, 3, 8, 5, 5);
  fill(context, PALETTE.leaf, 8, 5, 6, 8);
  fill(context, PALETTE.leafLight, 6, 7, 3, 2);
  fill(context, PALETTE.leafLight, 11, 6, 2, 3);
  const berries = Math.max(1, quantity);
  for (let index = 0; index < berries; index += 1) {
    fill(context, "#e85e54", 5 + index * 4, 10 + (index % 2), 2, 2);
  }
}

function drawTreeNode(context: PixelContext, quantity: number, sway: number) {
  fill(context, PALETTE.deepInk, 8, 9, 6, 17);
  fill(context, PALETTE.woodDark, 9, 8, 5, 17);
  fill(context, PALETTE.wood, 10, 9, 3, 15);
  fill(context, PALETTE.woodLight, 10, 11, 1, 3);
  fill(context, PALETTE.woodDark, 10, 16, 3, 2);
  fill(context, PALETTE.woodDark, 10, 21, 3, 2);
  const offset = sway ? 1 : 0;
  fill(context, PALETTE.leafDark, 7 + offset, 1, 9, 9);
  fill(context, PALETTE.leafDark, 2 + offset, 4, 12, 4);
  fill(context, PALETTE.leafDark, 0 + offset, 7, 8, 4);
  fill(context, PALETTE.leafDark, 13 + offset, 4, 8, 4);
  fill(context, PALETTE.leafDark, 17 + offset, 7, 5, 4);
  fill(context, PALETTE.leaf, 8 + offset, 2, 7, 6);
  fill(context, PALETTE.leaf, 4 + offset, 5, 9, 2);
  fill(context, PALETTE.leaf, 1 + offset, 8, 6, 2);
  fill(context, PALETTE.leaf, 14 + offset, 5, 6, 2);
  fill(context, PALETTE.leaf, 18 + offset, 8, 4, 2);
  fill(context, PALETTE.leafLight, 9 + offset, 3, 5, 2);
  fill(context, PALETTE.leafLight, 4 + offset, 6, 4, 1);
  if (quantity >= 3) {
    fill(context, PALETTE.fireLight, 9 + offset, 8, 2, 2);
    fill(context, PALETTE.fireLight, 13 + offset, 8, 2, 2);
  }
}

function drawStoneNode(context: PixelContext, quantity: number) {
  fill(context, PALETTE.deepInk, 3, 12, 14, 5);
  fill(context, PALETTE.stoneDark, 2, 9, 15, 6);
  fill(context, PALETTE.stone, 4, 6, 11, 8);
  fill(context, PALETTE.stoneLight, 6, 6, Math.max(3, quantity + 1), 2);
  fill(context, PALETTE.ink, 12, 11, 3, 2);
}

function drawShelter(context: PixelContext, stage: number) {
  if (stage === 0) {
    fill(context, PALETTE.sandDark, 4, 18, 24, 7);
    fill(context, PALETTE.stoneDark, 5, 19, 5, 4);
    fill(context, PALETTE.stone, 13, 18, 6, 5);
    fill(context, PALETTE.stoneDark, 22, 20, 5, 4);
    return;
  }
  fill(context, PALETTE.woodDark, 4, 20, 24, 6);
  fill(context, PALETTE.wood, 6, 10, 3, 15);
  fill(context, PALETTE.wood, 23, 10, 3, 15);
  fill(context, PALETTE.woodLight, 7, 11, 1, 12);
  if (stage === 1) {
    fill(context, PALETTE.rope, 5, 11, 22, 2);
    return;
  }
  fill(context, "#b9653b", 7, 11, 18, 13);
  fill(context, "#dd8750", 9, 12, 14, 3);
  fill(context, PALETTE.deepInk, 14, 17, 5, 7);
  if (stage === 2) {
    fill(context, PALETTE.rope, 4, 8, 24, 3);
    return;
  }
  fill(context, PALETTE.deepInk, 1, 8, 30, 5);
  fill(context, "#70432d", 2, 5, 28, 7);
  fill(context, "#9b5a35", 5, 2, 22, 8);
  fill(context, "#c27642", 8, 1, 16, 4);
  fill(context, PALETTE.rope, 13, 3, 6, 2);
  fill(context, PALETTE.fireLight, 16, 18, 2, 3);
}

function drawRoleIcon(context: PixelContext, role: AgentRole) {
  if (role === "forager") {
    fill(context, PALETTE.leafDark, 1, 2, 6, 5);
    fill(context, PALETTE.leafLight, 3, 1, 4, 3);
    fill(context, "#e85e54", 2, 5, 2, 2);
    return;
  }
  if (role === "woodcutter") {
    fill(context, PALETTE.stoneLight, 1, 1, 6, 3);
    fill(context, PALETTE.wood, 4, 3, 2, 5);
    return;
  }
  fill(context, PALETTE.stoneLight, 1, 1, 6, 2);
  fill(context, PALETTE.wood, 3, 3, 2, 5);
}

function drawCharacter(
  context: PixelContext,
  agent: ReplayAgent,
  direction: string,
  pose: string,
) {
  const skin = SKINS[agent.appearance.skin] ?? SKINS.golden;
  const hair = HAIR[agent.appearance.variant % HAIR.length];
  const accent = ACCENTS[agent.appearance.accent] ?? PALETTE.grassLight;
  const roleColor =
    agent.role === "forager"
      ? "#4b8c45"
      : agent.role === "woodcutter"
        ? "#a85836"
        : "#ca8c3c";
  const walk = pose === "walk-b" ? 1 : 0;
  const work = pose.startsWith("work");

  fill(context, PALETTE.deepInk, 7, 4, 11, 9);
  fill(context, skin[0], 6, 5, 12, 8);
  fill(context, skin[1], 7, 4, 10, 9);
  fill(context, skin[2], 8, 5, 7, 3);

  if (direction === "up") {
    fill(context, hair, 6, 3, 12, 8);
    fill(context, PALETTE.deepInk, 7, 3, 10, 3);
  } else {
    fill(context, hair, 6, 3, 12, 4);
    fill(context, hair, direction === "left" ? 6 : 15, 5, 3, 5);
    if (direction === "down") {
      fill(context, PALETTE.deepInk, 9, 8, 1, 1);
      fill(context, PALETTE.deepInk, 14, 8, 1, 1);
    }
  }

  if (agent.appearance.hair === "puffs" || agent.appearance.hair === "buns") {
    fill(context, hair, 4, 4, 3, 4);
    fill(context, hair, 17, 4, 3, 4);
  }
  if (agent.appearance.hair === "mohawk") {
    fill(context, hair, 10, 1, 4, 4);
  }
  if (agent.appearance.hair === "braid") {
    fill(context, hair, direction === "left" ? 16 : 6, 10, 2, 5);
  }

  fill(context, PALETTE.deepInk, 7, 12, 11, 8);
  fill(context, roleColor, 8, 11, 9, 8);
  fill(context, accent, 8, 12, 9, 2);
  if (agent.role === "forager") {
    fill(context, PALETTE.rope, 16, 14, 3, 5);
  } else if (agent.role === "woodcutter") {
    fill(context, PALETTE.woodLight, 8, 14, 2, 5);
  } else {
    fill(context, PALETTE.rope, 11, 13, 2, 6);
  }

  const leftLeg = pose === "walk-a" ? 1 : 0;
  const rightLeg = pose === "walk-b" ? 1 : 0;
  fill(context, PALETTE.deepInk, 8 - leftLeg, 19, 4, 4);
  fill(context, PALETTE.deepInk, 14 + rightLeg, 19, 4, 4);
  fill(context, accent, 9 - leftLeg, 19, 2, 3);
  fill(context, accent, 15 + rightLeg, 19, 2, 3);

  const armY = work ? 11 : 13 + walk;
  fill(context, skin[1], direction === "left" ? 5 : 17, armY, 3, 5);
  if (direction === "down" || direction === "up") {
    fill(context, skin[1], 5, armY, 3, 5);
    fill(context, skin[1], 17, armY, 3, 5);
  }
  if (work) {
    const swing = pose === "work-b";
    const toolX = direction === "left" ? 3 : 19;
    fill(context, PALETTE.wood, toolX, swing ? 8 : 11, 2, 9);
    fill(
      context,
      agent.role === "forager" ? PALETTE.rope : PALETTE.stoneLight,
      direction === "left" ? 1 : 18,
      swing ? 7 : 9,
      5,
      3,
    );
  }

  if (agent.appearance.accessory === "flower") {
    fill(context, "#f08aa2", 16, 4, 2, 2);
  } else if (agent.appearance.accessory === "headwrap") {
    fill(context, accent, 7, 4, 10, 2);
  } else if (agent.appearance.accessory === "feather") {
    fill(context, PALETTE.fireLight, 15, 1, 2, 4);
  } else if (agent.appearance.accessory === "bandana") {
    fill(context, accent, 6, 5, 12, 2);
  }
}

export function createPixelArtKit(scene: Phaser.Scene, agents: ReplayAgent[]) {
  (["water", "beach", "grass", "forest", "quarry"] as TerrainType[]).forEach(
    (terrain) => {
      for (let variant = 0; variant < 2; variant += 1) {
        canvasTexture(scene, `terrain-${terrain}-${variant}`, 16, 16, (context) =>
          drawTerrain(context, terrain, variant),
        );
      }
    },
  );

  for (let mask = 0; mask < 16; mask += 1) {
    canvasTexture(scene, `shore-${mask}`, 16, 16, (context) => {
      if (mask & 1) fill(context, PALETTE.foam, 0, 0, 16, 1);
      if (mask & 2) fill(context, PALETTE.foam, 15, 0, 1, 16);
      if (mask & 4) fill(context, PALETTE.foam, 0, 15, 16, 1);
      if (mask & 8) fill(context, PALETTE.foam, 0, 0, 1, 16);
    });
  }

  for (let quantity = 1; quantity <= 4; quantity += 1) {
    canvasTexture(scene, `resource-food-${quantity}`, 18, 18, (context) =>
      drawFoodNode(context, Math.min(3, quantity)),
    );
    for (let sway = 0; sway < 2; sway += 1) {
      canvasTexture(scene, `resource-wood-${quantity}-${sway}`, 22, 28, (context) =>
        drawTreeNode(context, quantity, sway),
      );
    }
    canvasTexture(scene, `resource-stone-${quantity}`, 20, 18, (context) =>
      drawStoneNode(context, quantity),
    );
  }

  for (let stage = 0; stage < 4; stage += 1) {
    canvasTexture(scene, `shelter-${stage}`, 32, 28, (context) =>
      drawShelter(context, stage),
    );
  }

  for (let frame = 0; frame < 2; frame += 1) {
    canvasTexture(scene, `campfire-${frame}`, 12, 14, (context) => {
      fill(context, PALETTE.deepInk, 1, 11, 10, 3);
      fill(context, PALETTE.wood, 2, 10, 8, 2);
      fill(context, PALETTE.fire, 4 - frame, 5, 5, 6);
      fill(context, PALETTE.fireLight, 5, 7 - frame, 3, 4);
    });
  }

  canvasTexture(scene, "pixel-shadow", 16, 6, (context) => {
    fill(context, "rgba(11,22,22,0.38)", 2, 1, 12, 4);
    fill(context, "rgba(11,22,22,0.5)", 4, 0, 8, 6);
  });
  canvasTexture(scene, "pixel-rain", 1, 5, (context) => {
    fill(context, "#9bd8d8", 0, 0, 1, 5);
  });
  canvasTexture(scene, "particle-pickup", 2, 2, (context) => {
    fill(context, PALETTE.fireLight, 0, 0, 2, 2);
  });
  canvasTexture(scene, "particle-spark", 3, 3, (context) => {
    fill(context, PALETTE.white, 1, 0, 1, 3);
    fill(context, PALETTE.white, 0, 1, 3, 1);
  });
  canvasTexture(scene, "tuft-grass", 8, 6, (context) => {
    fill(context, PALETTE.grassDark, 1, 3, 1, 3);
    fill(context, PALETTE.grassLight, 2, 1, 1, 5);
    fill(context, PALETTE.grassDark, 4, 2, 1, 4);
    fill(context, PALETTE.grassLight, 6, 3, 1, 3);
  });
  canvasTexture(scene, "shell", 6, 5, (context) => {
    fill(context, PALETTE.sandDark, 1, 1, 5, 4);
    fill(context, PALETTE.white, 1, 0, 4, 3);
    fill(context, "#e8a77e", 2, 1, 1, 2);
  });
  canvasTexture(scene, "stock-food", 10, 10, (context) => {
    fill(context, PALETTE.woodDark, 1, 4, 8, 6);
    fill(context, PALETTE.rope, 2, 5, 6, 1);
    fill(context, "#e85e54", 2, 2, 2, 3);
    fill(context, PALETTE.leafLight, 4, 1, 2, 3);
  });
  canvasTexture(scene, "stock-wood", 12, 9, (context) => {
    fill(context, PALETTE.woodDark, 1, 3, 10, 5);
    fill(context, PALETTE.woodLight, 2, 2, 8, 2);
    fill(context, PALETTE.rope, 5, 1, 2, 7);
  });
  canvasTexture(scene, "stock-stone", 12, 9, (context) => {
    fill(context, PALETTE.stoneDark, 1, 4, 10, 4);
    fill(context, PALETTE.stone, 3, 1, 6, 6);
    fill(context, PALETTE.stoneLight, 4, 2, 3, 2);
  });

  (["forager", "woodcutter", "builder"] as AgentRole[]).forEach((role) => {
    canvasTexture(scene, `role-${role}`, 8, 8, (context) =>
      drawRoleIcon(context, role),
    );
  });

  agents.forEach((agent) => {
    ["down", "up", "left", "right"].forEach((direction) => {
      ["idle", "walk-a", "walk-b", "work-a", "work-b"].forEach((pose) => {
        const key = characterTextureKey(agent.id, direction, pose);
        canvasTexture(scene, key, 24, 24, (context) =>
          drawCharacter(context, agent, direction, pose),
        );
      });
    });
  });
}

export function characterTextureKey(
  agentId: string,
  direction: string,
  pose: string,
) {
  return `agent-${agentId}-${direction}-${pose}`;
}

export function tileVariant(x: number, y: number) {
  return hash(x, y) % 2;
}

export function decorationVariant(x: number, y: number) {
  return hash(x, y, 31) % 17;
}
