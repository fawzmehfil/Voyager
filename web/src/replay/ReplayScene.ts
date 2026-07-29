import Phaser from "phaser";
import {
  characterTextureKey,
  createPixelArtKit,
  decorationVariant,
  tileVariant,
} from "./pixelArt";
import type {
  ReplayAgent,
  ReplayArtifact,
  ReplayCamp,
  ReplayFrame,
  ReplayResource,
  ReplayStatus,
  TerrainType,
} from "./types";

const TILE_SIZE = 48;
const ART_SCALE = 3;
const TICK_DURATION = 1000 / 12;

const ACHIEVEMENT_LABELS: Record<string, string> = {
  shelter_25_percent: "SHELTER FRAME RAISED",
  shelter_50_percent: "SHELTER HALF BUILT",
  shelter_complete: "SHELTER COMPLETE!",
  all_active_agents_alive_100: "100 STEPS · EVERYONE SAFE",
  all_roles_contributed: "ALL ROLES CONTRIBUTED",
  camp_food_buffer_10: "CAMP FOOD BUFFER · 10",
  first_storm_survived: "STORM SURVIVED!",
  food_security_100_steps: "FOOD SECURE · 100 STEPS",
  camp_food_buffer_20: "CAMP FOOD BUFFER · 20",
  shared_food_transfer: "FOOD SHARED ACROSS THE CREW",
};

const ACTION_LABELS: Record<string, string> = {
  gather_food: "+1 FOOD",
  gather_wood: "+1 WOOD",
  gather_stone: "+1 STONE",
  deposit_food: "DEPOSITING",
  deposit_wood: "DEPOSITING",
  deposit_stone: "DEPOSITING",
  withdraw_food: "PACKING FOOD",
  eat: "EATING",
};

interface ReplayHooks {
  onStatus: (status: ReplayStatus) => void;
}

interface AgentVisual {
  container: Phaser.GameObjects.Container;
  body: Phaser.GameObjects.Image;
  bubble: Phaser.GameObjects.Text;
  direction: string;
  lastX: number;
  lastY: number;
  bubbleCooldownStep: number;
}

export class ReplayScene extends Phaser.Scene {
  private replay: ReplayArtifact;
  private hooks: ReplayHooks;
  private frameIndex = 0;
  private accumulator = 0;
  private isPlaying = true;
  private soundEnabled = false;
  private audioContext?: AudioContext;
  private agentVisuals = new Map<string, AgentVisual>();
  private resourceVisuals = new Map<string, Phaser.GameObjects.Image>();
  private waterTiles: Phaser.GameObjects.Image[] = [];
  private treeTiles: Phaser.GameObjects.Image[] = [];
  private rain: Phaser.GameObjects.Image[] = [];
  private shelter?: Phaser.GameObjects.Image;
  private campfire?: Phaser.GameObjects.Image;
  private foodPile?: Phaser.GameObjects.Image;
  private woodPile?: Phaser.GameObjects.Image;
  private stonePile?: Phaser.GameObjects.Image;
  private stormShade?: Phaser.GameObjects.Rectangle;
  private toast?: Phaser.GameObjects.Text;
  private finale?: Phaser.GameObjects.Container;
  private waterFrame = 0;
  private foliageFrame = 0;
  private animationClock = 0;
  private currentStorm = false;
  private currentCamp: ReplayCamp;
  private currentAlive = 10;
  private firstDepositAgent = "";
  private sharingAgent = "";

  constructor(replay: ReplayArtifact, hooks: ReplayHooks) {
    super({ key: "VoyagerReplay" });
    this.replay = replay;
    this.hooks = hooks;
    this.currentCamp = replay.world.initial.camp;
  }

  create() {
    this.cameras.main.setBackgroundColor("#1f6f78");
    this.cameras.main.roundPixels = true;
    createPixelArtKit(this, this.replay.world.initial.agents);
    this.resolveStoryAgents();
    this.createWorld();
    this.createCamp();
    this.createAgents(this.replay.world.initial.agents);
    this.createWeather();
    this.createToast();
    this.setCameraImmediately();
    this.emitStatus();
  }

  update(_time: number, delta: number) {
    this.animationClock += delta;
    this.updateAmbientAnimation(delta);
    this.directCamera();
    if (!this.isPlaying || this.frameIndex >= this.replay.frames.length) {
      return;
    }

    this.accumulator += delta;
    while (
      this.accumulator >= TICK_DURATION &&
      this.frameIndex < this.replay.frames.length
    ) {
      this.accumulator -= TICK_DURATION;
      const frame = this.replay.frames[this.frameIndex];
      this.frameIndex += 1;
      this.applyFrame(frame);
    }
  }

  setPlaying(playing: boolean) {
    if (this.frameIndex >= this.replay.frames.length && playing) {
      this.restartReplay();
      return;
    }
    this.isPlaying = playing;
    this.emitStatus();
  }

  restartReplay() {
    this.frameIndex = 0;
    this.accumulator = 0;
    this.isPlaying = true;
    this.currentStorm = false;
    this.currentCamp = this.replay.world.initial.camp;
    this.currentAlive = this.replay.world.initial.agents.length;
    this.agentVisuals.clear();
    this.resourceVisuals.clear();
    this.waterTiles = [];
    this.treeTiles = [];
    this.rain = [];
    this.shelter = undefined;
    this.campfire = undefined;
    this.foodPile = undefined;
    this.woodPile = undefined;
    this.stonePile = undefined;
    this.stormShade = undefined;
    this.toast = undefined;
    this.finale = undefined;
    this.scene.restart();
  }

  setSoundEnabled(enabled: boolean) {
    this.soundEnabled = enabled;
    if (enabled && !this.audioContext) {
      this.audioContext = new AudioContext();
    }
    if (enabled) {
      void this.audioContext?.resume();
      this.playTone(440, 0.05);
    }
  }

  private createWorld() {
    const { width, height, initial } = this.replay.world;
    const worldWidth = width * TILE_SIZE;
    const worldHeight = height * TILE_SIZE;
    this.cameras.main.setBounds(-160, -120, worldWidth + 320, worldHeight + 240);

    initial.terrain.forEach((row, y) => {
      row.forEach((terrain, x) => {
        const variant = tileVariant(x, y);
        const tile = this.add
          .image(
            x * TILE_SIZE + TILE_SIZE / 2,
            y * TILE_SIZE + TILE_SIZE / 2,
            `terrain-${terrain}-${variant}`,
          )
          .setScale(ART_SCALE)
          .setDepth(terrain === "water" ? 0 : 10);
        if (terrain === "water") {
          this.waterTiles.push(tile);
          const mask = this.shoreMask(initial.terrain, x, y);
          if (mask > 0) {
            this.add
              .image(tile.x, tile.y, `shore-${mask}`)
              .setScale(ART_SCALE)
              .setDepth(12);
          }
        } else {
          this.addGroundDecoration(terrain, x, y);
        }
      });
    });

    initial.resources.forEach((resource) => this.upsertResource(resource));
  }

  private addGroundDecoration(terrain: TerrainType, x: number, y: number) {
    const variant = decorationVariant(x, y);
    if (
      (terrain === "grass" || terrain === "forest") &&
      variant < 3
    ) {
      this.add
        .image(
          x * TILE_SIZE + 9 + variant * 8,
          y * TILE_SIZE + 32,
          "tuft-grass",
        )
        .setScale(ART_SCALE)
        .setAlpha(0.7)
        .setDepth(20 + y);
    }
    if (terrain === "beach" && variant === 4) {
      this.add
        .image(x * TILE_SIZE + 19, y * TILE_SIZE + 27, "shell")
        .setScale(ART_SCALE)
        .setDepth(20 + y);
    }
  }

  private createCamp() {
    const camp = this.replay.world.initial.camp;
    const x = this.tileX(camp.x);
    const y = this.tileY(camp.y);
    this.shelter = this.add
      .image(x + 52, y - 36, "shelter-0")
      .setScale(ART_SCALE)
      .setOrigin(0.5, 0.78)
      .setDepth(2500 + y);
    this.campfire = this.add
      .image(x - 24, y + 16, "campfire-0")
      .setScale(ART_SCALE)
      .setDepth(2504 + y);
    this.foodPile = this.add
      .image(x - 66, y - 18, "stock-food")
      .setScale(ART_SCALE)
      .setDepth(2502 + y)
      .setVisible(false);
    this.woodPile = this.add
      .image(x - 66, y + 18, "stock-wood")
      .setScale(ART_SCALE)
      .setDepth(2502 + y)
      .setVisible(false);
    this.stonePile = this.add
      .image(x - 34, y + 36, "stock-stone")
      .setScale(ART_SCALE)
      .setDepth(2502 + y)
      .setVisible(false);

    const glow = this.add.graphics().setDepth(2498 + y);
    glow.fillStyle(0xf4a340, 0.08);
    glow.fillRect(x - 54, y - 12, 60, 60);
    glow.fillStyle(0xffd15c, 0.08);
    glow.fillRect(x - 42, y, 36, 36);
  }

  private createAgents(agents: ReplayAgent[]) {
    agents.forEach((agent) => {
      const shadow = this.add.image(0, 17, "pixel-shadow").setScale(2);
      const body = this.add
        .image(0, 0, characterTextureKey(agent.id, "down", "idle"))
        .setScale(2);
      const role = this.add
        .image(-17, -29, `role-${agent.role}`)
        .setScale(1.25);
      const name = this.add
        .text(-11, -31, agent.name.toUpperCase(), {
          fontFamily: '"Press Start 2P"',
          fontSize: "7px",
          color: "#fff3d2",
          backgroundColor: "#15231de6",
          padding: { x: 4, y: 3 },
        })
        .setOrigin(0, 0.5);
      const bubble = this.add
        .text(0, -48, "", {
          fontFamily: '"Press Start 2P"',
          fontSize: "7px",
          color: "#19251f",
          backgroundColor: "#fff0c6",
          padding: { x: 5, y: 4 },
        })
        .setOrigin(0.5, 1)
        .setVisible(false);
      const container = this.add.container(
        this.tileX(agent.x),
        this.tileY(agent.y),
        [shadow, body, role, name, bubble],
      );
      container.setDepth(3000 + this.tileY(agent.y));
      this.agentVisuals.set(agent.id, {
        container,
        body,
        bubble,
        direction: "down",
        lastX: agent.x,
        lastY: agent.y,
        bubbleCooldownStep: -100,
      });
    });
  }

  private createWeather() {
    this.stormShade = this.add
      .rectangle(0, 0, 1280, 720, 0x102735, 0)
      .setOrigin(0)
      .setScrollFactor(0)
      .setDepth(9000);
    for (let index = 0; index < 72; index += 1) {
      const drop = this.add
        .image(
          (index * 173 + 47) % 1280,
          (index * 97 + 31) % 720,
          "pixel-rain",
        )
        .setScale(3)
        .setAlpha(0.76)
        .setScrollFactor(0)
        .setDepth(9001)
        .setVisible(false);
      this.rain.push(drop);
    }
  }

  private createToast() {
    this.toast = this.add
      .text(640, 92, "", {
        fontFamily: '"Press Start 2P"',
        fontSize: "13px",
        color: "#fff0c6",
        backgroundColor: "#16261ef2",
        padding: { x: 14, y: 10 },
        align: "center",
      })
      .setOrigin(0.5)
      .setScrollFactor(0)
      .setDepth(9500)
      .setVisible(false);
  }

  private applyFrame(frame: ReplayFrame) {
    frame.resource_changes.forEach((resource) => this.upsertResource(resource));
    this.updateCamp(frame.camp);
    frame.agents.forEach((agent) => this.updateAgent(agent, frame.step));
    frame.new_achievements.forEach((achievement) =>
      this.showAchievement(achievement),
    );
    this.setStorm(frame.storm);
    this.currentCamp = frame.camp;
    this.currentAlive = frame.agents.filter((agent) => agent.alive).length;
    this.emitStatus();
    if (frame.step === this.replay.summary.world_steps) {
      this.finishReplay();
    }
  }

  private updateAgent(agent: ReplayAgent, step: number) {
    const visual = this.agentVisuals.get(agent.id);
    if (!visual) return;

    const deltaX = agent.x - visual.lastX;
    const deltaY = agent.y - visual.lastY;
    if (deltaX < 0) visual.direction = "left";
    if (deltaX > 0) visual.direction = "right";
    if (deltaY < 0) visual.direction = "up";
    if (deltaY > 0) visual.direction = "down";
    const moving = deltaX !== 0 || deltaY !== 0;
    const working = this.isWorkEvent(agent.event);
    const pose = working
      ? step % 2 === 0
        ? "work-a"
        : "work-b"
      : moving
        ? step % 2 === 0
          ? "walk-a"
          : "walk-b"
        : "idle";
    visual.body.setTexture(characterTextureKey(agent.id, visual.direction, pose));

    this.tweens.killTweensOf(visual.container);
    this.tweens.add({
      targets: visual.container,
      x: this.tileX(agent.x),
      y: this.tileY(agent.y),
      duration: moving ? 74 : 30,
      ease: "Sine.easeOut",
      onUpdate: () => {
        visual.container.setDepth(3000 + visual.container.y);
      },
    });

    const label = ACTION_LABELS[agent.event] ??
      (agent.event.startsWith("build_shelter") ? "BUILDING" : undefined);
    if (label && step - visual.bubbleCooldownStep >= 13) {
      visual.bubbleCooldownStep = step;
      this.showActionBubble(visual, label);
      this.spawnActionParticles(visual.container.x, visual.container.y, agent.event);
    }

    visual.lastX = agent.x;
    visual.lastY = agent.y;
    visual.container.setAlpha(agent.alive ? 1 : 0.35);
  }

  private updateCamp(camp: ReplayCamp) {
    const stage =
      camp.shelter_progress >= 1
        ? 3
        : camp.shelter_progress >= 0.5
          ? 2
          : camp.shelter_progress >= 0.25
            ? 1
            : 0;
    this.shelter?.setTexture(`shelter-${stage}`);
    this.foodPile?.setVisible(camp.stockpile.food > 0).setScale(
      ART_SCALE * Math.min(1.35, 0.75 + camp.stockpile.food / 40),
    );
    this.woodPile?.setVisible(camp.stockpile.wood > 0);
    this.stonePile?.setVisible(camp.stockpile.stone > 0);
  }

  private upsertResource(resource: ReplayResource) {
    const key = `${resource.x}:${resource.y}`;
    const existing = this.resourceVisuals.get(key);
    if (resource.type === "none" || resource.quantity <= 0) {
      if (existing) {
        existing.destroy();
        this.resourceVisuals.delete(key);
      }
      return;
    }
    const quantity = Phaser.Math.Clamp(resource.quantity, 1, 4);
    const texture =
      resource.type === "wood"
        ? `resource-wood-${quantity}-${this.foliageFrame}`
        : `resource-${resource.type}-${quantity}`;
    const node =
      existing ??
      this.add
        .image(this.tileX(resource.x), this.tileY(resource.y), texture)
        .setScale(ART_SCALE)
        .setOrigin(0.5, resource.type === "wood" ? 0.84 : 0.68)
        .setDepth(2000 + this.tileY(resource.y));
    node.setTexture(texture).setData("resourceType", resource.type);
    if (!existing) {
      this.resourceVisuals.set(key, node);
      if (resource.type === "wood") this.treeTiles.push(node);
    }
  }

  private showActionBubble(visual: AgentVisual, label: string) {
    visual.bubble.setText(label).setVisible(true).setAlpha(1).setY(-48);
    this.tweens.killTweensOf(visual.bubble);
    this.tweens.add({
      targets: visual.bubble,
      y: -56,
      alpha: 0,
      delay: 260,
      duration: 420,
      ease: "Stepped",
      onComplete: () => visual.bubble.setVisible(false),
    });
  }

  private spawnActionParticles(x: number, y: number, event: string) {
    const texture = event.startsWith("build") ? "particle-spark" : "particle-pickup";
    for (let index = 0; index < 3; index += 1) {
      const particle = this.add
        .image(x + index * 7 - 7, y - 13, texture)
        .setScale(2)
        .setDepth(6200);
      this.tweens.add({
        targets: particle,
        x: particle.x + (index - 1) * 10,
        y: particle.y - 20 - index * 4,
        alpha: 0,
        duration: 420,
        ease: "Stepped",
        onComplete: () => particle.destroy(),
      });
    }
  }

  private showAchievement(achievement: string) {
    const label = ACHIEVEMENT_LABELS[achievement];
    if (!label || !this.toast) return;
    this.toast.setText(label).setVisible(true).setAlpha(1).setY(92);
    this.tweens.killTweensOf(this.toast);
    this.tweens.add({
      targets: this.toast,
      y: 80,
      duration: 180,
      ease: "Sine.easeOut",
      yoyo: true,
      hold: 950,
      onComplete: () => this.toast?.setVisible(false),
    });
    this.playTone(achievement === "no_deaths_run" ? 660 : 520, 0.09);
  }

  private setStorm(storm: boolean) {
    if (this.currentStorm === storm) return;
    this.currentStorm = storm;
    this.tweens.add({
      targets: this.stormShade,
      alpha: storm ? 0.48 : 0,
      duration: 500,
      ease: "Sine.easeInOut",
    });
    this.rain.forEach((drop) => drop.setVisible(storm));
    if (storm) this.playTone(120, 0.15);
  }

  private updateAmbientAnimation(delta: number) {
    const waterFrame = Math.floor(this.animationClock / 480) % 2;
    if (waterFrame !== this.waterFrame) {
      this.waterFrame = waterFrame;
      this.waterTiles.forEach((tile) => {
        const parts = tile.texture.key.split("-");
        tile.setTexture(`terrain-water-${parts.at(-1) === "0" ? 1 : 0}`);
      });
      if (this.campfire) this.campfire.setTexture(`campfire-${waterFrame}`);
    }

    const foliageFrame = Math.floor(this.animationClock / 720) % 2;
    if (foliageFrame !== this.foliageFrame) {
      this.foliageFrame = foliageFrame;
      this.treeTiles.forEach((tree) => {
        if (!tree.active) return;
        const parts = tree.texture.key.split("-");
        tree.setTexture(
          `resource-wood-${parts[2]}-${foliageFrame}`,
        );
      });
    }

    if (this.currentStorm) {
      this.rain.forEach((drop, index) => {
        drop.x -= delta * 0.18;
        drop.y += delta * 0.46;
        if (drop.y > 730) drop.y = -20 - (index % 7) * 9;
        if (drop.x < -10) drop.x = 1290;
      });
    }
  }

  private directCamera() {
    const step = this.frameIndex;
    const camp = this.currentCamp;
    let targetX = this.tileX(camp.x);
    let targetY = this.tileY(camp.y);
    let targetZoom = 0.78;

    if (step < 24) {
      targetZoom = 0.68;
    } else if (step < 60) {
      const agent = this.agentVisuals.get(this.firstDepositAgent);
      if (agent) {
        targetX = agent.container.x;
        targetY = agent.container.y;
      }
      targetZoom = 1.12;
    } else if (step < 120) {
      targetZoom = 1.28;
      targetX += 24;
      targetY -= 12;
    } else if (step < 198) {
      const living = [...this.agentVisuals.values()];
      targetX =
        living.reduce((total, agent) => total + agent.container.x, 0) /
        living.length;
      targetY =
        living.reduce((total, agent) => total + agent.container.y, 0) /
        living.length;
      targetZoom = 0.82;
    } else if (step < 228) {
      targetZoom = 0.86;
    } else if (step < 270) {
      const agent = this.agentVisuals.get(this.sharingAgent);
      if (agent) {
        targetX = agent.container.x;
        targetY = agent.container.y;
      }
      targetZoom = 1.1;
    } else {
      targetZoom = 0.76;
      targetY -= 24;
    }

    const camera = this.cameras.main;
    camera.zoom = Phaser.Math.Linear(camera.zoom, targetZoom, 0.025);
    const desiredX = targetX - camera.width / (2 * camera.zoom);
    const desiredY = targetY - camera.height / (2 * camera.zoom);
    camera.scrollX = Phaser.Math.Linear(camera.scrollX, desiredX, 0.025);
    camera.scrollY = Phaser.Math.Linear(camera.scrollY, desiredY, 0.025);
  }

  private setCameraImmediately() {
    const camp = this.replay.world.initial.camp;
    const camera = this.cameras.main;
    camera.setZoom(0.68);
    camera.centerOn(this.tileX(camp.x), this.tileY(camp.y));
  }

  private resolveStoryAgents() {
    const depositFrame = this.replay.frames.find((frame) =>
      frame.agents.some((agent) => agent.event.startsWith("deposit_")),
    );
    this.firstDepositAgent =
      depositFrame?.agents.find((agent) => agent.event.startsWith("deposit_"))?.id ??
      "agent_0";
    const sharingStep = this.replay.summary.achievement_steps.shared_food_transfer;
    const sharingFrame = this.replay.frames[sharingStep - 1];
    this.sharingAgent =
      sharingFrame?.agents.find((agent) => agent.event === "eat")?.id ?? "agent_1";
  }

  private finishReplay() {
    this.isPlaying = false;
    const panel = this.add
      .rectangle(0, 0, 430, 118, 0x12231c, 0.94)
      .setStrokeStyle(4, 0xf1c866, 1);
    const title = this.add
      .text(0, -20, "10 / 10 SURVIVED", {
        fontFamily: '"Press Start 2P"',
        fontSize: "22px",
        color: "#fff0c6",
        align: "center",
      })
      .setOrigin(0.5);
    const caption = this.add
      .text(0, 26, "SHELTER COMPLETE · 16 ACHIEVEMENTS", {
        fontFamily: '"Press Start 2P"',
        fontSize: "9px",
        color: "#8fd68a",
        align: "center",
      })
      .setOrigin(0.5);
    this.finale = this.add
      .container(640, 382, [panel, title, caption])
      .setScrollFactor(0)
      .setDepth(9800)
      .setAlpha(0);
    this.tweens.add({
      targets: this.finale,
      alpha: 1,
      y: 360,
      duration: 520,
      ease: "Sine.easeOut",
    });
    this.playTone(720, 0.18);
    this.emitStatus();
  }

  private emitStatus() {
    this.hooks.onStatus({
      step: Math.min(this.frameIndex, this.replay.summary.world_steps),
      playing: this.isPlaying,
      ended: this.frameIndex >= this.replay.frames.length,
      storm: this.currentStorm,
      alive: this.currentAlive,
      camp: this.currentCamp,
    });
  }

  private playTone(frequency: number, duration: number) {
    if (!this.soundEnabled || !this.audioContext) return;
    const context = this.audioContext;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = "square";
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0.035, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      context.currentTime + duration,
    );
    oscillator.connect(gain).connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + duration);
  }

  private isWorkEvent(event: string) {
    return (
      event.startsWith("gather_") ||
      event.startsWith("build_shelter") ||
      event.startsWith("deposit_") ||
      event === "eat" ||
      event === "withdraw_food"
    );
  }

  private shoreMask(terrain: TerrainType[][], x: number, y: number) {
    const isLand = (sampleX: number, sampleY: number) =>
      terrain[sampleY]?.[sampleX] !== undefined &&
      terrain[sampleY][sampleX] !== "water";
    return (
      (isLand(x, y - 1) ? 1 : 0) |
      (isLand(x + 1, y) ? 2 : 0) |
      (isLand(x, y + 1) ? 4 : 0) |
      (isLand(x - 1, y) ? 8 : 0)
    );
  }

  private tileX(x: number) {
    return x * TILE_SIZE + TILE_SIZE / 2;
  }

  private tileY(y: number) {
    return y * TILE_SIZE + TILE_SIZE / 2;
  }
}
