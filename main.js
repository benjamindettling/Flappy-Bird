const isPortrait = window.matchMedia("(orientation: portrait)").matches;

let configPortrait = {
  // vertical setup (e.g., 360x640)
  scale: {
    mode: Phaser.Scale.ENVELOP,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: 288,
    height: 512,
  },
  physics: {
    default: "arcade",
    arcade: {
      gravity: { y: 1300 },
      debug: false,
      fps: 60,
    },
  },
  scene: {
    preload: preload,
    create: create,
    update: update,
  },
};

let configLandscape = {
  // horizontal setup (e.g., 640x360)
  scale: {
    mode: Phaser.Scale.ENVELOP,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: 1080,
    height: 512,
  },
  physics: {
    default: "arcade",
    arcade: {
      gravity: { y: 1300 },
      debug: false,
      fps: 60,
    },
  },
  scene: {
    preload: preload,
    create: create,
    update: update,
  },
};

let config = {
  type: Phaser.AUTO,
  scale: {
    mode: Phaser.Scale.ENVELOP,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: 288,
    height: 512,
  },
  physics: {
    default: "arcade",
    arcade: {
      gravity: { y: 1300 },
      debug: false,
      fps: 60,
    },
  },
  scene: {
    preload: preload,
    create: create,
    update: update,
  },
};

let jumpV = -400;
let speed = -150;
let spawnTime = 1500;

let isAIControlled = false;
let trainingMode = false;
let currentEpisodeSteps = 0;
let totalRewardThisEpisode = 0;
let lastObservation;
let lastAction;
let currentAction;

const lidarRayAngles = [-Math.PI / 4, 0, Math.PI / 4]; // diagonally up, straight, down
const lidarRayLength = 300; // how far each ray can "see"
const lidarRayStep = 5; // step in pixels per ray trace

let point;
let hit;
let wing;
let die;
let finalScoreText;

let score = 0;
let scoreText;
let isRefresh = false;
let hitPlayed = false;
let diePlayed = false;
let character;
let background;
let base;
let baseImage;
let baseHeight;
let baseWidth;

let gameStart = false;

import { DQNAgent } from "./ai.js";
const tf = window.tf;

// let game = new Phaser.Game(config);
let game = new Phaser.Game(isPortrait ? configPortrait : configLandscape);

let agent;

// Wait for Phaser to finish scene creation
window.addEventListener("load", () => {
  const scene = game.scene.scenes[0]; // Get the actual scene object

  document.addEventListener("keydown", async (event) => {
    if (event.key === "t" || event.key === "T") {
      // Flip training mode toggle
      isAIControlled = !isAIControlled;
      trainingMode = isAIControlled;

      if (!agent) {
        agent = new DQNAgent(7, 2, {
          // 7 = 4 features + 3 lidar rays
          gamma: 0.99,
          lr: 0.01,
          memoryCapacity: 10000,
          batchSize: 32,
        });
        console.log("🧠 New DQNAgent initialized.");
      }

      console.log(
        `🎯 Training ${
          trainingMode ? "enabled" : "paused"
        } — AI control: ${isAIControlled}`
      );
    }
    // Save model
    if (event.key === "s" || event.key === "S") {
      if (agent && agent.model) {
        await agent.model.save("indexeddb://flappybird-dqn");
        console.log("💾 Model saved to your Downloads folder.");
      } else {
        console.warn("⚠️ No model to save.");
      }
    }

    // Load model
    if (event.key === "l" || event.key === "L") {
      try {
        const loadedModel = await tf.loadLayersModel("/tfjs_model/model.json");

        if (agent) {
          agent.model = loadedModel;
          console.log("📂 Model loaded successfully!");
        } else {
          console.warn("⚠️ No agent defined. Define one first with 'T'.");
        }
      } catch (err) {
        console.error("❌ Failed to load model:", err);
      }
    }
  });
});

let epsilonStart = 0.5;
let epsilonEnd = 0.01;
let episodeCount = 0;
let totalEpisodes = 1000;
let maxStepsPerEpisode = 1000;

function getEpsilon(episode) {
  return (
    epsilonEnd + (epsilonStart - epsilonEnd) * (1 - episode / totalEpisodes)
  );
}

function getNextPipe(scene) {
  // Get the next upper pipe (ahead of the bird)
  const upperPipes = scene.upperPillars.getChildren();

  for (let i = 0; i < upperPipes.length; i++) {
    const pipe = upperPipes[i];
    if (pipe.x + pipe.width > character.x) {
      // Find the matching lower pipe
      const lowerPipes = scene.lowerPillars.getChildren();
      const lowerPipe = lowerPipes[i]; // Assumes matching order
      return { upper: pipe, lower: lowerPipe };
    }
  }

  // If no pipe is ahead, return a dummy far pipe
  return {
    upper: { x: character.x + 300, y: 200 },
    lower: { x: character.x + 300, y: 400 },
  };
}

function getObservation(scene) {
  const next = getNextPipe(scene);

  // Normalize each component:
  const birdY =
    (character.y - scene.scale.height / 2) / (scene.scale.height / 2); // from -1 (top) to +1 (bottom)
  const birdVY = character.body.velocity.y / 400; // roughly -1 (flap) to +1 (fall)
  const pipeDistX = (next.upper.x - character.x) / scene.scale.width; // 0 (overlap) to 1 (far away)

  const pipeGapY = (next.lower.y + next.upper.y) / 2;
  const pipeDistY = (pipeGapY - character.y) / scene.scale.height; // center difference, normalized

  const lidar = getLidarReadings(scene);

  return [
    Phaser.Math.Clamp(birdY, -1, 1),
    Phaser.Math.Clamp(birdVY, -1, 1),
    Phaser.Math.Clamp(pipeDistX, 0, 1),
    Phaser.Math.Clamp(pipeDistY, -1, 1),
    ...lidar,
  ];
}

function getLidarReadings(scene) {
  const readings = [];

  const pipes = [
    ...scene.upperPillars.getChildren(),
    ...scene.lowerPillars.getChildren(),
  ];
  const baseY = scene.scale.height;

  for (let angle of lidarRayAngles) {
    let rayDist = lidarRayLength;

    for (let d = 0; d < lidarRayLength; d += lidarRayStep) {
      const x = character.x + d * Math.cos(angle);
      const y = character.y + d * Math.sin(angle);

      // stop if off-screen
      if (y <= 0 || y >= baseY || x >= scene.scale.width) {
        rayDist = d;
        break;
      }

      // stop if ray intersects any pipe
      for (let pipe of pipes) {
        const bounds = pipe.getBounds();
        if (
          x >= bounds.left &&
          x <= bounds.right &&
          y >= bounds.top &&
          y <= bounds.bottom
        ) {
          rayDist = d;
          break;
        }
      }

      if (rayDist !== lidarRayLength) break;
    }

    // Normalize distance (0 = close obstacle, 1 = clear view)
    readings.push(rayDist / lidarRayLength);
  }

  return readings;
}

function computeReward(scene) {
  let reward = 0.1; // Frame bonus

  const next = getNextPipe(scene);

  const currentScore = scene.score;
  if (typeof computeReward.lastScore === "undefined") {
    computeReward.lastScore = currentScore;
  }

  if (currentScore > computeReward.lastScore) {
    reward += 5.0; // Pipe passed
    computeReward.lastScore = currentScore;
  }

  if (scene.character.y <= 0) {
    reward -= 1; // Ceiling penalty
  }

  if (scene.isGameOver) {
    reward -= 1.0; // Death penalty
  }

  if (scene.character.y < 50) {
    reward -= 1; // proximity to top penalty
  }

  const gapCenter = (next.lower.y + next.upper.y) / 2;
  const distToGap = Math.abs(gapCenter - character.y) / scene.scale.height; // normalized distance

  reward -= distToGap / scene.scale.height; // small penalty for being far from target

  return reward;
}

function preload() {
  this.load.image("background", "assets/GameObjects/background-day.png");
  this.load.image("character1", "assets/GameObjects/yellowbird-midflap.png");
  this.load.image("character2", "assets/GameObjects/yellowbird-downflap.png");
  this.load.image("character3", "assets/GameObjects/yellowbird-upflap.png");
  this.load.image("character4", "assets/GameObjects/yellowbird-fall.png");
  this.load.image("pillar", "assets/GameObjects/pipe-green.png");
  this.load.image("base", "assets/GameObjects/base.png");
  this.load.image("gameover", "assets/UI/gameover.png");
  this.load.image("score", "assets/UI/score.png");
  this.load.image("retry", "assets/UI/retry.png");
  this.load.image("startGame", "assets/UI/message.png");
  this.load.audio("score", "assets/SoundEffects/point.wav");
  this.load.audio("hit", "assets/SoundEffects/hit.wav");
  this.load.audio("wing", "assets/SoundEffects/wing.wav");
  this.load.audio("die", "assets/SoundEffects/die.wav");
}

function create() {
  background = this.add.tileSprite(
    0,
    0,
    this.scale.width,
    this.scale.height,

    "background"
  );
  background.setOrigin(0, 0);
  background.displayWidth = this.scale.width;
  background.displayHeight = this.scale.height;
  let baseImage = this.textures.get("base");
  let baseHeight = baseImage.getSourceImage().height;
  let baseWidth = baseImage.getSourceImage().width;
  base = this.add.tileSprite(
    this.scale.width / 2,
    this.scale.height - baseHeight / 2,
    this.scale.width,
    baseHeight,
    "base"
  );
  base.setOrigin(0.5, 1);
  base.y = this.scale.height;
  base.y = Math.round(base.y);

  this.physics.add.existing(base, true);
  base.setDepth(1);
  let startGameImage = this.add.image(
    this.scale.width / 2,
    this.scale.height / 3,
    "startGame"
  );
  startGameImage.setOrigin(0.5, 0.5);

  if (trainingMode) {
    startGameImage.setVisible(false);
  }

  character = this.physics.add.sprite(
    this.scale.width / 4,
    this.scale.height / 2,
    "character1"
  );
  character.setDepth(1);
  character.setCollideWorldBounds(true);
  character.body.allowGravity = false;
  this.character = character;

  gameStart = false;
  this.anims.create({
    key: "fly",
    frames: [
      { key: "character1" },
      { key: "character2" },
      { key: "character3" },
    ],
    frameRate: 9,
    repeat: -1,
  });
  this.anims.create({
    key: "fall",
    frames: [{ key: "character4" }],
    frameRate: 9,
    repeat: -1,
  });
  character.anims.play("fly", true);
  this.input.on("pointerdown", () => {
    if (this.isGameOver) {
      this.handleRestart();
    } else {
      if (!gameStart) {
        startGameImage.setVisible(false);
        this.handleGameStart();
      }
      this.handleFlap();
    }
  });
  this.input.keyboard.on("keydown-SPACE", () => {
    if (this.isGameOver) {
      this.handleRestart();
    } else {
      if (!gameStart) {
        startGameImage.setVisible(false);
        this.handleGameStart();
      }
      this.handleFlap();
    }
  });

  this.input.keyboard.on("keydown-UP", () => {
    if (this.isGameOver) {
      this.handleRestart();
    } else {
      if (!gameStart) {
        startGameImage.setVisible(false);
        this.handleGameStart();
      }
      this.handleFlap();
    }
  });
}

function update() {
  if (!this || !this.upperPillars || !this.lowerPillars) return;
  if (!this?.upperPillars?.children || !this?.lowerPillars?.children) return;
  // ✅ still allow training step even when game is over
  if (trainingMode && isAIControlled && this.isGameOver) {
    if (typeof lastObservation !== "undefined") {
      agent.memory.push(lastObservation, lastAction, -1, lastObservation, true);
    }
    const reward = computeReward(this);
    totalRewardThisEpisode += reward;

    console.log(
      `🧠 Episode ${
        episodeCount + 1
      } done — Reward: ${totalRewardThisEpisode.toFixed(
        2
      )}, Steps: ${currentEpisodeSteps}, Epsilon: ${getEpsilon(episodeCount)}`
    );
    currentEpisodeSteps = 0;
    totalRewardThisEpisode = 0;
    lastObservation = undefined;
    lastAction = undefined;

    episodeCount++;

    this.handleRestart();
  }

  if (!this.gameStart || this.isGameOver) return;

  // move base & background
  base.tilePositionX += 2.5;
  background.tilePositionX += 0.5;

  let scoreIncremented = false;

  [this.upperPillars, this.lowerPillars].forEach((group) => {
    group.children.iterate((pillar) => {
      if (!pillar) return;

      if (!pillar.hasPassed && pillar.x + pillar.width < character.x) {
        pillar.hasPassed = true;
        if (!scoreIncremented) {
          score++;
          scoreText.setText(score);
          if (!trainingMode) point.play();
          scoreIncremented = true;
        }
      }
      if (pillar.x + pillar.width < 0) {
        pillar.destroy();
      }
    });
  });
  scoreIncremented = false;
  if (this.pillarSpawnTime < this.time.now) {
    this.spawnPillarPair();
  }

  if (this.gameStart && isAIControlled && !this.isGameOver) {
    const obs = getObservation(this);

    currentAction = agent.act(obs, trainingMode ? getEpsilon(episodeCount) : 0);
    if (currentAction === 1) {
      this.handleFlap();
    }
  }

  if (isAIControlled && this.gameStart && !this.isGameOver) {
    const currentObs = getObservation(this);
    const reward = computeReward(this);
    totalRewardThisEpisode += reward;

    // Store in memory
    if (typeof lastObservation !== "undefined") {
      agent.memory.push(lastObservation, lastAction, reward, currentObs, false);
    }

    lastObservation = currentObs;
    lastAction = currentAction;

    currentEpisodeSteps++;
    agent.optimizeModel();
  }

  const velocityY = character.body.velocity.y;

  if (velocityY < 0) {
    // Jumping – instantly tilt up
    character.setAngle(-20);
  } else {
    // Falling – gradually tilt down based on velocity
    const minAngle = -20;
    const maxAngle = 60;
    const maxFallSpeed = 400; // cap fall speed used for tilt

    // Clamp velocity between 0 and maxFallSpeed
    const clampedVY = Phaser.Math.Clamp(velocityY, 0, maxFallSpeed);

    // Map velocityY to angle between minAngle and maxAngle
    const angle = Phaser.Math.Linear(
      minAngle,
      maxAngle,
      clampedVY / maxFallSpeed
    );
    character.setAngle(angle);
  }
}

Phaser.Scene.prototype.handleFlap = function () {
  if (!isRefresh && !this.isGameOver) {
    if (!trainingMode) wing.play();
    character.setVelocityY(jumpV);
  }
  isRefresh = false;
};

Phaser.Scene.prototype.handleGameStart = function () {
  if (gameStart) return;
  gameStart = true;
  this.gameStart = true;
  this.isGameOver = false;

  character.body.allowGravity = true;
  if (!trainingMode) {
    character.setVelocityY(jumpV);
  }

  this.upperPillars = this.physics.add.group();
  this.lowerPillars = this.physics.add.group();
  this.spawnPillarPair();
  this.physics.add.collider(
    character,
    this.upperPillars,
    hitPillar,
    null,
    this
  );
  this.physics.add.collider(
    character,
    this.lowerPillars,
    hitPillar,
    null,
    this
  );
  this.physics.add.collider(character, base, hitBase, null, this);
  scoreText = this.add.text(this.scale.width / 2, 30, "0", {
    fontSize: "32px",
    fontFamily: "Fantasy",
    fill: "white",
  });
  scoreText.setOrigin(0.5, 0.5);
  scoreText.setDepth(1);
  this.score = score;
  point = this.sound.add("score");
  hit = this.sound.add("hit");
  wing = this.sound.add("wing");
  die = this.sound.add("die");
  if (!trainingMode) wing.play();
};

Phaser.Scene.prototype.handleRestart = function () {
  this.isGameOver = false;
  this.isGameOver = false;

  score = 0;
  computeReward.lastScore = 0;
  gameStart = false;
  this.gameStart = false;
  this.scene.restart();
  hitPlayed = false;
  diePlayed = false;
  isRefresh = true;

  setTimeout(() => {
    if (trainingMode && isAIControlled) {
      const scene = game.scene.scenes[0];
      scene.handleGameStart();
    }
  }, 50);
};

Phaser.Scene.prototype.spawnPillarPair = function () {
  baseImage = this.textures.get("base");
  baseHeight = baseImage.getSourceImage().height;
  let pillarImage = this.textures.get("pillar");
  let pillarHeight = pillarImage.getSourceImage().height;
  let Offset = (Math.random() * pillarHeight) / 2;
  let k = Math.floor(Math.random() * 3) - 1;
  Offset = Offset * k;
  let gapHeight = (1 / 3) * (this.scale.height - baseHeight);
  let lowerY = 2 * gapHeight + pillarHeight / 2 + Offset;
  let upperY = gapHeight - pillarHeight / 2 + Offset;
  let upperPillar = this.upperPillars.create(
    this.scale.width,
    upperY,
    "pillar"
  );
  upperPillar.setAngle(180);
  let lowerPillar = this.lowerPillars.create(
    this.scale.width,
    lowerY,
    "pillar"
  );
  upperPillar.body.allowGravity = false;
  lowerPillar.body.allowGravity = false;

  upperPillar.setVelocityX(speed);
  lowerPillar.setVelocityX(speed);
  this.pillarSpawnTime = this.time.now + spawnTime;
  console.log("Pillar spawned at", this.time.now);
};

function hitBase(character, base) {
  if (!hitPlayed && !trainingMode) hit.play();

  character.anims.play("fall", true);
  base.body.enable = false;
  character.setVelocityX(0);
  character.setVelocityY(0);
  character.body.allowGravity = false;
  [this.upperPillars, this.lowerPillars].forEach((group) =>
    group.children.iterate((pillar) => (pillar.body.velocity.x = 0))
  );
  this.isGameOver = true;
  let gameOverImage = this.add.image(
    this.scale.width / 2,
    this.scale.height / 4,
    "gameover"
  );
  gameOverImage.setOrigin(0.5, 0.5);
  let scoreImage = this.add.image(
    this.scale.width / 2,
    this.scale.height,
    "score"
  );
  scoreImage.setOrigin(0.5, 0.5);
  finalScoreText = this.add.text(
    this.scale.width / 2,
    this.scale.height,
    score,
    { fontSize: "32px", fontFamily: "Fantasy", fill: "white" }
  );
  finalScoreText.setOrigin(0.5, 0.5);
  this.tweens.add({
    targets: [scoreImage, finalScoreText],
    y: (target) => {
      return target === scoreImage
        ? this.scale.height / 2.2
        : this.scale.height / 2.1;
    },
    ease: "Power1",
    duration: 500,
    repeat: 0,
    yoyo: false,
  });
  scoreText.destroy();
  let retryImage = this.add.image(
    this.scale.width / 2,
    this.scale.height / 1.5,
    "retry"
  );
  retryImage.setOrigin(0.5, 0.5);
  retryImage.setScale(0.25);
  retryImage.setInteractive();
  retryImage.on(
    "pointerdown",
    function () {
      this.handleRestart();
    },
    this
  );
}

function hitPillar(character, pillar) {
  if (!hitPlayed && !diePlayed) {
    if (!trainingMode) {
      hit.play();
      die.play();
    }
    hitPlayed = true;
    diePlayed = true;
  }
  character.anims.play("fall", true);
  pillar.body.enable = false;
  character.setVelocityX(0);
  [this.upperPillars, this.lowerPillars].forEach((group) =>
    group.children.iterate((pillar) => (pillar.body.velocity.x = 0))
  );
  this.isGameOver = true;
}

window.getObservation = getObservation;
