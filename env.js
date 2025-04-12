// env.js

export class GameEnvironment {
  constructor(scene) {
    this.scene = scene;
    this.resetRequested = false;
    this.prevAlive = true;
    this.frameAliveBonus = 0.1;
    this.pipeReward = 1.0;
    this.ceilingPenalty = -0.5;
    this.deathPenalty = -1.0;
    this.totalScore = 0;
    this.lastScore = 0;
  }

  reset() {
    this.scene.handleRestart(); // Reset scene state
    this.scene.handleGameStart(); // Ensure groups and physics are initialized
    this.prevAlive = true;
    this.totalScore = 0;
    this.lastScore = 0;
  }

  step(action) {
    if (action === 1 && !this.scene.isGameOver) {
      this.scene.handleFlap();
    }

    this.scene.physics.world.step(1 / 60);
    this.scene.update(); // run game logic that spawns pipes, moves base etc

    const observation = window.getObservation(this.scene);
    const done = this.scene.isGameOver;

    // 🛡️ Check for NaNs or undefined observations
    if (!Array.isArray(observation) || observation.some(Number.isNaN)) {
      console.warn("Skipping step: invalid observation", observation);
      return {
        observation: [0, 0, 0, 0],
        reward: 0,
        done: true,
      };
    }

    const reward = this._computeReward();

    this.prevAlive = !done;
    return { observation, reward, done };
  }

  _computeReward() {
    const isDead = this.scene.isGameOver;
    let reward = 0;

    // +0.1 for surviving
    reward += this.frameAliveBonus;

    // +1.0 if score increased
    const currentScore = this.scene.score;
    if (currentScore > this.lastScore) {
      reward += this.pipeReward;
      this.lastScore = currentScore;
    }

    // -0.5 if bird touches top (ceiling)
    if (this.scene.character.y <= 0) {
      reward += this.ceilingPenalty;
    }

    // -1.0 if bird dies
    if (isDead && this.prevAlive) {
      reward += this.deathPenalty;
    }

    return reward;
  }

  getObservation() {
    return window.getObservation(this.scene);
  }

  isDone() {
    return this.scene.isGameOver;
  }
}
