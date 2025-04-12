// ai.js
const tf = window.tf;

// ===============================
// Experience Replay Buffer
// ===============================
export class ExperienceReplay {
  constructor(capacity, batchSize) {
    this.capacity = capacity;
    this.batchSize = batchSize;
    this.buffer = [];
  }

  push(observation, action, reward, nextObservation, done) {
    if (this.buffer.length >= this.capacity) {
      this.buffer.shift(); // remove oldest
    }
    this.buffer.push({ observation, action, reward, nextObservation, done });
  }

  sample() {
    if (this.buffer.length < this.batchSize) return null;
    const batch = [];
    const indices = tf.util.createShuffledIndices(this.buffer.length);
    for (let i = 0; i < this.batchSize; i++) {
      batch.push(this.buffer[indices[i]]);
    }
    return batch;
  }

  size() {
    return this.buffer.length;
  }
}

// ===============================
// Q-Network
// ===============================
export function createQNet(inputSize, outputSize, hiddenSizes = [256, 256]) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: hiddenSizes[0],
      inputShape: [inputSize],
      activation: "relu",
    })
  );

  for (let i = 1; i < hiddenSizes.length; i++) {
    model.add(
      tf.layers.dense({
        units: hiddenSizes[i],
        activation: "relu",
      })
    );
  }

  model.add(
    tf.layers.dense({
      units: outputSize,
      activation: "linear",
    })
  );

  return model;
}

// ===============================
// Huber Loss
// ===============================
function huberLoss(yTrue, yPred) {
  const err = tf.sub(yTrue, yPred);
  const absErr = tf.abs(err);
  const quadratic = tf.minimum(absErr, 1.0);
  const linear = tf.sub(absErr, quadratic);
  return tf.add(tf.mul(0.5, tf.square(quadratic)), linear).mean();
}

// ===============================
// DQN Agent
// ===============================
export class DQNAgent {
  constructor(
    inputSize,
    outputSize,
    { gamma = 0.99, lr = 0.01, memoryCapacity = 10000, batchSize = 64 } = {}
  ) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.gamma = gamma;
    this.batchSize = batchSize;
    this.memory = new ExperienceReplay(memoryCapacity, batchSize);
    this.model = createQNet(inputSize, outputSize);
    this.optimizer = tf.train.adam(lr);
  }

  act(observation, epsilon) {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * 0.6 * this.outputSize); // explore
    } else {
      return tf.tidy(() => {
        const input = tf.tensor2d([observation]);
        const qValues = this.model.predict(input);
        return qValues.argMax(1).dataSync()[0]; // exploit
      });
    }
  }

  async optimizeModel() {
    if (this.memory.size() < this.batchSize) return;

    const batch = this.memory.sample();
    if (!batch) return;
    if (batch.some((e) => !e || !e.observation || !e.nextObservation)) {
      console.warn("Skipping batch with invalid entries", batch);
      return;
    }

    const { observations, actions, rewards, nextObservations, dones } =
      this._unpackBatch(batch);

    await this.optimizer.minimize(() => {
      const qValues = this.model.predict(observations);
      const actionMasks = tf.oneHot(actions, this.outputSize);
      const chosenQ = tf.sum(tf.mul(qValues, actionMasks), 1);

      const nextQValues = this.model.predict(nextObservations);
      const maxNextQ = nextQValues.max(1);
      const targetQ = tf.add(rewards, tf.mul(tf.sub(1, dones), maxNextQ));

      return huberLoss(targetQ, chosenQ);
    });

    tf.dispose([observations, actions, rewards, nextObservations, dones]);
  }

  _unpackBatch(batch) {
    if (!batch || batch.length === 0) {
      throw new Error("Attempted to unpack empty batch");
    }

    const obs = [];
    const acts = [];
    const rews = [];
    const nextObs = [];
    const dones = [];

    for (const e of batch) {
      // 🛡 Skip incomplete samples
      if (
        !Array.isArray(e.observation) ||
        !Array.isArray(e.nextObservation) ||
        typeof e.reward !== "number" ||
        typeof e.action !== "number"
      ) {
        console.warn("Skipping invalid experience:", e);
        continue;
      }

      obs.push(e.observation);
      acts.push(e.action);
      rews.push(e.reward);
      nextObs.push(e.nextObservation);
      dones.push(e.done ? 1 : 0);
    }

    if (obs.length === 0) {
      throw new Error("No valid data in batch after filtering");
    }

    return {
      observations: tf.tensor2d(obs),
      actions: tf.tensor1d(acts, "int32"),
      rewards: tf.tensor1d(rews),
      nextObservations: tf.tensor2d(nextObs),
      dones: tf.tensor1d(dones),
    };
  }
}
