export class Lidar {
  constructor(maxDistance) {
    this.maxDistance = maxDistance;
    this.collisions = new Array(180).fill([0, 0]);
  }

  scan(scene, playerX, playerY, playerAngle, upperPipes, lowerPipes, baseY) {
    const results = new Array(180).fill(0);
  }
}
