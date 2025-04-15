export class Lidar {
  constructor(maxDistance) {
    this.maxDistance = maxDistance;
    this.collisions = new Array(180).fill([0, 0]);
  }

  draw(scene, playerX, playerY, playerWidth, playerHeight, graphics) {
    for (let i = 0; i < this.collisions.length; i++) {
      const [endX, endY] = this.collisions[i];
      graphics.lineStyle(1, 0xff0000, 1);
      graphics.beginPath();
      graphics.moveTo(playerX, playerY);
      graphics.lineTo(endX, endY);
      graphics.strokePath();
    }
  }

  scan(
    playerX,
    playerY,
    playerRot,
    playerWidth,
    playerHeight,
    upperPipes,
    lowerPipes,
    groundY,
    scene
  ) {
    const result = new Array(180);
    const offsetX = playerX; //+ playerWidth;
    const offsetY = playerY; //+ playerHeight / 2;

    const visibleRot = Math.min(playerRot, 45); // replicate PLAYER_ROT_THR behavior

    const upper = [...upperPipes.getChildren()].sort((a, b) => a.x - b.x);
    const lower = [...lowerPipes.getChildren()].sort((a, b) => a.x - b.x);

    for (let i = 0; i < 180; i++) {
      const angleDeg = i - 90 + visibleRot;
      const rad = Phaser.Math.DegToRad(angleDeg);
      const dx = this.maxDistance * Math.cos(rad);
      const dy = this.maxDistance * Math.sin(rad);
      const endX = offsetX + dx;
      const endY = offsetY + dy;
      this.collisions[i] = [endX, endY];

      let closestCollision = { x: endX, y: endY, dist: this.maxDistance };

      // Ground collision (horizontal line at groundY)
      if (endY > groundY) {
        const t = (groundY - offsetY) / (endY - offsetY);
        const groundX = offsetX + t * (endX - offsetX);
        const dist = Math.sqrt(
          (groundX - offsetX) ** 2 + (groundY - offsetY) ** 2
        );

        if (dist < closestCollision.dist) {
          closestCollision.x = groundX;
          closestCollision.y = groundY;
          closestCollision.dist = dist;
        }
      }

      // Ceiling collision
      const ceilingY = 0; //set to f.e. 20 for debugging
      if (endY < ceilingY) {
        const t = (ceilingY - offsetY) / (endY - offsetY);
        const ceilingX = offsetX + t * (endX - offsetX);
        const dist = Math.sqrt(
          (ceilingX - offsetX) ** 2 + (ceilingY - offsetY) ** 2
        );

        if (dist < closestCollision.dist) {
          closestCollision.x = ceilingX;
          closestCollision.y = ceilingY;
          closestCollision.dist = dist;
        }
      }

      // Pipe collision
      for (let j = 0; j < Math.min(upper.length, lower.length); j++) {
        const up = upper[j].getBounds();
        const low = lower[j].getBounds();

        const collisionA = this.intersectRectLine(
          up,
          offsetX,
          offsetY,
          endX,
          endY
        );
        const collisionB = this.intersectRectLine(
          low,
          offsetX,
          offsetY,
          endX,
          endY
        );

        if (collisionA && collisionA.dist < closestCollision.dist) {
          closestCollision = collisionA;
        } else if (collisionB && collisionB.dist < closestCollision.dist) {
          closestCollision = collisionB;
        }
      }

      this.collisions[i] = [closestCollision.x, closestCollision.y];
      result[i] = closestCollision.dist / this.maxDistance;
    }

    return result;
  }

  intersectRectLine(rect, x1, y1, x2, y2) {
    const lines = [
      [rect.left, rect.top, rect.right, rect.top], // top
      [rect.right, rect.bottom, rect.left, rect.bottom], // bottom
      [rect.left, rect.bottom, rect.left, rect.top], // left
    ];

    let closest = null;
    for (const [x3, y3, x4, y4] of lines) {
      const pt = this.lineLineIntersection(x1, y1, x2, y2, x3, y3, x4, y4);
      if (pt) {
        const dx = x1 - pt.x;
        const dy = y1 - pt.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (!closest || dist < closest.dist) {
          closest = { x: pt.x, y: pt.y, dist };
        }
      }
    }
    return closest;
  }

  lineLineIntersection(x1, y1, x2, y2, x3, y3, x4, y4) {
    var ua,
      ub,
      denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if (denom === 0) return null;

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;

    if (ua < 0 || ua > 1 || ub < 0 || ub > 1) return null;

    return {
      x: x1 + ua * (x2 - x1),
      y: y1 + ua * (y2 - y1),
    };
  }
}
