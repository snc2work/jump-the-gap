import type { Frame } from "./model-types";

const SIZE = 32;

export function readCourse(frame: Frame) {
  const floor = frame.subarray(SIZE * (SIZE - 1));
  const right = floor.lastIndexOf(1);
  const left = floor.lastIndexOf(0, right) + 1;
  let surface = SIZE - 1;
  while (surface > 0 && frame[(surface - 1) * SIZE + right]) surface--;
  const terrain = frame.map((pixel, index) => index >= surface * SIZE ? pixel : 0);
  return { left, right, surface, terrain };
}

type Player = { x: number; foot: number };

function findPlayer(frame: Frame, terrain: Frame): Player | null | undefined {
  let top = -1;
  let bottom = -1;
  for (let i = 0; i < frame.length; i++) {
    if (terrain[i] && !frame[i]) return undefined;
    if (!frame[i] || terrain[i]) continue;
    if (top === -1) top = i;
    else if (bottom === -1 && i === top + SIZE) bottom = i;
    else return undefined;
  }
  if (top === -1) return null;
  if (bottom === -1) {
    if (top < SIZE * (SIZE - 1)) return undefined;
    bottom = top;
  }
  return { x: bottom % SIZE, foot: Math.floor(bottom / SIZE) + 1 };
}

export function rankFuture(frames: Frame[], { left, right, surface, terrain }: ReturnType<typeof readCourse>) {
  const positions = frames.map(frame => findPlayer(frame, terrain));
  const onPlatform = (player: Player | null | undefined): player is Player =>
    player != null && player.foot === surface && player.x >= left && player.x <= right;
  const last = positions.at(-1);
  const clarity = 1 - positions.filter(player => player === undefined).length / frames.length;
  if (onPlatform(last)) {
    const stopped = positions.slice(-3).every(player => onPlatform(player) && player.x === last.x);
    const margin = Math.min(last.x - left, right - last.x) / SIZE;
    return stopped
      ? { score: 100 + clarity + margin, outcome: "Stops on the ledge" }
      : { score: 50 + clarity + margin, outcome: "On the ledge at the end" };
  }
  const contact = positions.findIndex(onPlatform);
  if (contact >= 0) {
    const ranOff = positions.slice(contact + 1).some(player =>
      player && player.x > right && player.foot > surface);
    return { score: 20 + clarity, outcome: ranOff ? "Lands, then falls" : "Lands, then loses the ledge" };
  }
  const distance = Math.min(SIZE, ...positions.flatMap(player => player
    ? [Math.max(left - player.x, player.x - right, 0) + 2 * Math.abs(player.foot - surface)] : []));
  const outcome = clarity === 1 ? "Misses the ledge" : "No clear landing";
  return { score: (SIZE - distance) / SIZE + clarity, outcome };
}
