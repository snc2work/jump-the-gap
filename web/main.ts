import "./style.css";
import { rankFuture, readCourse } from "./rank";
import type { Action, Frame, History, InferenceRequest, Scene, WorkerRequest, WorkerResponse } from "./model-types";

function getElement<T extends Element>(selector: string, root: ParentNode = document): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
}
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));
const worker = new Worker(new URL("./model-worker.ts", import.meta.url), { type: "module" });
const pending = new Map<number, {
  resolve: (frames: Frame[]) => void;
  reject: (error: Error) => void;
}>();
let requestId = 0;
let failure: Error | undefined;

function abortRequests(message: string) {
  failure = new Error(message);
  for (const { reject } of pending.values()) reject(failure);
  pending.clear();
}

worker.addEventListener("error", event => abortRequests(event.message || "Model worker stopped."));
worker.addEventListener("messageerror", () => abortRequests("Cannot read the model worker response."));
worker.addEventListener("message", ({ data }: MessageEvent<WorkerResponse>) => {
  const request = pending.get(data.id);
  if (!request) return;
  pending.delete(data.id);
  if ("error" in data) request.reject(new Error(data.error));
  else request.resolve(data.frames);
});

function infer(payload: InferenceRequest): Promise<Frame[]> {
  if (failure) return Promise.reject(failure);
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...payload, id } satisfies WorkerRequest);
  });
}

function draw(canvas: HTMLCanvasElement, frame: Frame) {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas rendering is unavailable.");
  const image = ctx.createImageData(32, 32);
  for (let i = 0; i < frame.length; i++) {
    image.data.set(frame[i] ? [41, 48, 32, 255] : [186, 196, 162, 255], i * 4);
  }
  ctx.putImageData(image, 0, 0);
}

function decode(encoded: string, count: number): Frame[] {
  const bytes = atob(encoded);
  return Array.from({ length: count }, (_, frame) =>
    Uint8Array.from({ length: 1024 }, (_, index) =>
      (bytes.charCodeAt(frame * 128 + (index >> 3)) >> (7 - (index & 7))) & 1));
}

const choices = [
  { run: 0, brake: 10 },
  { run: 1, brake: 12 },
  { run: 2, brake: 10 },
];

const actionNames = ["Coast", "Run", "Jump", "Brake"];

function actionsFor({ run, brake }: (typeof choices)[number]): Action[] {
  const runSteps = Array.from({ length: run }, (): Action => 1);
  const afterJump = Array.from({ length: 20 }, (_, step): Action => step >= brake ? 3 : 0);
  return [...runSteps, 2, ...afterJump];
}

function describeActions(plan: Action[]) {
  const groups: { action: Action; count: number }[] = [];
  for (const action of plan) {
    const previous = groups.at(-1);
    if (previous?.action === action) previous.count++;
    else groups.push({ action, count: 1 });
  }
  return groups.map(({ action, count }) =>
    actionNames[action] + (count > 1 ? ` × ${count}` : "")).join(" → ");
}

const cards = choices.map((choice, index) => {
  const template = getElement<HTMLTemplateElement>("#future");
  const card = document.importNode(getElement<HTMLElement>("article", template.content), true);
  const canvas = getElement<HTMLCanvasElement>("canvas", card);
  const heading = getElement<HTMLHeadingElement>("h3", card);
  const result = getElement<HTMLElement>(".outcome", card);
  const title = `Future ${index + 1}`;
  const plan = actionsFor(choice);
  heading.textContent = title;
  canvas.setAttribute("aria-label", title);
  getElement<HTMLElement>(".plan", card).textContent = describeActions(plan);
  getElement("#futures").append(card);
  return { card, canvas, heading, result, title, plan };
});

async function play(canvas: HTMLCanvasElement, frames: Frame[], interval: number) {
  if (reducedMotion) {
    const last = frames.at(-1);
    if (last) draw(canvas, last);
    await sleep(450);
    return;
  }
  for (const frame of frames) {
    draw(canvas, frame);
    await sleep(interval);
  }
}

function randomScene(count: number, previous: number) {
  if (previous === -1 || count === 1) return Math.floor(Math.random() * count);
  const pick = Math.floor(Math.random() * (count - 1));
  return pick < previous ? pick : pick + 1;
}

async function runTurn(scene: Scene) {
  const status = getElement("#status");
  const current = getElement<HTMLCanvasElement>("#current");
  const history: History = {
    frames: decode(scene.observations, scene.actions.length + 1),
    actions: scene.actions,
  };
  const start = history.frames.at(-1);
  if (!start) throw new Error("The starting scene has no frames.");
  const course = readCourse(start);
  const thinker = getElement(".thinker");
  const source = getElement("#source");
  for (const { card, heading, result, title } of cards) {
    card.classList.remove("revealed", "selected");
    heading.textContent = title;
    result.textContent = "";
  }
  draw(current, start);
  source.textContent = "Starting frame from the recorded scene.";
  const past = describeActions(history.actions);
  getElement("#model-input").textContent = `Model input: ${history.frames.length} observed frames and their actions (${past}), followed by the plan below. The world model draws each next frame.`;
  status.textContent = "New cliff";
  thinker.classList.add("thinking");
  await sleep(320);

  const predictions = [];
  for (const [index, item] of cards.entries()) {
    status.textContent = `Imagining ${index + 1} of ${cards.length}…`;
    const frames = await infer({ type: "rollout", history, plan: item.plan });
    const evaluation = rankFuture(frames, course);
    predictions.push({ item, frames, ...evaluation });
    item.card.classList.add("revealed");
    await play(item.canvas, [start, ...frames], 52);
    item.result.textContent = evaluation.outcome;
  }

  const highScore = Math.max(...predictions.map(prediction => prediction.score));
  const bestOptions = predictions.filter(prediction => prediction.score === highScore);
  const best = bestOptions[Math.floor(Math.random() * bestOptions.length)];
  thinker.classList.remove("thinking");
  best.item.card.classList.add("selected");
  best.item.heading.textContent = best.item.title + " · Selected";
  const reason = bestOptions.length > 1 ? "Tied for best" : best.outcome;
  status.textContent = `${best.item.title} selected · ${reason}`;
  await sleep(460);
  status.textContent = "Taking the jump…";
  source.textContent = "Frames drawn by the world model.";
  await play(current, best.frames, 70);
  status.textContent = "Next cliff…";
  await sleep(650);
}

try {
  const modelBase = new URL(import.meta.env.BASE_URL + "model/", document.baseURI);
  const response = await fetch(new URL("seeds.json", modelBase));
  if (!response.ok) throw new Error(`Could not load starting scenes (HTTP ${response.status}).`);
  const seeds: Scene[] = await response.json();
  if (!seeds.length) throw new Error("No starting scenes.");
  await infer({ type: "load", model: new URL("world.onnx", modelBase).href });
  let previous = -1;
  while (true) {
    previous = randomScene(seeds.length, previous);
    await runTurn(seeds[previous]);
  }
} catch (error) {
  getElement("#status").textContent = "The demo stopped. Reload to try again.";
  console.error(error);
}
