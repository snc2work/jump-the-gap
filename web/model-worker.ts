import { env, InferenceSession, Tensor } from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import moduleUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import type { Action, Frame, History, WorkerRequest, WorkerResponse } from "./model-types";

const CONTEXT = 32;
const PIXELS = 32 * 32;
const MODES = 4;

env.wasm.numThreads = 1;
env.wasm.wasmPaths = { wasm: wasmUrl, mjs: moduleUrl };
const worker = self as unknown as DedicatedWorkerGlobalScope;
let session: InferenceSession | undefined;

function drawMode(weights: Float32Array) {
  const max = Math.max(...weights);
  const masses = weights.map(weight => Math.exp(weight - max));
  let threshold = Math.random() * masses.reduce((sum, weight) => sum + weight, 0);
  for (let mode = 0; mode < MODES - 1; mode++) {
    threshold -= masses[mode];
    if (threshold < 0) return mode;
  }
  return MODES - 1;
}

async function rollout(session: InferenceSession, history: History, plan: Action[]) {
  const frames = [...history.frames];
  const actions = [...history.actions];
  const future: Frame[] = [];
  const pixels = new Uint8Array(CONTEXT * PIXELS);
  const commands = new BigInt64Array(CONTEXT);
  for (const action of plan) {
    actions.push(action);
    const start = Math.max(0, frames.length - CONTEXT);
    const count = frames.length - start;
    pixels.fill(0);
    commands.fill(0n);
    for (let offset = 0; offset < count; offset++) {
      pixels.set(frames[start + offset], offset * PIXELS);
      commands[offset] = BigInt(actions[start + offset]);
    }
    const { weights, logits } = await session.run({
      frames: new Tensor("uint8", pixels, [1, CONTEXT, 32, 32]),
      actions: new Tensor("int64", commands, [1, CONTEXT]),
      index: new Tensor("int64", BigInt64Array.of(BigInt(count - 1)), [1]),
    });
    if (!(weights.data instanceof Float32Array) || !(logits.data instanceof Float32Array)) {
      throw new Error("Expected float32 model outputs.");
    }
    const mode = drawMode(weights.data);
    const frame = Uint8Array.from(logits.data.subarray(mode * PIXELS, (mode + 1) * PIXELS), logit => Number(logit > 0));
    frames.push(frame);
    future.push(frame);
  }
  return future;
}

worker.onmessage = async ({ data }: MessageEvent<WorkerRequest>) => {
  const { id } = data;
  try {
    if (data.type === "load") {
      session = await InferenceSession.create(data.model, { executionProviders: ["wasm"] });
      worker.postMessage({ id, frames: [] } satisfies WorkerResponse);
    } else {
      if (!session) throw new Error("The world model is not loaded.");
      const frames = await rollout(session, data.history, data.plan);
      worker.postMessage({ id, frames } satisfies WorkerResponse);
    }
  } catch (error) {
    worker.postMessage({ id, error: String(error) } satisfies WorkerResponse);
  }
};
