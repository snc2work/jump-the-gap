export type Frame = Uint8Array;
export type Action = 0 | 1 | 2 | 3;

export interface History {
  frames: Frame[];
  actions: Action[];
}

export interface Scene {
  observations: string;
  actions: Action[];
}

export type InferenceRequest =
  | { type: "load"; model: string }
  | { type: "rollout"; history: History; plan: Action[] };

export type WorkerRequest = InferenceRequest & { id: number };

export type WorkerResponse =
  | { id: number; frames: Frame[] }
  | { id: number; error: string };
