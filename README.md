# Jump the Gap

[日本語](README.ja.md)

A small **world model** that imagines three ways to cross a gap before taking the jump. It draws the next 32×32 frame from recent frames and actions, then feeds its own drawing back in to continue through takeoff, flight, and landing.

**[Watch the demo in your browser](https://snc2work.github.io/jump_the_gap/)**

## What you'll see

- **Current** shows a recorded scene from a course held out of training.
- **Possible Futures** compares three plans from that scene: jump now, run one more step, or run two more steps. Brake timing also varies across the plans.
- The demo favors a prediction where the jumper stops on the far ledge. It replays the chosen prediction on the main screen, then moves to a different course. Ties are broken at random.

## How the model works

A simulator generates the training clips. The Transformer learns from binary images and four actions: coast, run, jump, and brake. Each step produces four candidate images and samples one. The training clips include jumps with the same visible lead-up and action sequence but slightly different takeoff speeds, so more than one landing outcome can fit the same input.

The browser runs the three-layer model (779,684 parameters) with ONNX Runtime Web inside a Web Worker. The demo reads the generated frames to rank the three plans by their predicted landing outcomes.

## Results

The two models were trained for 2,000 steps each using the same minibatch order. For the rollout comparison, each model sampled 12 futures from each of 24 held-out scenes with a fixed action sequence.

| Metric | 2 layers | 3 layers |
| --- | ---: | ---: |
| Exact next frame | 74.9% | 77.4% |
| Drew a stop or a fall after landing | 13/24 | 12/24 |
| Drew both outcomes | 0/24 | 3/24 |
| Exactly matched a simulator clip for either outcome | 0/24 | 2/24 |

Copying the previous frame gives 38.9% exact next frames. The results are from a single run with seed 42.

## Run locally

Requires Node.js 22.12 or later and pnpm 10.33.2. The trained ONNX model and seven starting scenes are included.

```sh
cd web
pnpm install --frozen-lockfile
pnpm dev
```

Open the URL printed by Vite. To build the static site, run `pnpm build` from `web/`.

## Reproduce the model

Run these commands from the repository root to generate the data, train the three-layer model, evaluate it, and export the browser assets.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate.py data
.venv/bin/python train.py --steps 2000 --layers 3 --output data/compare-3layers.pt
.venv/bin/python evaluate.py --layers 3 --checkpoint data/compare-3layers.pt
.venv/bin/python export_web.py
```

`export_web.py` updates the assets under `web/public/model/`. To save a standalone replay of sampled futures, run `.venv/bin/python export_preview.py` and open `data/preview.html`.

## License

[MIT](LICENSE)
