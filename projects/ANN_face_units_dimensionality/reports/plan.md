# Plan: marimo notebook — face-selective unit dimensionality in task-optimized vs. random ResNet18

## Context

The goal is to test whether face-selective units in a task-optimized ANN (ImageNet-trained ResNet18) show distinctive effective-dimensionality (ED) properties for face images, compared to (a) an untrained/random ResNet18 and (b) a matched set of random, non-face-selective units in the same layer. This is a first exploratory pass, intended to live as a marimo notebook in the currently-empty `projects/ANN_face_units_dimensionality/` directory, which is clearly the intended home for this analysis (per the repo's per-project sandbox convention).

The `compcats` library (`comparing-category-selectivity/compcats`) already provides everything needed except the dimensionality metric itself:
- Model loading with hook-based activation extraction and disk caching (`BaseModel.generate_dataset_activations`)
- A Grill-Spector-style face-selectivity localizer with t-test thresholding (`compcats.localizer.localizer.get_froi_mask`)
- CelebA-HQ and (via a small custom wrapper) ImageNet dataset loaders

There is no existing "effective dimensionality" / participation-ratio code anywhere in the repo, and no existing marimo notebook does this analysis — the closest prior art is `preprint/dev/explore_marimo/explore.py`, which computes a d-prime face-selectivity metric on the same Grill-Spector mask and establishes the notebook's cell-structure/style conventions (underscore-prefixed loop locals, `mo.status.progress_bar`, notebook-local helper functions rather than library changes).

Methodology reference: [Elmoznino & Bonner, PLOS Comp Biol 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011792) defines "effective dimensionality" as the **participation ratio** of the PCA eigenspectrum of layer activations, `ED = (Σλᵢ)² / Σλᵢ²`, and extracts ResNet18 activations at the output of each of its 8 residual blocks. We adopt this ED formula and this layer granularity. We deliberately do **not** adopt their global-average-pooling step, since face-selective units in this analysis are defined at the native (channel, h, w) granularity used throughout `compcats`'s existing localizer machinery (confirmed with the user) — pooling first would change what a "unit" means and break compatibility with `localizer.get_froi_mask`.

**Known, documented deviation from the cited paper:** the paper uses N=10,000 ImageNet validation images (no face-specific dataset). CelebA-HQ val only has 2,000 images total (1,000 male + 1,000 female) on this machine. Because ED estimates are sensitive to sample size, comparing an ImageNet ED computed at N=10,000 against a CelebA-HQ ED computed at N=2,000 would confound the "general vs. face-specific" comparison with a pure N effect. The notebook therefore uses a matched N=2,000 for **both** datasets (full CelebA-HQ val, and a fixed-seed random subsample of ImageNet val) — this is called out explicitly as a config constant and a code comment so it's easy to revisit.

## Design decisions (confirmed with user)

1. **Models**: `resnet18_imagenet1kv1` (ImageNet-trained) and `resnet18_random` (untrained, default seed=42).
2. **Layers**: the 8 residual-block outputs, i.e. the final ReLU of each `BasicBlock` — `layer1.0.relu2`, `layer1.1.relu2`, `layer2.0.relu2`, `layer2.1.relu2`, `layer3.0.relu2`, `layer3.1.relu2`, `layer4.0.relu2`, `layer4.1.relu2`. (Verified these are NOT excluded by the `irrelevant_layer_regexes` in `resnet18_imagenet1kv1.py`/`resnet18_random.py`, so they're valid hookable layer names.)
3. **Face-selective units**: defined via `localizer.get_froi_mask(GRILLSPECTOR_DIR, 'grillspector', MODELS_DIR, model, layer, 'FFA', tthreshold=7)` — the library's existing t-test-based FFA contrast (faces > bodies/places/objects/characters/scrambled), unit granularity = raw (channel, h, w) positions, no pooling.
4. **Effective dimensionality**: participation ratio on raw (un-pooled) masked units, computed notebook-locally (not added to the library, matching the `estimate_dprime` precedent).
5. **Matched random baseline**: for each layer, 5 repeated random draws (fixed seeds) of non-face-selective units, same count as the face-selective set; report mean ± std of ED across repeats.
6. **Datasets**: ImageNet val subsampled to N=2,000 (fixed seed) via a small custom flat-directory dataset; CelebA-HQ val used in full (N=2,000, `ImageFolder`-based, already supported by `compcats.datasets.image.celebahq.get_val_dataset`).

## Environment & execution

- All work happens inside the `catsel` conda environment (`conda activate catsel`), which already has `torch`, `torchvision`, `pandas`, `plotly`, and `marimo` installed. No new environment/dependency setup needed.
- All model forward passes (activation generation) run on **MPS** (`DEVICE = 'mps'`), with `num_workers=0` on every `DataLoader`/`generate_dataset_activations` call — Apple Silicon multiprocessing + MPS tensors don't mix well with worker processes, so this avoids deadlocks/crashes.
- The notebook will be built interactively using the **marimo-pair** skill: cells added/run incrementally via `marimo edit`, so outputs (activation shapes, unit counts, plots) can be reviewed together as the notebook takes shape, rather than writing the whole file blind and verifying only at the end.

## File layout

New files under `comparing-category-selectivity/projects/ANN_face_units_dimensionality/`:
- `face_unit_dimensionality.py` — the marimo notebook (this is the only code file)
- `models/` — created at runtime; `MODELS_DIR` for activation/mask disk caching (same convention as `preprint/models`)
- `results/` — created at runtime; CSV output
- `reports/` — created at runtime; markdown files:
  - `plan.md` — copy of this approved plan, for provenance
  - `run_report.md` — regenerated at the end of each sweep: run config (device, N, seeds), per-layer face-selective unit counts per model, and a headline summary table of ED results
- `plots/` — created at runtime only if a plot is explicitly exported to a static image; the primary plots live inline in the notebook as interactive widgets (see below), not as saved files

No changes to the shared `compcats` library or to `preprint/paths.py`.

## Notebook cell plan

1. **Imports & paths cell**
   - `torch`, `numpy`, `pandas`, `matplotlib.pyplot`, `marimo as mo`
   - `from compcats.models import resnet18_imagenet1kv1, resnet18_random`
   - `from compcats.localizer import localizer`
   - `from compcats.datasets.image import celebahq`
   - `from compcats.datasets.image.custom import CustomDataset`
   - `sys.path.append('../../preprint'); import paths` (to reach `GRILLSPECTOR_LOCALIZER_DIR`, `CELEBAHQ_VAL_DIR`)
   - Local dirs: `MODELS_DIR = 'models'`, `RESULTS_DIR = 'results'`, `PLOTS_DIR = 'plots'` (created with `os.makedirs(..., exist_ok=True)`)
   - `IMAGENET_VAL_DIR = '/Users/adipani3/datasets/imagenet/val'` (hardcoded locally since `paths.IMAGENET_VAL_DIR` is `None` — the flat on-disk layout isn't `ImageFolder`-compatible anyway)

2. **Config cell**: `FROI = 'FFA'`, `TTHRESHOLD = 7`, `N_STIMULI = 2000`, `N_RANDOM_REPEATS = 5`, `RANDOM_SEED = 42`, `BLOCK_LAYERS = [...]` (the 8 names above), `DEVICE = 'mps'`, `NUM_WORKERS = 0`.

3. **Models cell**: instantiate `[('resnet18_trained', resnet18_imagenet1kv1.get_model()), ('resnet18_random', resnet18_random.get_model())]`.

4. **Datasets cell**:
   - CelebA-HQ: `celebahq.get_val_dataset(paths.CELEBAHQ_VAL_DIR, model.transforms)` with `dataset.name = 'celebahq'` set manually (mirrors `model_utils.py`'s pattern) — built per-model since `model.transforms` differs by model.
   - ImageNet: list all files in `IMAGENET_VAL_DIR`, `np.random.default_rng(RANDOM_SEED).choice(..., size=N_STIMULI, replace=False)`, wrap with `CustomDataset(IMAGENET_VAL_DIR, sampled_filenames, 'imagenetval2000', model.transforms)`.

5. **Effective dimensionality function cell** (notebook-local, docstring cites the PR formula and the paper):
   ```python
   def participation_ratio(activations):
       # activations: (n_samples, n_units)
       x = activations.double() - activations.double().mean(dim=0, keepdim=True)
       s = torch.linalg.svdvals(x)
       eig = s ** 2
       return ((eig.sum() ** 2) / (eig.square().sum())).item()
   ```

6. **Random-unit-matching helper cell**: given a boolean `face_mask` (shape = raw layer output trailing dims) and `n_repeats`, return `n_repeats` boolean masks of the same shape, each with the same number of `True` entries as `face_mask`, sampled without replacement from the `~face_mask` positions using seeded `np.random.default_rng`.

7. **Main sweep cell** (the core loop, wrapped in `mo.status.progress_bar`):
   ```
   for model_label, model in models:
       for layer in BLOCK_LAYERS:
           face_mask = localizer.get_froi_mask(GRILLSPECTOR_DIR, 'grillspector', MODELS_DIR, model, layer, FROI, TTHRESHOLD)
           n_face = face_mask.sum().item()
           if n_face == 0:
               log & skip  # expected for early blocks
               continue
           random_masks = sample_matched_random_masks(face_mask, N_RANDOM_REPEATS, seed=RANDOM_SEED)
           for dataset_label, dataset in [('imagenet', imagenet_ds[model_label]), ('celebahq', celebahq_ds[model_label])]:
               raw_activations = model.generate_dataset_activations(
                   layer, dataset, save_dir=f'{MODELS_DIR}/{model.name}/{dataset.name}/activations',
                   batch_size=256, device=DEVICE, num_workers=NUM_WORKERS
               )  # shape (N_STIMULI, C, H, W)
               flat = raw_activations.reshape(raw_activations.shape[0], -1)
               ed_face = participation_ratio(flat[:, face_mask.flatten()])
               rows.append({..., unit_type='face_selective', repeat=None, n_units=n_face, ed=ed_face})
               for i, rmask in enumerate(random_masks):
                   ed_rand = participation_ratio(flat[:, rmask.flatten()])
                   rows.append({..., unit_type='random_matched', repeat=i, n_units=n_face, ed=ed_rand})
   ```
   Result: a long-format list of dicts → `pd.DataFrame(rows)`.

8. **Save results cell**: `results_df.to_csv(f'{RESULTS_DIR}/face_unit_dimensionality.csv', index=False)`.

9. **Interactive plot cells** (using `plotly.express`, already in the `catsel` environment, rendered natively by marimo; plus `mo.ui` widgets for reactive filtering):
   - `mo.ui.dropdown`/`mo.ui.radio` to pick model and/or dataset, feeding a reactive downstream cell (so the user can flip between views live in the notebook rather than re-reading 4 static panels).
   - "# face-selective units vs. block index" — interactive line/bar chart per model (hover shows exact counts), to contextualize where the t>7 threshold yields a non-trivial count (early blocks will likely be empty/near-zero — expected).
   - Main ED chart: `px.line` with block index (1–8) on x, ED on y, colored by `unit_type` (face_selective vs. random_matched), faceted by model and dataset (`facet_row='model'`, `facet_col='dataset'`), error bars/bands on the random-matched line from the 5-repeat std, hover tooltips showing `n_units` per point. This is the main plot for testing the hypothesis, and stays interactive (zoom/hover/toggle series) rather than a flat static image.

10. **Reports cell**: build `reports/run_report.md` (markdown, written via plain Python string formatting) after the sweep — run config, a table of `n_face` selective units per (model, layer), and a summary table of face-selective vs. random-matched ED per (model, layer, dataset). Also copy this approved plan to `reports/plan.md` for provenance.

## Verification

- Build and verify the notebook cell-by-cell using the **marimo-pair** skill and `marimo edit` (inside `conda activate catsel`), reviewing intermediate outputs (activation shapes, `n_face` counts, interactive charts) together as we go, rather than writing the whole file blind.
- As a final headless check, run `marimo export script face_unit_dimensionality.py -o /tmp/check.py` to confirm the notebook runs end-to-end without errors — there's no live browser rendering to check against otherwise, per this repo's marimo verification convention.
- Sanity-check: confirm `n_face` is 0 for the earliest blocks and becomes non-zero by `layer4.*` for `resnet18_trained` (expected, consistent with known face-selectivity literature), and check whether `resnet18_random` yields any face-selective units at all under t>7 (a key part of the hypothesis test — if the random network shows few/no face-selective units, the ED comparison for that model will mostly be skipped, which is itself a valid and reportable finding).
- Confirm `results/face_unit_dimensionality.csv` and `reports/run_report.md` are written with the expected row/section counts and no unexpected NaNs.
