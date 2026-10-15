# Face-selective units and effective dimensionality: findings

**Source notebook**: `face_unit_dimensionality.py`
**Data**: `results/face_unit_dimensionality.csv` (2 models × 16 layers × 3 datasets × 6 unit-conditions, 576 rows)
**Models**: `resnet18_trained` (ImageNet-1k v1 pretrained), `resnet18_random` (untrained, seed=42)
**Layers**: all 16 ReLU layers of ResNet18 (relu1 + relu2 per residual block, 8 blocks)
**Datasets**: ImageNet-val, CelebA-HQ-val, AFHQv2-test, matched to N=1468 images each
**Units**: face-selective (Grill-Spector FFA localizer, t≥7) vs. an equal-size random-matched
non-selective baseline (5 repeats, mean reported)
**Metric**: effective dimensionality (ED) = participation ratio of the PCA eigenspectrum,
`ED = (Σλᵢ)² / Σλᵢ²`, computed on raw (un-pooled) masked units

---

## Finding 1 — Face-selective units compress dimensionality specifically on their preferred category

In the trained network, face-selective units represent face-like images (CelebA-HQ, AFHQv2) in a
markedly lower-dimensional code than they use for general images (ImageNet). Random, non-selective
units in the *same* trained network show the opposite pattern.

**Evidence** — mean ED ratio (face-dataset ÷ ImageNet), averaged across all 16 layers:

| Unit type | celebahq / imagenet | afhqv2 / imagenet |
|---|---|---|
| face_selective | **0.51** (median 0.41) | **0.45** (median 0.38) |
| random_matched | 1.13 (median 1.25) | 1.45 (median 1.47) |

Mean absolute ED (`resnet18_trained`): imagenet=73.8, celebahq=31.4, afhqv2=27.1 (face-selective
units) vs. imagenet=84.9, celebahq=84.3, afhqv2=105.1 (random-matched units).

In the untrained network (`resnet18_random`), both unit types stay close to a 1:1 ratio
(0.80–1.10), so this compression is not present before training — it is learned, not an
architectural artifact.

**Interpretation**: this is a real unit-type × dataset interaction, not a property of the face
datasets themselves being generically "lower-dimensional" (if it were, random units would compress
too — instead they expand). Face-selective units specifically collapse their preferred category
into a compact subspace.

---

## Finding 2 — Face-selective units keep a stimulus-general dimensionality "shape" across depth; generic units don't

For each unit population, we correlated its per-layer ED profile (16 points, one per layer) across
pairs of datasets, separately per model.

**Evidence** — Pearson r between per-layer ED on dataset pairs:

| Model | Dataset pair | face_selective | random_matched |
|---|---|---|---|
| resnet18_random | afhqv2 vs celebahq | r=+0.867 | r=+0.975 |
| resnet18_random | afhqv2 vs imagenet | r=+0.989 | r=+0.995 |
| resnet18_random | celebahq vs imagenet | r=+0.880 | r=+0.980 |
| resnet18_trained | afhqv2 vs celebahq | r=+0.860 | r=+0.919 |
| resnet18_trained | afhqv2 vs imagenet | r=+0.724 | **r=−0.064 (n.s., p=0.81)** |
| resnet18_trained | celebahq vs imagenet | r=+0.896 | **r=+0.189 (n.s., p=0.48)** |

(All non-flagged correlations significant at p<0.002, n=16 layers each.)

**Interpretation**:
- The untrained network shows uniformly high correlation everywhere (r=0.87–0.995) for both unit
  types. With random weights there is no learned structure — a layer's ED is set almost entirely by
  architecture (channel count, spatial extent, mask size), which is dataset-independent. This is the
  architectural null baseline.
- In the trained network, face-selective units keep that same high-correlation pattern across all
  three dataset pairs (r=0.72–0.90), including against ImageNet.
- Random-matched units in the trained network keep correlating with each other for the two
  face-like sets (celebahq vs afhqv2, r=0.92) but their relationship with ImageNet collapses to
  noise (r≈−0.06, r≈0.19, both n.s.).
- Proposed mechanism: training tunes the bulk (non-selective) population specifically to ImageNet's
  compositional statistics (diverse objects/scenes/viewpoints); their layer-by-layer ED profile
  becomes a signature of that particular training distribution and doesn't hold up on a
  structurally different, narrower stimulus set (single cropped, centered portrait/animal-face
  images). Face-selective units don't show this overfitting — their dimensionality profile is
  portable across all three stimulus distributions, including the one that decorrelates everyone
  else.
- celebahq and afhqv2 correlate with each other even for random units, in both models — plausibly
  because they share low-level compositional statistics (single subject, centered, uniform
  background) that are quite different from ImageNet's heterogeneity, independent of anything
  face-specific. This makes the ImageNet comparisons the more diagnostic ones.

---

## Interpretation, connected to the literature

**Two things are happening together**: face-selective units use a *compressed* code specifically on
faces (Finding 1), and *the shape of their dimensionality profile across depth is preserved*
regardless of what stimulus set they're shown (Finding 2) — including ImageNet, where the generic
population's profile is not preserved. A systematic compression with a preserved shape is not a
contradiction (scale a curve down by half and its correlation with the original stays high); the two
findings together paint one coherent picture: face-selective units carry a compact, stimulus-general
dimensionality signature, while the rest of the trained network's dimensionality structure is
comparatively overfit to ImageNet's specific compositional statistics.

**Classical face-space neuroscience predicts exactly the compression finding.**
[Chang & Tsao (2017, *Cell*)](https://www.cell.com/fulltext/S0092-8674(17)30538-X) found that macaque
face-patch cells encode facial identity along a remarkably compact ~50-dimensional linear "face
space" (25 shape + 25 appearance axes), with each cell's firing rate proportional to a single axis
projection — precise enough to reconstruct a face from population activity. This is the strongest
empirical precedent for "face-selective coding is intrinsically low-dimensional and structured,"
and it's qualitatively what we found: face-selective units collapsing to ~0.4–0.5× their
general-image dimensionality specifically on face-like stimuli.

**The dimensionality/generalization tradeoff literature explains *why* compression would also
produce portability.**
[Bernardi et al. (2020, *Cell*)](https://www.sciencedirect.com/science/article/pii/S0092867420312289),
"The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex," formalize a tradeoff that
maps closely onto Finding 2: representations expanded into high dimensionality support flexible,
fine-grained linear decoding *within* a training distribution, while representations compressed into
low-dimensional, abstract structure generalize *across* contexts. Generic units stay high-D and
tuned to ImageNet's specific structure (flexible but not portable); face-selective units compress
into a low-D, abstract code that survives distribution shift (less flexible but more general). This
is essentially the Bernardi tradeoff playing out inside a single trained CNN, split by
category-selectivity rather than by task variable.

**This sits in real tension with the paper whose ED methodology this notebook uses.**
[Elmoznino & Bonner (2024, *PLOS Comp Biol*)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011792)
argue that models with *higher* effective dimensionality (measured the same way, at the
whole-population level) better predict brain and behavioral data. Our result doesn't necessarily
contradict that (they measured whole populations on generic images; we measured a
functionally-defined subpopulation on its preferred category), but it raises a question their paper
doesn't address: is the "high-D is better" relationship actually carried by generic units, while
category-selective subpopulations behave in the opposite direction on their own preferred category?
That's a testable extension of their work that, as far as we found, nobody has run.

**This is squarely inside the debate the host lab's own paper is having.** This repository is built
around
[Dipani & Ratan Murty (2025, *Journal of Vision*)](https://jov.arvojournals.org/article.aspx?articleid=2809730),
"Category selectivity observed in the human FFA, PPA, and EBA is distinct from category selectivity
observed in artificial neural network units" — asking whether t-test-defined ANN "face-selective
units" are functionally/geometrically analogous to real category-selective brain regions, or only
superficially similar. This dimensionality result is a new data point for that question: a specific
axis (preferred-category compression + cross-distribution portability) where the ANN units *do*
converge with what's known about real face patches (Chang & Tsao), even if the broader paper's
thesis is that ANN and brain selectivity diverge on other measured properties. Worth checking
whether this reconciles as "distinct in tuning/geometry generally, but convergent on this
dimensionality signature specifically" — or whether closer inspection shows the ANN compression is
structurally different from the real face-space axes (e.g. not actually low-rank/linear in the same
way).

**Recent adjacent work.**
[Prince et al. (2024, *Science Advances*)](https://www.science.org/doi/10.1126/sciadv.adl1776) showed
category-selective units emerge in self-supervised (not just supervised) models, and that lesioning
them produces selective, dissociable behavioral deficits — part of a broader current push to ask not
just whether ANN units look selective, but whether they *function* like the brain regions they're
named after. This dimensionality result is a natural addition to that functional-comparison toolkit.

---

## Open question this result speaks to

**Is dimensionality expansion uniformly "good" / brain-like, or is there a selectivity-dependent
tradeoff** — where the field's general "higher effective dimensionality → better brain/behavior
alignment" claims (Elmoznino & Bonner) hold at the level of generic, non-selective populations, while
functionally specialized (category-selective) subpopulations achieve their specialization precisely
*by* compressing dimensionality on their preferred category, consistent with decades of face-space
and abstraction-geometry work (Chang & Tsao 2017; Bernardi et al. 2020)? This dataset can't settle
that question, but it's a clean, direct instance of it.

---

## Limitations

- Single architecture (ResNet18), single trained checkpoint, single random-network seed —
  correlational, not yet replicated.
- The macaque face-space work used a very different operationalization (single linear axes fit to
  real facial features) than participation ratio on a masked unit population; the connection drawn
  above is conceptual, not a direct quantitative match.
- CelebA-HQ and AFHQv2 share low-level compositional statistics (centered single subject, uniform
  background) that ImageNet lacks; some of the reported effects may partly reflect this
  composition difference rather than face-selectivity specifically — the random-unit control
  partially addresses this (it shows the opposite pattern), but a stimulus-statistics-matched
  control set would strengthen the claim further.
- fROI localization uses a single t-threshold (t≥7); unit counts per layer vary substantially
  (see `run_report.md`), which was not explicitly controlled for beyond the random-matched
  baseline being size-matched per layer.

## Suggested next steps

- Replicate across multiple random-network seeds and at least one additional trained architecture
  (e.g. ResNet50, a self-supervised/contrastive model per Prince et al. 2024) to check robustness.
- Directly test whether the ANN face-selective compression is structurally low-rank/axis-like in the
  way Chang & Tsao's macaque face space is (e.g. fit a small linear basis and check reconstruction
  quality), rather than only comparing scalar participation-ratio values.
- Split the Elmoznino & Bonner-style dimensionality-vs-brain-alignment analysis by category
  selectivity to test whether their "higher ED predicts better alignment" relationship is being
  carried by generic units alone.
