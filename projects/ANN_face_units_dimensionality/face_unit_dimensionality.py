import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    import datetime
    from itertools import combinations
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    import torch
    import marimo as mo
    from PIL import Image
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, Whisker
    from bokeh.layouts import gridplot
    from bokeh.palettes import Category10
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    # Custom packages
    from compcats.models import resnet18_imagenet1kv1, resnet18_random
    from compcats.localizer import localizer
    from compcats.localizer import utils as localizer_utils
    from compcats.datasets.image import celebahq, afhqv2

    # Custom scripts
    sys.path.append('../../preprint')
    import paths

    # Directories
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    REPORTS_DIR = 'reports'
    PLOTS_DIR = 'plots'
    for _d in (MODELS_DIR, RESULTS_DIR, REPORTS_DIR, PLOTS_DIR):
        os.makedirs(_d, exist_ok=True)

    GRILLSPECTOR_DIR = paths.GRILLSPECTOR_LOCALIZER_DIR
    CELEBAHQ_DIR = paths.CELEBAHQ_VAL_DIR
    AFHQV2_DIR = paths.AFHQV2_VAL_DIR
    # paths.IMAGENET_VAL_DIR is None (flat on-disk layout isn't ImageFolder-compatible
    # anyway), so this is hardcoded to the actual local path instead.
    IMAGENET_VAL_DIR = '/Users/adipani3/datasets/imagenet/val'
    return (
        AFHQV2_DIR,
        CDN,
        CELEBAHQ_DIR,
        Category10,
        ColumnDataSource,
        GRILLSPECTOR_DIR,
        HoverTool,
        IMAGENET_VAL_DIR,
        Image,
        MODELS_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
        Whisker,
        afhqv2,
        celebahq,
        datetime,
        file_html,
        figure,
        gridplot,
        combinations,
        localizer,
        localizer_utils,
        mo,
        np,
        os,
        pd,
        pearsonr,
        resnet18_imagenet1kv1,
        resnet18_random,
        torch,
    )


@app.cell
def _():
    # Run configuration
    FROI = 'FFA'
    TTHRESHOLD = 7
    # Matched across all three datasets (effective-dimensionality estimates are
    # sensitive to sample size). AFHQv2 test is the smallest available set
    # (493 cat + 491 dog + 483 wild = 1468), so it sets the ceiling; CelebA-HQ
    # val (2000 available) and ImageNet val (50000 available) are subsampled
    # down to match.
    N_STIMULI = 1468
    N_RANDOM_REPEATS = 5
    RANDOM_SEED = 42
    DEVICE = 'mps'
    NUM_WORKERS = 0  # MPS + multiprocessing workers don't mix well
    BATCH_SIZE = 256
    # Every ReLU inside every ResNet18 residual block (relu1 = after conv1/bn1,
    # relu2 = block output after the residual add), 16 total across the 8 blocks.
    SWEEP_LAYERS = [
        'layer1.0.relu1', 'layer1.0.relu2', 'layer1.1.relu1', 'layer1.1.relu2',
        'layer2.0.relu1', 'layer2.0.relu2', 'layer2.1.relu1', 'layer2.1.relu2',
        'layer3.0.relu1', 'layer3.0.relu2', 'layer3.1.relu1', 'layer3.1.relu2',
        'layer4.0.relu1', 'layer4.0.relu2', 'layer4.1.relu1', 'layer4.1.relu2',
    ]
    return (
        BATCH_SIZE,
        SWEEP_LAYERS,
        DEVICE,
        FROI,
        NUM_WORKERS,
        N_RANDOM_REPEATS,
        N_STIMULI,
        RANDOM_SEED,
        TTHRESHOLD,
    )


@app.cell
def _(resnet18_imagenet1kv1, resnet18_random):
    models = [
        ('resnet18_trained', resnet18_imagenet1kv1.get_model()),
        ('resnet18_random', resnet18_random.get_model()),
    ]
    return (models,)


@app.cell
def _(BATCH_SIZE, SWEEP_LAYERS, DEVICE, GRILLSPECTOR_DIR, MODELS_DIR, localizer_utils, mo, models):
    # Pre-generate & disk-cache Grill-Spector localizer activations for every
    # image category and all 16 swept layers, in one forward pass per category,
    # explicitly with num_workers=0. localizer.get_froi_mask() (used in the main
    # sweep below) calls model.generate_dataset_activations() internally without
    # exposing num_workers, so it silently defaults to num_workers=1 -- which
    # crashes under MPS + multiprocessing when run as a plain script. Priming the
    # cache here means get_froi_mask() always hits its "already cached" branch
    # and never spins up its own DataLoader workers.
    _grillspector_module = localizer_utils.get_localizerdataset_module('grillspector')
    with mo.status.progress_bar(total=len(models), title='Priming localizer activation cache') as _bar:
        for _model_label, _model in models:
            _imageclass_datasets = localizer_utils.get_all_imageclass_datasets(
                GRILLSPECTOR_DIR, _grillspector_module, _model.transforms
            )
            _activations_dir = f'{MODELS_DIR}/{_model.name}/grillspector/activations'
            for _imageclass_dataset in _imageclass_datasets.values():
                _model.generate_dataset_activations(
                    SWEEP_LAYERS, _imageclass_dataset, save_dir=_activations_dir,
                    batch_size=BATCH_SIZE, device=DEVICE, num_workers=0
                )
            _bar.update(subtitle=f'{_model_label} done')
    return


@app.cell
def _(torch):
    class ImageNetSubsetDataset(torch.utils.data.Dataset):
        """
        Flat-directory dataset for a fixed random subsample of ImageNet-val.

        Unlike compcats.datasets.image.custom.CustomDataset, this converts every
        image to RGB before returning it. A handful of ImageNet-val JPEGs are
        grayscale, which would otherwise produce a 1-channel tensor and break
        the model's 3-channel Normalize/conv1 step partway through the sweep.
        """
        def __init__(self, root_dir, image_names, dataset_name, transform):
            self.name = dataset_name
            self.root_dir = root_dir
            self.image_names = image_names
            self.transform = transform

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, idx):
            image = self._load_image(idx)
            if self.transform:
                image = self.transform(image)
            return image

        def _load_image(self, idx):
            import os
            img_path = os.path.join(self.root_dir, self.image_names[idx])
            from PIL import Image
            return Image.open(img_path).convert('RGB')

    return (ImageNetSubsetDataset,)


@app.cell
def _(
    AFHQV2_DIR,
    CELEBAHQ_DIR,
    ImageNetSubsetDataset,
    IMAGENET_VAL_DIR,
    N_STIMULI,
    RANDOM_SEED,
    afhqv2,
    celebahq,
    models,
    np,
    os,
    torch,
):
    def build_datasets(model):
        """
        Build the three stimulus sets for a given model's transforms, all
        matched to N_STIMULI images (the size of the smallest available set,
        AFHQv2 test):
        - AFHQv2 test, in full (1468 available: 493 cat + 491 dog + 483 wild)
        - CelebA-HQ val, a fixed-seed random N_STIMULI subsample (2000 available)
        - ImageNet val, a fixed-seed random N_STIMULI subsample (50000 available,
          flat directory)
        """
        afhqv2_dataset = afhqv2.get_val_dataset(AFHQV2_DIR, model.transforms)
        afhqv2_dataset.name = 'afhqv2'

        _celebahq_full = celebahq.get_val_dataset(CELEBAHQ_DIR, model.transforms)
        _celebahq_rng = np.random.default_rng(RANDOM_SEED)
        _celebahq_indices = _celebahq_rng.choice(
            len(_celebahq_full), size=N_STIMULI, replace=False
        ).tolist()
        # Subsetting changes the effective sample -- name it accordingly so it
        # gets its own activation cache key instead of colliding with a
        # differently-sized celebahq cache from an earlier run.
        celebahq_dataset = torch.utils.data.Subset(_celebahq_full, _celebahq_indices)
        celebahq_dataset.name = f'celebahq{N_STIMULI}'

        _all_files = sorted(
            f for f in os.listdir(IMAGENET_VAL_DIR)
            if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')
        )
        _imagenet_rng = np.random.default_rng(RANDOM_SEED)
        _sampled_files = _imagenet_rng.choice(_all_files, size=N_STIMULI, replace=False).tolist()
        imagenet_dataset = ImageNetSubsetDataset(
            IMAGENET_VAL_DIR, _sampled_files, f'imagenetval{N_STIMULI}', model.transforms
        )
        return {
            'afhqv2': afhqv2_dataset,
            'celebahq': celebahq_dataset,
            'imagenet': imagenet_dataset,
        }

    datasets_by_model = {
        model_label: build_datasets(model) for model_label, model in models
    }
    return (datasets_by_model,)


@app.cell
def _(torch):
    def participation_ratio(activations):
        """
        Effective dimensionality via the participation ratio of the PCA
        eigenspectrum (Elmoznino & Bonner, PLOS Comp Biol 2024):

            ED = (sum(lambda_i))^2 / sum(lambda_i^2)

        Computed via SVD of the mean-centered activation matrix rather than
        forming the covariance matrix explicitly (numerically more stable;
        the (n_samples - 1) normalization on the covariance cancels out in
        the ratio, so singular values can be used directly).

        Args:
            activations (torch.Tensor): shape (n_samples, n_units)

        Returns:
            float: effective dimensionality, in [1, n_units]
        """
        x = activations.double() - activations.double().mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(x)
        eigenvalues = singular_values ** 2
        return ((eigenvalues.sum() ** 2) / (eigenvalues.square().sum())).item()

    return (participation_ratio,)


@app.cell
def _(np, torch):
    def sample_matched_random_masks(face_mask, n_repeats, seed):
        """
        Draw `n_repeats` random boolean masks over the non-face-selective
        units, each with the same number of True entries as `face_mask`.

        Args:
            face_mask (torch.Tensor): boolean mask, any shape
            n_repeats (int): number of independent random draws
            seed (int): base seed; draw i uses seed + i for reproducibility

        Returns:
            list[torch.Tensor]: n_repeats boolean masks, same shape as face_mask
        """
        flat_face_mask = face_mask.flatten()
        n_selected = int(flat_face_mask.sum().item())
        candidate_indices = (~flat_face_mask).nonzero(as_tuple=False).flatten().numpy()

        random_masks = []
        for i in range(n_repeats):
            rng = np.random.default_rng(seed + i)
            chosen = rng.choice(candidate_indices, size=n_selected, replace=False)
            flat_mask = torch.zeros_like(flat_face_mask)
            flat_mask[chosen] = True
            random_masks.append(flat_mask.reshape(face_mask.shape))
        return random_masks

    return (sample_matched_random_masks,)


@app.cell
def _(
    BATCH_SIZE,
    SWEEP_LAYERS,
    DEVICE,
    FROI,
    GRILLSPECTOR_DIR,
    MODELS_DIR,
    NUM_WORKERS,
    N_RANDOM_REPEATS,
    RANDOM_SEED,
    TTHRESHOLD,
    datasets_by_model,
    localizer,
    mo,
    models,
    participation_ratio,
    sample_matched_random_masks,
):
    _result_rows = []
    _unit_count_rows = []
    _skipped_layers = []

    with mo.status.progress_bar(
        total=len(models) * len(SWEEP_LAYERS), title='Face-unit dimensionality sweep'
    ) as _bar:
        for _model_label, _model in models:
            for _layer in SWEEP_LAYERS:
                _layer_index = SWEEP_LAYERS.index(_layer) + 1

                _face_mask = localizer.get_froi_mask(
                    GRILLSPECTOR_DIR, 'grillspector', MODELS_DIR,
                    _model, _layer, FROI, TTHRESHOLD
                ).cpu()  # activations were generated on DEVICE (mps); numpy ops downstream need cpu
                _n_face = int(_face_mask.sum().item())
                _n_total = int(_face_mask.numel())
                _unit_count_rows.append({
                    'model': _model_label, 'layer': _layer, 'layer_index': _layer_index,
                    'n_face': _n_face, 'n_total': _n_total,
                })

                if _n_face == 0:
                    _skipped_layers.append((_model_label, _layer))
                    _bar.update(subtitle=f'{_model_label} {_layer}: 0 face-selective units, skipped')
                    continue

                _random_masks = sample_matched_random_masks(
                    _face_mask, N_RANDOM_REPEATS, RANDOM_SEED
                )
                _flat_face_mask = _face_mask.flatten()

                for _dataset_label, _dataset in datasets_by_model[_model_label].items():
                    _raw_activations = _model.generate_dataset_activations(
                        _layer, _dataset,
                        save_dir=f'{MODELS_DIR}/{_model.name}/{_dataset.name}/activations',
                        batch_size=BATCH_SIZE, device=DEVICE, num_workers=NUM_WORKERS
                    )
                    _flat = _raw_activations.reshape(_raw_activations.shape[0], -1).cpu()

                    _ed_face = participation_ratio(_flat[:, _flat_face_mask])
                    _result_rows.append({
                        'model': _model_label, 'layer': _layer, 'layer_index': _layer_index,
                        'dataset': _dataset_label, 'unit_type': 'face_selective', 'repeat': None,
                        'n_units': _n_face, 'n_total_units': _n_total,
                        'effective_dimensionality': _ed_face,
                    })

                    for _i, _rmask in enumerate(_random_masks):
                        _flat_rmask = _rmask.flatten()
                        _ed_rand = participation_ratio(_flat[:, _flat_rmask])
                        _result_rows.append({
                            'model': _model_label, 'layer': _layer, 'layer_index': _layer_index,
                            'dataset': _dataset_label, 'unit_type': 'random_matched', 'repeat': _i,
                            'n_units': _n_face, 'n_total_units': _n_total,
                            'effective_dimensionality': _ed_rand,
                        })

                _bar.update(subtitle=f'{_model_label} {_layer}: {_n_face}/{_n_total} face-selective units')

    results_rows = _result_rows
    unit_count_rows = _unit_count_rows
    skipped_layers = _skipped_layers
    return (results_rows, skipped_layers, unit_count_rows)


@app.cell
def _(RESULTS_DIR, pd, results_rows):
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(f'{RESULTS_DIR}/face_unit_dimensionality.csv', index=False)
    results_df
    return (results_df,)


@app.cell
def _(pd, unit_count_rows):
    unit_counts_df = pd.DataFrame(unit_count_rows)
    return (unit_counts_df,)


@app.cell
def _(results_df):
    summary_df = (
        results_df
        .groupby(['model', 'layer', 'layer_index', 'dataset', 'unit_type'], as_index=False)
        .agg(
            effective_dimensionality_mean=('effective_dimensionality', 'mean'),
            effective_dimensionality_std=('effective_dimensionality', 'std'),
            n_units=('n_units', 'first'),
        )
    )
    summary_df['effective_dimensionality_std'] = summary_df['effective_dimensionality_std'].fillna(0.0)
    return (summary_df,)


@app.cell
def _(CDN, file_html, mo):
    def render_bokeh(fig, height='450px'):
        """
        Render a Bokeh figure (or gridplot layout) as a marimo cell output.

        marimo has no built-in Bokeh formatter (unlike Plotly/Altair), and
        Bokeh's interactivity (hover, legend click-to-hide) depends on JS that
        only runs inside an iframe -- mo.Html/mo.as_html don't execute scripts.
        So this serializes the figure to a standalone HTML document (Bokeh JS
        pulled from CDN) and embeds it via mo.iframe.
        """
        html = file_html(fig, CDN, 'plot')
        return mo.iframe(html, height=height)

    return (render_bokeh,)


@app.cell
def _(Category10, ColumnDataSource, HoverTool, figure, render_bokeh, unit_counts_df):
    def make_unit_count_figure():
        p = figure(
            title='Face-selective unit count vs. network depth',
            x_axis_label='Layer index (1 = layer1.0.relu1, 16 = layer4.1.relu2)',
            y_axis_label='# face-selective units (t >= 7)',
            width=700, height=400,
        )
        _model_labels = sorted(unit_counts_df['model'].unique())
        _colors = Category10[3]
        for _i, _model_label in enumerate(_model_labels):
            _sub = unit_counts_df[unit_counts_df['model'] == _model_label].sort_values('layer_index')
            _src = ColumnDataSource(_sub)
            p.line('layer_index', 'n_face', source=_src, legend_label=_model_label, color=_colors[_i], line_width=2)
            p.scatter('layer_index', 'n_face', source=_src, legend_label=_model_label, color=_colors[_i], size=8)
        p.add_tools(HoverTool(tooltips=[
            ('Model', '@model'), ('Layer', '@layer'),
            ('# face-selective', '@n_face'), ('# total units', '@n_total'),
        ]))
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        return p

    fig_unit_counts = render_bokeh(make_unit_count_figure())
    fig_unit_counts
    return


@app.cell
def _(Category10, ColumnDataSource, HoverTool, Whisker, figure, gridplot, render_bokeh, summary_df):
    def make_ed_panel(model_label, dataset_label, unit_types, colors):
        p = figure(
            title=f'{model_label} — {dataset_label}',
            x_axis_label='Layer index', y_axis_label='Effective dimensionality',
            width=380, height=320,
        )
        for _unit_type, _color in zip(unit_types, colors):
            _sub = summary_df[
                (summary_df['model'] == model_label)
                & (summary_df['dataset'] == dataset_label)
                & (summary_df['unit_type'] == _unit_type)
            ].sort_values('layer_index').copy()
            _sub['ed_upper'] = _sub['effective_dimensionality_mean'] + _sub['effective_dimensionality_std']
            _sub['ed_lower'] = _sub['effective_dimensionality_mean'] - _sub['effective_dimensionality_std']
            _src = ColumnDataSource(_sub)
            p.line('layer_index', 'effective_dimensionality_mean', source=_src, legend_label=_unit_type, color=_color, line_width=2)
            p.scatter('layer_index', 'effective_dimensionality_mean', source=_src, legend_label=_unit_type, color=_color, size=6)
            p.add_layout(Whisker(source=_src, base='layer_index', upper='ed_upper', lower='ed_lower', line_color=_color))
        p.add_tools(HoverTool(tooltips=[
            ('Unit type', '@unit_type'), ('Layer', '@layer'), ('# units', '@n_units'),
            ('ED mean', '@effective_dimensionality_mean{0.00}'),
        ]))
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        return p

    def make_ed_grid():
        _model_labels = sorted(summary_df['model'].unique())
        _dataset_labels = sorted(summary_df['dataset'].unique())
        _unit_types = sorted(summary_df['unit_type'].unique())
        _colors = Category10[3][:len(_unit_types)]
        _grid = [
            [make_ed_panel(_m, _d, _unit_types, _colors) for _d in _dataset_labels]
            for _m in _model_labels
        ]
        return gridplot(_grid)

    fig_ed = render_bokeh(make_ed_grid(), height='700px')
    fig_ed
    return


@app.cell
def _(mo, summary_df):
    model_dropdown = mo.ui.dropdown(
        options=sorted(summary_df['model'].unique().tolist()),
        value=sorted(summary_df['model'].unique().tolist())[0],
        label='Model',
    )
    dataset_dropdown = mo.ui.dropdown(
        options=sorted(summary_df['dataset'].unique().tolist()),
        value=sorted(summary_df['dataset'].unique().tolist())[0],
        label='Dataset',
    )
    mo.hstack([model_dropdown, dataset_dropdown])
    return dataset_dropdown, model_dropdown


@app.cell
def _(
    Category10,
    ColumnDataSource,
    HoverTool,
    Whisker,
    dataset_dropdown,
    figure,
    model_dropdown,
    render_bokeh,
    summary_df,
):
    def make_focused_ed_figure():
        p = figure(
            title=f'ED — {model_dropdown.value} on {dataset_dropdown.value}',
            x_axis_label='Layer index', y_axis_label='Effective dimensionality',
            width=700, height=400,
        )
        _unit_types = sorted(summary_df['unit_type'].unique())
        _colors = Category10[3][:len(_unit_types)]
        for _unit_type, _color in zip(_unit_types, _colors):
            _sub = summary_df[
                (summary_df['model'] == model_dropdown.value)
                & (summary_df['dataset'] == dataset_dropdown.value)
                & (summary_df['unit_type'] == _unit_type)
            ].sort_values('layer_index').copy()
            _sub['ed_upper'] = _sub['effective_dimensionality_mean'] + _sub['effective_dimensionality_std']
            _sub['ed_lower'] = _sub['effective_dimensionality_mean'] - _sub['effective_dimensionality_std']
            _src = ColumnDataSource(_sub)
            p.line('layer_index', 'effective_dimensionality_mean', source=_src, legend_label=_unit_type, color=_color, line_width=2)
            p.scatter('layer_index', 'effective_dimensionality_mean', source=_src, legend_label=_unit_type, color=_color, size=6)
            p.add_layout(Whisker(source=_src, base='layer_index', upper='ed_upper', lower='ed_lower', line_color=_color))
        p.add_tools(HoverTool(tooltips=[
            ('Unit type', '@unit_type'), ('Layer', '@layer'), ('# units', '@n_units'),
            ('ED mean', '@effective_dimensionality_mean{0.00}'),
        ]))
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        return p

    fig_ed_focused = render_bokeh(make_focused_ed_figure())
    fig_ed_focused
    return


@app.cell
def _(summary_df):
    # Wide format: one row per (model, layer, unit_type), one column per
    # dataset's mean ED, so cross-dataset ED can be scattered directly.
    ed_wide_df = (
        summary_df
        .pivot_table(
            index=['model', 'layer', 'layer_index', 'unit_type'],
            columns='dataset', values='effective_dimensionality_mean',
        )
        .reset_index()
    )
    return (ed_wide_df,)


@app.cell
def _(
    Category10,
    ColumnDataSource,
    HoverTool,
    combinations,
    ed_wide_df,
    figure,
    gridplot,
    pearsonr,
    render_bokeh,
):
    def make_cross_dataset_panel(model_label, dataset_x, dataset_y, unit_types, colors):
        """
        Scatter layer-level effective dimensionality on dataset_x vs. dataset_y,
        one point per layer, for a single model. One series per unit type, each
        annotated with its Pearson correlation in the legend.
        """
        p = figure(
            title=f'{model_label}: {dataset_x} vs. {dataset_y}',
            x_axis_label=f'ED ({dataset_x})', y_axis_label=f'ED ({dataset_y})',
            width=380, height=320,
        )
        for _unit_type, _color in zip(unit_types, colors):
            _sub = ed_wide_df[
                (ed_wide_df['model'] == model_label) & (ed_wide_df['unit_type'] == _unit_type)
            ].dropna(subset=[dataset_x, dataset_y])
            _legend_label = _unit_type
            if len(_sub) >= 2:
                _r, _ = pearsonr(_sub[dataset_x], _sub[dataset_y])
                _legend_label = f'{_unit_type} (r={_r:.2f})'
            _src = ColumnDataSource(_sub)
            p.scatter(
                dataset_x, dataset_y, source=_src, color=_color, size=8,
                alpha=0.8, legend_label=_legend_label,
            )
        p.add_tools(HoverTool(tooltips=[
            ('Layer', '@layer'), ('Unit type', '@unit_type'),
            (dataset_x, f'@{dataset_x}{{0.00}}'), (dataset_y, f'@{dataset_y}{{0.00}}'),
        ]))
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        return p

    def make_cross_dataset_grid():
        _model_labels = sorted(ed_wide_df['model'].unique())
        _dataset_labels = sorted(
            c for c in ed_wide_df.columns
            if c not in ('model', 'layer', 'layer_index', 'unit_type')
        )
        _dataset_pairs = list(combinations(_dataset_labels, 2))
        _unit_types = sorted(ed_wide_df['unit_type'].unique())
        _colors = Category10[3][:len(_unit_types)]
        _grid = [
            [make_cross_dataset_panel(_m, _dx, _dy, _unit_types, _colors) for _m in _model_labels]
            for _dx, _dy in _dataset_pairs
        ]
        return gridplot(_grid)

    fig_cross_dataset = render_bokeh(make_cross_dataset_grid(), height='950px')
    fig_cross_dataset
    return


@app.cell
def _(
    BATCH_SIZE,
    SWEEP_LAYERS,
    DEVICE,
    FROI,
    NUM_WORKERS,
    N_RANDOM_REPEATS,
    N_STIMULI,
    RANDOM_SEED,
    REPORTS_DIR,
    TTHRESHOLD,
    datetime,
    mo,
    skipped_layers,
    summary_df,
    unit_counts_df,
):
    def build_run_report():
        """Assemble the markdown run report written to reports/run_report.md."""
        lines = []
        lines.append('# Face-selective unit dimensionality — run report')
        lines.append('')
        lines.append(f'Generated: {datetime.datetime.now().isoformat(timespec="seconds")}')
        lines.append('')
        lines.append('## Run configuration')
        lines.append('')
        lines.append(f'- fROI: {FROI}, t-threshold: {TTHRESHOLD}')
        lines.append(
            f'- Stimuli per dataset: {N_STIMULI} '
            '(AFHQv2 test in full; CelebA-HQ val and ImageNet val subsampled to match, fixed seed)'
        )
        lines.append(f'- Random-matched baseline repeats: {N_RANDOM_REPEATS} (seed={RANDOM_SEED})')
        lines.append(f'- Device: {DEVICE}, num_workers: {NUM_WORKERS}, batch_size: {BATCH_SIZE}')
        lines.append(f'- Layers swept (all ReLU layers, relu1 + relu2 per block): {", ".join(SWEEP_LAYERS)}')
        lines.append('')
        lines.append('## Face-selective unit counts per layer')
        lines.append('')
        lines.append('| Model | Layer | Layer index | # face-selective | # total units |')
        lines.append('|---|---|---|---|---|')
        for _, _row in unit_counts_df.sort_values(['model', 'layer_index']).iterrows():
            lines.append(
                f"| {_row['model']} | {_row['layer']} | {_row['layer_index']} | "
                f"{_row['n_face']} | {_row['n_total']} |"
            )
        lines.append('')
        if skipped_layers:
            lines.append(
                f'{len(skipped_layers)} (model, layer) combination(s) had 0 face-selective units '
                'at t >= 7 and were skipped in the dimensionality sweep below.'
            )
            lines.append('')
        lines.append('## Effective dimensionality summary')
        lines.append('')
        lines.append('| Model | Layer | Dataset | Unit type | # units | ED (mean) | ED (std) |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, _row in summary_df.sort_values(['model', 'layer_index', 'dataset', 'unit_type']).iterrows():
            lines.append(
                f"| {_row['model']} | {_row['layer']} | {_row['dataset']} | {_row['unit_type']} | "
                f"{_row['n_units']} | {_row['effective_dimensionality_mean']:.2f} | "
                f"{_row['effective_dimensionality_std']:.2f} |"
            )
        return '\n'.join(lines)

    run_report_text = build_run_report()
    with open(f'{REPORTS_DIR}/run_report.md', 'w', encoding='utf-8') as _f:
        _f.write(run_report_text)
    mo.md(run_report_text)
    return


if __name__ == "__main__":
    app.run()
