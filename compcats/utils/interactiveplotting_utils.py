"""
Utility functions for interactive plotting with Bokeh.
"""
import os
from typing import List, Optional
import numpy as np
import pandas as pd
from PIL import Image

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, HoverTool
from bokeh.palettes import Reds256
from bokeh.embed import file_html
from bokeh.resources import CDN

def save_stimuli_as_jpeg(stimuli_images, output_path, size=(75, 75), quality=85):
    """
    Resizes, compresses, and saves stimuli images as JPEGs into a specified directory.

    Args:
        stimuli_images (np.ndarray): The 4D array of stimuli images.
        output_path (str): The full path where images will be saved.
        size (tuple, optional): The target (width, height) for resizing. Defaults to (75, 75).
        quality (int, optional): The quality for JPEG compression (1-100). Defaults to 85.
    """
    # Check if directory already exists and has correct number of files
    if os.path.exists(output_path):
        # Count existing JPEG files
        existing_files = [f for f in os.listdir(output_path) 
                         if f.startswith("stimulus_") and f.endswith(".jpeg")]
        
        if len(existing_files) == len(stimuli_images):
            print(f"Skipping stimuli saving: {len(existing_files)} images already exist in '{output_path}/'")
            return
    
    print(f"Saving {len(stimuli_images)} images to '{output_path}/'...")
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    for i, img_array in enumerate(stimuli_images):
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        img = Image.fromarray(img_array)
        img = img.resize(size)

        # Save as JPEG with specified quality
        img_path = os.path.join(output_path, f"stimulus_{i}.jpeg")
        img.save(img_path, "JPEG", quality=quality)
    print("Images saved successfully.")

def create_interactive_heatmap(heatmap_data, stimuli_idxs, image_folder_path, vmin=None, vmax=None):
    """
    Creates a portable, interactive Bokeh heatmap. It displays the full
    symmetric matrix, but only the upper triangular part is interactive.
    This version uses an index array to map heatmap rows/cols to specific stimuli
    in a shared image folder.

    Args:
        heatmap_data (np.ndarray): A 2D NumPy array of shape (n_samples, n_samples)
                                   containing the values for the heatmap.
        stimuli_idxs (np.ndarray): A 1D NumPy array of shape (n_samples,) containing the
                                   integer indices of the stimuli being plotted.
        image_folder_path (str): The relative path to the shared folder where all stimuli images are stored.

    Returns:
        bokeh.plotting.figure: The Bokeh plot object.
    """
    n_samples = heatmap_data.shape[0]
    if n_samples != len(stimuli_idxs):
        raise ValueError("The length of stimuli_idxs must match the dimensions of heatmap_data.")

    # --- 1. Prepare data for TWO separate layers ---
    # Layer 1: The full matrix (for display only)
    full_matrix_data = {'x': [], 'y': [], 'value': []}
    # Layer 2: The upper triangle (for interactivity)
    upper_triangle_data = {'x': [], 'y': [], 'value': [], 'image1_path': [], 'image2_path': [], 'stim_idx_i': [], 'stim_idx_j': []}

    for i in range(n_samples):
        for j in range(n_samples):
            # Add data for the full matrix display
            full_matrix_data['x'].append(j)
            full_matrix_data['y'].append(i)
            full_matrix_data['value'].append(heatmap_data[i, j])

            # Only add data to the interactive layer for the strict upper triangle
            if j > i:
                # Get the actual stimulus index from the provided array
                stimulus_idx_i = stimuli_idxs[i]
                stimulus_idx_j = stimuli_idxs[j]

                upper_triangle_data['x'].append(j)
                upper_triangle_data['y'].append(i)
                upper_triangle_data['value'].append(heatmap_data[i, j])
                # Add original indices for display in tooltip
                upper_triangle_data['stim_idx_i'].append(stimulus_idx_i)
                upper_triangle_data['stim_idx_j'].append(stimulus_idx_j)
                # Construct the path using the image folder path and the actual stimulus index
                upper_triangle_data['image1_path'].append(f"{image_folder_path}/stimulus_{stimulus_idx_i}.jpeg")
                upper_triangle_data['image2_path'].append(f"{image_folder_path}/stimulus_{stimulus_idx_j}.jpeg")

    full_source = ColumnDataSource(data=full_matrix_data)
    upper_source = ColumnDataSource(data=upper_triangle_data)

    # --- 2. Define the Hover Tooltip ---
    TOOLTIPS = """
        <div style="padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
            <p style="font-weight: bold; margin: 0;">Plot Indices: (@y, @x)</p>
            <p style="font-weight: bold; margin: 0;">Stimuli Indices: (@stim_idx_i, @stim_idx_j)</p>
            <p style="margin: 0 0 10px 0;">Value: @value{0.00}</p>
            <div style="display: flex; gap: 10px; align-items: center;">
                <div>
                    <p style="margin: 0 0 5px 0; font-size: 12px; text-align: center;">Stimulus @stim_idx_i</p>
                    <img src="@image1_path" width="75" style="border: 1px solid #ddd; border-radius: 3px;">
                </div>
                <div>
                    <p style="margin: 0 0 5px 0; font-size: 12px; text-align: center;">Stimulus @stim_idx_j</p>
                    <img src="@image2_path" width="75" style="border: 1px solid #ddd; border-radius: 3px;">
                </div>
            </div>
        </div>
    """

    # --- 3. Create and configure the plot ---
    custom_palette = list(Reds256)
    custom_palette.reverse() # Light red for low values
    custom_palette[-1] = '#FFFFFF' # White for high values

    color_mapper = LinearColorMapper(
        palette=custom_palette, 
        low=vmin if vmin is not None else np.min([np.min(heatmap_data)-0.1, 0]), 
        high=vmax if vmax is not None else np.max(heatmap_data)+0.1
        # high=np.max(heatmap_data)+0.05 # Slightly extend the range for better color gradation
    )

    p = figure(
        title="Interactive Heatmap with Stimuli Images",
        x_range=(-0.5, n_samples - 0.5),
        y_range=(n_samples - 0.5, -0.5),
        tools="pan,wheel_zoom,reset,save",
        x_axis_location="above",
        height=1000,
        width=1200
    )
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None

    # --- 4. Draw the two layers ---
    # Draw the full matrix first (this will be the background)
    p.rect(
        x='x', y='y', width=1, height=1, source=full_source,
        fill_color={'field': 'value', 'transform': color_mapper},
        line_color=None
    )

    # Draw the interactive upper triangle on top. Capture the renderer object.
    interactive_renderer = p.rect(
        x='x', y='y', width=1, height=1, source=upper_source,
        fill_color={'field': 'value', 'transform': color_mapper},
        line_color=None
    )

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # --- 5. Add the HoverTool and specify it should ONLY apply to the interactive layer ---
    hover = HoverTool(tooltips=TOOLTIPS, renderers=[interactive_renderer])
    p.add_tools(hover)

    return p

def save_interactive_rdm(
    root_dir, rdm, stim_path, stimuli_indices, html_path,
    stimuli = None, vmin=None, vmax=None
):
    """
    Creates and saves an interactive RDM heatmap as an HTML file.
    
    Args:
        root_dir (str): The root directory where the HTML file will be saved.
        rdm (np.ndarray): A 2D NumPy array of shape (n_samples, n_samples)
                         containing the RDM values.
        stim_path (str): The relative path to the folder where stimuli images will be saved.
        stimuli_indices (np.ndarray): A 1D NumPy array containing the indices of the stimuli
                                     to be displayed in the heatmap.
        html_path (str): The relative path to save the HTML file.
        stimuli (np.ndarray, optional): A 4D NumPy array of shape (n_samples, height, width, 3)
                                      containing the stimuli images. Defaults to None.
        vmin (float, optional): Minimum value for colormap scaling. Defaults to None.
        vmax (float, optional): Maximum value for colormap scaling. Defaults to None.
    """
    # Error checking
    assert rdm.ndim == 2, "RDM must be a 2D array."
    assert rdm.shape[0] == rdm.shape[1], "RDM must be square."
    root_dir = root_dir.rstrip('/')
    # Append root_dir to paths
    html_path = html_path.lstrip('/')
    stim_path = stim_path.lstrip('/')
    html_path = f'{root_dir}/{html_path}'
    stim_path = f'{root_dir}/{stim_path}'
    # --- 1. Save stimuli images as JPEGs ---
    if stimuli is not None:
        assert stimuli.ndim == 4, "Stimuli must be a 4D array."
        assert rdm.shape[0] == stimuli.shape[0], "Number of stimuli must match RDM dimensions."
        save_stimuli_as_jpeg(stimuli, stim_path)
    # Find relative path to the stimuli folder from the HTML file location
    rel_path_to_stimuli = os.path.relpath(
        stim_path, start=os.path.dirname(html_path)
    )
    # --- 2. Create the interactive heatmap ---
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    rdm_interactive = create_interactive_heatmap(
        rdm,
        stimuli_indices,
        image_folder_path=rel_path_to_stimuli,
        vmin=vmin,
        vmax=vmax
    )
    html = file_html(
        rdm_interactive, CDN, "Interactive RDM"
    )
    with open(html_path, "w", encoding='utf-8') as f:
        f.write(html)
    print(f"Interactive RDM saved to '{html_path}'")

def create_interactive_scatter(
    df: pd.DataFrame,
    images_np: np.ndarray,
    image_folder_path: str,
    x_col: str,
    y_col: str,
    color_col: str,
    info_cols: Optional[List[str]] = None,
    output_html_path: Optional[str] = None,
    title: str = "Interactive Scatter Plot with Image Tooltips",
    tooltip_img_size: int = 120
):
    """
    Creates an interactive scatter plot with hover images loaded from disk instead of Base64.
    """
    # --- 1. Input validation ---
    assert len(df) == len(images_np), "DataFrame and images array must have the same length."
    df_plot = df.copy()
    if info_cols is None: info_cols = []

    # --- 2. Save images if not already saved ---
    os.makedirs(image_folder_path, exist_ok=True)
    save_stimuli_as_jpeg(images_np, image_folder_path)
    # Create file paths relative to HTML location (if specified)
    rel_paths = [f"{image_folder_path}/stimulus_{i}.jpeg" for i in range(len(images_np))]
    df_plot['image_path'] = rel_paths

    # --- 3. Build the ColumnDataSource ---
    source = ColumnDataSource(df_plot)

    # --- 4. Dynamic info tooltip HTML ---
    info_html = ""
    for col in info_cols:
        label = col.replace('_', ' ').title()
        info_html += f'<div><span style="font-weight:bold;">{label}:</span> @{col}</div>'

    TOOLTIP_HTML = f"""
        <div style="padding:5px;border:1px solid #ccc;">
            <div><img src="@image_path" width="{tooltip_img_size}" style="border:1px solid #aaa;"></div>
            {info_html}
            <div><span style="font-weight:bold;">Location:</span> (@{x_col}, @{y_col})</div>
        </div>
    """

    # --- 5. Color mapping (symmetric coolwarm) ---
    c_min, c_max = df_plot[color_col].min(), df_plot[color_col].max()
    limit = max(abs(c_min), abs(c_max)) or 1
    mapper = LinearColorMapper(palette=cc.coolwarm, low=-limit, high=limit)

    # --- 6. Plot creation ---
    hover = HoverTool(tooltips=TOOLTIP_HTML)
    p = figure(
        width=800, height=600,
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
        title=title, x_axis_label=x_col, y_axis_label=y_col
    )
    p.grid.visible = False

    # Add symmetry lines
    p.add_layout(Span(location=0, dimension='width', line_color='gray', line_dash='dashed', line_width=1.5))
    p.add_layout(Span(location=0, dimension='height', line_color='gray', line_dash='dashed', line_width=1.5))
    p.scatter(x=x_col, y=y_col, source=source, size=10,
              color={'field': color_col, 'transform': mapper},
              fill_alpha=0.7, line_color="gray", line_alpha=0.2)

    color_bar = ColorBar(color_mapper=mapper, label_standoff=12, title=color_col)
    p.add_layout(color_bar, 'right')

    # --- 7. Save or show ---
    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        html = file_html(p, CDN, title)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Plot saved to '{output_html_path}'")
    else:
        output_notebook()
        show(p)

    return p