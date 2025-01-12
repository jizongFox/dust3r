import typing as t
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

import plotly.graph_objects as go
import plotly.subplots as sp


class TypedView(TypedDict):
    img: Float[Tensor, "b 3 H W"]
    true_shape: Int[Tensor, "b 2"]
    instance: t.List[str]
    idx: t.List[int]


class TypedPred1(TypedDict):
    pts3d: Float[torch.Tensor, "b H W 3"]
    conf: Float[torch.Tensor, "b H W"]
    desc: Float[torch.Tensor, "b H W 24"]
    desc_conf: Float[torch.Tensor, "b H W"]


class TypedPred2(TypedDict):
    pts3d_in_other_view: Float[torch.Tensor, "b H W 3"]
    conf: Float[torch.Tensor, "b H W"]
    desc: Float[torch.Tensor, "b H W 24"]
    desc_conf: Float[torch.Tensor, "b H W"]


class TypedLossOutput(TypedDict):
    view1: TypedView
    view2: TypedView
    pred1: TypedPred1
    pred2: TypedPred2
    loss: torch.Tensor | None


class TypedLoadImageOutput(TypedDict):
    img: Float[Tensor, "1 3 H W"]
    true_shape: Int[np.ndarray, "1 2"]
    idx: int
    instance: str


def visualize_typed_loss_output_plotly(typed_loss_output: TypedLossOutput, save_path: Path | None = None):
    """
    Visualizes TypedLossOutput including images from view1 and view2, and predicted 3D point clouds using Plotly.
    Optimized for a compact layout.

    Args:
        typed_loss_output (TypedLossOutput): The loss output to visualize.
    """
    n_pairs = len(typed_loss_output["view1"]["img"]) // 2

    # Create image and 3D scatter plot subplots
    fig = sp.make_subplots(
        rows=n_pairs,
        cols=3,
        subplot_titles=[
            (
                f"View1 Image (Pair {i+1})"
                if j == 0
                else f"View2 Image (Pair {i+1})" if j == 1 else f"Predicted 3D Points (Pair {i+1})"
            )
            for i in range(n_pairs)
            for j in range(3)
        ],
        specs=[[{"type": "image"}, {"type": "image"}, {"type": "scatter3d"}] for _ in range(n_pairs)],
        # vertical_spacing=0.05,  # Reduce vertical spacing
    )

    for i in range(n_pairs):
        # Extracting and reshaping view1 image
        view1_img_tensor = typed_loss_output["view1"]["img"][i]  # Take the i-th batch
        view1_img = view1_img_tensor.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 3)

        # Extracting and reshaping view2 image
        view2_img_tensor = typed_loss_output["view2"]["img"][i]  # Take the i-th batch
        view2_img = view2_img_tensor.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 3)

        # Normalize images to [0, 255] and cast to uint8
        view1_img = ((view1_img + 1) / 2 * 255).astype("uint8")
        view2_img = ((view2_img + 1) / 2 * 255).astype("uint8")

        # Add view1 image
        fig.add_trace(go.Image(z=view1_img), row=i + 1, col=1)

        # Add view2 image
        fig.add_trace(go.Image(z=view2_img), row=i + 1, col=2)

        # Extracting and reshaping predicted point clouds
        pred1_pts3d = typed_loss_output["pred1"]["pts3d"][i].reshape(-1, 3).cpu().numpy()  # Shape: (H * W, 3)
        pred1_pts3d = pred1_pts3d[::5]  # Subsample more aggressively for compactness
        pred2_pts3d = (
            typed_loss_output["pred2"]["pts3d_in_other_view"][i].reshape(-1, 3).cpu().numpy()
        )  # Shape: (H * W, 3)
        pred2_pts3d = pred2_pts3d[::5]  # Subsample more aggressively for compactness

        # Add 3D scatter plot for pred1_pts3d
        fig.add_trace(
            go.Scatter3d(
                x=pred1_pts3d[:, 0],
                y=pred1_pts3d[:, 1],
                z=pred1_pts3d[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=pred1_pts3d[:, 2], colorscale="Viridis", opacity=0.7),
                name=f"Predicted Points (View1) Pair {i+1}",
            ),
            row=i + 1,
            col=3,
        )

        # Add 3D scatter plot for pred2_pts3d
        fig.add_trace(
            go.Scatter3d(
                x=pred2_pts3d[:, 0],
                y=pred2_pts3d[:, 1],
                z=pred2_pts3d[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=pred2_pts3d[:, 2], colorscale="Plasma", opacity=0.7),
                name=f"Predicted Points (View2) Pair {i+1}",
            ),
            row=i + 1,
            col=3,
        )

    # Ensure 3D scatter respects real-world aspect ratios
    fig.update_scenes(aspectmode="data")

    # Update layout for compactness
    fig.update_layout(
        title_text="View1 and View2 Images with Predicted 3D Points",
        title_x=0.5,
        height=200 * n_pairs,  # Reduced height for compactness
        margin=dict(l=10, r=10, t=30, b=10),  # Smaller margins
        legend=dict(font=dict(size=10)),  # Smaller legend font
    )
    # Save figure if save_path is provided
    if save_path:
        # Use Plotly's write_image to save at the highest quality
        fig.write_image(save_path, scale=2)  # Use scale=2 for higher resolution

    return fig
