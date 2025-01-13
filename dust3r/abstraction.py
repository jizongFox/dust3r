import typing as t
from itertools import product
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


def visualize_typed_loss_output_plotly_with_uncertainty(
    typed_loss_output: TypedLossOutput, save_path: Path | None = None
):
    """
    Visualizes TypedLossOutput including images from view1 and view2, their uncertainties (as images),
    and predicted 3D point clouds using Plotly.

    Args:
        typed_loss_output (TypedLossOutput): The loss output to visualize.
        save_path (Path | None): Path to save the figure (optional).
    """
    n_pairs = len(typed_loss_output["view1"]["img"]) // 2

    # Create image, uncertainty (as images), and 3D scatter plot subplots
    fig = sp.make_subplots(
        rows=n_pairs,
        cols=5,  # Adding columns for uncertainties
        subplot_titles=[
            (
                f"View1 Image (Pair {i+1})"
                if j == 0
                else (
                    f"View2 Image (Pair {i+1})"
                    if j == 1
                    else (
                        f"View1 Uncertainty (Pair {i+1})"
                        if j == 2
                        else f"View2 Uncertainty (Pair {i+1})" if j == 3 else f"Predicted 3D Points (Pair {i+1})"
                    )
                )
            )
            for i in range(n_pairs)
            for j in range(5)
        ],
        specs=[
            [{"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "scatter3d"}]
            for _ in range(n_pairs)
        ],
    )

    for i in range(n_pairs):
        # Extract and normalize view1 and view2 images
        view1_img_tensor = typed_loss_output["view1"]["img"][i]
        view1_img = ((view1_img_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
        view2_img_tensor = typed_loss_output["view2"]["img"][i]
        view2_img = ((view2_img_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")

        # Add view1 and view2 images
        fig.add_trace(go.Image(z=view1_img), row=i + 1, col=1)
        fig.add_trace(go.Image(z=view2_img), row=i + 1, col=2)

        # Extract uncertainty for view1 and view2
        view1_uncertainty = typed_loss_output["pred1"]["conf"][i].cpu().numpy()
        view2_uncertainty = typed_loss_output["pred2"]["conf"][i].cpu().numpy()

        # Normalize uncertainty to [0, 255] for image visualization
        view1_uncertainty_img = (
            (view1_uncertainty - view1_uncertainty.min()) / (view1_uncertainty.max() - view1_uncertainty.min()) * 255
        ).astype("uint8")
        view2_uncertainty_img = (
            (view2_uncertainty - view2_uncertainty.min()) / (view2_uncertainty.max() - view2_uncertainty.min()) * 255
        ).astype("uint8")

        # Add uncertainty images
        fig.add_trace(go.Image(z=np.stack([view1_uncertainty_img] * 3, axis=-1)), row=i + 1, col=3)
        fig.add_trace(go.Image(z=np.stack([view2_uncertainty_img] * 3, axis=-1)), row=i + 1, col=4)

        # Extract and subsample predicted 3D points for compactness
        pred1_pts3d = typed_loss_output["pred1"]["pts3d"][i].reshape(-1, 3).cpu().numpy()[::5]
        pred2_pts3d = typed_loss_output["pred2"]["pts3d_in_other_view"][i].reshape(-1, 3).cpu().numpy()[::5]

        # Add 3D scatter plots
        fig.add_trace(
            go.Scatter3d(
                x=pred1_pts3d[:, 0],
                y=pred1_pts3d[:, 1],
                z=pred1_pts3d[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=pred1_pts3d[:, 2], colorscale="Viridis", opacity=0.7),
                name=f"Predicted Points (View1) Pair {i+1}",
                showlegend=False,  # Disable legend for this trace
            ),
            row=i + 1,
            col=5,
        )
        fig.add_trace(
            go.Scatter3d(
                x=pred2_pts3d[:, 0],
                y=pred2_pts3d[:, 1],
                z=pred2_pts3d[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=pred2_pts3d[:, 2], colorscale="Plasma", opacity=0.7),
                name=f"Predicted Points (View2) Pair {i+1}",
                showlegend=False,  # Disable legend for this trace
            ),
            row=i + 1,
            col=5,
        )

    # Update layout for compactness
    fig.update_layout(
        title_text="View1 and View2 Images with Uncertainties and Predicted 3D Points",
        title_x=0.5,
        height=200 * n_pairs,  # Adjust height dynamically based on the number of pairs
        margin=dict(l=10, r=10, t=30, b=10),  # Smaller margins
        showlegend=False,  # Disable the legend globally
    )

    # Apply viewing direction for all 3D scatter plots
    for i in range(n_pairs):
        scene_id = f"scene{i + 1}"  # Generate scene IDs dynamically
        fig.update_layout(
            {scene_id: dict(camera=dict(eye=dict(x=0, y=-2.5, z=0)))}  # Set viewing direction along Y-axis
        )

    fig.update_scenes(aspectmode="data")  # Ensure 3D scatter respects real-world aspect ratios

    # Remove tick labels for all axes
    for row, col in product(range(1, n_pairs + 1), [1, 2, 3, 4]):
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    # Save figure if save_path is provided
    if save_path:
        fig.write_image(save_path, scale=5)  # Use scale=5 for high resolution

    return fig
