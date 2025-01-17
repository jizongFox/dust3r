# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from enum import Enum

from .modular_optimizer import ModularPointCloudOptimizer
from .optimizer import PointCloudOptimizer
from .pair_viewer import PairViewer
from ..abstraction import TypedLossOutput, TypedView, TypedPred1, TypedPred2


class GlobalAlignerMode(Enum):
    PointCloudOptimizer = "PointCloudOptimizer"
    ModularPointCloudOptimizer = "ModularPointCloudOptimizer"
    PairViewer = "PairViewer"


def global_aligner(dust3r_output: TypedLossOutput, device, mode=GlobalAlignerMode.PointCloudOptimizer, **optim_kw):
    # extract all inputs
    # view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()]
    view1: TypedView = dust3r_output["view1"]
    view2: TypedView = dust3r_output["view2"]
    pred1: TypedPred1 = dust3r_output["pred1"]
    pred2: TypedPred2 = dust3r_output["pred2"]
    # build the optimizer
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        net = PointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.ModularPointCloudOptimizer:
        net = ModularPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.PairViewer:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f"Unknown mode {mode}")

    return net
