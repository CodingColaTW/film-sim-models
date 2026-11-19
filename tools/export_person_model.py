"""Rebuild person_maskrcnn_v2_fp16.onnx from the torchvision COCO weights.

Needs (development machine only): torch, torchvision, onnx, onnxscript, numpy
    python tools/export_person_model.py [output.onnx]

Steps
 1. Load torchvision's Mask R-CNN ResNet50-FPN v2 with the COCO_V1 weights.
 2. Lower the number of proposals / detections so inference is quick on a CPU.
 3. Export to ONNX (opset 17, dynamic image size).
 4. Store the large weight tensors as float16 and cast them back to float32 when the model runs.
    This halves the file size; the computation itself stays float32, and detections were
    identical to the float32 model on the photos it was checked with.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

PROPOSALS = 300          # torchvision default: 1000
MAX_DETECTIONS = 30      # torchvision default: 100
SCORE_THRESHOLD = 0.3
MIN_TENSOR_SIZE = 1024   # smaller tensors stay float32


def export_float32(path: str) -> None:
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        box_score_thresh=SCORE_THRESHOLD,
        rpn_pre_nms_top_n_test=PROPOSALS,
        rpn_post_nms_top_n_test=PROPOSALS,
        box_detections_per_img=MAX_DETECTIONS,
    ).eval()
    model.transform.min_size = (768,)
    model.transform.max_size = 1024
    torch.onnx.export(
        model, ([torch.rand(3, 768, 1024)],), path, opset_version=17, dynamo=False,
        input_names=["image"], output_names=["boxes", "labels", "scores", "masks"],
        dynamic_axes={
            "image": {0: "c", 1: "h", 2: "w"},
            "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"},
            "masks": {0: "n", 2: "mh", 3: "mw"},
        },
    )


def store_weights_as_float16(source: str, target: str) -> int:
    model = onnx.load(source)
    graph = model.graph
    initializers, casts, converted = [], [], 0
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= MIN_TENSOR_SIZE:
            half = init.name + "__fp16"
            initializers.append(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), half))
            casts.append(helper.make_node("Cast", [half], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name))
            converted += 1
        else:
            initializers.append(init)
    nodes = list(graph.node)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.node[:]
    graph.node.extend(casts + nodes)       # the casts only depend on initializers, so they go first
    onnx.save(model, target)
    return converted


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "person_maskrcnn_v2_fp16.onnx"
    intermediate = target + ".fp32.tmp"
    export_float32(intermediate)
    count = store_weights_as_float16(intermediate, target)
    os.remove(intermediate)
    print(f"wrote {target}: {count} tensors stored as float16, {os.path.getsize(target) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
