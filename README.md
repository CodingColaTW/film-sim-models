# film-sim-models

Model files used by a desktop photo-editing app, published here so the app can download them on first use.
The files are **release attachments**, not part of the git history.

## person_maskrcnn_v2_fp16.onnx

Finds every person in a photo and returns one mask per person (used to build a "subject" mask).

| | |
|---|---|
| Release | [`v1`](https://github.com/CodingColaTW/film-sim-models/releases/tag/v1) |
| Size | 93,031,673 bytes (about 93 MB) |
| SHA-256 | `1142ce5dbfb005fe57281736a29de58e23c4f47a4c05dae491a281d8932dd22b` |
| Runtime | `onnxruntime` (CPU is enough; about 2 s for a 1024 px photo on a 10-core desktop CPU, measured with a warmed-up session) |

Always check the SHA-256 after downloading.

### Input and output

- Input `image`: `float32`, shape `(3, H, W)`, RGB, values in `[0, 1]`, no other normalisation. `H` and `W` may differ per call. It was tested with the long side at 1024 px, in landscape and portrait.
- Outputs, for the `n` detections found:
  - `boxes` `(n, 4)` as `x0, y0, x1, y1` in input pixels
  - `labels` `(n,)` COCO category ids; **person is `1`**
  - `scores` `(n,)`
  - `masks` `(n, 1, H, W)` probabilities in `[0, 1]`, already pasted to the input size

Keep detections with `labels == 1` and, for example, `scores >= 0.5`. The model keeps at most 30 detections above a score of 0.3.

### Where it comes from

- Architecture and weights: torchvision `maskrcnn_resnet50_fpn_v2`, weights `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` (trained on COCO).
- Converted with [`tools/export_person_model.py`](tools/export_person_model.py) in this repository:
  - ONNX opset 17, dynamic image size
  - proposals lowered from 1000 to 300 and detections from 100 to 30, for speed
  - large weight tensors stored as float16 and cast back to float32 at run time (computation stays float32)
- On five test photos the float16-stored model found the same number of people as the float32 model, and the masks differed by about 0.0001 on average. That is a small check, not a benchmark.

### Licences and what is not settled

- torchvision's code is BSD-3-Clause. torchvision states that its pre-trained weights "may have their own licenses or terms and conditions derived from the dataset used for training" and that users must decide whether they may use them for their purpose. It does not give the weights a separate licence.
- COCO: the annotations are licensed CC BY 4.0 (attribution required). The COCO consortium does not own the copyright of the images, which come from Flickr and fall under the Flickr terms of use. Lin et al., "Microsoft COCO: Common Objects in Context", 2014.
- Not checked: whether the ResNet-50 backbone was initialised from ImageNet-pretrained weights, and whether weights trained on COCO images count as derivative of those images. The second point is an open legal question.
- This repository gives no warranty. If you plan to use the file commercially, read the terms above yourself and ask someone qualified.

### Limits seen on stage photos

- People in the audience are detected as well; filter them if you only want the performers.
- A person shown on a projection screen can be detected.
- One small performer was missed in one of the five test photos.
- Mask edges are soft: the network predicts masks at 28 × 28 per box.

## Rebuilding the file

```
pip install torch torchvision onnx onnxscript numpy
python tools/export_person_model.py person_maskrcnn_v2_fp16.onnx
```

The result is not guaranteed to be byte-identical, so its SHA-256 can differ from the one above.
