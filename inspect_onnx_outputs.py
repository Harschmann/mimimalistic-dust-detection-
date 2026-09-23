"""
Run this on your corporate laptop, in the SAME Python environment that has
onnxruntime installed, pointing at your actual trained model.onnx.

Usage:
    python inspect_onnx_outputs.py path\\to\\model.onnx path\\to\\one_defect_image.png

It prints every output the model produces (name + shape) and, if you also
give it an image, runs the model on it and prints min/max/mean/std for
each output so we can tell which one is the real per-pixel anomaly map
(it should be 2D-ish and have real min-max spread, not a constant).
Paste the full printed output back to me.
"""
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python inspect_onnx_outputs.py path/to/model.onnx [path/to/image.png]")
    sys.exit(1)

onnx_path = sys.argv[1]
image_path = sys.argv[2] if len(sys.argv) > 2 else None

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

print("=" * 70)
print("INPUTS:")
for inp in session.get_inputs():
    print(f"  name={inp.name!r}  shape={inp.shape}  type={inp.type}")

print("\nOUTPUTS (declared in the graph):")
for out in session.get_outputs():
    print(f"  name={out.name!r}  shape={out.shape}  type={out.type}")
print("=" * 70)

if image_path:
    inp_meta = session.get_inputs()[0]
    shape = inp_meta.shape
    h = shape[2] if isinstance(shape[2], int) else 256
    w = shape[3] if isinstance(shape[3], int) else 256

    img = Image.open(image_path).convert("RGB").resize((w, h))
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...].astype("float32")

    outputs = session.run(None, {inp_meta.name: arr})
    out_names = [o.name for o in session.get_outputs()]

    print(f"\nRan on image: {image_path}  (resized to {w}x{h})")
    print("=" * 70)
    for name, val in zip(out_names, outputs):
        val = np.asarray(val)
        print(f"\nOutput {name!r}:")
        print(f"  shape = {val.shape}")
        print(f"  dtype = {val.dtype}")
        print(f"  min={val.min():.6f}  max={val.max():.6f}  mean={val.mean():.6f}  std={val.std():.6f}")
        flat = val.squeeze()
        if flat.ndim == 2:
            print(f"  (this one is 2D after squeeze -- shape {flat.shape} -- a real CANDIDATE for the anomaly map)")
        elif flat.ndim == 0 or (flat.ndim == 1 and flat.size == 1):
            print(f"  (this one is a single scalar -- likely the overall anomaly SCORE, not a map)")
    print("=" * 70)
else:
    print("\n(No image given -- pass one as a second argument to also see actual output values/stats.)")
