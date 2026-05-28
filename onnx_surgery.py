#!/usr/bin/env python3
"""
onnx_surgery.py — add intermediate tensors as graph outputs so ORT Web
surfaces their values during inference, enabling per-layer parity bisect
vs the WGSL walker.

Usage:
  python3 onnx_surgery.py /tmp/unet.onnx /tmp/unet_probed.onnx
"""
import sys, onnx
from onnx import helper, TensorProto

src = sys.argv[1]
dst = sys.argv[2]

m = onnx.load(src)
g = m.graph

# Collect all node output names and pick anchor points spread across the UNet.
all_outputs = []
for n in g.node:
    for o in n.output:
        if o:
            all_outputs.append(o)

# Heuristic anchor patterns spanning the UNet topology. Each matching tensor
# becomes a graph output so ORT surfaces it during session.run.
patterns = [
    # Input path
    ('conv_in', lambda s: '/conv_in/Conv_output_0' in s or s.endswith('conv_in/Conv_output_0')),
    # Time embedding
    ('time_emb', lambda s: 'time_embedding' in s and ('linear_2' in s or 'Linear_1' in s)),
    # Down path - each level
    ('down_0_resnet_end', lambda s: '/down_blocks.0/resnets.1/' in s and 'Add' in s),
    ('down_0_down', lambda s: '/down_blocks.0/downsamplers' in s and 'Conv_output_0' in s),
    ('down_1_resnet_end', lambda s: '/down_blocks.1/resnets.1/' in s and 'Add' in s),
    ('down_1_attn_end', lambda s: '/down_blocks.1/attentions.1/' in s and ('proj_out' in s or 'Add_1' in s)),
    ('down_1_down', lambda s: '/down_blocks.1/downsamplers' in s and 'Conv_output_0' in s),
    ('down_2_attn_end', lambda s: '/down_blocks.2/attentions.1/' in s and ('proj_out' in s or 'Add_1' in s)),
    # Mid block
    ('mid_attn', lambda s: '/mid_block/attentions.0/' in s and ('proj_out' in s or 'Add_1' in s)),
    ('mid_end', lambda s: '/mid_block/resnets.1/' in s and 'Add' in s),
    # Up path
    ('up_0_resnet_end', lambda s: '/up_blocks.0/resnets.2/' in s and 'Add' in s),
    ('up_1_attn_end', lambda s: '/up_blocks.1/attentions.2/' in s and ('proj_out' in s or 'Add_1' in s)),
    ('up_2_resnet_end', lambda s: '/up_blocks.2/resnets.2/' in s and 'Add' in s),
    # Output path
    ('conv_norm_out', lambda s: 'conv_norm_out' in s),
    ('conv_out', lambda s: '/conv_out/Conv_output_0' in s),
]

# Dedup and choose first match per pattern.
selected = []
used = set()
for label, pred in patterns:
    for name in all_outputs:
        if name in used: continue
        if pred(name):
            selected.append((label, name))
            used.add(name)
            break

print(f"Selected {len(selected)} anchors:")
for label, name in selected:
    print(f"  [{label}]  {name}")

# Add these as graph outputs. ORT Web will surface their runtime values.
existing_out_names = {o.name for o in g.output}
for label, name in selected:
    if name in existing_out_names: continue
    # Create a ValueInfoProto with unknown shape (ORT infers at runtime).
    vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
    g.output.append(vi)

# ONNX size may now exceed 2GB after save. Use save_model with save_as_external_data
# to keep it manageable. But the original is already ~1.65GB — we must keep that.
# Safer: just save — protobuf supports >2GB if we use external data, but for
# intermediate outputs we don't change tensor payloads, just value_info headers,
# so the file should remain ~same size.
try:
    onnx.save(m, dst)
except ValueError as e:
    # Protobuf >2GB — need external data format
    print(f"Standard save failed ({e}); saving with external data…")
    onnx.save_model(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='unet_probed_weights.bin', size_threshold=1024)

print(f"Wrote {dst}")
print(f"Probe tensor names (for the JS harness):")
print([name for _, name in selected])
