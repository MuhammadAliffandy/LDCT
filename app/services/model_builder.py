"""
model_builder.py
----------------
Rebuilds the Mod-Seg-SE(2) v2 architecture using tf.keras (TF 2.13)
and loads pre-trained weights from the Keras-3 extracted weights file
(model.weights.h5) using a custom h5py-based weight loader.

This fully bypasses keras.models.load_model() version-compatibility
issues on Apple Silicon (tensorflow-macos 2.13 + Keras 3 format weights).

Architecture mirrors LDCT-improved-se2.ipynb: build_mod_seg_se2_v2()
"""
import os
import logging
import numpy as np

logger = logging.getLogger("LDCT-ModelBuilder")

IMG_SIZE    = (256, 256)
NUM_CLASSES = 2


# ─── Architecture helpers (tf.keras) ──────────────────────────────────────────

def _se_block(x, ratio=8):
    """Squeeze-and-Excitation block — identical to the notebook."""
    import tensorflow as tf
    keras = tf.keras
    filters = x.shape[-1]
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Reshape((1, 1, filters))(se)
    se = keras.layers.Dense(filters // ratio, activation="relu",
                            kernel_initializer="he_normal", use_bias=False)(se)
    se = keras.layers.Dense(filters, activation="sigmoid",
                            kernel_initializer="he_normal", use_bias=False)(se)
    return keras.layers.Multiply()([x, se])


def _conv_bn_relu(x, filters, kernel_size=(3, 3)):
    """Conv2D → BatchNorm → ReLU — identical to the notebook."""
    import tensorflow as tf
    keras = tf.keras
    x = keras.layers.Conv2D(filters, kernel_size, padding="same",
                            kernel_initializer="he_normal", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def build_mod_seg_se2_v2(input_shape=(256, 256, 3), num_classes=NUM_CLASSES):
    """
    Improved Mod-Seg-SE(2) v2 — 3-level U-Net + SE blocks + dual head.
    Exact replica of build_mod_seg_se2_v2() in LDCT-improved-se2.ipynb.
    """
    import tensorflow as tf
    keras = tf.keras

    inputs = keras.layers.Input(shape=input_shape, name="image_input")

    # ── ENCODER ────────────────────────────────────────────────────────────────
    c1 = _conv_bn_relu(inputs, 32);  c1 = _conv_bn_relu(c1, 32);  c1 = _se_block(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = _conv_bn_relu(p1, 64);  c2 = _conv_bn_relu(c2, 64);  c2 = _se_block(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = _conv_bn_relu(p2, 128); c3 = _conv_bn_relu(c3, 128); c3 = _se_block(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # ── BOTTLENECK ─────────────────────────────────────────────────────────────
    bn = _conv_bn_relu(p3, 256); bn = _conv_bn_relu(bn, 256);  bn = _se_block(bn)

    # ── CLASSIFICATION HEAD ────────────────────────────────────────────────────
    cls = keras.layers.GlobalAveragePooling2D(name="gap_classification")(bn)
    cls = keras.layers.Dense(128, activation="relu",
                             kernel_initializer="he_normal")(cls)
    cls = keras.layers.BatchNormalization()(cls)
    cls = keras.layers.Dropout(0.4)(cls)
    cls = keras.layers.Dense(64, activation="relu",
                             kernel_initializer="he_normal")(cls)
    cls = keras.layers.Dropout(0.3)(cls)
    class_output = keras.layers.Dense(
        num_classes, activation="softmax", name="class_output"
    )(cls)

    # ── SEGMENTATION DECODER ───────────────────────────────────────────────────
    u3 = keras.layers.UpSampling2D((2, 2))(bn)
    d3 = keras.layers.Concatenate()([u3, c3])
    d3 = _conv_bn_relu(d3, 128); d3 = _conv_bn_relu(d3, 128); d3 = _se_block(d3)

    u2 = keras.layers.UpSampling2D((2, 2))(d3)
    d2 = keras.layers.Concatenate()([u2, c2])
    d2 = _conv_bn_relu(d2, 64);  d2 = _conv_bn_relu(d2, 64);  d2 = _se_block(d2)

    u1 = keras.layers.UpSampling2D((2, 2))(d2)
    d1 = keras.layers.Concatenate()([u1, c1])
    d1 = _conv_bn_relu(d1, 32);  d1 = _conv_bn_relu(d1, 32);  d1 = _se_block(d1)

    seg_output = keras.layers.Conv2D(
        1, (1, 1), activation="sigmoid", name="seg_output"
    )(d1)

    model = keras.Model(inputs=inputs, outputs=[class_output, seg_output])
    return model


# ─── Name remap: tf.keras 2.13 name → h5 (Keras-3) group name ─────────────────
#
# Root cause: tf.keras 2.13 creates the classification head layers (Dense 128,
# BatchNorm, Dense 64, Dense 2) BEFORE the decoder SE blocks in forward-pass
# order, giving them lower counter numbers. Keras 3 in contrast gives the decoder
# SE blocks priority and pushes class head dense/BN to higher indices.
#
# Verified by inspecting model.layers shapes vs h5 group shapes directly.
LAYER_REMAP = {
    # Dense layers (classification head numbered earlier in tf.keras than in Keras 3)
    "dense_8":     "dense_12",    # class Dense(128): (256,128)+(128,)
    "dense_9":     "dense_14",    # class Dense(64): (128,64)+(64,)
    "dense_10":    "dense_8",     # decoder d3 SE first: (128,16)
    "dense_11":    "dense_9",     # decoder d3 SE second: (16,128)
    "dense_12":    "dense_10",    # decoder d2 SE first: (64,8)
    "dense_13":    "dense_11",    # decoder d2 SE second: (8,64)
    "dense_14":    "dense_13",    # decoder d1 SE first: (32,4)
    # dense_15 → dense_15: coincidentally matches (4,32) ✓
    "class_output": "dense_16",   # final Dense(2): (64,2)+(2,)

    # BatchNormalization (class head BN is BN_8 in tf.keras, BN_14 in h5)
    "batch_normalization_8":  "batch_normalization_14",  # class head BN (128,)
    "batch_normalization_9":  "batch_normalization_8",   # d3 conv1 BN (128,)
    "batch_normalization_10": "batch_normalization_9",   # d3 conv2 BN (128,)
    "batch_normalization_11": "batch_normalization_10",  # d2 conv1 BN (64,)
    "batch_normalization_12": "batch_normalization_11",  # d2 conv2 BN (64,)
    "batch_normalization_13": "batch_normalization_12",  # d1 conv1 BN (32,)
    "batch_normalization_14": "batch_normalization_13",  # d1 conv2 BN (32,)

    # seg_output: explicitly named in training but stored as auto-name in h5
    "seg_output": "conv2d_14",    # final Conv2D(1,sigmoid): (1,1,32,1)+(1,)
}


# ─── Custom Keras-3 weight loader (name-based mapping) ────────────────────────

def load_keras3_weights(model, h5_path: str) -> None:
    """
    Load Keras-3 format weights (model.weights.h5) into a tf.keras model.

    Keras 3 and tf.keras 2.x use the SAME auto-naming convention for layers
    (conv2d, conv2d_1, batch_normalization, ...). We exploit this by matching
    the model's layer.name directly to the h5 file's group names.

    For each layer with weights:
        h5['layers'][layer.name]['vars'][0, 1, ...] → layer.weights

    This is fully robust to layer ordering differences.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        layers_grp = f["layers"]
        available_names = set(layers_grp.keys())

        assigned = 0
        skipped  = 0

        for layer in model.layers:
            if not layer.weights:
                continue

            lname = layer.name  # e.g. "conv2d_3", "batch_normalization_5"

            # Apply name remap (tf.keras 2.13 → Keras-3 h5 naming offset fix)
            h5_name = LAYER_REMAP.get(lname, lname)

            if h5_name not in available_names:
                logger.warning(
                    f"Layer '{lname}' not found in h5 file — keeping random init."
                )
                skipped += len(layer.weights)
                continue

            vg = layers_grp[h5_name]
            if "vars" not in vg:
                logger.warning(f"Layer '{lname}' (h5: '{h5_name}') has no 'vars' group — skipping.")
                skipped += len(layer.weights)
                continue

            vars_grp = vg["vars"]
            h5_vals  = [
                np.array(vars_grp[vi])
                for vi in sorted(vars_grp.keys(), key=lambda x: int(x))
            ]

            if len(h5_vals) != len(layer.weights):
                logger.warning(
                    f"'{lname}': expected {len(layer.weights)} weights, "
                    f"h5 has {len(h5_vals)} — skipping."
                )
                skipped += len(layer.weights)
                continue

            # Validate shapes before assigning
            mismatch = False
            for i, (val, w) in enumerate(zip(h5_vals, layer.weights)):
                if val.shape != tuple(w.shape):
                    logger.warning(
                        f"'{lname}' var {i}: shape mismatch "
                        f"h5={val.shape} vs model={tuple(w.shape)} — skipping layer."
                    )
                    mismatch = True
                    break

            if mismatch:
                skipped += len(layer.weights)
                continue

            layer.set_weights(h5_vals)
            assigned += len(layer.weights)

    logger.info(
        f"Weight loading complete: {assigned} weights assigned, "
        f"{skipped} skipped."
    )



def load_model_from_weights(weights_h5_path: str):
    """
    Build the Mod-Seg-SE(2) v2 model and load pre-trained Keras-3 weights.

    Args:
        weights_h5_path: Absolute path to model.weights.h5

    Returns:
        tf.keras.Model ready for inference.

    Raises:
        FileNotFoundError if the weights file does not exist.
    """
    if not os.path.exists(weights_h5_path):
        raise FileNotFoundError(f"Weights file not found: {weights_h5_path}")

    logger.info("Building Mod-Seg-SE(2) v2 architecture (tf.keras)...")
    model = build_mod_seg_se2_v2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=NUM_CLASSES,
    )

    logger.info(f"Loading Keras-3 weights from: {weights_h5_path}")
    load_keras3_weights(model, weights_h5_path)

    logger.info(
        f"✅ Model ready — {len(model.layers)} layers, "
        f"{model.count_params():,} parameters"
    )
    return model
