"""
heatmap_service.py
------------------
GradCAM & saliency map visualization for LDCT predictions.
Single Responsibility: Generate and overlay heatmaps on CT images.
"""
import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger("LDCT-HeatmapService")


def generate_gradcam(
    model,
    input_img: np.ndarray,
    target_class_idx: int = None,
) -> np.ndarray:
    """
    Bottleneck-layer GradCAM for the single-input multi-task Keras model.

    Matches the make_gradcam_heatmap() implementation in LDCT-improved-se2.ipynb:
      - Finds the last Conv2D layer at IMG_SIZE // 8 spatial resolution (bottleneck)
      - Computes gradients of class score w.r.t. feature map activations
      - Pools gradients → weights channels → produces localized attention map

    This gives much sharper and more spatially meaningful maps than the
    previous image-gradient (vanilla saliency) approach.

    Args:
        model: Full Keras model (best_mod_seg_se2_v2.keras)
        input_img: float32 array (1, 256, 256, 3) already in [0, 1]
        target_class_idx: Class index. If None, uses predicted class.

    Returns:
        heatmap: float32 array (H, W) normalized [0, 1]
    """
    import tensorflow as tf
    import tensorflow as _tf_mod; keras = _tf_mod.keras

    # ── Find bottleneck Conv2D layer (deepest, smallest spatial size) ──────────
    target_layer = None
    for layer in model.layers:
        if not isinstance(layer, keras.layers.Conv2D):
            continue
        try:
            shape = layer.output.shape   # symbolic TensorShape — always works
            h = shape[1]
            if h is not None and int(h) == 256 // 8:  # 32px for 256-input, 3 MaxPools
                target_layer = layer.name
        except Exception:
            pass

    # Fallback: last Conv2D with most filters (= bottleneck = 256 filters)
    if target_layer is None:
        max_f = 0
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                if layer.filters >= max_f:
                    max_f = layer.filters
                    target_layer = layer.name

    if target_layer is None:
        logger.warning("GradCAM: no Conv2D found — falling back to saliency.")
        return generate_saliency_map(model, input_img, target_class_idx)

    # ── Build sub-model: conv outputs + class head ────────────────────────────
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(target_layer).output,
                model.get_layer("class_output").output
            ]
        )
    except Exception as e:
        logger.warning(f"GradCAM sub-model failed ({e}) — falling back to saliency.")
        return generate_saliency_map(model, input_img, target_class_idx)

    img_tensor = tf.cast(input_img, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)

        if target_class_idx is None:
            target_class_idx = int(tf.argmax(predictions[0]).numpy())

        class_score = predictions[:, target_class_idx]

    # Gradients of class score w.r.t. conv feature map
    grads = tape.gradient(class_score, conv_outputs)  # (1, h, w, C)

    if grads is None:
        logger.warning("GradCAM: gradient is None — falling back to saliency.")
        return generate_saliency_map(model, input_img, target_class_idx)

    # Global Average Pool gradients → per-channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight channels by importance and collapse to 2D
    conv_out = conv_outputs[0]                                # (h, w, C)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]        # (h, w, 1)
    heatmap = tf.squeeze(heatmap).numpy()                     # (h, w)

    # ReLU + normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    logger.info(f"GradCAM: layer='{target_layer}', class={target_class_idx}")
    return heatmap.astype(np.float32)


def generate_saliency_map(
    model,
    input_img: np.ndarray,
    target_class_idx: int = None,
) -> np.ndarray:
    """
    Vanilla gradient saliency for single-input multi-task model.

    Args:
        model: Full Keras model
        input_img: float32 array (1, 256, 256, 3)
        target_class_idx: Class index. If None, uses predicted class.

    Returns:
        saliency: float32 array (H, W) normalized [0, 1]
    """
    import tensorflow as tf
    import tensorflow as _tf_mod; keras = _tf_mod.keras

    img_var = tf.Variable(tf.cast(np.array(input_img), tf.float32))

    with tf.GradientTape() as tape:
        preds = model(img_var)
        class_preds = preds[0]
        
        if target_class_idx is None:
            target_class_idx = int(tf.argmax(class_preds[0]).numpy())
        
        loss = class_preds[:, target_class_idx]

    grads = tape.gradient(loss, img_var)  # (1, H, W, 3)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()  # (H, W)

    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency.astype(np.float32)






def overlay_heatmap(
    orig_img: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a heatmap on the original image.

    Args:
        orig_img: RGB image (H, W, 3) uint8
        heatmap: float32 array (H', W') normalized [0,1]
        colormap: OpenCV colormap constant
        alpha: blend ratio for heatmap overlay

    Returns:
        overlaid: RGB image (H, W, 3) uint8
    """
    h, w = orig_img.shape[:2]

    # Resize heatmap to match original image
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_resized, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    orig_float = orig_img.astype(np.float32)
    heat_float = colored_heatmap.astype(np.float32)
    overlay = (1 - alpha) * orig_float + alpha * heat_float
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def generate_heatmap_from_image(image_path: str, mode: str = "gradcam") -> dict:
    """
    Full heatmap generation pipeline for the single-input multi-task model.

    Args:
        image_path: path to CT image
        mode: 'gradcam' or 'saliency'

    Returns:
        dict with:
        - heatmap_overlay_b64: base64 PNG of the overlaid attention heatmap
        - heatmap_raw_b64: base64 PNG of the raw attention heatmap
        - location_overlay_b64: base64 PNG of the predicted disease location mask
        - mode: which method was used
        - target_class: predicted class index
    """
    import os
    from app.services.model_loader import get_models, is_ready
    from app.services.feature_extractor import smart_preprocess

    if not is_ready():
        return {"error": "Models not loaded"}

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    try:
        models = get_models()

        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return {"error": "Cannot read image"}

        # ---- Prepare input — notebook-compatible (HU-windowed, float [0, 1]) ----
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        # smart_preprocess returns float32 [0,1]  — same as process_dicom() in notebook
        preprocessed = smart_preprocess(img_rgb)           # (256, 256, 3) float32 [0,1]
        input_batch  = np.expand_dims(preprocessed, axis=0)  # (1, 256, 256, 3)

        # Get predicted outputs to extract class and segmentation mask
        keras_outputs = models["keras"].predict(input_batch, verbose=0)
        class_preds = keras_outputs[0][0]
        seg_mask_batch = keras_outputs[1]

        target_class = int(np.argmax(class_preds))

        # ---- Attention Map (GradCAM or Saliency) --------------------------------
        if mode == "saliency":
            heatmap = generate_saliency_map(models["keras"], input_batch, target_class)
        else:
            heatmap = generate_gradcam(models["keras"], input_batch, target_class)

        # Overlay Attention Map on the preprocessed image (uint8 for display)
        display_img = (preprocessed * 255).astype(np.uint8)  # [0,255] RGB
        overlaid_attention = overlay_heatmap(display_img, heatmap)
        _, buf_attention = cv2.imencode(".png", cv2.cvtColor(overlaid_attention, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buf_attention).decode("utf-8")

        # Raw Attention Colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(heatmap, (256, 256))), cv2.COLORMAP_JET
        )
        _, buf_raw = cv2.imencode(".png", heatmap_colored)
        raw_heatmap_b64 = base64.b64encode(buf_raw).decode("utf-8")

        # ---- Segmentation Mask Overlays -----------------------------------------
        location_b64 = None
        seg_mask_b64 = None
        try:
            seg_mask_2d = seg_mask_batch[0, :, :, 0]   # (256, 256) float32 sigmoid

            # Normalize for display
            seg_display = seg_mask_2d
            if seg_display.max() > 0:
                seg_display = seg_display / seg_display.max()

            # ── Location overlay (hot colormap) — kept for backwards compat. ──
            loc_overlaid = overlay_heatmap(
                display_img, seg_display, colormap=cv2.COLORMAP_HOT, alpha=0.5
            )
            _, loc_buf = cv2.imencode(".png", cv2.cvtColor(loc_overlaid, cv2.COLOR_RGB2BGR))
            location_b64 = base64.b64encode(loc_buf).decode("utf-8")

            # ── Raw segmentation sigmoid as hot PNG (new field) ───────────────
            seg_colored = cv2.applyColorMap(
                np.uint8(255 * cv2.resize(seg_display, (256, 256))), cv2.COLORMAP_HOT
            )
            _, seg_buf = cv2.imencode(".png", seg_colored)
            seg_mask_b64 = base64.b64encode(seg_buf).decode("utf-8")

        except Exception as loc_e:
            logger.error(f"Segmentation overlay error: {loc_e}", exc_info=True)

        logger.info(f"Heatmap generated: mode={mode}, target_class={target_class}")

        return {
            "heatmap_overlay_b64": heatmap_b64,
            "heatmap_raw_b64":     raw_heatmap_b64,
            "location_overlay_b64": location_b64,
            "seg_mask_b64":         seg_mask_b64,   # raw seg sigmoid — new field
            "mode": mode,
            "target_class": target_class,
        }

    except Exception as e:
        logger.error(f"Heatmap generation error: {e}", exc_info=True)
        return {"error": str(e)}
