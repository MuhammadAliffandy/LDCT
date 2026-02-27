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
    all_inputs: list,
    target_class_idx: int = None,
) -> np.ndarray:
    """
    Gradient-based class activation map for the 3-input hybrid Keras model.

    Uses image-space gradients via tf.Variable — compatible with Keras 3.x
    multi-input models where mid-computation tensor watching is not supported.

    Args:
        model: Full Keras model with 3 inputs [image, scaled_feats, umap_feats]
        all_inputs: [image_batch (1,256,256,3), scaled_feats (1,20), umap_feats (1,2)]
        target_class_idx: Class index. If None, uses predicted class.

    Returns:
        heatmap: float32 array (H, W) normalized [0, 1]
    """
    import tensorflow as tf

    # Wrap the IMAGE as a tf.Variable so TF automatically tracks gradients
    img_var = tf.Variable(
        np.array(all_inputs[0]).astype(np.float32), trainable=True
    )
    # Other inputs are constants
    other = [tf.constant(np.array(x), dtype=tf.float32) for x in all_inputs[1:]]

    with tf.GradientTape() as tape:
        # img_var is a Variable — tape watches it automatically
        preds = model([img_var] + other)

        if target_class_idx is None:
            target_class_idx = int(tf.argmax(preds[0]).numpy())

        loss = preds[:, target_class_idx]

    # d_loss / d_image  →  shape (1, H, W, 3)
    grads = tape.gradient(loss, img_var)

    if grads is None:
        logger.warning("GradCAM: gradient is None — returning uniform heatmap.")
        return np.ones((256, 256), dtype=np.float32) * 0.5

    # Collapse channels: take mean of absolute gradients per spatial location
    # shape → (H, W)
    grads_np = grads.numpy()[0]                           # (H, W, 3)
    heatmap = np.mean(np.abs(grads_np), axis=-1)          # (H, W)

    # ReLU + normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap.astype(np.float32)


def generate_saliency_map(
    model,
    all_inputs: list,
    target_class_idx: int = None,
) -> np.ndarray:
    """
    Vanilla gradient saliency for 3-input hybrid model.

    Args:
        model: Full Keras model with 3 inputs
        all_inputs: [image_batch, scaled_feats, umap_feats]
        target_class_idx: Class index. If None, uses predicted class.

    Returns:
        saliency: float32 array (H, W) normalized [0, 1]
    """
    import tensorflow as tf

    img_var = tf.Variable(tf.cast(np.array(all_inputs[0]), tf.float32))
    other = [tf.cast(np.array(x), tf.float32) for x in all_inputs[1:]]

    with tf.GradientTape() as tape:
        preds = model([img_var] + other)
        if target_class_idx is None:
            target_class_idx = int(tf.argmax(preds[0]).numpy())
        loss = preds[:, target_class_idx]

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
    Full heatmap generation pipeline from an image file path.

    Args:
        image_path: path to CT image
        mode: 'gradcam' or 'saliency'

    Returns:
        dict with:
        - heatmap_b64: base64 PNG of the overlaid heatmap
        - mode: which method was used
        - target_class: predicted class index
    """
    import os
    from app.services.model_loader import get_models, is_ready
    from app.services.feature_extractor import smart_preprocess, extract_handcrafted

    if not is_ready():
        return {"error": "Models not loaded"}

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    try:
        import scipy.stats

        models = get_models()

        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return {"error": "Cannot read image"}

        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        processed_img = smart_preprocess(img_rgb)

        # Prepare inputs for Keras model
        input_float = processed_img.astype(np.float32) / 255.0
        input_batch = np.expand_dims(input_float, axis=0)

        raw_feats = extract_handcrafted(processed_img)
        feat_array = np.array(raw_feats).reshape(1, -1)
        scaled_feats = models["scaler"].transform(feat_array)
        umap_feats = models["umap"].transform(scaled_feats)

        # Get predicted class for targeted heatmap
        keras_proba = models["keras"].predict(
            [input_batch, scaled_feats, umap_feats], verbose=0
        )[0]
        target_class = int(np.argmax(keras_proba))

        # Build the 3-input list required by the hybrid model
        all_inputs = [input_batch, scaled_feats, umap_feats]

        # Generate heatmap using all 3 inputs
        if mode == "saliency":
            heatmap = generate_saliency_map(models["keras"], all_inputs, target_class)
        else:
            heatmap = generate_gradcam(models["keras"], all_inputs, target_class)


        # Overlay
        overlaid = overlay_heatmap(processed_img, heatmap)

        # Encode to base64
        _, buf = cv2.imencode(".png", cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buf).decode("utf-8")

        # Also encode raw heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(heatmap, (256, 256))), cv2.COLORMAP_JET
        )
        _, buf2 = cv2.imencode(".png", heatmap_colored)
        raw_heatmap_b64 = base64.b64encode(buf2).decode("utf-8")

        logger.info(f"Heatmap generated: mode={mode}, target_class={target_class}")

        return {
            "heatmap_overlay_b64": heatmap_b64,
            "heatmap_raw_b64": raw_heatmap_b64,
            "mode": mode,
            "target_class": target_class,
        }

    except Exception as e:
        logger.error(f"Heatmap generation error: {e}", exc_info=True)
        return {"error": str(e)}
