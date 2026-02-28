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
    Gradient-based class activation map for the single-input multi-task Keras model.

    Uses image-space gradients via tf.Variable — compatible with Keras 3.x
    where mid-computation tensor watching is not supported.

    Args:
        model: Full Keras model (mod_seg_se2_model.h5)
        input_img: float32 array (1, 256, 256, 3)
        target_class_idx: Class index. If None, uses predicted class.

    Returns:
        heatmap: float32 array (H, W) normalized [0, 1]
    """
    import tensorflow as tf

    img_var = tf.Variable(
        np.array(input_img).astype(np.float32), trainable=True
    )

    with tf.GradientTape() as tape:
        # Puts the image tensor through the model. 
        # For the multi-task model, outputs are [class_preds, seg_preds]
        preds = model(img_var)
        class_preds = preds[0]

        if target_class_idx is None:
            target_class_idx = int(tf.argmax(class_preds[0]).numpy())

        # Grab the scalar loss value for the given class
        loss = class_preds[:, target_class_idx]

    # Compute gradient of class score with respect to input image
    grads = tape.gradient(loss, img_var)

    if grads is None:
        logger.warning("GradCAM: gradient is None — returning uniform heatmap.")
        return np.ones((256, 256), dtype=np.float32) * 0.5

    # Collapse channels: take mean of absolute gradients
    grads_np = grads.numpy()[0]                           # (H, W, 3)
    heatmap = np.mean(np.abs(grads_np), axis=-1)          # (H, W)

    # ReLU + normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

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

        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        processed_img = smart_preprocess(img_rgb)

        # Prepare inputs for Keras model
        input_float = processed_img.astype(np.float32) / 255.0
        input_batch = np.expand_dims(input_float, axis=0)

        # Get predicted outputs to extract class and segmentation mask
        keras_outputs = models["keras"].predict(input_batch, verbose=0)
        class_preds = keras_outputs[0][0]
        seg_mask_batch = keras_outputs[1]

        target_class = int(np.argmax(class_preds))

        # Generate Attention Map (GradCAM or Saliency)
        if mode == "saliency":
            heatmap = generate_saliency_map(models["keras"], input_batch, target_class)
        else:
            heatmap = generate_gradcam(models["keras"], input_batch, target_class)

        # Overlay Attention Map
        overlaid_attention = overlay_heatmap(processed_img, heatmap)
        _, buf_attention = cv2.imencode(".png", cv2.cvtColor(overlaid_attention, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buf_attention).decode("utf-8")

        # Raw Attention Map
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(heatmap, (256, 256))), cv2.COLORMAP_JET
        )
        _, buf_raw = cv2.imencode(".png", heatmap_colored)
        raw_heatmap_b64 = base64.b64encode(buf_raw).decode("utf-8")

        # --- Disease Location Overlay (from segmentation mask) ---
        location_b64 = None
        try:
            seg_mask_2d = seg_mask_batch[0, :, :, 0]
            if seg_mask_2d.max() > 0:
                seg_mask_2d = seg_mask_2d / seg_mask_2d.max()

            # Overlay using a hot colormap to differentiate from conventional GradCAM
            loc_overlaid = overlay_heatmap(
                processed_img, seg_mask_2d, colormap=cv2.COLORMAP_HOT, alpha=0.5
            )
            _, loc_buf = cv2.imencode(".png", cv2.cvtColor(loc_overlaid, cv2.COLOR_RGB2BGR))
            location_b64 = base64.b64encode(loc_buf).decode("utf-8")
        except Exception as loc_e:
            logger.error(f"Disease Locator inference error: {loc_e}", exc_info=True)

        logger.info(f"Heatmap generated: mode={mode}, target_class={target_class}")

        return {
            "heatmap_overlay_b64": heatmap_b64,
            "heatmap_raw_b64": raw_heatmap_b64,
            "location_overlay_b64": location_b64,
            "mode": mode,
            "target_class": target_class,
        }

    except Exception as e:
        logger.error(f"Heatmap generation error: {e}", exc_info=True)
        return {"error": str(e)}
