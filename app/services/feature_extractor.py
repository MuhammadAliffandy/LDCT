"""
feature_extractor.py
--------------------
Handcrafted feature extraction pipeline.
Single Responsibility: Image preprocessing + Wavelet & GLCM features.
"""
import numpy as np
import cv2
import pywt
import scipy.stats
import logging

logger = logging.getLogger("LDCT-FeatureExtractor")

IMG_SIZE = (256, 256)
WAVELET = "db1"

# Feature names for interpretability
FEATURE_NAMES = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Entropy",
    "LH_Mean", "LH_Std", "LH_Var", "LH_Entropy",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Entropy",
    "HH_Mean", "HH_Std", "HH_Var", "HH_Entropy",
    "HH_Energy",
    "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity",
]


def apply_hu_window_png(gray: np.ndarray) -> np.ndarray:
    """
    Approximate HU windowing for already-rendered CT PNG images.

    Because uploaded PNGs don't carry raw HU values, we use percentile-based
    soft clipping (p1–p99) to replicate the dynamic range compression that
    HU windowing applies during DICOM training.

    This matches the effect of apply_hu_window() in LDCT-improved-se2.ipynb.

    Args:
        gray: Grayscale image (H, W) as float32 or uint8

    Returns:
        Normalized float32 image in [0, 1]
    """
    img = gray.astype(np.float32)
    p_low  = np.percentile(img, 1)
    p_high = np.percentile(img, 99)
    if p_high - p_low < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low)    # → [0, 1]
    return img.astype(np.float32)


def smart_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a CT image to match the LDCT-improved-se2.ipynb training pipeline.

    Mirrors process_dicom() from the notebook:
      1. Convert to grayscale
      2. Apply percentile windowing (approximates HU lung window on rendered PNGs)
      3. Resize to 256×256
      4. Stack to 3-channel RGB float32 in [0, 1]

    Args:
        img: RGB or grayscale image array (any size, uint8 or float32)

    Returns:
        (256, 256, 3) float32 in [0, 1]  — ready for model.predict()
    """
    if img is None:
        return None

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # HU-style windowing
    windowed = apply_hu_window_png(gray)

    # Resize to 256×256
    resized = cv2.resize(windowed, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    # Stack to 3-channel float RGB (matches np.stack([img]*3, axis=-1) in notebook)
    return np.stack([resized] * 3, axis=-1).astype(np.float32)  # (256, 256, 3) in [0,1]



def _wavelet_band_stats(band: np.ndarray) -> list:
    """Compute statistical features from a single wavelet sub-band."""
    flat = np.abs(band.flatten()) + 1e-6
    return [
        float(np.mean(band)),
        float(np.std(band)),
        float(np.var(band)),
        float(scipy.stats.entropy(flat)),
    ]


def extract_handcrafted(img: np.ndarray) -> list:
    """
    Extract 20 handcrafted features from a preprocessed RGB image.

    Features:
    - 4 Wavelet sub-bands × 4 stats (Mean, Std, Var, Entropy) = 16
    - HH Energy = 1
    - GLCM Contrast, Dissimilarity, Homogeneity = 3
    Total = 20

    Args:
        img: RGB image array (H, W, 3) or grayscale (H, W)

    Returns:
        List of 20 float features
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Wavelet Decomposition (Level 1, db1)
    coeffs = pywt.dwt2(gray.astype(np.float32), WAVELET)
    LL, (LH, HL, HH) = coeffs

    feats = []
    for band in [LL, LH, HL, HH]:
        feats.extend(_wavelet_band_stats(band))

    # HH sub-band energy
    feats.append(float(np.sum(np.square(HH))))

    # GLCM Texture Features
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(
        gray_norm,
        distances=[5],
        angles=[0, np.pi / 4, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True,
    )
    feats.append(float(graycoprops(glcm, "contrast").mean()))
    feats.append(float(graycoprops(glcm, "dissimilarity").mean()))
    feats.append(float(graycoprops(glcm, "homogeneity").mean()))

    return feats  # 20 features


def features_to_dict(feats: list) -> dict:
    """Convert flat feature list to named dict for API response."""
    return {
        name: round(val, 6)
        for name, val in zip(FEATURE_NAMES, feats)
    }
