import cv2
import numpy as np
import math
import argparse
from pathlib import Path


def ensure_odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def show(img, title=None, is_bgr=True, figsize=(7, 5)):
    """Display an image in Jupyter (OpenCV loads BGR by default)."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        if is_bgr:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def binarize_otsu_with_offset(gray_blur: np.ndarray, otsu_offset: int = 5):
    """
    Otsu threshold + a small positive offset.
    The offset helps suppress background texture that may create false connections.
    """
    thr, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr2 = min(255, thr + int(otsu_offset))
    _, th = cv2.threshold(gray_blur, thr2, 255, cv2.THRESH_BINARY)

    # Ensure particles are white (255)
    mean_fg = float(np.mean(gray_blur[th == 255])) if np.any(th == 255) else 0.0
    mean_bg = float(np.mean(gray_blur[th == 0])) if np.any(th == 0) else 0.0
    if mean_fg < mean_bg:
        th = cv2.bitwise_not(th)
    return th


def cleanup_binary(th: np.ndarray, k_small: int = 3, open_iter: int = 1, close_iter: int = 2):
    """Morphological open/close to remove small speckles and smooth regions."""
    k_small = ensure_odd(k_small)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    out = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=int(open_iter))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
    return out


def watershed_split(img_bgr: np.ndarray, binary_mask: np.ndarray, k_small: int, dist_ratio: float, dilate_iter: int = 3):
    """
    Split touching objects using distance transform + watershed.

    markers_ws:
      -1: watershed boundary
       1: background (typically)
      >1: object labels
    """
    k_small = ensure_odd(k_small)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))

    sure_bg = cv2.dilate(binary_mask, kernel, iterations=int(dilate_iter))

    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_norm = dist / (dist.max() + 1e-9)

    _, sure_fg = cv2.threshold(dist_norm, float(dist_ratio), 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers_ws = cv2.watershed(img_bgr.copy(), markers)
    return markers_ws


def labels_to_list(markers_ws: np.ndarray):
    """Collect valid object labels from watershed output."""
    labs = np.unique(markers_ws)
    return [int(x) for x in labs if x > 1]


def draw_highlight(img_bgr: np.ndarray, markers_ws: np.ndarray, selected_labels, thickness: int = 2, seed: int = 0):
    """Draw colored contours for selected labels."""
    out = img_bgr.copy()
    rng = np.random.default_rng(int(seed))
    for pid in selected_labels:
        mask = (markers_ws == pid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(x) for x in rng.integers(0, 255, size=3))
        cv2.drawContours(out, contours, -1, color, int(thickness))
    return out


def put_text_bottom_right(
    img_bgr: np.ndarray,
    lines,
    margin: int = 12,
    line_gap: int = 20,
    font_scale: float = 0.55,
    thickness: int = 1,
    color=(255, 255, 255),          # white (BGR)
    outline: bool = True,           # subtle outline for readability
    outline_color=(0, 0, 0),
    outline_thickness: int = 2
):
    """Render multi-line text at the bottom-right, right-aligned."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [str(s) for s in lines]
    sizes = []
    max_w = 0
    max_h = 0
    for s in lines:
        (tw, th), base = cv2.getTextSize(s, font, font_scale, thickness)
        sizes.append((tw, th, base))
        max_w = max(max_w, tw)
        max_h = max(max_h, th)

    y_last = h - margin
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i]
        tw, th, base = sizes[i]
        x = w - margin - tw
        y = y_last - (len(lines) - 1 - i) * int(line_gap)

        if outline:
            cv2.putText(out, s, (x, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(out, s, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return out


def coin_props_from_labels(markers_ws: np.ndarray):
    """
    Estimate per-object properties from watershed labels:
    - centroid (cx, cy)
    - area (pixels)
    - radius ~= sqrt(area / pi)
    """
    props = {}
    for pid in labels_to_list(markers_ws):
        mask = (markers_ws == pid)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        area = int(xs.size)
        cx = float(xs.mean())
        cy = float(ys.mean())
        r = math.sqrt(area / math.pi)
        props[pid] = (cx, cy, r, area)
    return props


def nonoverlap_by_circle(props: dict, overlap_alpha: float = 1.095):
    """
    Decide overlap by circle distance:
    if d < overlap_alpha * (r1 + r2), treat as touching/overlapping.
    """
    labels = list(props.keys())
    overlap = set()
    for i in range(len(labels)):
        a = labels[i]
        cxa, cya, ra, _ = props[a]
        for j in range(i + 1, len(labels)):
            b = labels[j]
            cxb, cyb, rb, _ = props[b]
            d = math.hypot(cxa - cxb, cya - cyb)
            if d < float(overlap_alpha) * (ra + rb):
                overlap.add(a)
                overlap.add(b)
    non = [pid for pid in labels if pid not in overlap]
    return non, list(overlap)


def count_particles(
    img_bgr: np.ndarray,
    roi=None,                # (x, y, w, h) or None

    # Preprocess / threshold
    blur_ksize=5,            # Gaussian blur kernel size
    clahe_clip=None,         # set to ~2.0-3.0 if you need local contrast
    otsu_offset=5,           # push threshold up a bit to reduce false bridges

    # Morphology
    k_small=3,
    open_iter=1,
    close_iter=2,

    # Watershed
    dist_ratio=0.42,
    dilate_iter=3,

    # Non-overlap rule
    overlap_alpha=1.095,

    # Visualization
    contour_thickness=2,
    seed=0,

    # Text style
    text_font_scale=0.55,
    text_thickness=1,
    text_margin=12,
    text_line_gap=20,
    return_debug=True
):
    """
    Returns:
      num1, num2, vis_all, vis_nonoverlap, debug(dict)
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    img = img_bgr.copy()
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w].copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if clahe_clip is not None:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    blur_ksize = ensure_odd(blur_ksize)
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    th = binarize_otsu_with_offset(gray_blur, otsu_offset=otsu_offset)
    binary = cleanup_binary(th, k_small=k_small, open_iter=open_iter, close_iter=close_iter)

    markers_ws = watershed_split(
        img_bgr=img,
        binary_mask=binary,
        k_small=k_small,
        dist_ratio=dist_ratio,
        dilate_iter=dilate_iter
    )

    particle_labels = labels_to_list(markers_ws)
    num1 = len(particle_labels)

    props = coin_props_from_labels(markers_ws)
    nonoverlap_labels, overlap_labels = nonoverlap_by_circle(props, overlap_alpha=overlap_alpha)
    num2 = len(nonoverlap_labels)

    vis_all = draw_highlight(img, markers_ws, particle_labels, thickness=contour_thickness, seed=seed)
    vis_all = put_text_bottom_right(
        vis_all,
        [f"Num1 (all): {num1}", f"Num2 (non-overlap): {num2}"],
        margin=text_margin,
        line_gap=text_line_gap,
        font_scale=text_font_scale,
        thickness=text_thickness,
        color=(255, 255, 255),
        outline=True
    )

    vis_non = draw_highlight(img, markers_ws, nonoverlap_labels, thickness=contour_thickness, seed=seed)
    vis_non = put_text_bottom_right(
        vis_non,
        [f"Num2 (non-overlap only): {num2}"],
        margin=text_margin,
        line_gap=text_line_gap,
        font_scale=text_font_scale,
        thickness=text_thickness,
        color=(255, 255, 255),
        outline=True
    )

    debug = {}
    if return_debug:
        ws_boundary = img.copy()
        ws_boundary[markers_ws == -1] = (0, 0, 255)

        debug = {
            "img_used": img,
            "gray": gray,
            "gray_blur": gray_blur,
            "binary": binary,
            "markers_ws": markers_ws,
            "ws_boundary_vis": ws_boundary,
            "particle_labels": particle_labels,
            "nonoverlap_labels": nonoverlap_labels,
            "overlap_labels": overlap_labels,
            "props": props
        }

    return num1, num2, vis_all, vis_non, debug


def run_particle_count(image_path: str, output_dir: str = "outputs", show_plots: bool = False, roi=None):
    """Run particle counting and save result visualizations."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num1, num2, vis_all, vis_non, dbg = count_particles(img, roi=roi)

    all_file = output_path / "particles_all.png"
    nonoverlap_file = output_path / "particles_nonoverlap.png"
    binary_file = output_path / "particles_binary.png"
    watershed_file = output_path / "particles_watershed_boundary.png"

    cv2.imwrite(str(all_file), vis_all)
    cv2.imwrite(str(nonoverlap_file), vis_non)
    cv2.imwrite(str(binary_file), dbg["binary"])
    cv2.imwrite(str(watershed_file), dbg["ws_boundary_vis"])

    if show_plots:
        show(vis_all, "All (Num1)")
        show(vis_non, "Non-overlap only (Num2)")
        show(dbg["binary"], "Binary", is_bgr=False)
        show(dbg["ws_boundary_vis"], "Watershed boundary")

    return {
        "num_all": num1,
        "num_nonoverlap": num2,
        "all": str(all_file),
        "nonoverlap": str(nonoverlap_file),
        "binary": str(binary_file),
        "watershed": str(watershed_file),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Count particles with thresholding and watershed splitting.")
    parser.add_argument("--input", default="data/222.jpg", help="Input image path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated images.")
    parser.add_argument("--show", action="store_true", help="Display diagnostic plots after processing.")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional region of interest.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_particle_count(args.input, args.output_dir, args.show, tuple(args.roi) if args.roi else None)
    print(f"Num1 (all): {result['num_all']}")
    print(f"Num2 (non-overlap): {result['num_nonoverlap']}")
    print(f"Saved visualization: {result['all']}")
