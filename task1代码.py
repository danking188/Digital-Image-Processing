import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# -------------------------- Core Configuration --------------------------
img_path = "data/FigP0520.tif"  # Input image path

# Crosshair parameters for PSF estimation
CROSS_LEN = 30    # Crosshair arm length (pixels)
CROSS_W   = 3     # Crosshair arm width (pixels)
ROI_SIZE  = 260   # ROI size for crosshair detection
TM_SCALE  = 5     # Template matching scale
PATCH_HALF = 90   # Half-size of crosshair patch

# PSF sigma search config
SIGMA_SEARCH = (1.0, 8.0)
SIGMA_STEP_COARSE = 0.20
SIGMA_STEP_FINE   = 0.05

# Deconvolution params (Wiener + TV-RL)
WIENER_K = 0.008
RL_ITERS = 48
RL_MAX_RATIO = 4.0
RL_TV_WEIGHT = 0.009
RL_TV_EVERY  = 3
RL_TV_ITERS  = 10

# Frangi vesselness params
FRANGI_SIGMAS_THIN = [0.8, 1.2, 1.7, 2.3]
FRANGI_SIGMAS_ALL  = [0.8, 1.2, 1.7, 2.3, 3.2, 4.2]
FRANGI_BETA = 0.5
FRANGI_C    = 15.0

# -------------------------- Basic Utility Functions --------------------------
def load_grayscale01(path: str) -> np.ndarray:
    """Load grayscale image and normalize to [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 1.0
    img = img / mx if mx > 1.0 else img
    return np.clip(img, 0.0, 1.0)

def to_u8(img01: np.ndarray) -> np.ndarray:
    """Convert [0,1] float to 8-bit uint8"""
    return (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def to_u16(img01: np.ndarray) -> np.ndarray:
    """Convert [0,1] float to 16-bit uint16"""
    return (np.clip(img01, 0, 1) * 65535.0 + 0.5).astype(np.uint16)

def from_u16(u16: np.ndarray) -> np.ndarray:
    """Convert 16-bit uint16 back to [0,1] float"""
    return u16.astype(np.float32) / 65535.0

def disk_kernel(r: int) -> np.ndarray:
    """Generate disk-shaped morphological kernel"""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))

# -------------------------- Crosshair & PSF Estimation --------------------------
def make_cross_template(length=30, width=3, val=255, scale=5) -> np.ndarray:
    """Create scaled crosshair template for matching"""
    L = length * scale
    W = width * scale
    pad = 20 * scale
    sz = L + 2 * pad
    tmpl = np.zeros((sz, sz), np.uint8)
    c = sz // 2
    hl = L // 2
    hw = W // 2
    tmpl[c - hl:c + hl + 1, c - hw:c + hw + 1] = val
    tmpl[c - hw:c + hw + 1, c - hl:c + hl + 1] = val
    return tmpl

def find_crosshair_by_template(img01: np.ndarray, roi_size=260, scale=5):
    """Locate crosshair center using template matching"""
    h, w = img01.shape
    y0 = max(0, h - roi_size)
    x0 = max(0, w - roi_size)

    roi = to_u8(img01[y0:h, x0:w])
    roi_up = cv2.resize(roi, (roi.shape[1]*scale, roi.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    tmpl = make_cross_template(CROSS_LEN, CROSS_W, val=255, scale=scale)

    res = cv2.matchTemplate(roi_up, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, max_loc = cv2.minMaxLoc(res)

    cy_up = max_loc[1] + tmpl.shape[0] / 2.0
    cx_up = max_loc[0] + tmpl.shape[1] / 2.0
    return float(y0 + cy_up/scale), float(x0 + cx_up/scale), float(maxv)

def psf_gaussian(ksize: int, sx: float, sy: float):
    """Generate 2D Gaussian PSF (normalized to sum=1)"""
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1)/2.0
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-0.5 * ((xx/max(sx,1e-6))**2 + (yy/max(sy,1e-6))**2)).astype(np.float32)
    h /= (float(h.sum()) + 1e-12)  # Normalize PSF
    return h

def estimate_sigma_2d(obs_patch01: np.ndarray, width=3, length=30,
                      sigma_min=1.0, sigma_max=8.0,
                      step_coarse=0.2, step_fine=0.05):
    """Estimate 2D Gaussian sigma of PSF (2-stage search)"""
    thr = 0.55 * float(obs_patch01.max())
    ys, xs = np.where(obs_patch01 >= thr)
    cy = float(ys.mean()) if len(ys) else obs_patch01.shape[0]/2.0
    cx = float(xs.mean()) if len(xs) else obs_patch01.shape[1]/2.0
    cyi, cxi = int(round(cy)), int(round(cx))

    win = 81
    half = win // 2
    y0 = max(0, cyi-half); y1 = min(obs_patch01.shape[0], cyi+half+1)
    x0 = max(0, cxi-half); x1 = min(obs_patch01.shape[1], cxi+half+1)
    obs = obs_patch01[y0:y1, x0:x1]
    obs = cv2.copyMakeBorder(
        obs,
        top=max(0, half-cyi), bottom=max(0, (cyi+half)-(obs_patch01.shape[0]-1)),
        left=max(0, half-cxi), right=max(0, (cxi+half)-(obs_patch01.shape[1]-1)),
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    sharp = make_sharp_cross(win, length=length, width=width)

    def search(step, sx0=None, sy0=None, radius=0.6):
        best = None
        if sx0 is None:
            sx_list = np.arange(sigma_min, sigma_max + 1e-12, step)
            sy_list = np.arange(sigma_min, sigma_max + 1e-12, step)
        else:
            sx_list = np.arange(max(sigma_min, sx0-radius), min(sigma_max, sx0+radius) + 1e-12, step)
            sy_list = np.arange(max(sigma_min, sy0-radius), min(sigma_max, sy0+radius) + 1e-12, step)

        for sx in sx_list:
            for sy in sy_list:
                k = int(max(21, (6*max(sx, sy) + 1)))
                if k % 2 == 0: k += 1
                psf = psf_gaussian(k, sx, sy)
                pred = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REPLICATE)
                mse = fit_ab_mse(pred, obs)
                if best is None or mse < best[0]:
                    best = (mse, float(sx), float(sy), int(k))
        return best

    b1 = search(step_coarse)
    b2 = search(step_fine, sx0=b1[1], sy0=b1[2], radius=0.6)
    return b2[1], b2[2], b2[3]

# -------------------------- Deconvolution --------------------------
def wiener_deconv(img01: np.ndarray, psf: np.ndarray, K=0.008, eps=1e-7):
    """Wiener deconvolution (initial guess for RL)"""
    H, W = img01.shape
    kh, kw = psf.shape
    psf_pad = np.zeros((H, W), np.float32)
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, -kh//2, axis=0)
    psf_pad = np.roll(psf_pad, -kw//2, axis=1)

    Hf = np.fft.fft2(psf_pad)
    G  = np.fft.fft2(img01)
    F = np.conj(Hf) * G / (np.abs(Hf)**2 + K + eps)
    f = np.fft.ifft2(F).real.astype(np.float32)
    return np.clip(f, 0, 1)

def richardson_lucy_tv(img01: np.ndarray, psf: np.ndarray, iters=48, clip_ratio=4.0,
                       init=None, tv_weight=0.009, tv_every=3, tv_iters=10, eps=1e-7):
    """RL deconvolution with periodic TV regularization (anti-ringing)"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    psf = np.clip(psf, 0, None).astype(np.float32)
    psf /= (psf.sum() + 1e-12)
    psf_flip = psf[::-1, ::-1].copy()

    est = np.clip(img if init is None else init, eps, 1).astype(np.float32)

    for i in range(iters):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE)
        rel = img / (conv + eps)
        rel = np.clip(rel, 0, clip_ratio)
        est *= cv2.filter2D(rel, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
        est = np.clip(est, 0, 1)

        # Apply TV denoising periodically
        if tv_weight > 0 and (i + 1) % tv_every == 0:
            est = tv_denoise_chambolle(est, weight=tv_weight, n_iter=tv_iters)

    return est

# -------------------------- Vessel Enhancement --------------------------
def frangi_vesselness_2d(img01: np.ndarray, sigmas, beta=0.5, c=15.0, bright_ridges=True):
    """Multi-scale Frangi filter for 2D vesselness detection"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    vessel = np.zeros_like(img, dtype=np.float32)

    for s in sigmas:
        sm = cv2.GaussianBlur(img, (0,0), s)
        Ixx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3) * (s**2)
        Iyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3) * (s**2)
        Ixy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3) * (s**2)

        trace = Ixx + Iyy
        det = Ixx*Iyy - Ixy*Ixy
        tmp = np.sqrt(np.maximum(trace*trace/4.0 - det, 0.0))

        l1 = trace/2.0 - tmp
        l2 = trace/2.0 + tmp

        swap = np.abs(l1) > np.abs(l2)
        l1s = l1.copy(); l2s = l2.copy()
        l1s[swap], l2s[swap] = l2[swap], l1[swap]

        cond = (l2s < 0) if bright_ridges else (l2s > 0)
        l2abs = np.abs(l2s) + 1e-12
        Rb = np.abs(l1s) / l2abs
        S = np.sqrt(l1s*l1s + l2s*l2s)

        V = np.exp(-(Rb*Rb) / (2.0*beta*beta)) * (1.0 - np.exp(-(S*S) / (2.0*c*c)))
        V[~cond] = 0.0
        vessel = np.maximum(vessel, V.astype(np.float32))

    vessel = vessel - vessel.min()
    vessel = vessel / (vessel.max() - vessel.min() + 1e-12)
    return vessel

# -------------------------- Utility Functions --------------------------
def apply_clahe(img01: np.ndarray, clip=1.35, grid=(8,8)):
    """Apply CLAHE for local contrast enhancement"""
    u8 = to_u8(img01)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(u8).astype(np.float32) / 255.0

def gamma_correct(img01: np.ndarray, gamma=1.0):
    """Apply gamma correction"""
    return np.clip(img01, 0, 1) ** gamma

def robust_rescale(img01: np.ndarray, p_lo=0.5, p_hi=99.5):
    """Robust intensity rescaling using percentiles"""
    x = img01.astype(np.float32)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi - lo < 1e-6:
        return np.clip(x, 0, 1)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def extract_patch(img01: np.ndarray, cy: float, cx: float, half: int):
    """Extract square patch centered at (cy, cx)"""
    h, w = img01.shape
    cyi, cxi = int(round(cy)), int(round(cx))
    y0, y1 = max(0, cyi - half), min(h, cyi + half + 1)
    x0, x1 = max(0, cxi - half), min(w, cxi + half + 1)
    return img01[y0:y1, x0:x1]

def make_sharp_cross(size: int, length=30, width=3) -> np.ndarray:
    """Generate ideal sharp crosshair pattern"""
    img = np.zeros((size, size), np.float32)
    c = size // 2
    hl = length // 2
    hw = width // 2
    img[c - hl:c + hl + 1, c - hw:c + hw + 1] = 1.0
    img[c - hw:c + hw + 1, c - hl:c + hl + 1] = 1.0
    return img

def fit_ab_mse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Calculate MSE after linear fitting (a*pred + b)"""
    p = pred.reshape(-1).astype(np.float64)
    y = obs.reshape(-1).astype(np.float64)
    pm, ym = p.mean(), y.mean()
    a = (((p - pm) * (y - ym)).sum()) / (((p - pm) ** 2).sum() + 1e-12)
    b = ym - a * pm
    err = y - (a*p + b)
    return float((err*err).mean())

def edgetaper(img01: np.ndarray, psf: np.ndarray, taper=0.12):
    """Apply edge tapering to reduce deconvolution artifacts"""
    h, w = img01.shape
    a = int(max(1, round(taper * w)))
    b = int(max(1, round(taper * h)))

    def rc(n):
        t = np.linspace(0, np.pi, n, dtype=np.float32)
        return 0.5 * (1 - np.cos(t))

    wx = np.ones(w, np.float32)
    wy = np.ones(h, np.float32)
    wx[:a] = rc(a); wx[-a:] = rc(a)[::-1]
    wy[:b] = rc(b); wy[-b:] = rc(b)[::-1]
    w2 = wy[:, None] * wx[None, :]

    blurred = cv2.filter2D(img01, -1, psf, borderType=cv2.BORDER_REPLICATE)
    return w2 * img01 + (1 - w2) * blurred

def tv_denoise_chambolle(img01: np.ndarray, weight=0.009, n_iter=10):
    """Chambolle's TV denoising (anisotropic)"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    u = img.copy()
    px = np.zeros_like(u); py = np.zeros_like(u)
    tau = 0.125
    w = max(weight, 1e-6)

    for _ in range(n_iter):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        px_new = px + (tau / w) * ux
        py_new = py + (tau / w) * uy
        norm = np.maximum(1.0, np.sqrt(px_new*px_new + py_new*py_new))
        px = px_new / norm
        py = py_new / norm
        divp = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = img + w * divp
    return np.clip(u, 0, 1)

def make_soft_mask(v, gamma=1.9, dilate_r=2):
    """Create soft mask from vesselness map"""
    v = np.clip(v, 0, 1) ** gamma
    if dilate_r > 0:
        k = disk_kernel(dilate_r)
        vd = cv2.dilate(to_u8(v), k)
        v = vd.astype(np.float32) / 255.0
    v = cv2.GaussianBlur(v, (0,0), 1.0)
    return np.clip(v, 0, 1)

def gabor_detail(img01: np.ndarray, thetas=8, sigmas=(1.0, 1.6), gamma=0.45, lambd_factor=3.2):
    """Multi-orientation Gabor filter for detail enhancement"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    resp_max = np.zeros_like(img, np.float32)
    theta_list = [i * np.pi / thetas for i in range(thetas)]

    for s in sigmas:
        ksize = int(max(9, 6*s + 1))
        if ksize % 2 == 0: ksize += 1
        lambd = max(2.0, lambd_factor * s)

        for th in theta_list:
            k = cv2.getGaborKernel((ksize, ksize), sigma=s, theta=th, lambd=lambd,
                                   gamma=gamma, psi=0, ktype=cv2.CV_32F)
            r = cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
            r = np.maximum(r, 0)
            resp_max = np.maximum(resp_max, r)

    resp_max -= resp_max.min()
    resp_max /= (resp_max.max() + 1e-12)
    return resp_max

def bg_flatten_vessel_weighted(img01: np.ndarray, vessel_w: np.ndarray, open_r=25, blend=0.65, posonly=True):
    """Vessel-weighted background flattening"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    w = np.clip(vessel_w, 0, 1).astype(np.float32)

    u = to_u16(img)
    bg_u = cv2.morphologyEx(u, cv2.MORPH_OPEN, disk_kernel(open_r))
    bg = from_u16(bg_u)

    flat = img - bg
    if posonly:
        flat = np.maximum(flat, 0)

    flat_n = robust_rescale(flat, p_lo=1.0, p_hi=99.5)
    alpha = np.clip(blend, 0, 1) * w
    out = (1 - alpha) * img + alpha * flat_n
    return np.clip(out, 0, 1), bg, flat_n

def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Generate line-shaped morphological kernel"""
    if length % 2 == 0:
        length += 1
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    ang = np.deg2rad(angle_deg)
    dx = int(round(np.cos(ang) * c))
    dy = int(round(np.sin(ang) * c))
    p1 = (c - dx, c - dy)
    p2 = (c + dx, c + dy)
    cv2.line(k, p1, p2, 1, 1)
    return k

def multiscale_line_tophat(img01: np.ndarray, lengths=(9,13), angles=(0,45,90,135)):
    """Multi-scale line top-hat filtering"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    u = to_u16(img)
    acc = np.zeros_like(img, np.float32)

    for L in lengths:
        for ang in angles:
            lk = line_kernel(L, ang)
            th_u = cv2.morphologyEx(u, cv2.MORPH_TOPHAT, lk)
            th = from_u16(th_u)
            acc = np.maximum(acc, th)

    acc = robust_rescale(acc, p_lo=0.5, p_hi=99.5)
    return np.clip(acc, 0, 1)

def fix_pepper_and_holes_in_vessels(img01: np.ndarray, vessel_w: np.ndarray,
                                   med_k=3, pepper_thr=0.03,
                                   close_r=1, hole_thr=0.02, blend=1.0):
    """Fix pepper noise/holes in vessel regions"""
    img = np.clip(img01, 0, 1).astype(np.float32)
    w = (np.clip(vessel_w, 0, 1) > 0.35).astype(np.uint8)

    u16 = to_u16(img)
    try:
        med_u = cv2.medianBlur(u16, med_k)
        med = from_u16(med_u)
    except cv2.error:
        med_u8 = cv2.medianBlur(to_u8(img), med_k)
        med = med_u8.astype(np.float32) / 255.0

    pepper = ((med - img) > pepper_thr).astype(np.uint8)
    pepper = (pepper & w).astype(np.uint8)

    close_u = cv2.morphologyEx(u16, cv2.MORPH_CLOSE, disk_kernel(close_r))
    close = from_u16(close_u)
    holes = ((close - img) > hole_thr).astype(np.uint8)
    holes = (holes & w).astype(np.uint8)

    fix_mask = ((pepper | holes) > 0).astype(np.float32)
    fix_mask = cv2.GaussianBlur(fix_mask, (0,0), 0.8)
    fix_mask = np.clip(fix_mask, 0, 1) * np.clip(blend, 0, 1)

    out = img*(1 - fix_mask) + close*fix_mask
    out = np.clip(out, 0, 1)

    return out, pepper.astype(np.float32), holes.astype(np.float32)

def guided_filter(I, p, r=8, eps=6e-4):
    """Edge-preserving guided filter"""
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    ksize = (2*r+1, 2*r+1)

    mean_I  = cv2.boxFilter(I, -1, ksize, borderType=cv2.BORDER_REPLICATE)
    mean_p  = cv2.boxFilter(p, -1, ksize, borderType=cv2.BORDER_REPLICATE)
    mean_Ip = cv2.boxFilter(I*p, -1, ksize, borderType=cv2.BORDER_REPLICATE)
    cov_Ip  = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(I*I, -1, ksize, borderType=cv2.BORDER_REPLICATE)
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a, -1, ksize, borderType=cv2.BORDER_REPLICATE)
    mean_b = cv2.boxFilter(b, -1, ksize, borderType=cv2.BORDER_REPLICATE)

    q = mean_a*I + mean_b
    return np.clip(q, 0, 1)

def sharpen_on_mask(img01: np.ndarray, mask01: np.ndarray, sigma=0.9, amount=0.6, detail_clip=0.06):
    """Edge-aware sharpening restricted to mask regions"""
    base = np.clip(img01, 0, 1).astype(np.float32)
    m = np.clip(mask01, 0, 1).astype(np.float32)

    blur = cv2.GaussianBlur(base, (0,0), sigma)
    detail = np.clip(base - blur, -detail_clip, detail_clip)
    out = base + amount * m * detail
    return np.clip(out, 0, 1)

# -------------------------- Main Pipeline --------------------------
if __name__ == "__main__":
    # Load image
    img = load_grayscale01(img_path)

    # Locate crosshair & estimate PSF
    cy, cx, score = find_crosshair_by_template(img, roi_size=ROI_SIZE, scale=TM_SCALE)
    obs_patch = extract_patch(img, cy, cx, half=PATCH_HALF)
    sx, sy, ksize = estimate_sigma_2d(
        obs_patch, width=CROSS_W, length=CROSS_LEN,
        sigma_min=SIGMA_SEARCH[0], sigma_max=SIGMA_SEARCH[1],
        step_coarse=SIGMA_STEP_COARSE, step_fine=SIGMA_STEP_FINE
    )
    psf = psf_gaussian(ksize, sx, sy)

    # Pre-denoise + edge taper
    den_u8 = cv2.fastNlMeansDenoising(to_u8(img), None, h=3, templateWindowSize=7, searchWindowSize=21)
    den = den_u8.astype(np.float32) / 255.0
    den_taper = edgetaper(den, psf, taper=0.12)

    # Wiener + TV-RL deconvolution
    wien = wiener_deconv(den_taper, psf, K=WIENER_K)
    rest = richardson_lucy_tv(
        den_taper, psf,
        iters=RL_ITERS, clip_ratio=RL_MAX_RATIO, init=wien,
        tv_weight=RL_TV_WEIGHT, tv_every=RL_TV_EVERY, tv_iters=RL_TV_ITERS
    )

    # Vesselness detection + soft mask
    v_thin = frangi_vesselness_2d(rest, FRANGI_SIGMAS_THIN, beta=FRANGI_BETA, c=FRANGI_C, bright_ridges=True)
    v_all  = frangi_vesselness_2d(rest, FRANGI_SIGMAS_ALL,  beta=FRANGI_BETA, c=FRANGI_C, bright_ridges=True)
    v_thin_m = make_soft_mask(v_thin, gamma=1.9, dilate_r=2)
    v_all_m  = make_soft_mask(v_all,  gamma=1.9, dilate_r=2)

    # Background flattening
    stage = rest.copy()
    stage, bg_est, flat_map = bg_flatten_vessel_weighted(
        stage, vessel_w=v_all_m, open_r=25, blend=0.65, posonly=True
    )

    # Gabor detail enhancement
    if True:
        gd = gabor_detail(stage, thetas=8, sigmas=[1.0, 1.6], gamma=0.45, lambd_factor=3.2)
        stage = np.clip(stage + 0.14 * v_thin_m * gd, 0, 1)

    # Line top-hat enhancement
    th_map = multiscale_line_tophat(stage, lengths=[9,13], angles=[0,45,90,135])
    stage = np.clip(stage + 0.22 * v_thin_m * th_map, 0, 1)

    # Fix pepper noise/holes
    stage, pepper_map, hole_map = fix_pepper_and_holes_in_vessels(
        stage, vessel_w=v_all_m,
        med_k=3, pepper_thr=0.030,
        close_r=1, hole_thr=0.020,
        blend=1.0
    )

    # Guided filter background suppression
    if True:
        bg = guided_filter(stage, stage, r=8, eps=6e-4)
        bg_u8 = cv2.fastNlMeansDenoising(to_u8(bg), None, h=4, templateWindowSize=7, searchWindowSize=21)
        bg = bg_u8.astype(np.float32) / 255.0
        preserve = np.clip(v_all_m, 0, 1) ** 2.4
        stage = preserve * stage + (1 - preserve) * bg
        stage = np.clip(stage, 0, 1)

    # Edge-aware sharpening
    thin_bin = (v_thin_m > 0.35).astype(np.uint8) * 255
    thin_d = cv2.dilate(thin_bin, disk_kernel(1), iterations=1)
    thin_e = cv2.erode(thin_bin, disk_kernel(1), iterations=1)
    thin_edge_band = ((thin_d > 0) & (thin_e == 0)).astype(np.float32)
    thin_edge_band = cv2.GaussianBlur(thin_edge_band, (0,0), 0.8)
    thin_edge_band = np.clip(thin_edge_band, 0, 1)
    stage = sharpen_on_mask(stage, thin_edge_band, sigma=0.9, amount=0.60, detail_clip=0.06)

    # Final tone adjustment
    out = robust_rescale(stage, p_lo=0.5, p_hi=99.5)
    if True:
        out = apply_clahe(out, clip=1.35, grid=(8,8))
    out = gamma_correct(out, 0.95)

    # Save results
    cv2.imwrite("FigP0520_clear_vessel_fix3.png", to_u8(out))
    cv2.imwrite("FigP0520_rest_tvrl.png", to_u8(rest))
    cv2.imwrite("FigP0520_psf.png", to_u8(psf / (psf.max() + 1e-12)))

    # Visualize steps
    plt.figure(figsize=(22,10))
    plt.subplot(2,6,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(2,6,2); plt.imshow(rest, cmap="gray"); plt.title("TV-RL"); plt.axis("off")
    plt.subplot(2,6,3); plt.imshow(flat_map, cmap="gray"); plt.title("Flatten map"); plt.axis("off")
    plt.subplot(2,6,4); plt.imshow(th_map, cmap="gray"); plt.title("Line Top-hat"); plt.axis("off")
    plt.subplot(2,6,5); plt.imshow(out, cmap="gray"); plt.title("Final"); plt.axis("off")  
    plt.subplot(2,6,6); plt.imshow(psf, cmap="gray"); plt.title(f"PSF sx={sx:.2f}, sy={sy:.2f}, k={ksize}"); plt.axis("off")  
    plt.tight_layout(); plt.show()