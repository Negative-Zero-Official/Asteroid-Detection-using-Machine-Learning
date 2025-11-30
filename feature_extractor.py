import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares

# Helpers

def gaussian_psf(size, sigma, center=None):
    n = size
    if center is None:
        cx = cy = (n - 1) / 2.0
    else:
        cx, cy = center
    y, x = np.mgrid[0:n, 0:n]
    g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))
    g /= g.sum() + 1e-12
    return g

def robust_bg_stats_local(img, cx, cy, r_in=8, r_out=11, grids=None):
    if grids is not None:
        y, x = grids
    else:
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
    r_sq = (x - cx)**2 + (y - cy)**2
    mask = (r_sq >= r_in**2) & (r_sq <= r_out**2)
    ring = img[mask]
    if ring.size < 25:
        ring = img.ravel()
    med = np.median(ring)
    mad = np.median(np.abs(ring - med))
    rms = 1.4826 * mad if mad > 0 else np.std(ring)
    return med, rms

def first_second_moments_nonneg(img, grids=None):
    I = img - np.min(img)
    I = np.maximum(I, 0)
    S = I.sum()
    if S <= 0:
        h, w = I.shape
        return ((w-1)/2.0, (h-1)/2.0), (1.0, 1.0, 1.0)
    if grids is not None:
        y, x = grids
    else:
        y, x = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    inv_S = 1.0 / S
    xbar = (I * x).sum() * inv_S
    ybar = (I * y).sum() * inv_S
    dx = x - xbar
    dy = y - ybar
    xx = (I * dx * dx).sum() * inv_S
    yy = (I * dy * dy).sum() * inv_S
    xy = (I * dx * dy).sum() * inv_S
    return (xbar, ybar), (xx, yy, xy)

def aperture_fluxes(img, radii, center, grids=None):
    if grids is not None:
        y, x = grids
    else:
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
    cx, cy = center
    r_sq = (x - cx)**2 + (y - cy)**2
    # Vectorized aperture flux computation
    radii_sq = np.array(radii) ** 2
    sums = np.array([img[r_sq <= r2].sum() for r2 in radii_sq])
    return sums

# Main Extractor for one alert

def extract_alert_features(
    sci, ref, diff,
    fwhm_alert=None,
    cutout_size=None,
    apertures=(1.5, 2.5, 3.5),
    multiscale_sigmas=(1.0, 2.0, 4.0)
):
    feats = {}
    H, W = diff.shape

    # Precompute grids once for all operations
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    grids = (y_grid, x_grid)
    cx0, cy0 = (W - 1) * 0.5, (H - 1) * 0.5
    
    # Background on difference (local)
    bg_mean, bg_rms = robust_bg_stats_local(diff, cx0, cy0, grids=grids)
    feats['diff_bg_mean'] = float(bg_mean)
    feats['diff_bg_rms'] = float(bg_rms)
    
    clipped_diff = np.maximum(diff - bg_mean, 0)
    
    # Pick PSF sigma
    if fwhm_alert is not None and fwhm_alert > 0:
        psf_sigma = fwhm_alert * 0.42466090014400953  # 1/2.355
    else:
        (cx_tmp, cy_temp), (xx, yy, _) = first_second_moments_nonneg(clipped_diff, grids=grids)
        psf_sigma = np.sqrt(np.sqrt(xx * yy)) + 1e-6
    psf_size = int(max(15, 6 * psf_sigma)) | 1
    psf = gaussian_psf(psf_size, psf_sigma)
    
    # Matched filter on diff
    mf = ndi.correlate(diff, psf, mode='constant', cval=bg_mean)
    peak_val = mf.max()
    py, px = np.unravel_index(np.argmax(mf), mf.shape)
    post_sigma = bg_rms * np.sqrt((psf**2).sum())
    mf_snr = peak_val / (post_sigma + 1e-12)
    
    feats.update({
        'mf_peak' : float(peak_val),
        'mf_snr' : float(mf_snr),
        'mf_peak_x' : float(px),
        'mf_peak_y' : float(py)
    })
    
    # Second-moment shape on positive clip of diff
    (cx, cy), (xx, yy, xy) = first_second_moments_nonneg(clipped_diff, grids=grids)
    cov = np.array([[xx + 1e-12, xy], [xy, yy + 1e-12]])
    evals = np.linalg.eigvalsh(cov)
    evals = np.maximum(evals, 1e-12)
    a, b = np.sqrt(evals)
    inv_b = 1.0 / (b + 1e-6)
    elong = (a + 1e-6) * inv_b
    roundness = 1.0 - b / a
    fwhm_mom = 2.3548 * np.sqrt(np.sqrt(xx * yy))
    
    feats.update({
        'centroid_x' : float(cx),
        'centroid_y' : float(cy),
        'm_xx' : float(xx),
        'm_yy' : float(yy),
        'm_xy' : float(xy),
        'fwhm_moment' : float(fwhm_mom),
        'elongation' : float(elong),
        'roundness' : float(roundness)
    })
    
    # Sharpness / Concentration
    ap_flux = aperture_fluxes(clipped_diff, apertures, (cx, cy), grids=grids)
    ap0, ap1, ap2 = ap_flux[0], ap_flux[1], ap_flux[2]
    feats.update({
        'ap_r1' : float(ap0),
        'ap_r2' : float(ap1),
        'ap_r3' : float(ap2),
        'conc_r1_r2' : float((ap0 + 1e-9) / (ap1 + 1e-9)),
        'conc_r2_r3' : float((ap1 + 1e-9) / (ap2 + 1e-9))
    })
    
    # Positive/Negative lobe symmetry (dipole rejection)
    pos_sum = float(np.sum(clipped_diff))
    neg_sum = float(np.sum(np.clip(bg_mean - diff, 0, None)))
    feats.update({
        'pos_sum' : pos_sum,
        'neg_sum' : neg_sum,
        'pos_neg_ratio' : float((pos_sum + 1e-6) / (neg_sum + 1e-6))
    })
    
    # PSF fit amplitude and chi^2 on diff
    inv_2sigma_sq = 1.0 / (2.0 * psf_sigma**2)
    pg = np.exp(-(((x_grid - cx)**2 + (y_grid - cy)**2) * inv_2sigma_sq))
    pg *= 1.0 / (pg.sum() + 1e-12)
    pg = np.nan_to_num(pg, nan=0.0, posinf=0.0, neginf=0.0)
    
    data = diff
    w = 1.0 / (bg_rms + 1e-6)
    
    def resid(params):
        A, C = params
        r = (w * (data - (A * pg + C))).ravel()
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)
    
    # Robust initial guess for A
    A0 = ap2 if np.isfinite(ap2) else peak_val
    res = least_squares(resid, x0=[float(A0), 0.0], max_nfev=200)
    A, C = res.x
    chi2 = float(np.sum(resid(res.x)**2))
    dof = max(1, data.size - 2)
    feats.update({
        'psf_fit_smp' : float(A),
        'psf_fit_bg' : float(C),
        'psf_fit_chi2' : chi2,
        'psf_fit_chi2_dof' : chi2 / dof
    })
    
    # matched-filter flux estimate & error
    num = np.sum((data - C) * pg)
    den = np.sum(pg**2) + 1e-12
    mf_flux = num / den
    mf_flux_err = bg_rms / np.sqrt(den)
    feats.update({
        'mf_flux' : float(mf_flux),
        'mf_flux_err' : float(mf_flux_err),
        'mf_flux_snr' : float(mf_flux / (mf_flux_err + 1e-12))
    })
    
    # Local Crowding
    r_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    mask_outer = r_sq > 20.25  # 4.5^2
    threshold = bg_mean + 5.0 * bg_rms
    feats['crowding_ngt5sigma'] = int(np.sum((diff > threshold) & mask_outer))
    
    # Cross-Image Consistency
    def centroid_and_rflux(img):
        if img is None:
            return (np.nan, np.nan), np.nan
        m, _ = robust_bg_stats_local(img, cx, cy, grids=grids)
        img_clip = np.maximum(img - m, 0)
        (cxx, cyy), _ = first_second_moments_nonneg(img_clip, grids=grids)
        f = aperture_fluxes(img_clip, apertures, (cxx, cyy), grids=grids)[-1]
        return (cxx, cyy), f
    
    (csx, csy), sci_r3 = centroid_and_rflux(sci)
    (ctx, cty), ref_r3 = centroid_and_rflux(ref)
    
    feats.update({
        'science_centroid_dx' : float(csx - cx) if np.isfinite(csx) else np.nan,
        'science_centroid_dy' : float(csy - cy) if np.isfinite(csy) else np.nan,
        'template_centroid_dx' : float(ctx - cx) if np.isfinite(ctx) else np.nan,
        'template_centroid_dy' : float(cty - cy) if np.isfinite(cty) else np.nan,
        'science_flux_r3' : float(sci_r3) if np.isfinite(sci_r3) else np.nan,
        'template_flux_r3' : float(ref_r3) if np.isfinite(ref_r3) else np.nan,
        'science_template_flux_ratio' : float((sci_r3 + 1e-9) / (ref_r3 + 1e-9)) if np.isfinite(sci_r3) and np.isfinite(ref_r3) else np.nan
    })
    
    # Multi-Scale (DoG) energies on diff
    for s in multiscale_sigmas:
        g1 = ndi.gaussian_filter(diff, s)
        g2 = ndi.gaussian_filter(diff, 2.0 * s)
        feats[f'DoG_energy_sigma_{s:.1f}'] = float(np.sum(np.abs(g1 - g2)))
    
    return feats