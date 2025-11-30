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

def robust_bg_stats_local(img, cx, cy, r_in=8, r_out=11):
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    ring = img[(r >= r_in) & (r <= r_out)]
    if ring.size < 25:
        ring = img.ravel()
    med = np.median(ring)
    mad = np.median(np.abs(ring - med))
    rms = 1.4826 * mad if mad > 0 else np.std(ring)
    return med, rms

def first_second_moments_nonneg(img):
    I = img - np.min(img)
    I[I < 0] = 0
    if I.sum() <= 0:
        h,w = I.shape
        return ((w-1)/2.0, (h-1)/2.0), (1.0, 1.0, 1.0)
    y, x = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    S = I.sum()
    xbar = (I * x).sum() / S
    ybar = (I * y).sum() / S
    xx = (I * (x - xbar)**2).sum() / S
    yy = (I * (y - ybar)**2).sum() / S
    xy = (I * (x - xbar)*(y - ybar)).sum() / S
    return (xbar, ybar), (xx, yy, xy)

def aperture_fluxes(img, radii, center):
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    cx, cy, = center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    sums = []
    for rad in radii:
        sums.append(img[r <= rad].sum())
    return np.array(sums)

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

    cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
    
    # Background on difference (local)
    bg_mean, bg_rms = robust_bg_stats_local(diff, cx0, cy0)
    feats['diff_bg_mean'] = float(bg_mean)
    feats['diff_bg_rms'] = float(bg_rms)
    
    clipped_diff = np.clip(diff - bg_mean, 0, None)
    
    # Pick PSF sigma
    if fwhm_alert is not None and fwhm_alert > 0:
        psf_sigma = float(fwhm_alert) / 2.355
    else:
        # quick estimate from second moments of positive clip
        (cx_tmp, cy_temp), (xx, yy, _) = first_second_moments_nonneg(clipped_diff)
        psf_sigma = float(np.sqrt(np.sqrt(xx * yy)) + 1e-6)
    psf_size = int(max(15, 6 * psf_sigma)) | 1 # odd size
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
    (cx, cy), (xx, yy, xy) = first_second_moments_nonneg(clipped_diff)
    # FIX: correct covariance construction
    cov = np.array([[xx, xy], [xy, yy]]) + 1e-12*np.eye(2)
    evals, _ = np.linalg.eigh(cov)
    a, b = np.sqrt(np.maximum(evals, 1e-12)) # sigma major/minor
    elong = (a + 1e-6) / (b + 1e-6)
    roundness = 1.0 - b / a
    fwhm_mom = 2.3548 * float(np.sqrt(np.sqrt(xx * yy)))
    
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
    ap_flux = aperture_fluxes(clipped_diff, apertures, (cx, cy))
    feats.update({
        'ap_r1' : float(ap_flux[0]),
        'ap_r2' : float(ap_flux[1]),
        'ap_r3' : float(ap_flux[2]),
        'conc_r1_r2' : float((ap_flux[0] + 1e-9) / (ap_flux[1] + 1e-9)),
        'conc_r2_r3' : float((ap_flux[1] + 1e-9) / (ap_flux[2] + 1e-9))
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
    # # Fit A * PSF(cx, cy) + C with center fixed at (cx, cy)
    # psf_grid = gaussian_psf(H if H%2==1 else H-1, psf_sigma, center=(cx, cy))
    # Build PSF on the same HxW grid to avoid crop/pad issues
    y, x = np.mgrid[0:H, 0:W]
    # Center at (cy, cx) in (y,x) coordinates
    pg = np.exp(-(((x - cx)**2 + (y - cy)**2) / (2.0 * psf_sigma**2)))
    pg /= pg.sum() + 1e-12
    # Ensure finite
    pg = np.nan_to_num(pg, nan=0.0, posinf=0.0, neginf=0.0)
        
    data = diff
    w = 1.0 / (bg_rms + 1e-6)
    
    def resid(params):
        A, C = params
        r = (w * (data - (A * pg + C))).ravel()
        # Guard against non-finite values
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)
    
    # Robust initial guess for A
    A0 = ap_flux[-1]
    if not np.isfinite(A0):
        # fallback to matched-filter flux estimate at peak
        A0 = peak_val
    C0 = 0.0
    res = least_squares(resid, x0=[float(A0), float(C0)], max_nfev=200)
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
    ring = diff.copy()
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    ring[r <= 4.5] = bg_mean
    k = 5.0
    feats['crowding_ngt5sigma'] = int((ring > (bg_mean + k*bg_rms)).sum())
    
    # Cross-Image Consistency
    def centroid_and_rflux(img):
        if img is None:
            return (np.nan, np.nan), np.nan
        
        m, _ = robust_bg_stats_local(img, cx, cy)
        (cxx, cyy), _ = first_second_moments_nonneg(np.clip(img - m, 0, None))
        f = aperture_fluxes(np.clip(img - m, 0, None), apertures, (cxx, cyy))[-1]
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
        band = g1 - g2
        feats[f'DoG_energy_sigma_{s:.1f}'] = float(np.sum(np.abs(band)))
    
    return feats