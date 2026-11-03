import cv2

def check(img_bgr, blur_t=40.0, lo=35, hi=225):
    """Quality gate: blur (Laplacian variance) + brightness. Returns (ok, reason)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(g, cv2.CV_64F).var() < blur_t: return False, "blurry"
    m = g.mean()
    if m < lo: return False, "too_dark"
    if m > hi: return False, "too_bright"
    return True, "ok"
