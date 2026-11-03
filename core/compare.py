import cv2, numpy as np
def align(prev_bgr, cur_bgr):
    """ORB feature matching + homography to register old scan onto new."""
    o = cv2.ORB_create(2000); a, b = cv2.cvtColor(prev_bgr, 6), cv2.cvtColor(cur_bgr, 6)
    k1, d1 = o.detectAndCompute(a, None); k2, d2 = o.detectAndCompute(b, None)
    if d1 is None or d2 is None: return cv2.resize(prev_bgr, cur_bgr.shape[1::-1])
    mt = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, True).match(d1, d2), key=lambda m: m.distance)[:100]
    if len(mt) < 10: return cv2.resize(prev_bgr, cur_bgr.shape[1::-1])
    H, _ = cv2.findHomography(np.float32([k1[m.queryIdx].pt for m in mt]), np.float32([k2[m.trainIdx].pt for m in mt]), cv2.RANSAC)
    return cv2.warpPerspective(prev_bgr, H, cur_bgr.shape[1::-1]) if H is not None else cv2.resize(prev_bgr, cur_bgr.shape[1::-1])

def change(area_prev, area_cur):
    d = area_cur - area_prev
    return "Worsening" if d > 3 else "Improving" if d < -3 else "Stable", d
