import numpy as np
def image_risk(d, p):
    if d == "tb": return float(p[1])
    if d == "dr": return float((p * np.array([0, .25, .5, .75, 1])).sum())
    return float(p[1] * .5 + p[2])

def questionnaire_risk(answers):  # answers: {question: 0/1/2 severity}
    return min(1.0, sum(answers.values()) / (2 * max(1, len(answers))) * 1.5)

def triage(img_r, q_r, conf, prev_risk=None, min_conf=0.6):
    """Fusion (weighted baseline; swap in LogReg/XGBoost once you have labelled data)."""
    if conf < min_conf: return "REVIEW", img_r
    r = .7 * img_r + .3 * q_r
    if prev_risk is not None and r - prev_risk > .15: r = max(r, .65)  # auto-upgrade on worsening
    return ("RED" if r >= .65 else "AMBER" if r >= .35 else "GREEN"), r
