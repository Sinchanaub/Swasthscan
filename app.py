import cv2, numpy as np, sqlite3, uuid, json, streamlit as st
from core import quality, models, triage as T, notes, compare
st.set_page_config("SwasthScan", "🩺")
db = sqlite3.connect("records.db", check_same_thread=False)
db.execute("create table if not exists r(pid, disease, risk, color, area, ts default current_timestamp)")
st.title("🩺 SwasthScan"); st.caption("Screening aid, not a diagnosis.")
lang = st.sidebar.selectbox("Language", ["en", "hi", "te"])
if not st.sidebar.checkbox("Patient consent obtained"): st.warning("Consent required."); st.stop()
pid = st.sidebar.text_input("Anonymized patient ID", f"P{uuid.uuid4().hex[:6]}")
d = st.radio("Scan type", ["tb", "dr", "oral"], format_func={"tb": "Chest X-ray", "dr": "Retina", "oral": "Mouth"}.get, horizontal=True)
f = st.camera_input("Capture") or st.file_uploader("…or upload", ["jpg", "jpeg", "png"])
QS = {"tb": ["Cough > 2 weeks", "Fever / night sweats", "Weight loss", "TB contact"],
      "dr": ["Diabetes > 10 yrs", "High recent sugar", "Blurred vision"],
      "oral": ["Tobacco / gutka / paan", "Alcohol", "Non-healing ulcer", "Difficulty swallowing"]}
ans = {q: st.select_slider(q, [0, 1, 2], format_func=["No", "Some", "Yes"].__getitem__) for q in QS[d]}
if f and st.button("Analyze"):
    bgr = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), 1); rgb = cv2.cvtColor(bgr, 4)
    ok, why = quality.check(bgr)
    if not ok: st.error(f"Retake photo: {why}"); st.stop()
    m, trained = models.load(d)
    if not trained: st.warning("⚠ DEMO MODE: weights/%s.pt not found – predictions are meaningless until trained." % d)
    x = models.prep(rgb); p, sd = models.predict_mc(m, x); k = int(p.argmax())
    conf = float(p[k] - sd[k]); cam, vis = models.gradcam(m, x, rgb, k)
    ir = T.image_risk(d, p); prev = db.execute("select risk, area from r where pid=? and disease=? order by ts desc", (pid, d)).fetchone()
    color, risk = T.triage(ir, T.questionnaire_risk(ans), conf, prev[0] if prev else None)
    area = float((cam > .5).mean() * 100)
    c1, c2 = st.columns(2); c1.image(rgb, "Scan"); c2.image(vis, "Grad-CAM")
    st.subheader({"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴", "REVIEW": "🟣"}[color] + " " + color)
    st.success(notes.ADVICE[lang][color]); st.text_area("Finding note (editable)", notes.finding_note(d, models.CFG[d][1][k], conf, cam))
    if prev:
        s, dl = compare.change(prev[1], area); st.info(f"Vs last visit: {s} ({dl:+.0f}% area). Previous risk {prev[0]:.2f} → {risk:.2f}")
        hist = [r[0] for r in db.execute("select risk from r where pid=? and disease=? order by ts", (pid, d))] + [risk]; st.line_chart(hist)
    st.caption(notes.DISCLAIMER[lang]); db.execute("insert into r(pid,disease,risk,color,area) values(?,?,?,?,?)", (pid, d, risk, color, area)); db.commit()
