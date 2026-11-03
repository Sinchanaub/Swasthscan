ADVICE = {
 "en": {"GREEN": "Low risk. Routine check-up.", "AMBER": "Follow-up within 3 months.",
        "RED": "Urgent: visit a hospital now.", "REVIEW": "Needs specialist review."},
 "hi": {"GREEN": "कम जोखिम। नियमित जाँच कराएँ।", "AMBER": "3 महीने के भीतर दोबारा जाँच।",
        "RED": "तुरंत अस्पताल जाएँ।", "REVIEW": "विशेषज्ञ की समीक्षा आवश्यक है।"},
 "te": {"GREEN": "తక్కువ ప్రమాదం. సాధారణ పరీక్ష చేయించుకోండి.", "AMBER": "3 నెలల్లో మళ్లీ పరీక్ష చేయించుకోండి.",
        "RED": "వెంటనే ఆసుపత్రికి వెళ్లండి.", "REVIEW": "నిపుణుల సమీక్ష అవసరం."}}
DISCLAIMER = {"en": "Screening aid, not a diagnosis. Confirm with a doctor.",
              "hi": "यह केवल स्क्रीनिंग सहायता है, निदान नहीं। डॉक्टर से पुष्टि करें।",
              "te": "ఇది స్క్రీనింగ్ సహాయం మాత్రమే, నిర్ధారణ కాదు. వైద్యుడిని సంప్రదించండి."}

def finding_note(d, label, conf, cam):
    """Rule-based template (safe, predictable) filled from heatmap."""
    ys, xs = (cam > .5).nonzero(); area = 100 * len(ys) / cam.size
    if not len(ys): return f"{d.upper()}: {label} ({conf:.0%}). No focal region highlighted."
    h, w = cam.shape; side = "left" if xs.mean() < w / 2 else "right"; part = "upper" if ys.mean() < h / 2 else "lower"
    return f"{d.upper()}: {label} ({conf:.0%}). Region of interest in {part}-{side} of image, ~{area:.0f}% of frame. Approximate, not a clinical measurement."
