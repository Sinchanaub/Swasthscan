# SwasthScan – offline-first AI screening (TB · DR · Oral cancer)
Screening aid, not a diagnosis.
```
pip install -r requirements.txt
python train.py tb data/tb      # repeat for dr, oral (ImageFolder: train/ val/)
python export_onnx.py tb        # INT8 ONNX for on-device use
streamlit run app.py
```
Without trained weights the app runs in clearly-flagged DEMO MODE.
Datasets: Shenzhen/Montgomery/TBX11K, APTOS/IDRiD, Kaggle oral sets – verify licenses.
Not built yet: Telugu/Hindi TTS, PDF report, SMS, dashboard, specialist queue, federated slide.
