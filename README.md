# StoreScout
> AI-driven geographic market intelligence and store location recommendation platform for Brazilian quick-commerce and retail expansion.

**Live Demo URL:** [https://storescout.onrender.com](https://storescout-tn9y.onrender.com/) *(Placeholder: update if a custom domain or alternate suffix is assigned upon Render deployment)*

## How It Works
```text
+-----------------------------------------------------------------------------------------+
|                                    HOW STORESCOUT WORKS                                 |
|                                                                                         |
|  [User Inquiry]   --> "sao paulo me electronics ke liye 5 din delivery wale best zones" |
|         |                                                                               |
|         v                                                                               |
|  [ONNX INT8 NLP]  --> Fast in-process intent classification (sub-10ms CPU latency)      |
|         |                                                                               |
|         v                                                                               |
|  [Slot Extraction]--> RapidFuzz entity extraction (City: Sao Paulo, Cat: Telephony, 5d) |
|         |                                                                               |
|         v                                                                               |
|  [Data & K-Means] --> Query precomputed regional market clusters & fulfillment latency  |
|         |                                                                               |
|         v                                                                               |
|  [Response & Map] --> Data-backed recommendation metrics + interactive geospatial map   |
+-----------------------------------------------------------------------------------------+
```

## System Architecture & Memory Optimization (Render Free Tier: 512 MB)
- In-process execution using `onnxruntime` (CPUExecutionProvider) and `tokenizers` exclusively.
- Zero PyTorch and zero Hugging Face Transformers dependencies in production.
- Model weights dynamically quantized to INT8 (~22-64 MB).
- Operational RAM: ~220-260 MB (well below Render's 512 MB ceiling).

## Local Development & Setup
```bash
git clone https://github.com/GitGuru-sudo/storescout.git
cd storescout
pip install -r requirements.txt
streamlit run app.py
```

## Live Deployment on Render Free Tier
1. Connect this GitHub repository as a **Render Blueprint** using `render.yaml`.
2. Add your live Render URL as a GitHub Repository Variable: `APP_URL = "https://storescout.onrender.com"`.
3. The `.github/workflows/keepalive.yml` workflow automatically pings `/_stcore/health` every 10 minutes to prevent cold boot hibernation.

## Limits
1. **Brazil-Only Data:** Trained and calibrated strictly on Brazilian e-commerce transaction logs (Olist dataset 2016-2018).
2. **No Customer Age:** There is no customer age column in the dataset; all intelligence derives strictly from geographic, transactional, product, and delivery attributes.
3. **Estimates, Not Facts:** Recommended clusters, candidate zip codes, and delivery estimations are statistical heuristics to guide retail decisions, not guaranteed financial forecasts.

## License
MIT License - Copyright (c) 2026 GitGuru-sudo
