# Deploy Credit Radar to Streamlit Community Cloud

The app serves pre-computed outputs, so deployment installs only `streamlit` and `pandas` and starts in seconds.

## 1. Run it locally first (VS Code terminal)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Check all three tabs: the dashboard (click a locality, a lane, then a user), the Scores explorer, and Method and results.

## 2. Push to GitHub
Create an empty GitHub repo (public is simplest), then from this folder:
```bash
git init
git add .
git commit -m "Credit Radar Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Everything in this repo is synthetic. **Never commit real Paytm data or credentials to a public repo.**

## 3. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub (authorise access to the repo when asked).
2. Click **Create app** / **New app**.
3. Repository: `<your-username>/<repo-name>`, Branch: `main`, Main file path: `app.py`.
4. Optional: choose a subdomain; under **Advanced settings** you can pick the Python version.
5. Click **Deploy**. The first build takes a few minutes. You get a public `*.streamlit.app` link.

Later changes: commit and push to `main`; Community Cloud redeploys automatically.

## Refreshing the model outputs (optional, local only)
```bash
pip install -r requirements-pipeline.txt
bash run_all.sh          # regenerates data/, out/, dashboard/index.html (~1.5 min)
git add -A && git commit -m "Refresh outputs" && git push
```

## Troubleshooting
- **App stuck on "in the oven" or fails to start:** open *Manage app → logs*. Most common causes: wrong main file path (must be `app.py`) or a missing `requirements.txt` at the repo root.
- **Dashboard tab is blank or cut off:** the page is embedded in an iframe at a fixed height (`embed_html(..., height=2200)` in `app.py`). Adjust it if needed.
- **`st.iframe` not found:** old Streamlit; the app falls back to `components.html` automatically.
