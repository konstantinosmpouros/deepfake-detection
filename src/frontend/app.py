"""Streamlit frontend for the deepfake-detection app.

Talks to the FastAPI backend over HTTP and POLLS GET /status (no persistent
connection): pick a pipeline -> backend clears the GPU and warms it -> the UI
polls until it reports `warm` -> upload an image -> predict.

Run (backend must be up):  streamlit run src/frontend/app.py
"""
from __future__ import annotations

import base64
import os
import time

import requests
import streamlit as st

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")
POLL_SECONDS = 1.0

st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️", layout="centered")

BADGE = {"warm": "🟢 warm", "warming": "🟡 warming", "cold": "⚪ cold"}


def api_get(path: str):
    return requests.get(f"{BACKEND}{path}", timeout=10)


def api_post(path: str, **kw):
    return requests.post(f"{BACKEND}{path}", timeout=120, **kw)


def get_status():
    try:
        r = api_get("/status")
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


st.title("🕵️ Deepfake / AI-image detector")
st.caption("Pick a pipeline, wait for it to warm up on the GPU, then upload an image.")

status, err = get_status()
if err:
    st.error(f"Backend not reachable at {BACKEND}.\n\n`{err}`\n\n"
             f"Start it with: `uvicorn src.backend.main:app --port 8000`")
    st.stop()

infos = {p["key"]: p for p in status["pipelines"]}
resident = status["resident"]
busy = status["busy"]

# ---- Sidebar: pipeline selector + live state -------------------------------
with st.sidebar:
    st.header("Pipelines")
    st.caption(f"device: `{status['device']}`")

    selectable = [k for k, p in infos.items() if p["available"]]
    unavailable = [k for k, p in infos.items() if not p["available"]]

    if not selectable:
        st.warning("No trained weights found yet. Train a pipeline (notebooks 04–13) "
                   "so its checkpoint appears under `artifacts/<pipeline>/models/`.")
    else:
        default_idx = selectable.index(resident) if resident in selectable else 0
        choice = st.radio(
            "Select a pipeline to load",
            options=selectable,
            index=default_idx,
            format_func=lambda k: infos[k]["label"],
            disabled=busy,
        )
        if st.button("Load on GPU", disabled=busy, use_container_width=True):
            r = api_post("/select", json={"key": choice})
            if r.ok and r.json().get("accepted"):
                st.toast(f"Warming {infos[choice]['label']}…")
            else:
                detail = (r.json() or {}).get("detail", r.text)
                st.warning(f"Could not select: {detail}")
            time.sleep(0.3)
            st.rerun()

    st.divider()
    st.subheader("State")
    for key, p in infos.items():
        line = f"{BADGE.get(p['state'], p['state'])} — {p['label']}"
        if not p["available"]:
            line += "  · _no weights_"
        st.markdown(line)
    if unavailable:
        st.caption("Pipelines without committed weights can't be loaded yet.")

# ---- Poll while a warm-up is in progress -----------------------------------
if busy or any(p["state"] == "warming" for p in infos.values()):
    warming = next((p["label"] for p in infos.values() if p["state"] == "warming"), "pipeline")
    note = ""
    if any(p["state"] == "warming" and p["downloads_backbone"] for p in infos.values()):
        note = " (first load downloads the frozen backbone — this can take a while)"
    with st.spinner(f"Warming **{warming}** on the GPU…{note}"):
        time.sleep(POLL_SECONDS)
    st.rerun()

# ---- Main panel: upload → image → tabs (Prediction / Explainability) -------
METHOD_BLURB = {
    "grad-cam": "**Grad-CAM** — warmer regions pushed the score toward *fake*.",
    "attention-rollout": "**Attention rollout** — patches the ViT's `[CLS]` token attends to.",
    "patch-attention": "**Per-patch MIL attention** — a greener header marks the patches the pooling weighted most.",
}


def render_prediction(out: dict):
    final = out["final"]
    c1, c2 = st.columns(2)
    c1.metric("Verdict", final["label"].upper())
    c2.metric("p(fake)", f"{final['p_fake']:.3f}")
    st.progress(min(max(final["p_fake"], 0.0), 1.0))
    comps = out.get("components", [])
    if len(comps) > 1:
        st.subheader("Per-component scores")
        st.bar_chart({c["name"]: c["p_fake"] for c in comps})
        st.table([{"component": c["name"], "label": c["label"],
                   "p_fake": round(c["p_fake"], 4)} for c in comps])
    st.caption(f"Saved to `{out['saved_to']}`")


def render_explanation(ex: dict):
    if ex is None:
        st.info("Run a prediction to generate an explanation.")
        return
    if not ex.get("available"):
        st.info(ex.get("reason") or "No spatial explanation is available for this pipeline.")
        return
    png = base64.b64decode(ex["overlay_png_b64"])
    st.image(png, caption=ex.get("method"), use_container_width=True)
    st.markdown(METHOD_BLURB.get(ex.get("method"), ""))


if resident and infos.get(resident, {}).get("state") == "warm":
    st.success(f"**{infos[resident]['label']}** is warm and ready.")
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])

    if upload is not None:
        st.image(upload, caption=upload.name, use_container_width=True)
        img_bytes = upload.getvalue()
        # New image or new pipeline → drop stale results.
        sig = (resident, hash(img_bytes))
        if st.session_state.get("sig") != sig:
            st.session_state["sig"] = sig
            st.session_state.pop("pred", None)
            st.session_state.pop("explain", None)

        if st.button("Predict", type="primary", use_container_width=True):
            files = {"image": (upload.name, img_bytes, upload.type or "image/png")}
            with st.spinner("Running inference + explanation…"):
                rp = api_post("/predict", files=files)
                if not rp.ok:
                    st.error(f"Prediction failed ({rp.status_code}): {rp.json().get('detail', rp.text)}")
                else:
                    st.session_state["pred"] = rp.json()
                    re_ = api_post("/explain", files={"image": (upload.name, img_bytes,
                                                                upload.type or "image/png")})
                    st.session_state["explain"] = re_.json() if re_.ok else {
                        "available": False, "reason": f"explanation failed ({re_.status_code})"}

        if st.session_state.get("pred"):
            tab_pred, tab_explain = st.tabs(["🔎 Prediction", "🧠 Explainability"])
            with tab_pred:
                render_prediction(st.session_state["pred"])
            with tab_explain:
                render_explanation(st.session_state.get("explain"))
else:
    st.info("Select a pipeline in the sidebar and click **Load on GPU** to begin.")
