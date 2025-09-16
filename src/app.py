import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import torch
from torchvision import transforms as T
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile

mlflow.set_tracking_uri("file:/app/mlruns")
experiment = mlflow.get_experiment_by_name("toxic-plant-classification")

client = MlflowClient()

st.title("🌱 Toxic Plant Classification")

experiments = mlflow.search_experiments()
exp_map = {e.name: e.experiment_id for e in experiments}
exp_name = st.selectbox("Select experiment", list(exp_map.keys()))
exp_id = exp_map[exp_name]

runs_df = mlflow.search_runs([exp_id], order_by=["attributes.start_time DESC"], max_results=20)
if runs_df.empty:
    st.warning("No runs found")
    st.stop()
st.dataframe(runs_df[["run_id","metrics.val_accuracy","params.model","params.epochs", "params.lr"]])

selected = st.multiselect("Select run(s) to compare", runs_df["run_id"].tolist(), max_selections=3)

def to_df(history):
    return pd.DataFrame([(m.step,m.value) for m in history], columns=["step","value"])

if selected:
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    for r in selected:
        for metric, ax in [("train_loss", axs[0]), ("val_loss", axs[0]), ("train_accuracy", axs[1]), ("val_accuracy", axs[1])]:
            hist = client.get_metric_history(r, metric)
            if hist:
                df = to_df(hist)[: -1]
                linestyle = "-" if "train" in metric else "--"
                ax.plot(df["step"], df["value"], linestyle=linestyle, label=f"{r[:6]}-{metric}")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].set_title("Accuracy"); axs[1].legend()
    st.pyplot(fig)

st.header("Inference Demo")
run_id = st.selectbox("Select run for inference", runs_df["run_id"].tolist())
uploaded = st.file_uploader("Upload plant image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224,224))
    st.image(img, caption="Uploaded Image")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(tmp.name)

    st.write("Loading model...")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    transform = T.Compose([T.Resize((224,224)),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])
    arr = transform(img).unsqueeze(0)

    CLASS_NAMES = {0: 'Nontoxic', 1: 'Toxic'}
    
    with torch.no_grad():
        out = model(arr)
        probs = torch.softmax(out, dim=1).numpy()[0]
        pred_class_idx = int(np.argmax(probs))
        pred_class_name = CLASS_NAMES.get(pred_class_idx, "Unknown")
        pred_class_prob = probs[pred_class_idx]
    
    st.write(f"The plant is most likely {pred_class_name} with probability {pred_class_prob: .4f}")