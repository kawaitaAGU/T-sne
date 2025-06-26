import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

st.set_page_config(page_title="国家試験問題クラスタ可視化（重心付き）", layout="wide")
st.title("🧠 国家試験問題クラスタ可視化（重心付き）")

full_file = "embeddings_with_subject.pkl"
part_files = [f"embeddings_with_subject_part{i}.pkl" for i in range(1, 9)]

# ✅ 再結合処理（なければ）
if not os.path.exists(full_file):
    st.warning("❗ embeddings_with_subject.pkl が見つかりません。結合処理を開始します…")

    missing_files = [f for f in part_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"❌ 以下のファイルが見つかりません: {', '.join(missing_files)}")
        st.stop()

    all_embeddings = []
    all_dfs = []

    for file in part_files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            all_embeddings.append(data["embeddings"])
            all_dfs.append(data["df"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    df = pd.concat(all_dfs, ignore_index=True)

    with open(full_file, "wb") as f:
        pickle.dump({"embeddings": embeddings, "df": df}, f)

    st.success("✅ embeddings_with_subject.pkl を結合・保存しました。")

# ✅ 正常に結合 or 既存ファイルからロード
else:
    with open(full_file, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    df = data["df"]

# ✅ t-SNE 実行
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(embeddings)

df["x"] = tsne_result[:, 0]
df["y"] = tsne_result[:, 1]

# ✅ 科目選択
subjects = sorted(df["推定科目"].dropna().unique())
selected_subject = st.selectbox("表示する科目を選んでください：", subjects)

# ✅ 色分けと重心
df["色分け"] = df["推定科目"].apply(lambda x: "選択科目" if x == selected_subject else "その他")
overall_center = df[["x", "y"]].mean()
subject_center = df[df["推定科目"] == selected_subject][["x", "y"]].mean()

# ✅ プロット
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="色分け",
    color_discrete_map={"選択科目": "red", "その他": "lightgray"},
    hover_data={"設問": True, "推定科目": True, "x": False, "y": False},
    title=f"t-SNE 可視化：{selected_subject}（赤）+ 全体重心（黒×）+ 科目重心（青×）",
    width=1000,
    height=800
)

# ✅ 重心を追加
fig.add_trace(go.Scatter(
    x=[overall_center["x"]],
    y=[overall_center["y"]],
    mode='markers',
    marker=dict(symbol='x', size=14, color='black'),
    name='全体の重心'
))

fig.add_trace(go.Scatter(
    x=[subject_center["x"]],
    y=[subject_center["y"]],
    mode='markers',
    marker=dict(symbol='x', size=18, color='blue'),
    name=f'{selected_subject} の重心'
))

st.plotly_chart(fig, use_container_width=True)
st.markdown("🖱️ 各点にマウスをかざすと、その問題文と科目が表示されます。")
