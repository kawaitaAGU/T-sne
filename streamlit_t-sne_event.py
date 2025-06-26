import os
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

st.set_page_config(page_title="t-SNE 可視化（重心付き）", layout="wide")
st.title("🧠 国家試験問題クラスタ可視化（重心付き）")

# ✅ ファイルがなければ 8分割から結合して作成
if not os.path.exists("embeddings_with_subject.pkl"):
    st.warning("❗ embeddings_with_subject.pkl が見つかりません。結合処理を開始します...")
    part_files = [f"embeddings_part_{i}.pkl" for i in range(8)]

    merged_embeddings = []
    merged_df = None

    for i, file in enumerate(part_files):
        if not os.path.exists(file):
            st.error(f"❌ ファイルが見つかりません: {file}")
            st.stop()
        with open(file, "rb") as f:
            part = pickle.load(f)
            merged_embeddings.extend(part["embeddings"])
            if merged_df is None:
                merged_df = part["df"]
            else:
                merged_df = pd.concat([merged_df, part["df"]], ignore_index=True)

    with open("embeddings_with_subject.pkl", "wb") as f:
        pickle.dump({"embeddings": merged_embeddings, "df": merged_df}, f)

    st.success("✅ embeddings_with_subject.pkl を作成しました。")

# ✅ データ読み込み
with open("embeddings_with_subject.pkl", "rb") as f:
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

# ✅ 色分け列の作成
df["色分け"] = df["推定科目"].apply(lambda x: "選択科目" if x == selected_subject else "その他")

# ✅ 重心計算
overall_center = df[["x", "y"]].mean()
subject_center = df[df["推定科目"] == selected_subject][["x", "y"]].mean()

# ✅ 主プロット作成
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="色分け",
    color_discrete_map={"選択科目": "red", "その他": "lightgray"},
    hover_data={
        "設問": True,
        "推定科目": True,
        "x": False,
        "y": False
    },
    title=f"t-SNE 可視化：{selected_subject}（赤）+ 全体重心（黒×）+ 科目重心（青×）",
    width=1000,
    height=800
)

# ✅ 重心マーカー追加
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

# ✅ 表示
st.plotly_chart(fig, use_container_width=True)
st.markdown("🖱️ 各点にマウスをかざすと、その問題文と科目が表示されます。")