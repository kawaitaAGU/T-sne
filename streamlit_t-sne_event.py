import streamlit as st
import pickle
import io
import re
import os
import pandas as pd
import numpy as np

st.set_page_config(page_title="pkl再結合", layout="wide")
st.title("📦 8個のpklファイルを結合して保存")

uploaded_files = st.file_uploader(
    "part1.pkl 〜 part8.pkl の全ファイルをアップロードしてください（順不同）",
    type="pkl",
    accept_multiple_files=True
)

def extract_number(file):
    match = re.search(r'\d+', file.name)
    return int(match.group()) if match else -1

if uploaded_files and len(uploaded_files) == 8:
    # 数字順でソート（例：part1, part2, ..., part8）
    sorted_files = sorted(uploaded_files, key=extract_number)

    # 各パートを読み込んで統合
    full_embeddings = []
    full_df = pd.DataFrame()

    for file in sorted_files:
        bytes_data = file.read()
        part = pickle.loads(bytes_data)
        full_embeddings.append(part["embeddings"])
        full_df = pd.concat([full_df, part["df"]], ignore_index=True)

    # 結合
    merged_embeddings = np.vstack(full_embeddings)

    # 保存
    merged_data = {
        "embeddings": merged_embeddings,
        "df": full_df
    }

    with open("embeddings_with_subject.pkl", "wb") as f:
        pickle.dump(merged_data, f)

    st.success("✅ embeddings_with_subject.pkl を生成しました。")
    st.download_button(
        label="📥 結合済みファイルをダウンロード",
        data=pickle.dumps(merged_data),
        file_name="embeddings_with_subject.pkl",
        mime="application/octet-stream"
    )
else:
    st.info("🔄 8個の .pkl ファイルをすべてアップロードしてください。")
