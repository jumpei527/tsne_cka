import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
import numpy as np

# サイドバーでパラメータを選択
st.sidebar.title('パラメータ設定')

# t-SNEのPerplexityをスライダーで選択（5から50まで、ステップ5）
perplexity = st.sidebar.slider('t-SNE Perplexity', min_value=5, max_value=50, step=5, value=30)

# t-SNE用のクラスタ数をスライダーで選択（2から10まで）
n_clusters_tsne = st.sidebar.slider('t-SNE クラスタ数 (K-Means)', min_value=2, max_value=10, step=1, value=3)

# UMAP用のクラスタ数をスライダーで選択（2から10まで）
n_clusters_umap = st.sidebar.slider('UMAP クラスタ数 (K-Means)', min_value=2, max_value=10, step=1, value=3)

# CSVファイルの読み込み
@st.cache_data
def load_data():
    # CKAマトリックスの読み込み
    cka_df = pd.read_csv('outputs/qa2/cka_matrix/cka_matrix.csv', header=None)
    cka_matrix_np = cka_df.values[1:, 1:].astype(float)
    labels = cka_df.values[1:, 0]
    labels = [label.replace('.csv', '') for label in labels]

    return cka_matrix_np, labels

cka_matrix_np, labels = load_data()

# t-SNEによる次元削減（2次元）
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
transformed_data_tsne = tsne.fit_transform(cka_matrix_np)

# UMAPによる次元削減（2次元）
umap_model = UMAP(n_components=2, random_state=42)
transformed_data_umap = umap_model.fit_transform(cka_matrix_np)

# t-SNEのクラスタリング
kmeans_tsne = KMeans(n_clusters=n_clusters_tsne, random_state=42)
clusters_tsne = kmeans_tsne.fit_predict(transformed_data_tsne)

# UMAPのクラスタリング
kmeans_umap = KMeans(n_clusters=n_clusters_umap, random_state=42)
clusters_umap = kmeans_umap.fit_predict(transformed_data_umap)

# t-SNEの結果をデータフレームに変換
tsne_df = pd.DataFrame(transformed_data_tsne, columns=['Dim 1', 'Dim 2'])
tsne_df['Label'] = labels  # ラベルを追加
tsne_df['Cluster'] = clusters_tsne  # クラスタラベルを追加

# UMAPの結果をデータフレームに変換
umap_df = pd.DataFrame(transformed_data_umap, columns=['Dim 1', 'Dim 2'])
umap_df['Label'] = labels  # ラベルを追加
umap_df['Cluster'] = clusters_umap  # クラスタラベルを追加

# t-SNEのプロットを作成
fig_tsne = go.Figure()
fig_tsne.add_trace(go.Scatter(
    x=tsne_df['Dim 1'],
    y=tsne_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color='blue',  # 統一色に設定
        line=dict(width=1, color='black')
    ),
    text=tsne_df['Label'],
    hovertemplate='%{text}<extra></extra>'  # 「精度」に関する部分を削除
))

fig_tsne.update_layout(
    title='t-SNE 可視化（クラスタリング結果）',
    xaxis_title='Dim 1',
    yaxis_title='Dim 2',
    width=800,
    height=600
)

# UMAPのプロットを作成
fig_umap = go.Figure()
fig_umap.add_trace(go.Scatter(
    x=umap_df['Dim 1'],
    y=umap_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color='blue',  # 統一色に設定
        line=dict(width=1, color='black')
    ),
    text=umap_df['Label'],
    hovertemplate='%{text}<extra></extra>'  # 「精度」に関する部分を削除
))

fig_umap.update_layout(
    title='UMAP 可視化（クラスタリング結果）',
    xaxis_title='Dim 1',
    yaxis_title='Dim 2',
    width=800,
    height=600
)

# Streamlitで表示
st.title('CKA行列の可視化（クラスタリング結果）')

# t-SNEのグラフを表示
st.subheader('t-SNE 可視化')
st.plotly_chart(fig_tsne, use_container_width=True)

# UMAPのグラフを表示
st.subheader('UMAP 可視化')
st.plotly_chart(fig_umap, use_container_width=True)

# t-SNEによる次元削減（3次元）
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity, max_iter=1000)
transformed_data_tsne_3d = tsne_3d.fit_transform(cka_matrix_np)

# UMAPによる次元削減（3次元）
umap_model_3d = UMAP(n_components=3, random_state=42)
transformed_data_umap_3d = umap_model_3d.fit_transform(cka_matrix_np)

# 3次元データフレームの作成
tsne_df_3d = pd.DataFrame(transformed_data_tsne_3d, columns=['Dim 1', 'Dim 2', 'Dim 3'])
tsne_df_3d['Label'] = labels

umap_df_3d = pd.DataFrame(transformed_data_umap_3d, columns=['Dim 1', 'Dim 2', 'Dim 3'])
umap_df_3d['Label'] = labels

# 3D t-SNEプロット
fig_tsne_3d = go.Figure(data=[go.Scatter3d(
    x=tsne_df_3d['Dim 1'],
    y=tsne_df_3d['Dim 2'],
    z=tsne_df_3d['Dim 3'],
    mode='markers',
    marker=dict(
        size=6,
        color='blue',  # 統一色に設定
        opacity=0.8,
        line=dict(width=1, color='black')
    ),
    text=tsne_df_3d['Label'],
    hovertemplate='%{text}<extra></extra>'  # 「精度」に関する部分を削除
)])

fig_tsne_3d.update_layout(
    title='3D t-SNE 可視化（クラスタリング結果）',
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    ),
    width=800,
    height=800
)

# 3D UMAPプロット
fig_umap_3d = go.Figure(data=[go.Scatter3d(
    x=umap_df_3d['Dim 1'],
    y=umap_df_3d['Dim 2'],
    z=umap_df_3d['Dim 3'],
    mode='markers',
    marker=dict(
        size=6,
        color='blue',  # 統一色に設定
        opacity=0.8,
        line=dict(width=1, color='black')
    ),
    text=umap_df_3d['Label'],
    hovertemplate='%{text}<extra></extra>'  # 「精度」に関する部分を削除
)])

fig_umap_3d.update_layout(
    title='3D UMAP 可視化（クラスタリング結果）',
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    ),
    width=800,
    height=800
)

# Streamlitでの表示
st.title('3次元可視化')

# 3D t-SNEプロットの表示
st.subheader('3D t-SNE 可視化')
st.plotly_chart(fig_tsne_3d, use_container_width=True)

# 3D UMAPプロットの表示
st.subheader('3D UMAP 可視化')
st.plotly_chart(fig_umap_3d, use_container_width=True)
