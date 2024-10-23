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
    df = pd.read_csv('outputs/qa2/cka_matrix/cka_matrix.csv', header=None)
    cka_matrix_np = df.values[1:, 1:].astype(float)  # データ部分（最初の行と列を削除）
    labels = df.values[1:, 0]                      # ラベル（最初の列）
    labels = [label.replace('.csv', '') for label in labels]  # .csvを除外
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

# カラーマップの作成 (手動で色を定義)
color_map = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
cluster_colors_tsne = [color_map[i % len(color_map)] for i in clusters_tsne]
cluster_colors_umap = [color_map[i % len(color_map)] for i in clusters_umap]

# t-SNEのプロットを作成
fig_tsne = go.Figure()
fig_tsne.add_trace(go.Scatter(
    x=tsne_df['Dim 1'],
    y=tsne_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=cluster_colors_tsne,  # クラスタラベルで色分け
        showscale=False
    ),
    text=tsne_df['Label'],
    hoverinfo='text'
))
fig_tsne.update_layout(
    title='t-SNEによるクラスタリング付き可視化結果',
    xaxis=dict(
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,
        showgrid=False,
        automargin=True
    ),
    yaxis=dict(
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,
        showgrid=False,
        automargin=True
    ),
    plot_bgcolor='white',
    width=600,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50)
)

# UMAPのプロットを作成
fig_umap = go.Figure()
fig_umap.add_trace(go.Scatter(
    x=umap_df['Dim 1'],
    y=umap_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=cluster_colors_umap,  # クラスタラベルで色分け
        showscale=False
    ),
    text=umap_df['Label'],
    hoverinfo='text'
))
fig_umap.update_layout(
    title='UMAPによるクラスタリング付き可視化結果',
    xaxis=dict(
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,
        showgrid=False,
        automargin=True
    ),
    yaxis=dict(
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,
        showgrid=False,
        automargin=True
    ),
    plot_bgcolor='white',
    width=600,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50)
)

# Streamlitで表示
st.title('CKA行列の可視化')

# t-SNEのグラフを表示
st.subheader('t-SNE 可視化')
st.plotly_chart(fig_tsne, use_container_width=True)

# t-SNEクラスタごとのモデル名を表示
st.subheader('t-SNE クラスタごとのモデル名')
for cluster_num in range(n_clusters_tsne):
    st.markdown(f'<span style="color:{color_map[cluster_num % len(color_map)]};">**クラスタ {cluster_num + 1}:**</span>', unsafe_allow_html=True)
    for label in tsne_df[tsne_df['Cluster'] == cluster_num]['Label']:
        st.markdown(f'<span style="color:{color_map[cluster_num % len(color_map)]};">{label}</span>', unsafe_allow_html=True)

# UMAPのグラフを表示
st.subheader('UMAP 可視化')
st.plotly_chart(fig_umap, use_container_width=True)

# UMAPクラスタごとのモデル名を表示
st.subheader('UMAP クラスタごとのモデル名')
for cluster_num in range(n_clusters_umap):
    st.markdown(f'<span style="color:{color_map[cluster_num % len(color_map)]};">**クラスタ {cluster_num + 1}:**</span>', unsafe_allow_html=True)
    for label in umap_df[umap_df['Cluster'] == cluster_num]['Label']:
        st.markdown(f'<span style="color:{color_map[cluster_num % len(color_map)]};">{label}</span>', unsafe_allow_html=True)
