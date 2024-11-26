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
    cka_matrix_np = cka_df.values[1:, 1:].astype(float)  # データ部分（最初の行と列を削除）
    labels = cka_df.values[1:, 0]                      # ラベル（最初の列）
    labels = [label.replace('.csv', '') for label in labels]  # .csvを除外

    # 改善データの読み込み
    improvements_df = pd.read_csv('improvements.csv')
    improvements_dict = dict(zip(improvements_df['Model'], improvements_df['Improvement']))

    return cka_matrix_np, labels, improvements_dict

cka_matrix_np, labels, improvements_dict = load_data()

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

# 改善度をデータフレームに追加
tsne_df['Improvement'] = tsne_df['Label'].map(improvements_dict)
umap_df['Improvement'] = umap_df['Label'].map(improvements_dict)

# カラーマッピングの準備
# 正の改善度は青、負の改善度は赤、データがない場合は白
# PlotlyのRdBu_rカラースケールを使用（反転して青が正、赤が負に）
color_scale = 'RdBu_r'

# カラーバリューを定義（NaNは0に設定して白に近づける）
# 適切な範囲を設定するために、改善度の最大絶対値を取得
max_improvement = max(tsne_df['Improvement'].max(), umap_df['Improvement'].max())
min_improvement = min(tsne_df['Improvement'].min(), umap_df['Improvement'].min())

# PlotlyではNaNは色で表現できないため、白色にするために改善度がNaNの場合は0に設定
tsne_df['Improvement_plot'] = tsne_df['Improvement'].fillna(0)
umap_df['Improvement_plot'] = umap_df['Improvement'].fillna(0)

# マーカーの色を改善度に基づいて設定
tsne_df['Color'] = tsne_df['Improvement_plot']
umap_df['Color'] = umap_df['Improvement_plot']

# カラーバーの設定
colorbar = dict(
    title="Improvement",
    tickmode='linear',
    tick0=min_improvement,
    dtick=(max_improvement - min_improvement) / 10
)

# t-SNEのプロットを作成
fig_tsne = go.Figure()
fig_tsne.add_trace(go.Scatter(
    x=tsne_df['Dim 1'],
    y=tsne_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=tsne_df['Color'],
        colorscale=color_scale,
        colorbar=colorbar,
        cmin=min_improvement,
        cmax=max_improvement,
        showscale=True,
        line=dict(width=1, color='black')  # マーカーの枠線を追加
    ),
    text=tsne_df['Label'] + f" (Improvement: {tsne_df['Improvement']})",
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
    width=800,
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
        color=umap_df['Color'],
        colorscale=color_scale,
        colorbar=colorbar,
        cmin=min_improvement,
        cmax=max_improvement,
        showscale=True,
        line=dict(width=1, color='black')  # マーカーの枠線を追加
    ),
    text=umap_df['Label'] + f" (Improvement: {umap_df['Improvement']})",
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
    width=800,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50)
)

# Streamlitで表示
st.title('CKA行列の可視化')

# t-SNEのグラフを表示
st.subheader('t-SNE 可視化')
st.plotly_chart(fig_tsne, use_container_width=True)

# UMAPのグラフを表示
st.subheader('UMAP 可視化')
st.plotly_chart(fig_umap, use_container_width=True)

# クラスタごとのモデル名と改善度を表示
st.subheader('モデルごとの精度向上度')

# モデルごとの精度向上度を表形式で表示
st.write("以下は各モデルの精度向上度です。正の値は向上、負の値は低下を示します。")

# 改善度データフレームの作成
improvement_display_df = improvements_df.copy()
improvement_display_df = improvement_display_df.sort_values(by='Improvement', ascending=False)

# スタイルを適用して色付け
def color_improvement(val):
    if pd.isna(val):
        color = 'white'
    elif val > 0:
        color = f'rgba(0, 0, 255, {min(val / max_improvement, 1)})'  # 青色の透明度
    elif val < 0:
        color = f'rgba(255, 0, 0, {min(-val / abs(min_improvement), 1)})'  # 赤色の透明度
    else:
        color = 'white'
    return f'background-color: {color}'

styled_df = improvement_display_df.style.applymap(color_improvement, subset=['Improvement'])
st.dataframe(styled_df, width=700, height=600)
