import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP

# CSVファイルの読み込み
df = pd.read_csv('outputs/qa2/cka_matrix/cka_matrix.csv', header=None)
cka_matrix_np = df.values[1:, 1:]  # データ部分（最初の行と列を削除）
labels = df.values[1:, 0]          # ラベル（最初の列）

# t-SNEによる次元削減（2次元）
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
transformed_data_tsne = tsne.fit_transform(cka_matrix_np)

# t-SNEの結果をデータフレームに変換
tsne_df = pd.DataFrame(transformed_data_tsne, columns=['Dim 1', 'Dim 2'])
tsne_df['Label'] = labels  # ラベルを追加

# UMAPによる次元削減（2次元）
umap_model = UMAP(n_components=2, random_state=42)
transformed_data_umap = umap_model.fit_transform(cka_matrix_np)

# UMAPの結果をデータフレームに変換
umap_df = pd.DataFrame(transformed_data_umap, columns=['Dim 1', 'Dim 2'])
umap_df['Label'] = labels  # ラベルを追加

# グラフの幅と高さを設定
graph_width = 600
graph_height = 600

# t-SNEのプロットを作成
fig_tsne = go.Figure()
fig_tsne.add_trace(go.Scatter(
    x=tsne_df['Dim 1'],
    y=tsne_df['Dim 2'],
    mode='markers',
    marker=dict(size=10),
    text=tsne_df['Label'],
    hoverinfo='text'
))
fig_tsne.update_layout(
    title='t-SNEによるCKA行列の可視化結果',
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
    width=graph_width,
    height=graph_height,
    margin=dict(l=50, r=50, b=50, t=50)
)

# UMAPのプロットを作成
fig_umap = go.Figure()
fig_umap.add_trace(go.Scatter(
    x=umap_df['Dim 1'],
    y=umap_df['Dim 2'],
    mode='markers',
    marker=dict(size=10),
    text=umap_df['Label'],
    hoverinfo='text'
))
fig_umap.update_layout(
    title='UMAPによるCKA行列の可視化結果',
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
    width=graph_width,
    height=graph_height,
    margin=dict(l=50, r=50, b=50, t=50)
)

# Streamlitで表示
st.title('CKA行列の可視化')

# グラフを縦に並べて表示
st.plotly_chart(fig_tsne, use_container_width=True)
st.plotly_chart(fig_umap, use_container_width=True)
