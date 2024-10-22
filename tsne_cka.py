import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# CSVファイルの読み込み
df = pd.read_csv('outputs/qa2/cka_matrix/cka_matrix.csv', header=None)
cka_matrix_np = df.values[1:, 1:]  # データ部分（最初の行と列を削除）
labels = df.values[1:, 0]  # ラベル（最初の列）

# t-SNEによる次元削減（2次元）
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
transformed_data = tsne.fit_transform(cka_matrix_np)

# t-SNEの結果をデータフレームに変換
tsne_df = pd.DataFrame(transformed_data, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['Label'] = labels  # ラベルを追加

# X軸とY軸の範囲を揃えてバランスを取る
x_min, x_max = tsne_df['t-SNE 1'].min(), tsne_df['t-SNE 1'].max()
y_min, y_max = tsne_df['t-SNE 2'].min(), tsne_df['t-SNE 2'].max()

# 軸範囲を少し広げて、点が枠線に被らないようにする
padding = 0.1 * max(x_max - x_min, y_max - y_min)
range_min = min(x_min, y_min) - padding
range_max = max(x_max, y_max) + padding

# 中心で交差する座標軸を設定するためにPlotly Graph Objectを使用
fig = go.Figure()

# 散布図を作成
fig.add_trace(go.Scatter(
    x=tsne_df['t-SNE 1'],
    y=tsne_df['t-SNE 2'],
    mode='markers',
    marker=dict(size=10),
    text=tsne_df['Label'],  # ホバー時にラベルを表示
    hoverinfo='text'
))

# 座標軸の設定（中央で交差するように調整し、余分な線を除去）
fig.update_layout(
    title='t-SNEのラベル付き可視化結果',
    xaxis=dict(
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,  # グラフの両側に線を引く
        range=[range_min, range_max],  # X軸の範囲を手動で設定
        showgrid=False,  # グリッドを非表示
        automargin=True  # 自動的に余白を調整
    ),
    yaxis=dict(
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,  # グラフの両側に線を引く
        range=[range_min, range_max],  # Y軸の範囲を手動で設定
        showgrid=False,  # グリッドを非表示
        automargin=True  # 自動的に余白を調整
    ),
    plot_bgcolor='white',  # 背景を白に設定
    width=600,  # グラフの幅
    height=600,  # グラフの高さ
    margin=dict(l=50, r=50, b=50, t=50)  # マージンを調整して余白を確保
)

# Streamlitで表示
st.title('t-SNEによるCKA行列の可視化結果')
st.plotly_chart(fig)
