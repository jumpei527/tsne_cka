import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

input_texts = 'qa1'

# CSVファイルの読み込み
df = pd.read_csv(f'outputs/{input_texts}/cka_matrix/cka_matrix.csv', header=None)
cka_matrix_np = df.values[1:, 1:]  # データ部分（最初の行と列を削除）
labels = df.values[1:, 0]  # ラベル（最初の列）

# t-SNEによる次元削減（2次元）
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
transformed_data = tsne.fit_transform(cka_matrix_np)

# t-SNEの結果をデータフレームに変換
tsne_df = pd.DataFrame(transformed_data, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['Label'] = labels  # ラベルを追加

# Plotlyを使ってインタラクティブなプロットを作成
fig = px.scatter(tsne_df, x='t-SNE 1', y='t-SNE 2', hover_name='Label',
                 title='t-SNE with Interactive Labels')

# Streamlitで表示
st.title('t-SNE Visualization with Labels')
st.plotly_chart(fig)
