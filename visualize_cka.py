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

    # ファインチューニング前後の改善データの読み込み
    try:
        improvements_before_df = pd.read_csv('improvements_before.csv')
        improvements_after_df = pd.read_csv('improvements_after.csv')
        
        # カラム名が異なる可能性があるため、最初の数値カラムを使用
        improvement_col_before = improvements_before_df.select_dtypes(include=[np.number]).columns[0]
        improvement_col_after = improvements_after_df.select_dtypes(include=[np.number]).columns[0]
        
        improvements_before_dict = dict(zip(improvements_before_df['Model'], improvements_before_df[improvement_col_before]))
        improvements_after_dict = dict(zip(improvements_after_df['Model'], improvements_after_df[improvement_col_after]))
    except Exception as e:
        st.warning(f"改善度データの読み込みに失敗しました: {str(e)}")
        improvements_before_dict = {label: 0 for label in labels}
        improvements_after_dict = {label: 0 for label in labels}

    return cka_matrix_np, labels, improvements_before_dict, improvements_after_dict
cka_matrix_np, labels, improvements_before_dict, improvements_after_dict = load_data()

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

# 改善度をデータフレームに追加（ファインチューニング前）
tsne_df['Improvement_Before'] = tsne_df['Label'].map(improvements_before_dict)
umap_df['Improvement_Before'] = umap_df['Label'].map(improvements_before_dict)

# 改善度をデータフレームに追加（ファインチューニング後）
tsne_df['Improvement_After'] = tsne_df['Label'].map(improvements_after_dict)
umap_df['Improvement_After'] = umap_df['Label'].map(improvements_after_dict)

# 改善度の最大・最小値を取得（全体での最大・最小を計算）
max_improvement = max(tsne_df[['Improvement_Before', 'Improvement_After']].max().max(),
                      umap_df[['Improvement_Before', 'Improvement_After']].max().max())
min_improvement = min(tsne_df[['Improvement_Before', 'Improvement_After']].min().min(),
                      umap_df[['Improvement_Before', 'Improvement_After']].min().min())

# カラーマッピングの関数を定義
def get_color_values(improvements, max_improvement, min_improvement):
    colors = []
    for val in improvements:
        if pd.isna(val):
            colors.append('rgba(255, 255, 255, 1)')  # 白
        else:
            # 値を0-1の範囲に正規化
            normalized = (val - min_improvement) / (max_improvement - min_improvement)
            # jetに似たカラーマップを実装
            if normalized < 0.25:
                r, g, b = 0, 4 * normalized, 1
            elif normalized < 0.5:
                r, g, b = 0, 1, 1 - 4 * (normalized - 0.25)
            elif normalized < 0.75:
                r, g, b = 4 * (normalized - 0.5), 1, 0
            else:
                r, g, b = 1, 1 - 4 * (normalized - 0.75), 0
            colors.append(f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)')
    return colors
# ファインチューニング前の色を計算
tsne_colors_before = get_color_values(tsne_df['Improvement_Before'], max_improvement, min_improvement)
umap_colors_before = get_color_values(umap_df['Improvement_Before'], max_improvement, min_improvement)

# ファインチューニング後の色を計算
tsne_colors_after = get_color_values(tsne_df['Improvement_After'], max_improvement, min_improvement)
umap_colors_after = get_color_values(umap_df['Improvement_After'], max_improvement, min_improvement)

# t-SNEのプロットを作成（ファインチューニング前）
fig_tsne_before = go.Figure()
fig_tsne_before.add_trace(go.Scatter(
    x=tsne_df['Dim 1'],
    y=tsne_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=tsne_df['Improvement_Before'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(
            title='精度',
            thickness=15,
            len=0.7,
        ),
        line=dict(width=1, color='black')
    ),
    text=tsne_df['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
))

# t-SNEのプロットを作成（ファインチューニング後）
fig_tsne_after = go.Figure()
fig_tsne_after.add_trace(go.Scatter(
    x=tsne_df['Dim 1'],
    y=tsne_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=tsne_df['Improvement_After'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(
            title='精度',
            thickness=15,
            len=0.7,
        ),
        line=dict(width=1, color='black')
    ),
    text=tsne_df['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
))

# UMAPのプロットを作成（ファインチューニング前）
fig_umap_before = go.Figure()
fig_umap_before.add_trace(go.Scatter(
    x=umap_df['Dim 1'],
    y=umap_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=umap_df['Improvement_Before'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(
            title='精度',
            thickness=15,
            len=0.7,
        ),
        line=dict(width=1, color='black')
    ),
    text=umap_df['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
))

# UMAPのプロットを作成（ファインチューニング後）
fig_umap_after = go.Figure()
fig_umap_after.add_trace(go.Scatter(
    x=umap_df['Dim 1'],
    y=umap_df['Dim 2'],
    mode='markers',
    marker=dict(
        size=10,
        color=umap_df['Improvement_After'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(
            title='精度',
            thickness=15,
            len=0.7,
        ),
        line=dict(width=1, color='black')
    ),
    text=umap_df['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
))

# Streamlitで表示
st.title('CKA行列の可視化（ファインチューニング前後の比較）')

# t-SNEのグラフを表示（前後）
st.subheader('t-SNE 可視化（ファインチューニング前）')
st.plotly_chart(fig_tsne_before, use_container_width=True)

st.subheader('t-SNE 可視化（ファインチューニング後）')
st.plotly_chart(fig_tsne_after, use_container_width=True)

# UMAPのグラフを表示（前後）
st.subheader('UMAP 可視化（ファインチューニング前）')
st.plotly_chart(fig_umap_before, use_container_width=True)

st.subheader('UMAP 可視化（ファインチューニング後）')
st.plotly_chart(fig_umap_after, use_container_width=True)


# ... existing code ...

# t-SNEによる次元削減（3次元）
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity, max_iter=1000)
transformed_data_tsne_3d = tsne_3d.fit_transform(cka_matrix_np)

# UMAPによる次元削減（3次元）
umap_model_3d = UMAP(n_components=3, random_state=42)
transformed_data_umap_3d = umap_model_3d.fit_transform(cka_matrix_np)

# 3次元データフレームの作成
tsne_df_3d = pd.DataFrame(transformed_data_tsne_3d, columns=['Dim 1', 'Dim 2', 'Dim 3'])
tsne_df_3d['Label'] = labels
tsne_df_3d['Improvement_Before'] = tsne_df['Improvement_Before']
tsne_df_3d['Improvement_After'] = tsne_df['Improvement_After']

umap_df_3d = pd.DataFrame(transformed_data_umap_3d, columns=['Dim 1', 'Dim 2', 'Dim 3'])
umap_df_3d['Label'] = labels
umap_df_3d['Improvement_Before'] = umap_df['Improvement_Before']
umap_df_3d['Improvement_After'] = umap_df['Improvement_After']

# 3D t-SNEプロット（ファインチューニング前）
fig_tsne_3d_before = go.Figure(data=[go.Scatter3d(
    x=tsne_df_3d['Dim 1'],
    y=tsne_df_3d['Dim 2'],
    z=tsne_df_3d['Dim 3'],
    mode='markers',
    marker=dict(
        size=6,
        color=tsne_df_3d['Improvement_Before'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(title='精度'),
        opacity=0.8
    ),
    text=tsne_df_3d['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
)])

fig_tsne_3d_before.update_layout(
    title='3D t-SNE 可視化（ファインチューニング前）',
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    ),
    width=800,
    height=800
)

# 3D t-SNEプロット（ファインチューニング後）
fig_tsne_3d_after = go.Figure(data=[go.Scatter3d(
    x=tsne_df_3d['Dim 1'],
    y=tsne_df_3d['Dim 2'],
    z=tsne_df_3d['Dim 3'],
    mode='markers',
    marker=dict(
        size=6,
        color=tsne_df_3d['Improvement_After'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(title='精度'),
        opacity=0.8
    ),
    text=tsne_df_3d['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
)])

fig_tsne_3d_after.update_layout(
    title='3D t-SNE 可視化（ファインチューニング後）',
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    ),
    width=800,
    height=800
)

# 3D UMAPプロット（ファインチューニング前）
fig_umap_3d_before = go.Figure(data=[go.Scatter3d(
    x=umap_df_3d['Dim 1'],
    y=umap_df_3d['Dim 2'],
    z=umap_df_3d['Dim 3'],
    mode='markers',
    marker=dict(
        size=6,
        color=umap_df_3d['Improvement_Before'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(title='精度'),
        opacity=0.8
    ),
    text=umap_df_3d['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
)])

fig_umap_3d_before.update_layout(
    title='3D UMAP 可視化（ファインチューニング前）',
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    ),
    width=800,
    height=800
)

# 3D UMAPプロット（ファインチューニング後）
fig_umap_3d_after = go.Figure(data=[go.Scatter3d(
    x=umap_df_3d['Dim 1'],
    y=umap_df_3d['Dim 2'],
    z=umap_df_3d['Dim 3'],
    mode='markers',
    marker=dict(
        size=6,
        color=umap_df_3d['Improvement_After'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(title='精度'),
        opacity=0.8
    ),
    text=umap_df_3d['Label'],
    hovertemplate='%{text}<br>精度: %{marker.color:.3f}<extra></extra>'
)])

fig_umap_3d_after.update_layout(
    title='3D UMAP 可視化（ファインチューニング後）',
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    ),
    width=800,
    height=800
)

# Streamlitでの表示（既存の2Dプロットの後に追加）
st.title('3次元可視化')

# 3D t-SNEプロットの表示
st.subheader('3D t-SNE 可視化（ファインチューニング前）')
st.plotly_chart(fig_tsne_3d_before, use_container_width=True)

st.subheader('3D t-SNE 可視化（ファインチューニング後）')
st.plotly_chart(fig_tsne_3d_after, use_container_width=True)

# 3D UMAPプロットの表示
st.subheader('3D UMAP 可視化（ファインチューニング前）')
st.plotly_chart(fig_umap_3d_before, use_container_width=True)

st.subheader('3D UMAP 可視化（ファインチューニング後）')
st.plotly_chart(fig_umap_3d_after, use_container_width=True)

# # 改善度データフレームの作成と表示（ファインチューニング前後）
# st.subheader('モデルごとの精度向上度（ファインチューニング前後）')

# # 精度変化のグラフを作成
# st.subheader('モデルごとの精度変化')

# # 変化率のデータを準備
# fig_changes = go.Figure()

# # Before のデータ点
# fig_changes.add_trace(go.Scatter(
#     x=improvement_display_df['Model'],
#     y=improvement_display_df['Improvement_Before'],
#     mode='lines+markers',
#     name='ファインチューニング前',
#     line=dict(color='blue'),
#     marker=dict(size=8)
# ))

# # After のデータ点
# fig_changes.add_trace(go.Scatter(
#     x=improvement_display_df['Model'],
#     y=improvement_display_df['Improvement_After'],
#     mode='lines+markers',
#     name='ファインチューニング後',
#     line=dict(color='red'),
#     marker=dict(size=8)
# ))

# # グラフのレイアウト設定
# fig_changes.update_layout(
#     xaxis=dict(
#         title='モデル',
#         tickangle=45,
#         showline=True,
#         linecolor='black',
#         mirror=True,
#         showgrid=False
#     ),
#     yaxis=dict(
#         title='精度',
#         showline=True,
#         linecolor='black',
#         mirror=True,
#         showgrid=True,
#         gridcolor='lightgray'
#     ),
#     plot_bgcolor='white',
#     width=800,
#     height=500,
#     margin=dict(l=50, r=50, b=100, t=50),
#     legend=dict(
#         yanchor="top",
#         y=0.99,
#         xanchor="right",
#         x=0.99
#     )
# )

# # グラフを表示
# st.plotly_chart(fig_changes, use_container_width=True)

# # 改善度データフレームの結合
# improvement_display_df = pd.DataFrame({
#     'Model': labels,
#     'Improvement_Before': tsne_df['Improvement_Before'],
#     'Improvement_After': tsne_df['Improvement_After']
# })

# # データフレームを表示
# st.dataframe(improvement_display_df, width=700, height=600)
