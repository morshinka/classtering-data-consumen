import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
import streamlit as st 

# Membaca data dari file CSV
df = pd.read_csv('Mall_Customers.csv')

# Mengganti nama kolom untuk kemudahan
df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)

# Memilih kolom yang relevan
X = df[['Income', 'Score']]

# Judul dan tampilan data pada Streamlit
st.title("Visualisasi KMeans Clustering")
st.write("Data yang digunakan:")
st.dataframe(X.style.highlight_max(axis=0))

# Inisialisasi list untuk menyimpan inertia dari KMeans
cluster = []

# Menghitung inertia untuk berbagai jumlah cluster
for i in range(1,11):
    km = KMeans(n_clusters = i).fit(X)
    cluster.append(km.inertia_)
    
# Membuat plot untuk menemukan elbow
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(x=list(range(1,11)), y=cluster, ax=ax)
ax.set_title('Elbow Method untuk Menentukan Jumlah Cluster yang Optimal')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('Inertia')

# Menambahkan penjelasan pada elbow
elbow_point1 = 3  # Nilai elbow pertama
elbow_value1 = cluster[elbow_point1-1]
ax.annotate('Elbow 1', xy=(elbow_point1, elbow_value1), xytext=(elbow_point1+1, elbow_value1+50),
             arrowprops=dict(facecolor='red', shrink=0.05))

elbow_point2 = 5  # Nilai elbow kedua
elbow_value2 = cluster[elbow_point2-1]
ax.annotate('Elbow 2', xy=(elbow_point2, elbow_value2), xytext=(elbow_point2+1, elbow_value2+50),
             arrowprops=dict(facecolor='blue', shrink=0.05))

# Menampilkan plot pada Streamlit
st.pyplot(fig)

# Sidebar untuk input pengguna
st.sidebar.subheader("Pengaturan Cluster")
clust = st.sidebar.slider("Pilih jumlah cluster:", 1, 10, 3)

# Fungsi untuk melakukan KMeans clustering dan menampilkan hasilnya
def K_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['cluster'] = kmean.labels_
    
    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(data=X, x='Income', y='Score', hue='cluster', style='cluster', sizes=(100, 200), palette=sns.color_palette('hls', n_clust), ax=ax)

    # Menambahkan label pada centroid
    for label in X['cluster'].unique():
        ax.annotate(label, (X[X['cluster'] == label]['Income'].mean(),
            X[X['cluster'] == label]['Score'].mean()),
            horizontalalignment='center',
            verticalalignment='center',
            size=20, weight='bold', color='black')
    
    st.header("Hasil KMeans Clustering")
    st.pyplot(fig)
    st.write("Data dengan label cluster:")
    st.dataframe(X.style.applymap(lambda x: 'background-color : yellow' if x in X['cluster'].unique() else ''))

# Memanggil fungsi KMeans dengan jumlah cluster yang dipilih pengguna
K_means(clust)
