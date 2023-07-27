#!/usr/bin/env python
# coding: utf-8

# # CLUSTERING MENGGUNAKAN K-Means

# ### ⛳Objective: 
# Untuk mengkategorikan negara menggunakan faktor sosial ekonomi dan kesehatan yang menentukan pembangunan negara secara keseluruhan.
# 
# ### 📒Tentang Organisasi:
# HELP International adalah LSM kemanusiaan internasional yang berkomitmen untuk memerangi kemiskinan dan menyediakan fasilitas dan bantuan dasar bagi masyarakat di negara-negara terbelakang saat terjadi bencana dan bencana alam.
# 
# ### 🍡Permasalahan:
# HELP International telah berhasil mengumpulkan sekitar $ 10 juta. Saat ini, CEO LSM perlu memutuskan bagaimana menggunakan uang ini secara strategis dan efektif. Jadi, CEO harus mengambil keputusan untuk memilih negara yang paling membutuhkan bantuan. 
# 
# ### 📌Fitur data:
# `Negara` : Nama negara
# 
# `Kematian_anak`: Kematian anak di bawah usia 5 tahun per 1000 kelahiran
# 
# `Ekspor` : Ekspor barang dan jasa perkapita
# 
# `Kesehatan`: Total pengeluaran kesehatan perkapita
# 
# `Impor`: Impor barang dan jasa perkapita
# 
# `Pendapatan`: Penghasilan bersih perorang
# 
# `Inflasi`: Pengukuran tingkat pertumbuhan tahunan dari Total GDP 
# 
# `Harapan_hidup`: Jumlah tahun rata-rata seorang anak yang baru lahir akan hidup jika pola kematian saat ini tetap sama
# 
# `Jumlah_fertiliti`: Jumlah anak yang akan lahir dari setiap wanita jika tingkat kesuburan usia saat ini tetap sama
# 
# `GDPperkapita`: GDP per kapita. Dihitung sebagai Total GDP dibagi dengan total populasi. 

# In[21]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


#input data
df = pd.read_csv('Downloads/Data_Negara_HELP.csv')
df


# ## EDA

# In[23]:


#menghapus nilai kosong
df = df.dropna()


# In[24]:


#melihat statistik deskriptif dari data
df.describe()


# In[190]:


#mendeteksi adanya outlier
figure, axis = plt.subplots(3, 3)
  
axis[0, 0].boxplot(df['Kematian_anak'])
axis[0, 0].set_title("Kematian Anak")

axis[0, 1].boxplot(df['Ekspor'])
axis[0, 1].set_title("Ekspor")

axis[0, 2].boxplot(df['Kesehatan'])
axis[0, 2].set_title("Kesehatan")

axis[1, 0].boxplot(df['Impor'])
axis[1, 0].set_title("Impor")

axis[1, 1].boxplot(df['Pendapatan'])
axis[1, 1].set_title("Pendapatan")

axis[1, 2].boxplot(df['Inflasi'])
axis[1, 2].set_title("Inflasi")

axis[2, 0].boxplot(df['Harapan_hidup'])
axis[2, 0].set_title("Harapan_hidup")

axis[2, 1].boxplot(df['Jumlah_fertiliti'])
axis[2, 1].set_title("Jumlah_fertiliti")

axis[2, 2].boxplot(df['Harapan_hidup'])
axis[2, 2].set_title("Harapan_hidup")
  
plt.show()


# In[191]:


#melakukan penghapusan outlier
percentile25 = df['Pendapatan'].quantile(0.25)
percentile75 = df['Pendapatan'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

df[df['Pendapatan'] > upper_limit]
df[df['Pendapatan'] < lower_limit]

new_df = df[df['Pendapatan'] < upper_limit]
new_df = df[df['Pendapatan'] > lower_limit]
new_df.shape


# In[192]:


#pengecekan apakah outlier masih ada
plt.boxplot(new_df['Pendapatan'])


# In[242]:


#melakukan penghapusan outlier dikarekan masih terdeteksi adanya outlier
percentile25 = new_df['Pendapatan'].quantile(0.25)
percentile75 = new_df['Pendapatan'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Pendapatan'] > upper_limit]
new_df[new_df['Pendapatan'] < lower_limit]

new_df = new_df[new_df['Pendapatan'] < upper_limit]
new_df = new_df[new_df['Pendapatan'] > lower_limit]
new_df.shape


# In[243]:


plt.boxplot(new_df['Pendapatan'])


# In[217]:


percentile25 = new_df['Inflasi'].quantile(0.25)
percentile75 = new_df['Inflasi'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Inflasi'] > upper_limit]
new_df[new_df['Inflasi'] < lower_limit]

new_df = new_df[new_df['Inflasi'] < upper_limit]
new_df = new_df[new_df['Inflasi'] > lower_limit]
new_df.shape


# In[220]:


plt.boxplot(new_df['Inflasi'])


# In[222]:


percentile25 = new_df['Kematian_anak'].quantile(0.25)
percentile75 = new_df['Kematian_anak'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Kematian_anak'] > upper_limit]
new_df[new_df['Kematian_anak'] < lower_limit]

new_df = new_df[new_df['Kematian_anak'] < upper_limit]
new_df = new_df[new_df['Kematian_anak'] > lower_limit]
new_df.shape


# In[223]:


plt.boxplot(new_df['Kematian_anak'])


# In[228]:


percentile25 = new_df['Ekspor'].quantile(0.25)
percentile75 = new_df['Ekspor'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Ekspor'] > upper_limit]
new_df[new_df['Ekspor'] < lower_limit]

new_df = new_df[new_df['Ekspor'] < upper_limit]
new_df = new_df[new_df['Ekspor'] > lower_limit]
new_df.shape


# In[229]:


plt.boxplot(new_df['Ekspor'])


# In[234]:


percentile25 = new_df['Kesehatan'].quantile(0.25)
percentile75 = new_df['Kesehatan'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Kesehatan'] > upper_limit]
new_df[new_df['Kesehatan'] < lower_limit]

new_df = new_df[new_df['Kesehatan'] < upper_limit]
new_df = new_df[new_df['Kesehatan'] > lower_limit]
new_df.shape


# In[235]:


plt.boxplot(new_df['Kesehatan'])


# In[237]:


percentile25 = new_df['Impor'].quantile(0.25)
percentile75 = new_df['Impor'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Impor'] > upper_limit]
new_df[new_df['Impor'] < lower_limit]

new_df = new_df[new_df['Impor'] < upper_limit]
new_df = new_df[new_df['Impor'] > lower_limit]
new_df.shape


# In[238]:


plt.boxplot(new_df['Impor'])


# In[245]:


percentile25 = new_df['Jumlah_fertiliti'].quantile(0.25)
percentile75 = new_df['Jumlah_fertiliti'].quantile(0.75)

iqr = percentile75-percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

new_df[new_df['Jumlah_fertiliti'] > upper_limit]
new_df[new_df['Jumlah_fertiliti'] < lower_limit]

new_df = new_df[new_df['Jumlah_fertiliti'] < upper_limit]
new_df = new_df[new_df['Jumlah_fertiliti'] > lower_limit]
new_df.shape


# In[246]:


plt.boxplot(new_df['Jumlah_fertiliti'])


# In[247]:


#pengecekan akhir untuk melihat keseluruhan data tidak memiliki outlier
figure, axis = plt.subplots(3, 3)
  
axis[0, 0].boxplot(new_df['Kematian_anak'])
axis[0, 0].set_title("Kematian Anak")

axis[0, 1].boxplot(new_df['Ekspor'])
axis[0, 1].set_title("Ekspor")

axis[0, 2].boxplot(new_df['Kesehatan'])
axis[0, 2].set_title("Kesehatan")

axis[1, 0].boxplot(new_df['Impor'])
axis[1, 0].set_title("Impor")

axis[1, 1].boxplot(new_df['Pendapatan'])
axis[1, 1].set_title("Pendapatan")

axis[1, 2].boxplot(new_df['Inflasi'])
axis[1, 2].set_title("Inflasi")

axis[2, 0].boxplot(new_df['Harapan_hidup'])
axis[2, 0].set_title("Harapan_hidup")

axis[2, 1].boxplot(df['Jumlah_fertiliti'])
axis[2, 1].set_title("Jumlah_fertiliti")

axis[2, 2].boxplot(new_df['Harapan_hidup'])
axis[2, 2].set_title("Harapan_hidup")

plt.show()


# In[249]:


new_df


# ## Clustering

# In[296]:


#mengambil kolom angka
dfcluster = new_df.iloc[:, 1:10]
dfcluster.head()


# In[297]:


#melakukan standarisasi agar data cocok/fit
scaled_dfcluster = StandardScaler().fit_transform(dfcluster)
print(scaled_dfcluster[:5])


# In[298]:


#membuat plot WSS (elbow method) untuk menentukan jumlah n cluster yang optimal
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_dfcluster)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[323]:


#berdasarkan plot didapatkan n cluster optimal adalah 4
kmeans = KMeans(init="random", n_clusters=4, n_init=10, random_state=1)
kmeans.fit(scaled_dfcluster)


# In[324]:


#menggabungkan hasil cluster dengan data awal
new_df['cluster'] = kmeans.labels_
new_df


# In[306]:


#mengambil cluster 1
new_df.loc[new_df['cluster'] == 0]


# In[325]:


#mengambil cluster 2
new_df.loc[new_df['cluster'] == 1]


# In[326]:


#mengambil cluster 3
new_df.loc[new_df['cluster'] == 2]


# In[308]:


#mengambil cluster 4
new_df.loc[new_df['cluster'] == 3]


# In[309]:


#mencari banyak cluster 1
len(dfcluster[dfcluster['cluster'] == 0])


# In[310]:


#mencari banyak cluster 2
len(dfcluster[dfcluster['cluster'] == 1])


# In[311]:


#mencari banyak cluster 3
len(dfcluster[dfcluster['cluster'] == 2])


# In[312]:


#mencari banyak cluster 4
len(dfcluster[dfcluster['cluster'] == 3])


# In[319]:


#mencari rata-rata dari setiap kolom pada cluster 1
cluster_1 = new_df.loc[new_df['cluster'] == 0]
cluster_1.mean()


# In[320]:


#mencari rata-rata dari setiap kolom pada cluster 2
cluster_2 = new_df.loc[new_df['cluster'] == 1]
cluster_2.mean()


# In[321]:


#mencari rata-rata dari setiap kolom pada cluster 3
cluster_3 = new_df.loc[new_df['cluster'] == 2]
cluster_3.mean()


# In[327]:


#mencari rata-rata dari setiap kolom pada cluster 4
cluster_4 = new_df.loc[new_df['cluster'] == 3]
cluster_4.mean()


# ### 🌍Kesimpulan
# berdasarkan hasil clustering, negara-negara tersebut dapat dikategorikan menjadi 4 cluster dengan cluster yang paling membutuhkan bantuan adalah cluster 1 yaitu,

# In[329]:


cluster_1['Negara']


# In[ ]:




