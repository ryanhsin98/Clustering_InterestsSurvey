#!/usr/bin/env python
# coding: utf-8

# In[66]:


#置入所需的城市套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[67]:


# Load in the data
path = 'C:/Users/Ryan Hsin/Desktop/清大/四下/ML/hw4/'
df = pd.read_csv(path+"InterestsSurvey.csv")


# In[68]:


#檢查遺漏值
null = df.isnull().sum()
print('number of missing value')
print(null)


# In[69]:


#Data pre-processing

#遺漏值處理
null = df.isnull().sum()
print('number of missing value')
print(null)

#遺漏值用0填補
df = df.fillna(0)


# In[70]:


#Data pre-processing

#刪除離群值
#有人填太少興趣或太多興趣，難以達到聚類的目的

df['grand_tot_interests'].describe()
#平均是37.31，最小值是1，最大值是104

#找到Q1,Q3
q1, q3 = np.percentile(df['grand_tot_interests'], [25, 75])
print(f"Q1 is: {q1}, Q3 is: {q3}\n")
#Q1=28,Q3=48


# In[71]:


#繪製箱型圖，確認離群值是否存在
df['grand_tot_interests'].plot.box(title="Box Chart")
plt.grid(linestyle="--", alpha=0.3)
plt.show()


# In[72]:


#找到離群值的判斷上下界
above = q3 + 1.5 * (q3 - q1)
below = q1 - 1.5 * (q3 - q1)
print(f"Above is: {above}, Below is: {below}\n")

#above是78,below是-2


# In[73]:


#過濾掉離群值
df = df[~(np.abs(df['grand_tot_interests'] > 78))]
df = df[~(np.abs(df['grand_tot_interests'] < -2))]


# In[74]:


#繪製箱型圖，確認離群值是否刪除成功
df['grand_tot_interests'].plot.box(title="Box Chart")
plt.grid(linestyle="--", alpha=0.3)
plt.show()


# In[75]:


#Data pre-processing

#group成類別變數
one_hot = pd.get_dummies(df['group'])
df = df.drop('group',axis = 1)
# Join the encoded df
df = df.join(one_hot)


# In[76]:


#降維
#捨去low variance的feature
#想要去掉那些超過80%的樣本都取值為0或1的所有特徵，變異數為p(1-p)，伯努力分布.
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selector(data, threshold=(.8 * (1 - .8))):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

df = variance_threshold_selector(df)
df.shape #(6331, 53)


# In[77]:


#降維

# PCA
x = df.drop(['grand_tot_interests'],axis=1)

#判斷PCA要取幾個主成分
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(x)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

#結論要取兩個


# In[78]:


#取兩個PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)


# In[79]:


#繪製兩大主成分背後的原始資料特徵
pcs = np.array(pca.components_) # (n_comp, n_features)
df_pc = pd.DataFrame(pcs, columns=df.columns[1:])
df_pc.index = [f"第{c}主成分" for c in ['一','二']]
df_pc.style    .background_gradient(cmap='bwr_r', axis=None)    .format("{:.2}")


# In[80]:


#繪製兩大主成分為雙軸下的散布圖
plt.scatter(PCA_components[0], PCA_components[1], s=5, alpha=1, color='pink')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


# In[81]:


#聚類

#判斷K-means應要分成幾群的k值
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='pink')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#結論是要分為四群


# In[82]:


#K-means
# 用 KMeans 在資料中找出 4個分組
kmeans = KMeans(n_clusters=4)
kmeans.fit(PCA_components.iloc[:,:3])
# 預測 label
dy = kmeans.predict(PCA_components.iloc[:,:3])
plt.rcParams['font.size'] = 14
plt.figure(figsize=(16, 8))

# 根據分成的 3組來畫出資料
plt.subplot(111)
plt.title('KMeans=4 groups')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.scatter(PCA_components[0], PCA_components[1], c=dy, cmap=plt.cm.Set1)
# 顯示圖表
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




