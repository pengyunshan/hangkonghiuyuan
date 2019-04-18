# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:30:54 2017

@author: Frank
"""

import pandas as pd
import os
os.chdir('xxxx')

def reducedim(data,dim = 2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = dim)
    new_data = pca.fit_transform(data)
    return new_data
    
def kmeans_cluster(data,n):
    from sklearn.cluster import KMeans
    pred_y = KMeans(n_clusters = n).fit_predict(data)
    return pred_y

def test_scale_pca(unscale_data,scale_data):
   
    #不经过降维的标准化数据与未标准化数据
    import time
    time0 = time.time()
    y_hat_unscale = kmeans_cluster(unscale_data,3)
    time1 = time.time()
    y_hat_scale = kmeans_cluster(scale_data,3)
    time2 = time.time()
   
    #经过降维的标准化数据与未标准化数据
    unscale_date_pca = reducedim(unscale_data)
    scale_data_pca = reducedim(scale_data)
    
    time3 = time.time()
    y_uns_pca = kmeans_cluster(unscale_date_pca,3)
    time4 = time.time()
    y_sca_pca = kmeans_cluster(scale_data_pca,3)
    time5 = time.time()
    
    del1 = time1 - time0
    del2 = time2 - time1
    del3 = time4 - time3
    del4 = time5 - time4
    
    print(del1,del2,del3,del4)
    #4.8437418937683105 3.8906936645507812 2.7812838554382324 1.9531242847442627
    x0 = (y_hat_unscale == y_hat_scale).sum()/len(y_hat_scale)
    x1 = (y_uns_pca == y_sca_pca).sum()/len(y_hat_scale)
    x2 = (y_hat_unscale == y_uns_pca).sum()/len(y_hat_scale)
    x3 = (y_hat_scale == y_sca_pca).sum()/len(y_hat_scale)
    print(x0,x1,x2,x3)
    #0.330634406554 0.335873499714 0.835921127834 0.331983869943
    #使用降维并且标准化的之后的数据效果变得灰常差，所以最后采用方差筛选后未进行标准化的数据

    return y_hat_unscale

def hiera_clustering(data): 
    from sklearn.cluster import AgglomerativeClustering
    agg_cluster = AgglomerativeClustering()
    y_hat = agg_cluster.fit_predict(data)
    return y_hat

def dbscan_clustering(data):
	from sklearn.cluster import DBSCAN
	y_hat = DBSCAN(eps = 0.3,min_samples=5).fit_predict(data)
	return y_hat

def gauss_mix(data):
    from sklearn import mixture
    gaus = mixture.GaussianMixture(n_components = 3, covariance_type='full').fit(data)
    y_hat = gaus.predict(data)
    return y_hat


if __name__ == "__main__":
#	unscale_data = pd.read_csv('num_filted_unscale.csv')
#	scale_data = pd.read_csv('num_filter_scale.csv')
	# y_hat = test_scale_pca(unscale_data,scale_data)
#	new_data = reducedim(unscale_data)
#	y_hat2 = dbscan_clustering(new_data)
#	y_hat3 = gauss_mix(new_data)
    
#对用户进行聚类分析，只采用部分数据,这些数据通过人工挑选

    cluster_data = data[['FLIGHT_COUNT','BASE_POINTS_SUM','ELITE_POINTS_SUM_YR_2','EXPENSE_SUM_YR_1',
    'EXPENSE_SUM_YR_2','SEG_KM_SUM']]

    from sklearn.preprocessing import Imputer
    imp = Imputer(strategy='most_frequent',axis = 0)
    cluster_data = imp.fit_transform(cluster_data)

    y_hat_gaus = gauss_mix(cluster_data)
    
    y_gau_des = [False if i == 0 else True for i in y_hat_gaus]
    y_gau_data = data[y_gau_des]
    y_gau_des = y_gau_data[['FLIGHT_COUNT','BASE_POINTS_SUM','ELITE_POINTS_SUM_YR_2','EXPENSE_SUM_YR_1',
    'EXPENSE_SUM_YR_2','SEG_KM_SUM']].describe()
    y_gau_des.loc['mean',:]
    
    
    y_hat = kmeans_cluster(cluster_data ,  n=2)
    
    y_hat_db = dbscan_clustering(cluster_data)
    y_hat_db = pd.DataFrame(y_hat_db)
    y_hat_db_info = [ False if i == -1 else True for i in y_hat_db[0]]
    
    des_vip = data[ y_hat_db_info]
    des_vip_des = des_vip[['FLIGHT_COUNT','BASE_POINTS_SUM','ELITE_POINTS_SUM_YR_2','EXPENSE_SUM_YR_1',
    'EXPENSE_SUM_YR_2','SEG_KM_SUM']].describe()
    des_vip_des.loc['mean',:]
    
    x = data[y_hat.astype(bool)]
    xx = x['BASE_POINTS_SUM']
    des_vip = x[['FLIGHT_COUNT','BASE_POINTS_SUM','ELITE_POINTS_SUM_YR_2','EXPENSE_SUM_YR_1',
    'EXPENSE_SUM_YR_2','SEG_KM_SUM']].describe()
    des_vip.loc['mean',:]
