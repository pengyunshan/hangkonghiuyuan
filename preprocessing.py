# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:43:54 2017

@author: Frank
"""
import os
os.chdir("F:/研究生/我爱数据处理/统计软件考试  2017年秋/")
import pandas as pd
import numpy as np

def load_data(path=None):
    """加载数据并生成标签
    @param 数据地址
    @return data
    """
    data = pd.read_excel(path,sheetname = 1,header = 0,
                     index_col = 0,parse_dates = True)
    #生成标签
    data['LABEL'] = data['Ration_L1Y_BPS'].map(
            lambda x:0 if x < 0.2 else 1 if (x >= 0.2)&(x < 0.5) else 2)
    #删除无效数据
    data = data.drop(['WORK_CITY','WORK_PROVINCE','WORK_COUNTRY','Ration_P1Y_BPS','Ration_L1Y_BPS'],axis = 1)
    #设置索引
    data= data.set_index(np.arange(len(data)))
    #分割数据为分类、数值、时间类型
    time_col = ['FFP_DATE','FIRST_FLIGHT_DATE','LOAD_TIME','LAST_FLIGHT_DATE','TRANSACTION_DATE']
    cat_col = ['GENDER','age','FFP_TIER', ]
    label_col = ['LABEL']
    
    data_cat = data[cat_col]
    data_time = data[time_col] 
    data_num = data.drop(time_col + cat_col + label_col,axis = 1)
    label = data[label_col]
    return data_num,data_cat,data_time,label

def gender_encoder_imputer(data):
    """对性别进行labelencoder和onehotencoder
    @param data
    @return none
    """ 
    gender = data['GENDER'].apply(lambda x:{'男':0,'女':1,np.nan:0}[x])
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder()
    gender = one_hot_encoder.fit_transform(gender.values.reshape(-1,1))
    gender = pd.DataFrame(gender.toarray(),columns = ['M','F'])
    return gender

def explory_anlysis(data):
    """ 对数据列之间是否存在加和关系
    @param data
    @return none
    """
    temp = data[['FLIGHT_COUNT_QTR_1', 'FLIGHT_COUNT_QTR_2', 'FLIGHT_COUNT_QTR_3',
       'FLIGHT_COUNT_QTR_4', 'FLIGHT_COUNT_QTR_5', 'FLIGHT_COUNT_QTR_6',
       'FLIGHT_COUNT_QTR_7', 'FLIGHT_COUNT_QTR_8']]

    temp.sum(axis = 1)==data['FLIGHT_COUNT']
    #返回全是true，
    temp = data[['BASE_POINTS_SUM_QTR_1', 'BASE_POINTS_SUM_QTR_2',
       'BASE_POINTS_SUM_QTR_3', 'BASE_POINTS_SUM_QTR_4',
       'BASE_POINTS_SUM_QTR_5', 'BASE_POINTS_SUM_QTR_6',
       'BASE_POINTS_SUM_QTR_7', 'BASE_POINTS_SUM_QTR_8']]
    temp.sum(axis = 1) == data['BASE_POINTS_SUM']
    #返回全是true
      
def time_one_hot(data):
    """对类别数据进行onehot编码，对年龄数据分组onehot，提取时间序列的年/月,输出到io
    @param data
    @return none
    """
    #提取时间数据的年和月 
    FFP_DATE_YEAR = data['FFP_DATE'].apply(lambda x: x.year)
    FFP_DATE_MONTH = data['FFP_DATE'].apply(lambda x: x.month)
    
    FIRST_FLIGHT_DATE_YEAR = data['FIRST_FLIGHT_DATE'].apply(lambda x:x.year)
    FIRST_FLIGHT_DATE_MONTH = data['FIRST_FLIGHT_DATE'].apply(lambda x:x.month)
    
    LAST_FLIGHT_DATE_YEAR = data['LAST_FLIGHT_DATE'].apply(lambda x:x.year)
    LAST_FLIGHT_DATE_MONTH = data['LAST_FLIGHT_DATE'].apply(lambda x:x.month)
    
    data_time = pd.concat([LAST_FLIGHT_DATE_YEAR,LAST_FLIGHT_DATE_MONTH,
              FIRST_FLIGHT_DATE_YEAR,FIRST_FLIGHT_DATE_MONTH,FFP_DATE_YEAR,FFP_DATE_MONTH] ,axis = 1)
#    data_time.to_csv('data_time.csv')
    return data_time

def age_encoder(data):
    #缺失值用众数填充
    from sklearn.preprocessing import Imputer
    imp = Imputer(strategy='most_frequent',axis = 0)
    #离散化
    age = imp.fit_transform(data['age'].values.reshape(-1,1))
    age = pd.cut(age.flatten(),bins = [0,14,22,35,45,55,110])    
    age = age.astype(str)
    #onehotencoder
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    label_encoder = LabelEncoder()
    age = label_encoder.fit_transform(age)
    one_hot = OneHotEncoder()
    age = one_hot.fit_transform(age.reshape(-1,1))
    age = age.toarray()
    age = pd.DataFrame(age,columns = ['age'+str(i) for i in range(6)])
    
    #对级别数据onehot
    tier_cat = one_hot.fit_transform(data['FFP_TIER'].values.reshape(-1,1))
    tier_cat = pd.DataFrame(tier_cat.toarray(),columns = ['FFP_TIER'+str(i) for i in range(3)])
  
    return age,tier_cat

def reductdim(data):
    """对数据降维,利用方差选择法筛选数据，用最大最小法进行筛选
    @param
    @return
    """
    #缺省值填充
    from sklearn.preprocessing import Imputer
    imp = Imputer(strategy='most_frequent',axis = 0)
    filled_data = imp.fit_transform(data)
    #方差选择法
    from sklearn.feature_selection import VarianceThreshold
    variance_threshold = VarianceThreshold(threshold = 3)
    num_filter = variance_threshold.fit_transform(filled_data)
    #数据放缩
    from sklearn.preprocessing import scale
    num_filter_scale = scale(num_filter)
    return num_filter, num_filter_scale

if __name__ == '__main__':
    data_num,data_cat,data_time,label = load_data('国内某航空公司会员数据.xls')
    
    gender = gender_encoder_imputer(data_cat)
    age ,tier_cat = age_encoder(data_cat)
    time = time_one_hot(data_time)
    
    time_filter,time_filter_scale = reductdim(time)
    
    data_cat = pd.concat([age,tier_cat,time,gender],axis = 1)
    num_filter,num_filter_scale = reductdim(data_num)
    
    