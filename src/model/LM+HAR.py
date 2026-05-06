# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:15:06 2022

@author: 12404
"""
###要多取22天+270个数据

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from rich.progress import track
import scipy as sc
import time
import os


data = pd.read_excel('./data/上证50高频数据.xlsx')
data_day = pd.read_csv('./data/上证50指数历史数据.csv')
data = data.rename(columns={'时间':'日期'})
data['日期'] = data['日期'].apply(lambda x:str(x))
data_day = data_day.rename(columns={'日期':'日期'})
data_day['日期'] = data_day['日期'].apply(lambda x:time.strftime('%Y%m%d',time.strptime(x,'%d/%m/%Y')))#修改日期格式 
data = data.set_index('日期')
data_day = data_day.set_index('日期')
###参数(需要自己设置!!!)：
t = 48
k = 270
date = '2023-03-01 09:30:00 '   #设置需要识别跳的开始日期
date_day = '20230301'    #与date对应设置
signif = 0.05   #显著性水平
###从'2017-07-07 09:35:00'号开始计算跳：
stock = data.iloc[:,3].dropna()
stock_day = data_day.iloc[:,0].dropna()                            
###############################################
r = np.log(stock).diff(1)
r_day = np.log(stock_day).diff(1)
r_new = r.loc[date:]   ###为了设置n
sign = np.sign(r)   ###跳符号
bvt = abs(r)*abs(r.shift(1))
sig = np.sqrt((bvt.rolling((k-1)).mean()))
L = r/((sig.shift(1)))  #%检验统计量L
###识别LM-参数设置：
n = len(r_new)
C = (2/math.pi)**0.5
Sn = 1/(C*(2*np.log(n))**0.5)
Cn = ((2*np.log(n))**0.5)/C - np.log(math.pi*np.log(n))/(2*C*(2*np.log(n))**0.5)
beta_star = -np.log(-np.log(1-signif))    #求拒绝域
T = (abs(L)-Cn)/Sn   #Test statistics
for i in range(len(T)):
    if  pd.isna(T[i]) != True:
        if T[i] > beta_star:
            T[i] = 1
        else:
            T[i] = 0
    else:
        T[i] = T[i]
#此步输出为0 or 1(上跳) or -1（下跳）：
J = T
for i in range(len(T)):
    if pd.isna(T[i]) != True:
        J[i] = int(T[i])*int(sign[i])
    else:
        J[i] = T[i]
        


####对跳跃示性函数取绝对值(方便后文代码)：
I_abs = abs(J)
I_abs = I_abs.reset_index()
I_abs['交易日'] = I_abs['日期'].apply(lambda x:x.split(' ')[0])  
I_abs['交易日'] = I_abs['交易日'].apply(lambda x:time.strftime('%Y%m%d',time.strptime(x,'%Y-%m-%d')))#修改日期格式    
I_abs['分时'] = I_abs['日期'].apply(lambda x:x.split(' ')[1])
I_abs_pivot = pd.pivot_table(I_abs,index=['分时'],columns=['交易日'],values=[(data.iloc[:,3]).name])[(data.iloc[:,3]).name]                           #############################################
I_abs_pivot=I_abs_pivot.loc[I_abs['分时'].unique(),:]  ###分布图


####计算每天跳的个数（不取绝对值）：
I = J  
I = I.reset_index()
I['交易日'] = I['日期'].apply(lambda x:x.split(' ')[0])  
I['交易日'] = I['交易日'].apply(lambda x:time.strftime('%Y%m%d',time.strptime(x,'%Y-%m-%d')))#修改日期格式    
I['分时'] = I['日期'].apply(lambda x:x.split(' ')[1])
I_pivot = pd.pivot_table(I,index=['分时'],columns=['交易日'],values=[(data.iloc[:,3]).name])[(data.iloc[:,3]).name]                           #############################################
I_pivot=I_pivot.loc[I['分时'].unique(),:]  ###分布图
#计算每个交易日的跳跃个数
I_sum = (((I_pivot))**2).sum()
I_sum.rename(data.iloc[:,3].name, inplace = True) ###每天跳的个数

#第T天第i个离散样本跳大小
k = abs(J)*r
k = k.reset_index()
k['交易日'] = k['日期'].apply(lambda x:x.split(' ')[0])
k['交易日'] = k['交易日'].apply(lambda x:time.strftime('%Y%m%d',time.strptime(x,'%Y-%m-%d')))#修改日期格式   
k['分时'] = k['日期'].apply(lambda x:x.split(' ')[1])
k_pivot = pd.pivot_table(k,index=['分时'],columns=['交易日'],values=[(data.iloc[:,3]).name])[(data.iloc[:,3]).name]                           #############################################
k_pivot=k_pivot.loc[k['分时'].unique(),:]  ###分布图
#计算每个交易日的跳跃大小平方和
k_sum = ((k_pivot)**2).sum()
#计算每个交易日的跳跃大小和
k_sum_r = (k_pivot).sum()    
k_sum.rename(data.iloc[:,3].name, inplace = True) ###每天跳的平方和


###RV的计算：
#stock_RV = r_new.values.reshape(t,int(len(r_new)/t))
stock_RV = r.reset_index()
stock_RV['交易日'] = stock_RV['日期'].apply(lambda x:x.split(' ')[0]) 
stock_RV['交易日'] = stock_RV['交易日'].apply(lambda x:time.strftime('%Y%m%d',time.strptime(x,'%Y-%m-%d')))#修改日期格式     
stock_RV['分时'] = stock_RV['日期'].apply(lambda x:x.split(' ')[1])
stock_RV_pivot = pd.pivot_table(stock_RV,index=['分时'],columns=['交易日'],values=[(data.iloc[:,3]).name])[(data.iloc[:,3]).name]                        #############################################
stock_RV_pivot=stock_RV_pivot.loc[stock_RV['分时'].unique(),:]    ###分布图
#计算每个交易日的日度已实现波动率
RV_sum_d = ((stock_RV_pivot)**2).sum()
RV_sum_w =RV_sum_d.rolling((5)).mean()
RV_sum_m =RV_sum_d.rolling((22)).mean()
RV_sum_d.rename(data.iloc[:,3].name, inplace = True)   #####################################



###跳跃分解：
no_jump_count = 48 - I_sum
no_jump_R2 = ((1-I_abs_pivot)*(stock_RV_pivot**2)).sum()   ###没有跳的R^2的和
meanvol = no_jump_R2/no_jump_count     ###未发生跳跃的平均波动
meanvol.dropna(inplace=True)
k2 = k_pivot**2
PJ = k_pivot**2          
for i in range(PJ.shape[1]):
    for t in range(len(PJ)):
        if  pd.isna(PJ.iloc[t,i]) != True:
            if k2.iloc[t,i] == 0:
                PJ.iloc[t,i] = k2.iloc[t,i]
            else:
                PJ.iloc[t,i] = k2.iloc[t,i] - meanvol[i]
        else:
            PJ.iloc[t,i] = PJ.iloc[t,i]
JSVt = PJ.sum()  ###跳跃波动
CSVt_d = RV_sum_d - JSVt   ###连续跳跃波动
CSVt_w = CSVt_d.rolling((5)).mean()
CSVt_m = CSVt_d.rolling((22)).mean()
JSVt_pre = I_pivot*PJ 
JSVt_pre_z = I_pivot*PJ 
JSVt_pre_f = I_pivot*PJ 
for i in range(JSVt_pre.shape[1]):
    for t in range(len(JSVt_pre)):
        if JSVt_pre.iloc[t,i] <= 0:
            JSVt_pre_f.iloc[t,i] = abs(JSVt_pre.iloc[t,i])
            JSVt_pre_z.iloc[t,i] = 0
        else:
            JSVt_pre_z.iloc[t,i] = JSVt_pre.iloc[t,i]
            JSVt_pre_f.iloc[t,i] = 0

#正向波动：
JSVt_zheng_d = JSVt_pre_z.sum()
JSVt_zheng_w = JSVt_zheng_d.rolling((5)).mean()
JSVt_zheng_m = JSVt_zheng_d.rolling((22)).mean()
#负向波动：
JSVt_fu_d = JSVt_pre_f.sum()
JSVt_fu_w = JSVt_fu_d.rolling((5)).mean()
JSVt_fu_m = JSVt_fu_d.rolling((22)).mean()
###下行风险：
#跳跃收益率：
jrt_d = k_sum_r
crt_d = r_day - jrt_d
for i in range(len(jrt_d)):
    if jrt_d[i] > 0:
        jrt_d.iloc[i] = 0
    else:
        jrt_d.iloc[i] = jrt_d.iloc[i]
for i in range(len(crt_d)):
    if crt_d[i] > 0:
        crt_d.iloc[i] = 0
    else:
        crt_d.iloc[i] = crt_d.iloc[i]
#滞后周月的下行风险：
jrt_w = jrt_d.rolling((5)).mean()
jrt_m = jrt_d.rolling((22)).mean()
crt_w = crt_d.rolling((5)).mean()
crt_m = crt_d.rolling((22)).mean()  
jrt_d = (jrt_d.shift(1)).loc[date_day:]
jrt_w = (jrt_w.shift(1)).loc[date_day:]
jrt_m = (jrt_m.shift(1)).loc[date_day:]
crt_d = (crt_d.shift(1)).loc[date_day:]
crt_w = (crt_w.shift(1)).loc[date_day:]
crt_m = (crt_m.shift(1)).loc[date_day:]    
jump_sum = pd.concat([CSVt_d,CSVt_w,CSVt_m,JSVt_zheng_d,JSVt_zheng_w,JSVt_zheng_m,JSVt_fu_d,JSVt_fu_w,JSVt_fu_m],axis =1)
###滞后一天：
jump_sum = jump_sum.shift(1)
jump_sum = jump_sum.loc[date_day:]
RV_sum_now = RV_sum_d.copy(deep=True)
RV_sum_d = (RV_sum_d.shift(1)).loc[date_day:]
RV_sum_w = (RV_sum_w.shift(1)).loc[date_day:]
RV_sum_m = (RV_sum_m.shift(1)).loc[date_day:]
r_now = r_day.loc[date_day:]
r_d = (r_day.shift(1)).loc[date_day:]
sum = pd.concat([jump_sum,RV_sum_d,RV_sum_w,RV_sum_m,RV_sum_now,r_now,r_d],axis =1)
sum.columns = ['CSVt_d','CSVt_w','CSVt_m','JSVt_zheng_d','JSVt_zheng_w','JSVt_zheng_m','JSVt_fu_d','JSVt_fu_w','JSVt_fu_m','RV_sum_d','RV_sum_w','RV_sum_m','RV_sum_now','r_now','r_d']
sum.to_csv('./data/LM+HAR_new[voltility].csv') 

