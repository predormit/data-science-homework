import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv('/git/final project/data/nba_2022-23_all_stats_with_salary.csv')
dat_cor=data.loc[:,['Salary','Age','GP','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','OWS','DWS','WS','BPM']]
coor=dat_cor.corr()
sns.heatmap(coor,square=True, linewidths=0.02, annot=False)
#seaborn中的heatmap函数，是将多维度数值变量按数值大小进行交叉热图展示。