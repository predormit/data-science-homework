# 题目：对2022-2023赛季NBA球员薪水及状态的数据分析与可视化
# 一、项目概述
## 1.1数据集介绍：
### 数据来源于Kaggle上名为NBA Player Salaries (2022-23 Season)的数据集。数据链接为。https://www.kaggle.com/datasets/jamiewelsh2/nba-player-salaries-2022-23-season 。数据集包含了2022-2023赛季所有球员的薪水以及详细的个人比赛数据。

## 1.2研究意义：
### NBA，作为当今最热门的篮球联赛，随着其快速发展，nba球员的薪资也水涨船高。我们不由得思考，在当今联盟的比赛节奏中，球员的哪些数据更能决定其薪资水平，联盟球员的薪资分布情况。并进一步根据当前赛季的工资帽占比推测出球员的薪资水平。

# 二、数据清洗和预处理
## 1.数据导入
### 先导入需要用到的库。pandas库用来处理数据，对数据进行读取，清洗，分组，聚合，合并等。numpy库进行数值计算，如矩阵，向量等。matplotlib库与seaborn库，用于绘制各类图形。
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
### 从数据集中读出数据
    # 读出所有数据
    data = pd.read_csv('/git/final project/data/nba_2022-23_all_stats_with_salary.csv')
    # 删除没有意义的一列
    data.drop(data.columns[0],axis=1,inplace=True)
    data.columns

    # 包含以下球员数据
    Index(['Player Name', 'Salary', 'Position', 'Age', 'Team', 'GP', 'GS', 'MP',
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
        'PF', 'PTS', 'Total Minutes', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%',
        'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS',
        'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP'],
        dtype='object')

### 查看数据前五行
    data.head()     

![](./pic/head.png)
### 我们可以看到该数据的前五行是按2022-23赛季薪资排序的，分别是库里、沃尔、威少、詹姆斯和杜兰特。而数据包含51列，即不同维度的技术统计。此份数据提供了这400+球员的众多项比赛数据，我希望通过数据分析来发现其中的有趣的信息。

### 对主要球员数据的解释：
### Player Name 球员姓名、Salary 薪资、Position 司职位置
### Age 年龄、Team 球队、GP 出场次数
### GS 首发次数、MP 场均出场时间、FG 场均投篮命中数
### FGA 场均出手次数、FG% 投篮命中率、3P 3分球命中球数
### 3PA 3分球出手次数、3P% 3分球命中率、2P 2分球命中球数
### 2PA 2分球出手次数、2P% 2分球命中率=、eFG% 有效命中率=
### FT 罚球命中数、FTA 罚球次数、FT% 罚球命中率
### ORB 进攻篮板、DRB 防守篮板、TRB 总篮板
### AST 助攻数、STL 抢断数、BLK 盖帽
### TOV 失误、PF 个人犯规、PTS 得分数
### Total Minutes 总时长、PER 球员效率值、TS% 真实命中率
### FTrFree 罚球率、OWS 进攻贡献、DWS 防守贡献
### WS 贡献、OBPM 进攻正负值、DBPM 防守正负值
### BPM 正负值、VORP 球员不可替代值

## 2.数据清洗
### 查看数据集中的缺失值数量
    data.isna().sum()

    Player Name       0
    Salary            0
    Position          0
    Age               0
    Team              0
    GP                0
    GS                0
    MP                0
    FG                0
    FGA               0
    FG%               1
    3P                0
    3PA               0
    3P%              13
    2P                0
    2PA               0
    2P%               4
    eFG%              1
    FT                0
    FTA               0
    FT%              23
    ORB               0
    DRB               0
    TRB               0
    AST               0
    STL               0
    BLK               0
    TOV               0
    PF                0
    PTS               0
    Total Minutes     0
    PER               0
    TS%               1
    3PAr              1
    FTr               1
    ORB%              0
    DRB%              0
    TRB%              0
    AST%              0
    STL%              0
    BLK%              0
    TOV%              0
    USG%              0
    OWS               0
    DWS               0
    WS                0
    WS/48             0
    OBPM              0
    DBPM              0
    BPM               0
    VORP              0
    dtype: int64
### 查看数据集中各数据类型，大部分都是浮点数。
    Player Name       object
    Salary             int64
    Position          object
    Age                int64
    Team              object
    GP                 int64
    GS                 int64
    MP               float64
    FG               float64
    FGA              float64
    FG%              float64
    3P               float64
    3PA              float64
    3P%              float64
    2P               float64
    2PA              float64
    2P%              float64
    eFG%             float64
    FT               float64
    FTA              float64
    FT%              float64
    ORB              float64
    DRB              float64
    TRB              float64
    AST              float64
    STL              float64
    BLK              float64
    TOV              float64
    PF               float64
    PTS              float64
    Total Minutes      int64
    PER              float64
    TS%              float64
    3PAr             float64
    FTr              float64
    ORB%             float64
    DRB%             float64
    TRB%             float64
    AST%             float64
    STL%             float64
    BLK%             float64
    TOV%             float64
    USG%             float64
    OWS              float64
    DWS              float64
    WS               float64
    WS/48            float64
    OBPM             float64
    DBPM             float64
    BPM              float64
    VORP             float64
    dtype: object
### 观察数据集发现，缺失值多半是概率值，缺失的原因是因为分母分子都为0.因此把缺失值都换为0就行。
    data.fillna(0,inplace=True)
    data.isna().sum()

# 三、基本数据分析
## 1.描述统计
    data.describe()
![](./pic/describe.png)
### 从这部分统计我们可以看出该赛季平均年薪为841万美元，最高年薪为4807万美元。球员平均年龄为25.8岁，年龄最大的达到了42岁，最小的只有19岁。一名球员一赛季平均能出场48.23场，平均每场19.8分钟。

## 2.球员薪资分布统计，年龄分布统计
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,6))
    #绘制直方图
    data['Salary'].hist(bins = 20,
        histtype = 'bar',
        align = 'left',
        orientation = 'vertical',
        alpha=0.5,
        density =True)
    data['Salary'].plot(kind='kde',style='k--')
    plt.xlabel('SALARY')
![](./pic/salary.png)

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,6))
    data['Age'].hist(bins = 20,
        histtype = 'bar',
        align = 'left',
        orientation = 'vertical',
        alpha=0.5,
        density =True)
    data['Age'].plot(kind='kde',style='k--')
    plt.xlabel('AGE')
![](./pic/age.png)
### 可以看到大部分球员的工资还是较低的，拿高薪的球员只占少数.而球员的年龄分布类似于正态分布，主要集中在25，26岁这个年龄。

## 3.探索性分析
### (1).球员年龄和球员薪水之间的关系
    dat1=data.loc[:,['WS','Salary','Age','PTS']]
    sns.jointplot(x=dat1.Salary,y=dat1.Age,data=dat1,kind='kde') 
### 使用sns库中的jointplot函数双变量绘图，得出球员薪水与年龄之间的关系。根据下图可以得出，大部分年轻球员的年薪都在1000万美金以下。少部分年龄在25岁以下的明星球员可以拿到近4000万的年薪。联盟中持高薪的球员还是集中在30岁左右这个年龄段。
![](./pic/ageandsalary.png)