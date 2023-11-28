import pandas as pd

# 读取CSV文件
df = pd.read_csv('acs2015_census_tract_data.csv')

# 丢弃指定的列
columns_to_drop = ['CensusTract', 'IncomePerCap', 'IncomePerCapErr', 'IncomeErr']
df = df.drop(columns=columns_to_drop)

# 打印处理后的数据框的前几行
print(df.head())
df.to_csv('all_data.csv',index=False)