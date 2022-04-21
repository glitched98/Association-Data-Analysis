# Association Analysis
# Omar Syed - 500809837
import pandas as pd
import numpy as pf
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


dataSet = pd.read_csv('GroceryStoreDataSet.csv', names = ['Items'], sep = ',')
print(dataSet.head(11))

ds = list(dataSet["Items"].apply(lambda x:x.split(",") ))

transE = TransactionEncoder()
print(ds)

transD = transE.fit(ds).transform(ds)
dataSet = pd.DataFrame(transD, columns=transE.columns_)

dataSet = dataSet.replace(False, 0)
dataSet = dataSet.replace(True, 1)
print(dataSet.head(11))

# Application of Apriori

dataSet = apriori(dataSet, min_support = 0.01, use_colnames= True, verbose = 1)
print(dataSet.head(6))

# Association

dataSetAssociationRule = association_rules(dataSet, metric= "confidence", min_threshold = 0.05)
print(dataSetAssociationRule.head())

support = dataSetAssociationRule.support.to_numpy()
confidence = dataSetAssociationRule.confidence.to_numpy()

plt.figure(figsize=(4,5))
plt.title('Association Analysis')
plt.ylabel('Confidance')
plt.xlabel('Support')
sns.regplot(x=support, y=confidence, fit_reg=False)
plt.show()

