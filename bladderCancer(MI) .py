#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from random import sample
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ### Pre-processing of data and Normalizing the columns for Machine Learning

# In[2]:


# read in the data

data = pd.read_csv('bladderRNASeq.txt', sep = '\t')


# In[3]:


data


# In[4]:


data = data.drop(['Entrez_Gene_Id'], axis=1)
data


# In[5]:


# remove genes which were NaN

data = data.dropna(subset=['Hugo_Symbol'])
data


# In[6]:


# transpose the data so that genes are columns and rows are samples

data = np.transpose(data)
data


# In[7]:


# rename columns for dataframe

columns = data.iloc[0]


# In[8]:


data = data[1:]


# In[9]:


data.columns = columns
data


# ### Matching sample ID with metastasis status

# In[10]:


metastasis = pd.read_csv('bladderClinicalData.txt', sep = '\t')


# In[11]:


metastasis


# In[12]:


# obtain a list of sample ID

idList = metastasis['Sample ID'].tolist()
print(len(idList))


# In[13]:


metStatus = metastasis['American Joint Committee on Cancer Metastasis Stage Code'].tolist()


# In[14]:


dataID = data.index.tolist()


# In[15]:


# create column for metastasis status in data dataframe
listDataStatus = []
for name in dataID:
    if name in idList:
        index = idList.index(name)
        listDataStatus.append(metStatus[index])
        


# In[16]:


listDataStatus


# In[17]:


data['status'] = listDataStatus


# In[18]:


data


# In[19]:


# filter 


# In[20]:


statusData = data[data['status'].isin(['M0','M1'])]


# In[21]:


statusData


# In[22]:


m0 = statusData[statusData['status'] == 'M0']
m1 = statusData[statusData['status'] == 'M1']


# In[23]:



m0 = m0.drop('status', axis = 1)


# In[24]:


m1 = m1.drop('status', axis = 1)


# In[25]:


m0


# In[26]:


m1


# In[27]:


print(len(m0.columns))


# In[28]:


print(len(m1.columns))


# In[29]:


# make both dataframes have the same number of rows

newM1 = m1.sample(195, replace = True)


# In[30]:


newM1


# ### Calculating Mutual Information Score and ranking top 2000 genes

# In[31]:


# calculate mutual information scores

geneMI = []
for idx, gene in enumerate(m0.columns):
    
    mi = mutual_info_score(m0.iloc[:,idx], newM1.iloc[:,idx])
    geneMI.append(mi)
    


# In[32]:


geneMI


# In[37]:


top2000GeneIndicesMI = np.argsort(geneMI)[-2000:]


# In[38]:


top2000GenesMI = m0.columns[list(top2000GeneIndicesMI)]


# In[39]:


topCorrGenesMI = top2000GenesMI[::-1]


# In[40]:


topCorrGenesMI


# ### Using leave-one-out cross validation to get accuracy score for Logistic Regression

# In[41]:


filteredData = statusData.loc[:,topCorrGenesMI]


# In[42]:


filteredDataType = pd.concat([filteredData,statusData.iloc[:,-1]], axis=1, ignore_index=True)


# In[43]:


filteredDataType


# In[44]:


filteredDataType = filteredDataType.reset_index(drop=True)


# In[45]:


filteredDataType.reset_index(drop=True)


# ### Using Logistic Regression

# In[ ]:




finalAcc = []


for g in range(2000):
    train = filteredDataType


    predictionsLr = []
    actual = []
    
    for sampleID in range(206):

        trainCV = train.copy()
   

        testData = trainCV.iloc[sampleID]
        trainData = trainCV.drop(sampleID)
        k = (g+1)
        
        scaler = StandardScaler()
        
        trX = trainData.iloc[:,:k] 
        trY = trainData.iloc[:,-1]

        teX = testData.iloc[:k].values.reshape(1,-1)
        teY = testData.iloc[-1]
        
        trXScaled = scaler.fit_transform(trX)
        teXScaled = scaler.transform(teX)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(trXScaled, trY)
        yPredLr = lr.predict(teXScaled)
        predictionsLr.append(yPredLr)
        actual.append(teY)
  
    print(predictionsLr)
    pLr = accuracy_score(predictionsLr, actual)
    finalAcc.append(pLr)
    print(pLr)


# In[ ]:


finalAcc


# ### Using Random Forest

# In[ ]:




finalAccRF = []


for g in range(2000):
    train = filteredDataType


    predictionsLr = []
    actual = []
    
    for sampleID in range(206):

        trainCV = train.copy()
   

        testData = trainCV.iloc[sampleID]
        trainData = trainCV.drop(sampleID)
        k = (g+1)
        
        scaler = StandardScaler()
        
        trX = trainData.iloc[:,:k] 
        trY = trainData.iloc[:,-1]

        teX = testData.iloc[:k].values.reshape(1,-1)
        teY = testData.iloc[-1]
                           
        
        trXScaled = scaler.fit_transform(trX)
        teXScaled = scaler.transform(teX)

        lr = RandomForestClassifier()
        lr.fit(trXScaled, trY)
        yPredLr = lr.predict(teXScaled)
        predictionsLr.append(yPredLr)
        actual.append(teY)
  
    print(predictionsLr)
    pLr = accuracy_score(predictionsLr, actual)
    finalAccRF.append(pLr)
    print(pLr)


# In[ ]:



finalAccRF


# In[ ]:


plt.figure(figsize=(10, 6)) 

genes = [i for i in range(103)]

plt.errorbar(genes,finalAcc,linestyle='-', marker='^', color='green')
plt.errorbar(genes,finalAccRF,linestyle='-', marker='^', color='blue')

plt.title("Plot of accuracy over number of ranked genes")


p1 = mlines.Line2D([], [], color='green', marker='^', linestyle='-',
                          markersize=10, label='Accuracy for Logistic Regression (MI)')

p2 = mlines.Line2D([], [], color='blue', marker='^', linestyle='-',
                          markersize=10, label='Accuracy for Random Forest (MI)')




plt.legend(handles=[p1,p2])




plt.xlabel("Number of Genes",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[ ]:





# In[ ]:




