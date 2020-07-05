#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
train=pd.read_csv('Data_Train2.csv')
train.head()


# In[113]:


train.isnull().sum()


# In[114]:


train.dropna(how='any',axis=0,inplace=True)


# In[115]:


train['Date_Journey']=pd.to_datetime(train.Date_of_Journey)


# In[116]:


from datetime import datetime, date


# In[117]:


train['Weekday']=train.Date_Journey.dt.day_name()


# In[118]:


train.head(3)


# In[119]:


train['Peakdayflag']=0


# In[120]:


CatVarList=[0,1,2,3,4,5,6]
for i in CatVarList:
    print(i)
if(i==4)|(i==5)|(i==6):
    train.loc[(train.Date_Journey.dt.weekday==i),'Peakdayflag']=1


# # Total stops

# In[121]:


import numpy as np
train['Total_Dur'] = pd.to_timedelta(
    np.where(train['Duration'].str.count(':') == 1, train['Duration'] + ':00', train['Duration']))


# In[122]:


train['Total_Stops']=train['Total_Stops'].replace('non-stop','0 stops')


# In[123]:


import re
def ExtractNumber(x):
    z=re.findall("\d+",x)
    return z[0]


# In[124]:


train['Total_Stops']=train['Total_Stops'].apply(ExtractNumber)


# In[125]:


train.head(10)


# # Airline

# In[126]:


train['LowCostTag']=0

CatVarList=['Indigo','Air India','Jet Airways','SpiceJet',
           'Multiple carriers','GoAir','Vistara','Air Asia',
           'Vistara Premium economy','Jet Airways Business',
           'Multiple carriers Premium economy','Trujet']

for i in CatVarList:
    print(i)
    
    if i== 'Indigo':
        train.loc[(train.Airline == i),'LowCostTag'] = 1
    elif i== 'Air India':
        train.loc[(train.Airline == i),'LowCostTag'] = 0
    elif i== 'SpiceJet':
        train.loc[(train.Airline == i),'LowCostTag'] = 1
    elif i== 'Multiple carriers':
        train.loc[(train.Airline == i),'LowCostTag'] = 0
    elif i== 'GoAir':
        train.loc[(train.Airline == i),'LowCostTag'] = 1
    elif i== 'Air Asia':
        train.loc[(train.Airline == i),'LowCostTag'] = 1
    elif i== 'Vistara':
        train.loc[(train.Airline == i),'LowCostTag'] = 1
    elif i== 'Vistara Premium economy':
        train.loc[(train.Airline == i),'LowCostTag'] = 0
    elif i== 'Jet Airways Business':
        train.loc[(train.Airline == i),'LowCostTag'] = 0
    elif i== 'Multiple carriers Premium economy':
        train.loc[(train.Airline == i),'LowCostTag'] = 0
    elif i== 'Trujet':
        train.loc[(train.Airline == i),'LowCostTag'] = 1


# In[127]:


train.head(3)


# In[128]:


z=train.groupby('Airline')['Airline','LowCostTag']
print(z.sum())


# # DURATION

# In[129]:


train.head(3)


# In[130]:


train.dtypes


# In[131]:


train['Total_Dur'] = train['Total_Dur'].dt.total_seconds().div(60).astype(int)


# In[132]:


train.head(5)


# # Departure Time

# In[133]:


def rationalise_deptime(my_list):
    if len(my_list) !=1:
        print(my_list[0],my_list[1])
    try:
        hour = float(my_list[0])
        minute=float(my_list[1])
        return(hour+minute/60)
    except:
        return float(my_list[0])
train['Dep_Time']=train.Dep_Time.str.split(':').apply(rationalise_deptime)


# In[134]:


train.head(3)


# # Departure Flag

# In[135]:


def deftimeflag(mylist):
    if ((mylist)>=6)&((mylist)<11):
        return 'Peak'
    elif ((mylist)>=11)&((mylist)<17):
        return 'Non Peak'
    elif ((mylist)>=17)&((mylist)<21):
        return 'Peak'
    else:
        return 'Non Peak'
    
train['Departure_Flag']=train.Dep_Time.apply(deftimeflag)


# In[136]:


train.head()


# # Encoding

# In[137]:


train.sort_index(axis=1,inplace=True)


# In[138]:


train.columns


# In[28]:


#AllColList=['Additional_Info','Airline','']

CatColList=['Dep_Time','Peakdayflag']


# In[139]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train['Airline']=label_encoder.fit_transform(train['Airline'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)


# In[140]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train['Route']=label_encoder.fit_transform(train['Route'])
le_name_mapping1 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping1)


# In[141]:


train['Source']=label_encoder.fit_transform(train['Source'])
le_name_mapping2 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping2)


# In[142]:


train['Destination']=label_encoder.fit_transform(train['Destination'])
le_name_mapping3 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping3)


# In[143]:


train['Departure_Flag']=label_encoder.fit_transform(train['Departure_Flag'])
le_name_mapping4 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping4)


# In[144]:


train['Additional_Info']=label_encoder.fit_transform(train['Additional_Info'])
le_name_mapping5 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping5)


# In[145]:


train.LowCostTag.value_counts()


# In[146]:


train.Peakdayflag.value_counts()


# In[74]:


train.Dep_Time.value_counts()


# In[147]:


train.drop(['Route'],axis=1,inplace=True)


# In[148]:


train.drop(['Dep_Time'],axis=1,inplace=True)


# In[149]:


train.head(5)


# In[29]:


# In[150]:


train.dtypes


# # Considering only relevant columns

# In[97]:




# In[151]:


AllColList=['Additional_Info','Airline','Destination','LowCostTag','Source','Total_Dur','Total_Stops',
           'Departure_Flag','Peakdayflag','Price']


# In[152]:


train.columns


# In[153]:


train.head(4)


# In[154]:


#For train data

train=train.loc[:,AllColList]
train.sort_index(axis=1,inplace=True)


# In[155]:


train.dtypes


# In[156]:


train.Total_Stops=pd.to_numeric(train.Total_Stops, errors='coerce')


# In[157]:


train.head(2)


# In[158]:


X1=train.loc[:,train.columns != 'Price']
y1=train.Price

from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,random_state=20)


# In[159]:


# Train model and metrices
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=600,random_state=20)

model.fit(X1_train,y1_train)

y_pred=model.predict(X1_test)
from sklearn.metrics import mean_squared_error
from math import sqrt

mse=mean_squared_error(y1_test,y_pred)
msq=1-mean_squared_error(np.log(y1_test),np.log(y_pred))

print('\rmse',mse)
print('\rmse',msq)


# In[161]:


from sklearn import metrics
print("R2 score =", round(metrics.r2_score(y1_test,y_pred), 2))


# In[162]:
import pickle
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
result=model.predict([[2,3,1,5,1,1,3,120,3]])
print(model.predict([[2,3,1,5,1,1,3,120,3]]))
print(result[0])






