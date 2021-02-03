#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


leads=pd.read_csv("E:/Decode_Lectures/Case Study/Case Study_4/Leads.csv")
leads


# # Data Cleaning

# In[3]:


leads.isna().sum()


# In[4]:


#Drop the column have more then 3000 missing value or null value
for col in leads.columns:
    if leads [col].isnull().sum()>3000:
        leads.drop(col,axis=1,inplace=True)


# In[5]:


leads.shape


# In[6]:


leads.drop(["City","Country"],axis=1,inplace=True)


# In[7]:


leads.isna().sum()/leads.shape[0]*100


# In[8]:


leads["Newspaper"].value_counts()


# In[9]:


leads["Newspaper"].astype("category").value_counts()


# In[10]:


for column in leads:
    print(leads[column].astype("category").value_counts())
    print("---------------------------------------------")


# In[11]:


leads.info()


# In[12]:


leads["Lead Profile"].astype("category").value_counts()


# In[13]:


leads["Specialization"].astype("category").value_counts()


# In[14]:


#Drop Unwanted columns
leads.drop(["How did you hear about X Education","Search","Magazine","Do Not Call","X Education Forums",
           "Digital Advertisement","Newspaper Article","Through Recommendations","Receive More Updates About Our Courses",
           "Update me on Supply Chain Content","Get updates on DM Content",
            "I agree to pay the amount through cheque","Newspaper","What matters most to you in choosing a course"],axis=1,inplace=True)
leads.drop([],axis=1,inplace=True)


# In[15]:


leads.info()


# In[16]:


leads.isnull().sum()


# In[17]:


leads.drop(["Lead Profile"],axis=1,inplace=True)


# In[18]:


leads.isnull().sum()


# In[22]:


leads.shape


# In[27]:


#Keep "What is your current occupation" this column and drop null rows
leads=leads[~pd.isnull(leads["What is your current occupation"])]


# In[28]:


leads.shape


# In[31]:


leads.isnull().sum()


# In[32]:


leads=leads[~pd.isnull(leads["TotalVisits"])]


# In[33]:


leads=leads[~pd.isnull(leads["Lead Source"])]
leads=leads[~pd.isnull(leads["Specialization"])]


# In[34]:


leads.isnull().sum()


# In[35]:


leads.shape


# In[36]:


leads.drop(["Prospect ID","Lead Number"],axis=1,inplace=True)


# In[37]:


leads


# In[38]:


leads.dtypes


# In[39]:


leads.select_dtypes(include=object)


# In[41]:


temp=leads.loc[:,leads.dtypes=="object"]


# In[42]:


temp.dtypes


# In[43]:


temp.columns


# In[45]:


dummy=pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
       'Specialization', 'What is your current occupation',
       'A free copy of Mastering The Interview', 'Last Notable Activity']],drop_first=True)


# In[46]:


dummy


# In[47]:


#Merging the data set
leads=pd.concat([leads,dummy],axis=1)
leads


# In[48]:


leads.dtypes


# In[50]:


leads=leads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
       'Specialization', 'What is your current occupation',
       'A free copy of Mastering The Interview', 'Last Notable Activity'],axis=1)


# In[51]:


leads.head()


# In[56]:


(leads.dtypes=="object").sum()


# In[57]:


x=leads.drop(["Converted"],axis=1)


# In[58]:


x.head()


# In[64]:


y=leads["Converted"]


# In[65]:


leads["Converted"].value_counts()


# In[66]:


from sklearn.model_selection import train_test_split


# In[68]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100,stratify=y)


# In[72]:


from sklearn.preprocessing import MinMaxScaler


# In[73]:


scaler=MinMaxScaler()


# In[75]:


x_train[["TotalVisits","Total Time Spent on Website","Page Views Per Visit"]]=scaler.fit_transform(
    x_train[["TotalVisits","Total Time Spent on Website","Page Views Per Visit"]])


# In[76]:


x_train.head()


# In[78]:


from sklearn.linear_model import LogisticRegression


# In[79]:


logreg=LogisticRegression()


# In[82]:


from sklearn.feature_selection import RFE


# In[83]:


rfe=RFE(logreg,74)


# In[84]:


rfe=RFE(logreg,15)


# In[85]:


rfe.fit(x_train,y_train)


# In[86]:


list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# In[88]:


col=x_train.columns[rfe.support_]


# In[89]:


x_train=x_train[col]


# In[90]:


import statsmodels.api as sm


# In[91]:


x_train_sm=sm.add_constant(x_train)


# In[96]:


logm2=sm.GLM(y_train,x_train_sm,family=sm.families.Binomial())


# In[98]:


res=logm2.fit()


# In[99]:


res.summary()


# In[101]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[103]:


vif=pd.DataFrame()
vif["Features"]=x_train.columns
vif["VIF"]=[variance_inflation_factor(x_train.values,i)for i in range(x_train.shape[1])]
vif["VIF"]=round(vif["VIF"])
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[105]:


x_train.drop("What is your current occupation_Housewife",axis=1,inplace=True)


# In[108]:


logm1=sm.GLM(y_train,(sm.add_constant(x_train)),family=sm.families.Binomial())


# In[113]:


logm1.fit().summary()


# In[130]:


y_train_pred=res.predict(x_train_sm)


# In[131]:


y_train_pred[:10]


# In[132]:


y_train_pred.shape


# In[133]:


y_train_pred=y_train_pred.values.reshape(-1)


# In[134]:


y_train_pred[:10]


# In[135]:


y_train_pred_final=pd.DataFrame({"Converted":y_train.values,"conversion_prob":y_train_pred})


# In[137]:


y_train_pred_final.head(10)


# In[139]:


y_train_pred_final["Predicted"]=y_train_pred_final.conversion_prob.map(lambda x:1 if x>0.5 else 0)


# In[141]:


y_train_pred_final.head()


# In[151]:


from sklearn import metrics 


# In[152]:


confusion=metrics.confusion_matrix(y_train_pred_final.Converted,y_train_pred_final.Predicted)


# In[153]:


confusion


# In[146]:


#Predicted                   Not_Converted           Converted
#Actual
#Not_Converted                  TN                      FP
#Converted                      FN                     TP


# In[156]:


metrics.accuracy_score(y_train_pred_final.Converted,y_train_pred_final.Predicted)


# In[157]:


TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]


# In[159]:


#Specificity
TN/(TN+FP)


# In[158]:


#Sensitivity
TP/(TP+FN)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




