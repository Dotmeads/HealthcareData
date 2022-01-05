#!/usr/bin/env python
# coding: utf-8

# # Analysis of Healthcare Cost 

# In[1]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_ff01f6f8ec354d1a95f15ed75d210eb5 = 'https://s3.us.cloud-object-storage.appdomain.cloud'
else:
    endpoint_ff01f6f8ec354d1a95f15ed75d210eb5 = 'https://s3.private.us.cloud-object-storage.appdomain.cloud'

client_ff01f6f8ec354d1a95f15ed75d210eb5 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='MqNJR9OhfgnmBkp527bFpTQ_kDvCE1e78mBuhRgRu8og',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_ff01f6f8ec354d1a95f15ed75d210eb5)

body = client_ff01f6f8ec354d1a95f15ed75d210eb5.get_object(Bucket='healthcaredataanalysisholmusk-donotdelete-pr-2ufppwmslovbhx',Key='bill_amount.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

bill_amount_df = pd.read_csv(body)
bill_amount_df.info()


# In[2]:


body = client_ff01f6f8ec354d1a95f15ed75d210eb5.get_object(Bucket='healthcaredataanalysisholmusk-donotdelete-pr-2ufppwmslovbhx',Key='bill_id.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

bill_id_df = pd.read_csv(body)
bill_id_df.info()


# In[3]:


body = client_ff01f6f8ec354d1a95f15ed75d210eb5.get_object(Bucket='healthcaredataanalysisholmusk-donotdelete-pr-2ufppwmslovbhx',Key='demographics.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

demographics_df = pd.read_csv(body)
demographics_df.info()


# In[4]:


#conversion of demographic entries respectively 
demographics_df['gender'] = demographics_df['gender'].replace(['m','f'],['Male','Female'])
demographics_df['resident_status'] = demographics_df['resident_status'].replace(['Singapore citizen'],'Singaporean')
demographics_df['race'] = demographics_df['race'].replace(['chinese','India'],['Chinese','Indian'])


# In[5]:


body = client_ff01f6f8ec354d1a95f15ed75d210eb5.get_object(Bucket='healthcaredataanalysisholmusk-donotdelete-pr-2ufppwmslovbhx',Key='clinical_data.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

clinical_data_df = pd.read_csv(body)
clinical_data_df.info()


# In[6]:


#rename id to patient id to ensure consistency for merge later on
clinical_data_df.rename(columns = {"id":"patient_id"},inplace = True)


# In[7]:


#Merge the bill related files together using the bill_id and ensure all rows are included.
bill_df = pd.merge(bill_amount_df,bill_id_df, on='bill_id')
bill_df.info()


# In[8]:


#Create a column for merge later 
bill_df['patient_admission_id'] = bill_df['patient_id'] + " " + bill_df['date_of_admission']


# In[9]:


#The number of rows in this bill_df is significantly higher 
#We see each patient has more than one bill ID in the raw file and conduct a check
bill_df.value_counts('patient_id')


# In[10]:


#Merge clinical_data with demographics on the patient_id
clinical_data_df = pd.merge(clinical_data_df,demographics_df, on='patient_id')

#Create column for merge later
clinical_data_df['patient_admission_id'] = clinical_data_df['patient_id'] + " " + clinical_data_df['date_of_admission']

clinical_data_df.head(10)


# In[11]:


import datetime as datetime

#Create Useful variables for analysis

#length of stay in hospital
clinical_data_df['date_of_discharge'] = pd.to_datetime(clinical_data_df['date_of_discharge'])
clinical_data_df['date_of_admission'] = pd.to_datetime(clinical_data_df['date_of_admission'])
clinical_data_df['length_of_stay'] = clinical_data_df['date_of_discharge'] - clinical_data_df['date_of_admission']

#BMI to account for height and weight instead of conventional normalization 
clinical_data_df['BMI'] = round(clinical_data_df['weight']/((clinical_data_df['height']/100)**2),1)

#Age based on the year between date of admission and DOB 
clinical_data_df['date_of_birth'] = pd.to_datetime(clinical_data_df['date_of_birth'])
clinical_data_df['age'] = (clinical_data_df['date_of_admission'] - clinical_data_df['date_of_birth']).astype('<m8[Y]')

clinical_data_df.head(10)


# In[12]:


#Ensure all medical_history is converted to integers for consistency
clinical_data_df['medical_history_2'] = clinical_data_df['medical_history_2'].astype("Int64")
clinical_data_df['medical_history_5'] = clinical_data_df['medical_history_5'].astype("Int64")

clinical_data_df.head(10)


# In[13]:


#conversion of Yes and No entries to 1 and 0 respectively 
clinical_data_df['medical_history_3'] = clinical_data_df['medical_history_3'].replace(['Yes','No'],[1,0])
#conversion of medical_history_3 to integer from object
clinical_data_df["medical_history_3"] = clinical_data_df["medical_history_3"].astype(str).astype(int) 
clinical_data_df.dtypes


# In[14]:


#create a new column to group bills from the same patient and admission date 
bill_df['patient_admission_id'] = bill_df['patient_id'] + " " + bill_df['date_of_admission']
print(f"number of unique patient admissions: {len(bill_df['patient_admission_id'].unique())}")
bill_df.head()


# We see the unique patient admission now tally with the number of entries in clinical_data_df. So we can proceed to create the same variable in the  clinical_data_df later on. Before that, we will merge the bill amount on the same patient_admission_id we have created. 

# In[15]:


patient_df = pd.merge(clinical_data_df,bill_df, on='patient_admission_id')


# In[16]:


patient_df2 = patient_df.groupby('patient_admission_id',as_index=False).sum([['amount']]) 
patient_df2.rename(columns = {"amount":"total_amount"},inplace = True)
patient_df2


# In[17]:


patient_final_df = pd.merge(patient_df, patient_df2[['patient_admission_id','total_amount']], on='patient_admission_id')


# In[18]:


#After obtaining the total amount, we add another column to account for amount per admission day.
#This accounts for length of stay contributing to healthcare cost.

patient_final_df['length_of_stay'] = patient_final_df['length_of_stay'].astype('timedelta64[D]').astype(int)

patient_final_df['average_amount_per_day'] = (patient_final_df['total_amount'])/(patient_final_df['length_of_stay'])


# In[19]:


patient_final_df.drop(['amount','bill_id','patient_id_y','date_of_admission_y'],axis = 1,inplace = True)
patient_final_df.drop_duplicates(subset=None, keep='first',inplace = True)
patient_final_df.reset_index()


# ## Analysis of Healthcare Data

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeRegressor


# In[21]:


patient_final_df.corr(method = 'spearman')['average_amount_per_day'].sort_values()


# In[22]:


patient_final_df.corr(method = 'spearman')['total_amount'].sort_values()


# In[23]:


plt.figure(figsize=(16, 10))
sns.heatmap(patient_final_df.corr())


# ### Resident Status on Healthcare Cost 

# In[24]:


#we start of by examining the demographics of the patient on the bill amount 

fig, axes = plt.subplots(1, 2,figsize=(12,6))
fig.suptitle('Resident Status')

# Count
sns.countplot(x = 'resident_status',data = patient_final_df, ax = axes[0])
axes[0].set_title('Resident Status Type Count')

# Type vs Amount 
sns.boxplot(x = 'resident_status' , y = 'average_amount_per_day',data = patient_final_df, ax=axes[1])
axes[1].set_title('Resident Status vs Average Amount')


# ### Gender on Healthcare Cost 

# In[25]:


fig, axes = plt.subplots(1, 2,figsize=(12,6))
fig.suptitle('Gender')

# Count
sns.countplot(x = 'gender',data = patient_final_df, ax = axes[0])
axes[0].set_ylim(bottom=0, top=2000)
axes[0].set_title('Gender Type Count')

# Type vs Amount 
sns.stripplot(x = 'gender' , y = 'average_amount_per_day',data = patient_final_df, ax=axes[1])
axes[1].set_title('Gender vs Average Amount') 


# ### Race on Healthcare Cost 

# In[26]:


fig, axes = plt.subplots(1, 3,figsize=(18,6))
fig.suptitle('Race')

# Count
sns.countplot(x = 'race',data = patient_final_df, ax = axes[0])
axes[0].set_title('Race Type Count')

# Type vs Amount (scatter)
sns.stripplot(x = 'race' , y = 'average_amount_per_day',data = patient_final_df, ax=axes[1])
axes[1].set_title('Race vs Average Amount')

# Type vs Amount (box)
sns.boxplot(x = 'race' , y = 'average_amount_per_day',data = patient_final_df, ax=axes[2])
axes[2].set_title('Race vs Average Amount')


# In[27]:


Chinese = patient_final_df[patient_final_df['race']=='Chinese']['average_amount_per_day']
Malay = patient_final_df[patient_final_df['race']=='Malay']['average_amount_per_day'].dropna()
Indian = patient_final_df[patient_final_df['race']=='Indian']['average_amount_per_day'].dropna()
Others = patient_final_df[patient_final_df['race']=='Others']['average_amount_per_day'].dropna()

race_df = pd.concat([Chinese,Malay,Indian,Others],axis = 1)
race_df.columns = ['Chinese','Malay','Indian','Others']


# In[28]:


import scipy.stats as stats
from scipy.stats import anderson

#Determine if there is normal distribution in the races dataset 
for col in race_df.columns: 
    result = (anderson(race_df[col].dropna(), dist='norm'))
    print(f"A-D statistic: {result[0]}")
print(f"Critical values: {result[1]}")
print(f"Significance levels: {result[2]}")


# In[29]:


#We conduct ANOVA to examine is the difference among races is statistically significant. 

#Use Box Cox to bring the distribution close to Gaussian distribution for ANOVA to be performed
for col in race_df.columns: 
    race_df[col],fitted_lambda = stats.boxcox(race_df[col])

F, p = stats.f_oneway(race_df['Chinese'].dropna(),race_df['Malay'].dropna(),race_df['Indian'].dropna(),race_df['Others'].dropna())
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.3f' % (F, p))


# As p > 0.05, we do not reject the null hypothesis. Therefore we cannot conclude there is a significant difference between the race dataset.

# ### Age on Healthcare Cost 

# In[30]:


plt.figure(figsize = (12,9))
sns.regplot(x= "age", y= "average_amount_per_day", data = patient_final_df)
#set y limit to exclude outliers 
plt.set_ylim = [(0,10000)]


# In[31]:


conditions = [
    (patient_final_df['age'] <= 30),
    (patient_final_df['age'] <= 40),
    (patient_final_df['age'] <= 50),
    (patient_final_df['age'] <= 60),
    (patient_final_df['age'] <= 70),
    (patient_final_df['age'] <= 80),
    (patient_final_df['age'] > 80)
    ]

# create a list of the values we want to assign for each condition
values = ['below 30', '31-40', '41-50', '51-60','61-70','71-80','above 80']

# create a new column and use np.select to assign values to it using our lists as arguments
patient_final_df['age group'] = np.select(conditions, values)

plt.figure(figsize = (12,9))
sns.boxplot(x = "age group", y= "average_amount_per_day", data = patient_final_df, order = values)
plt.ylim(top = 10000)


# In[32]:


#Breakdown the effects from the clinical data 

#BMI on Healthcare Cost 
plt.figure(figsize=(12, 9))
sns.regplot(x= "BMI", y= "average_amount_per_day", data = patient_final_df)
plt.ylim(top = 15000)
patient_final_df[["average_amount_per_day","BMI"]].corr()


# In[33]:


#BMI beyond 25 is considered unhealthy so we break these patients into 2 groups 
BMI_df = patient_final_df.copy()
BMI_df['Healthy'] = np.where(BMI_df['BMI']>=25,'unhealthy', 'healthy')
BMI_df

#Plot graph related to Healthy column
fig, axes = plt.subplots(1, 3,figsize=(16,6))
fig.suptitle('Unhealthy (BMI>25) vs Healthy Patient (BMI<=25) Type')

# Count
sns.countplot(x = 'Healthy',data = BMI_df, ax = axes[0])
axes[0].set_title('Unhealthy vs Healthy Patient Count')

# Type vs Amount (scatter)
sns.stripplot(x = 'Healthy' , y = 'average_amount_per_day',data = BMI_df, ax=axes[1])
axes[1].set_title('Patient Type vs Average Amount')

# Type vs Amount (box)
sns.boxplot(x = 'Healthy' , y = 'average_amount_per_day',data = BMI_df, ax=axes[2])
axes[2].set_title('Patient Type vs Average Amount')

BMI_df.groupby('Healthy')['average_amount_per_day'].mean()


# ### Lab Results on Average Amount 

# In[34]:


#lab_result vs average amount per day 
lab_results_df = patient_final_df[['lab_result_1', 'lab_result_2','lab_result_3','average_amount_per_day']]
lab_results_df.corr()

#Plot graph for Lab Results 
fig, axes = plt.subplots(3, 1,figsize=(10,16))
fig.suptitle('Lab results vs Average Amount')

axes[0].scatter(x='lab_result_1',y='average_amount_per_day', data=lab_results_df)
axes[0].set_title('Lab Result 1')
axes[0].set_ylim([0, 15000])

axes[1].scatter(x='lab_result_2',y='average_amount_per_day', data=lab_results_df)
axes[1].set_title('Lab Result 2')
axes[1].set_ylim([0, 15000])

axes[2].scatter(x='lab_result_3',y='average_amount_per_day', data=lab_results_df)
axes[2].set_title('Lab Result 3')
axes[2].set_ylim([0, 15000])


# The tables shows there is weak or no relationship between the Lab results and Average Amount spent, nor is there relationship between the Lab results. 

# ### Preopt Medication on Healthcare Cost 

# In[35]:


preop_med = ['preop_medication_1', 'preop_medication_2','preop_medication_3','preop_medication_4', 'preop_medication_5','preop_medication_6']
    
pre_med_0 = []
pre_med_1 = []

for i in preop_med:  
    preop_medication_df = patient_final_df[[i,'average_amount_per_day']]
    x = preop_medication_df.groupby(i)['average_amount_per_day'].mean().values
    pre_med_0.append(x[0])
    pre_med_1.append(x[1])

dict = {'Pre-opt med': preop_med, '0': pre_med_0, '1': pre_med_1} 

df = pd.DataFrame(dict).set_index(['Pre-opt med'])
df.plot(kind = 'bar',figsize = (8,8))
plt.xticks(rotation = 45)
plt.ylim([1000,2300])

df


# In[36]:


preop_medication_df = patient_final_df[['preop_medication_1', 'preop_medication_2','preop_medication_3', 
                    'preop_medication_4', 'preop_medication_5','preop_medication_6','average_amount_per_day']]

preop_medication_df = preop_medication_df.astype(str)

preop_medication_df['combination'] = preop_medication_df[['preop_medication_1', 'preop_medication_2','preop_medication_3', 
                    'preop_medication_4', 'preop_medication_5','preop_medication_6']].agg('-'.join, axis=1)

preop_medication_df['average_amount_per_day'] = preop_medication_df['average_amount_per_day'].astype(float)

preop_medication_df2 = preop_medication_df.groupby('combination').mean().sort_values(by = 'average_amount_per_day')


# In[37]:


fig, axes = plt.subplots(2, 1,figsize=(18,18))

# Count
sns.countplot(x = 'combination',order = preop_medication_df2.index,data = preop_medication_df, ax = axes[0])
axes[0].set_title('Combination Count')
axes[0].tick_params(axis='x', rotation=90)

# Type vs Amount (scatter)
sns.stripplot(x = 'combination' , y = 'average_amount_per_day',order = preop_medication_df2.index, data = preop_medication_df, ax=axes[1])
axes[1].set_title('Combination vs Average Amount')
axes[1].tick_params(axis='x', rotation=90)
axes[1].set_ylim(0, 15000)

#preop_medication_df['combination'].value_counts().index


# ### Symptoms on Healthcare Cost 

# In[38]:


symptom_df = patient_final_df[['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5','average_amount_per_day']]

symptom_df = symptom_df.astype(str)

symptom_df['combination'] = symptom_df[['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5']].agg('-'.join, axis=1)

symptom_df['average_amount_per_day'] = symptom_df['average_amount_per_day'].astype(float)

symptom_df2 = symptom_df.groupby('combination').mean().sort_values(by = 'average_amount_per_day')


# In[39]:


fig, axes = plt.subplots(2, 1,figsize=(18,18))

# Count
sns.countplot(x = 'combination',order = symptom_df2.index,data = symptom_df, ax = axes[0])
axes[0].set_title('Combination Count')
axes[0].tick_params(axis='x', rotation=90)

# Type vs Amount (scatter)
sns.boxplot(x = 'combination' , y = 'average_amount_per_day',order = symptom_df2.index, data = symptom_df, ax=axes[1])
axes[1].set_title('Combination vs Average Amount')
axes[1].tick_params(axis='x', rotation=90)


# ### Medical History on Healthcare Cost 

# In[40]:


#Filter patient file by dropping null values to be used for training data
filtered_patient_final_df = patient_final_df.dropna()


# In[41]:


#Separate out the entries where medical_history_2 is null 
med2_test = patient_final_df[patient_final_df['medical_history_2'].isnull()]

#Separate out the entries where ONLY medical_history_5 is null 
med5_test = patient_final_df[patient_final_df['medical_history_5'].isnull()]
med5_test =  med5_test.dropna(subset=['medical_history_2'])

#Separate out the entries where ONLY medical_history_2 is null 
med2_test =  med2_test.dropna(subset=['medical_history_5'])

#Separate out the entries where medical_history_2, 5 is null 
med2n5_test = med5_test[med5_test['medical_history_2'].isnull()]

#Filter patient file by dropping entries where medical_history_2, 5 is null
filtered_med_patient_final_df = patient_final_df.drop(med2n5_test.index)


# In[42]:


med2_test_filtered = med2_test[['medical_history_1', 'medical_history_3',
       'medical_history_4', 'medical_history_5', 'medical_history_6',
       'medical_history_7', 'preop_medication_1', 'preop_medication_2',
       'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
       'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
       'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
       'lab_result_3', 'length_of_stay', 'BMI', 'age','total_amount']]

med2_test_filtered = pd.concat([med2_test_filtered, pd.get_dummies(med2_test['race']),pd.get_dummies(med2_test['gender']),
                pd.get_dummies(med2_test['resident_status'])], axis=1)


# In[43]:


#Create Decision Tree Model to predict null values in medical_history_2 

X2 = filtered_patient_final_df[['medical_history_1', 'medical_history_3',
       'medical_history_4', 'medical_history_5', 'medical_history_6',
       'medical_history_7', 'preop_medication_1', 'preop_medication_2',
       'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
       'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
       'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
       'lab_result_3', 'length_of_stay', 'BMI', 'age','total_amount']]

X2 = pd.concat([X2, pd.get_dummies(filtered_patient_final_df['race']),pd.get_dummies(filtered_patient_final_df['gender']),
                pd.get_dummies(filtered_patient_final_df['resident_status'])], axis=1)

y = filtered_patient_final_df['medical_history_2'].astype(int).values

X2 = preprocessing.StandardScaler().fit(X2).transform(X2)

x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# In[44]:


for d in range(1,10):
    dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = d).fit(x_train, y_train)
    dt_yhat = dt.predict(x_test)
    print("For depth = {}  the accuracy score is {} ".format(d, accuracy_score(y_test, dt_yhat)))


# The best value of depth is d = 5 

# In[45]:


#Creating the best decision tree with depth = 5 
best_dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5,class_weight ='balanced').fit(x_train, y_train)

print("Train set Accuracy (Jaccard): ", jaccard_score(y_train, best_dt_model.predict(x_train)))
print("Test set Accuracy (Jaccard): ", jaccard_score(y_test, best_dt_model.predict(x_test)))

print("Train set Accuracy (F1): ", f1_score(y_train, best_dt_model.predict(x_train), average='weighted'))
print("Test set Accuracy (F1): ", f1_score(y_test, best_dt_model.predict(x_test), average='weighted'))


# In[46]:


#Create a Logistic Regression Model for comparison 
for k in ('lbfgs', 'saga', 'liblinear', 'newton-cg', 'sag'):
    lr_model = LogisticRegression(C = 0.01, solver = k).fit(x_train, y_train)
    lr_yhat = lr_model.predict(x_test)
    y_prob = lr_model.predict_proba(x_test)
    print('When Solver is {}, logloss is : {}'.format(k, log_loss(y_test, y_prob)))


# In[47]:


best_lr_model = LogisticRegression(C = 0.01, solver = 'lbfgs', class_weight = 'balanced').fit(x_train, y_train)

print("Train set Accuracy (Jaccard): ", jaccard_score(y_train, best_lr_model.predict(x_train),pos_label = 1))
print("Test set Accuracy (Jaccard): ", jaccard_score(y_test, best_lr_model.predict(x_test), pos_label = 1))

print("Train set Accuracy (F1): ", f1_score(y_train, best_lr_model.predict(x_train), average='weighted'))
print("Test set Accuracy (F1): ", f1_score(y_test, best_lr_model.predict(x_test), average='weighted'))


# In[48]:


#Create svm model for comparison
from sklearn import svm 
for k in ('linear', 'poly', 'rbf','sigmoid'):
    svm_model = svm.SVC( kernel = k).fit(x_train,y_train)
    svm_yhat = svm_model.predict(x_test)
    print("For kernel: {}, the f1 score is: {}".format(k,f1_score(y_test,svm_yhat, average='weighted')))


# In[49]:


best_svm = svm.SVC(kernel='sigmoid', class_weight = 'balanced').fit(x_train,y_train)
best_svm


# In[50]:


print("Train set Accuracy (Jaccard): ", jaccard_score(y_train, best_svm.predict(x_train)))
print("Test set Accuracy (Jaccard): ", jaccard_score(y_test, best_svm.predict(x_test)))

print("Train set Accuracy (F1): ", f1_score(y_train, best_svm.predict(x_train), average='weighted'))
print("Test set Accuracy (F1): ", f1_score(y_test, best_svm.predict(x_test), average='weighted'))


# In[51]:


med2_test_filtered['medical_history_2'] = best_lr_model.predict(med2_test_filtered)


# In[52]:


med2_test = med2_test.merge(med2_test_filtered['medical_history_2'], left_index=True, right_index=True)


# In[53]:


#Rename the columns 
med2_test.drop(columns = ['medical_history_2_x'])
med2_test.rename(columns = {"medical_history_2_y":"medical_history_2"},inplace = True) 


# In[54]:


#Repeat the process for medical_history_5
med5_test_filtered = med5_test[['medical_history_1', 'medical_history_2',
       'medical_history_3', 'medical_history_4', 'medical_history_6',
       'medical_history_7', 'preop_medication_1', 'preop_medication_2',
       'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
       'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
       'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
       'lab_result_3', 'length_of_stay', 'BMI', 'age','total_amount']]

med5_test_filtered = pd.concat([med5_test_filtered, pd.get_dummies(med5_test['race']),pd.get_dummies(med5_test['gender']),
                pd.get_dummies(med5_test['resident_status'])], axis=1)


# In[55]:


#Create Logistic Model to predict null values in medical_history_5

X3 = filtered_patient_final_df[['medical_history_1', 'medical_history_2',
       'medical_history_3', 'medical_history_4', 'medical_history_6',
       'medical_history_7', 'preop_medication_1', 'preop_medication_2',
       'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
       'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
       'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
       'lab_result_3', 'length_of_stay', 'BMI', 'age','total_amount']]

X3 = pd.concat([X3, pd.get_dummies(filtered_patient_final_df['race']),pd.get_dummies(filtered_patient_final_df['gender']),
                pd.get_dummies(filtered_patient_final_df['resident_status'])], axis=1)

y = filtered_patient_final_df['medical_history_5'].astype(int).values

X3 = preprocessing.StandardScaler().fit(X3).transform(X3)

x_train, x_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# In[56]:


for k in ('lbfgs', 'saga', 'liblinear', 'newton-cg', 'sag'):
    lr_model = LogisticRegression(C = 0.01, solver = k).fit(x_train, y_train)
    lr_yhat = lr_model.predict(x_test)
    y_prob = lr_model.predict_proba(x_test)
    print('When Solver is {}, logloss is : {}'.format(k, log_loss(y_test, y_prob)))


# In[57]:


#Creating the log regression model 
best_lr_model = LogisticRegression(C = 0.01, solver = 'liblinear', class_weight = 'balanced').fit(x_train, y_train)

print("Train set Accuracy (Jaccard): ", jaccard_score(y_train, best_lr_model.predict(x_train)))
print("Test set Accuracy (Jaccard): ", jaccard_score(y_test, best_lr_model.predict(x_test)))

print("Train set Accuracy (F1): ", f1_score(y_train, best_lr_model.predict(x_train), average='weighted'))
print("Test set Accuracy (F1): ", f1_score(y_test, best_lr_model.predict(x_test), average='weighted'))


# In[58]:


med5_test_filtered['medical_history_5'] = best_lr_model.predict(med5_test_filtered)


# In[59]:


med5_test = med5_test.merge(med5_test_filtered['medical_history_5'], left_index=True, right_index=True)
#Rename the columns 
med5_test.rename(columns = {"medical_history_5_y":"medical_history_5"},inplace = True) 
med5_test.drop(columns = ['medical_history_5_x'],inplace = True)


# In[60]:


filtered_patient_final_df = filtered_patient_final_df.append([med5_test,med2_test])


# In[70]:


from sklearn.linear_model import LinearRegression

X = filtered_patient_final_df[['medical_history_1', 'medical_history_2',
       'medical_history_3', 'medical_history_4', 'medical_history_5','medical_history_6',
       'medical_history_7', 'preop_medication_1', 'preop_medication_2',
       'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
       'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
       'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
       'lab_result_3', 'BMI', 'age']]

X = pd.concat([X, pd.get_dummies(filtered_patient_final_df['race']),pd.get_dummies(filtered_patient_final_df['gender']),
                pd.get_dummies(filtered_patient_final_df['resident_status'])], axis=1)

y = filtered_patient_final_df['average_amount_per_day'].astype(int).values

X1 = X.columns

X = preprocessing.StandardScaler().fit(X).transform(X)


# In[71]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# In[72]:


dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)

coeff_df = pd.DataFrame(dtr.feature_importances_, X1, columns=['feature importance'])
coeff_df = coeff_df.sort_values(by=['feature importance'])
coeff_df['Type'] = coeff_df.index


# In[73]:


plt.figure(figsize=(8, 10))
sns.barplot(x = 'feature importance', y = 'Type' , data = coeff_df, color = 'blue')
plt.title('Feature Importance vs Type')


# In[65]:


medical_history = ['medical_history_1', 'medical_history_2','medical_history_3',
              'medical_history_4', 'medical_history_5','medical_history_6','medical_history_7']
med_0 = []
med_1 = []

for x in medical_history: 
    medical_history_df = filtered_patient_final_df[[x,'average_amount_per_day']]
    g = medical_history_df.groupby(x)['average_amount_per_day'].mean().values
    med_0.append(g[0])
    med_1.append(g[1])

dict = {'medical_history': medical_history, '0': med_0, '1': med_1} 
    
df = pd.DataFrame(dict).transpose()

df = pd.DataFrame(dict).set_index(['medical_history'])
df.plot(kind = 'bar',figsize = (8,8))
plt.xticks(rotation = 45)
plt.ylim([1000,3000])

df


# In[ ]:




