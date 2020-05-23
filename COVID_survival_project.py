#!/usr/bin/env python
# coding: utf-8

# # Imports and Functions

# In[47]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
import re

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, confusion_matrix, classification_report, fbeta_score,  r2_score

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

from sklearn.model_selection import (GridSearchCV,
                                     train_test_split)
from sklearn.metrics import (classification_report,
                             mean_squared_error,
                             mean_absolute_error)
from matplotlib import pyplot as plt
import xgboost as xgb
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import manifold 


# In[48]:


def test_results(y_test, y_test_pred, y_test_pred_pr):

    print("*********TEST RESULTS*********")

    auc_plotting(y_test, y_test_pred_pr)
    auc_score, accuracy, precision, recall, f1 = results_summary(y_test, y_test_pred, y_test_pred_pr)

    print("Classification Report:")
    print(sklearn.metrics.classification_report(y_test, y_test_pred))

    return auc_score, accuracy, precision, recall, f1


def auc_plotting(y_test, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr) # compute area under the curve
#     auc_score = sklearn.metrics.roc_auc_score(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show() 
    
def results_summary(y_test,y_pred, y_test_pred_pr):
    auc_score = sklearn.metrics.roc_auc_score(y_test, y_test_pred_pr)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    print("AUC %s " % auc_score)
    print("Accuracy %s " % accuracy)
    print("Precision %s " % precision)
    print("Recall %s " % recall )
#     print("Kappa %s " % kappa)
    print("F1 %s " % f1)
#     print("R2 Score %s " % r2)
#     print("Confusion Matrix:")
#     confusion_matrix(y_test, y_pred)
    
    print("Confusion Matrix:")
    # Confusion Matrix for evaluaiton of results
    cnf_matrix_log = confusion_matrix(y_test, y_pred)
    class_names=['Active','Churned'] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Blues" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
    return auc_score, accuracy, precision, recall, f1


# In[49]:


# data = pd.read_csv('https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.csv', encoding = 'latin',  index_col = False)
data = pd.read_csv('nCoV2019/latest_data/latestdata.csv', encoding = 'Latin')


# In[50]:


data.shape


# # Pre-processing

# In[51]:


df = data.drop(columns=['longitude', 'latitude', 'geo_resolution', 'source', 'notes_for_discussion',
                       'admin_id', 'admin1', 'admin2', 'admin3', 'data_moderator_initials', 
                       'sequence_available', 'city', 'province'])
df = df.rename(columns={"lives_in_Wuhan": "con_to_Wuhan"})


# In[52]:


df['sex'] = df['sex'].replace('female', 1)
df['sex'] = df['sex'].replace('male', 0)

df['con_to_Wuhan'] = df['con_to_Wuhan'].replace('yes', 1)
df['con_to_Wuhan'] = df['con_to_Wuhan'].replace('no', 0)
df['con_to_Wuhan'] = df['con_to_Wuhan'].replace('no, work in Wuhan', 1)

df['travel_history_binary'] = df['travel_history_binary'].astype(str) 
df['chronic_disease_binary'] = df['chronic_disease_binary'].astype(str)

df['travel_history_binary'] = df['travel_history_binary'].replace('True', 1)
df['travel_history_binary'] = df['travel_history_binary'].replace('False', 0)

df['chronic_disease_binary'] = df['chronic_disease_binary'].replace('False', 0)
df['chronic_disease_binary'] = df['chronic_disease_binary'].replace('True', 1)

df['travel_history_binary'] = df['travel_history_binary'].replace('nan', np.nan)

df = df.set_index('ID')


# In[53]:


df.isnull().sum().sort_values(ascending = False).plot(kind='bar')


# In[54]:


df['chronic_disease'] = df['chronic_disease'].replace('nan', -1)
df['chronic_disease'] = df['chronic_disease'].replace(r'^\s*$', -1, regex=True)
df['chronic_disease'] = df['chronic_disease'].replace(np.nan, -1)

delimiters = ";", ",", ":"
no_chronic_disease = []
regexPattern = '|'.join(map(re.escape, delimiters))

for index, row in df.iterrows():
    if( row.chronic_disease is -1 or 'http' in row.chronic_disease):
        no_chronic_disease.append(-1)
    else:
        no_chronic_disease.append(len(re.split(regexPattern, row.chronic_disease)))

df['Number of chronic diseases'] =  no_chronic_disease

df['Number of chronic diseases'].value_counts()


# In[55]:


df['Hypertension'] = df['chronic_disease'].str.contains('hypertension|hypertensive')
df['Hypertension'] = df['Hypertension'].apply(lambda x: 1 if x == True else 0)

df['Diabetes'] = df['chronic_disease'].str.contains('diabetes')
df['Diabetes'] = df['Diabetes'].apply(lambda x: 1 if x == True else 0)

df['Heart'] = df['chronic_disease'].str.contains('heart|coronary|bypass|cardiac|cardio')
df['Heart'] = df['Heart'].apply(lambda x: 1 if x == True else 0)

df['Kidney'] = df['chronic_disease'].str.contains('kidney|Kidney')
df['Kidney'] = df['Kidney'].apply(lambda x: 1 if x == True else 0)

df.drop(columns = ['chronic_disease'], inplace = True)


# In[56]:


df = pd.get_dummies(df, columns=['country'])


# In[57]:


df['outcome'] = df['outcome'].replace(['died', 'Death', 'Deceased', 'Died', 'dead', 'death', 'Dead'], 'Died')

df['outcome'] = df['outcome'].replace(['Critical condition', 'critical condition', 'severe illness', 'severe', 
                                       'critical condition, intubated as of 14.02.2020', 'unstable'], 'Critical')

df['outcome'] = df['outcome'].replace(['recovered', 'Recovered', 'recovering at home 03.03.2020', 'not hospitalized',
                                      'Discharged', 'Discharged from hospital', 'released from quarantine', 
                                       'discharge', 'discharged', 'Alive'],'Recovered')

df['outcome'] = df['outcome'].replace(['stable', 'stable condition', 'Stable', 'Under treatment', 
                                       'Symptoms only improved with cough. Currently hospitalized for follow-up.',
                                      'Receiving Treatment', 'treated in an intensive care unit (14.02.2020)'], 'Stable')

df = df[~df.outcome.str.contains("http", na = False)]
df = df[~df.outcome.str.contains("gov", na = False)]
df = df[~df.outcome.str.contains("State", na = False)]


# In[58]:


df['outcome'].value_counts()


# In[59]:


df.shape


# In[60]:


df['outcome'] = df['outcome'].replace(['Stable', 'Recovered'], 1)
df['outcome'] = df['outcome'].replace(['Died', 'Critical'], 0)


# In[61]:


df['outcome'].value_counts()


# In[62]:


df = df.dropna(subset =['outcome'])


# In[63]:


# Cleaning age column

#function to return age categories
def bucket_age(age):
    if(age == -1):
          return 'unknown'
    if(age <= 9):
        return 'child'
    if(age <=19):
        return 'adolescent'
    if(age <=59):
        return 'adult'
    if (age <=100):
        return 'senior'
    else:
        return 'unknown'
      

# replacing null values
df['age'] = df['age'].replace('nan', -1)

# creating new column
age_category = []

for index, row in df.iterrows():

    if(isinstance(row.age, float)):
         age_category.append(bucket_age(row.age))
            
    elif( row.age is -1 ):
        age_category.append(bucket_age(-1))
    
    elif('.' in row.age):
        age = (float(row.age))
        age_category.append(bucket_age(age))
        
    elif('-' in row.age):
        ages= (row.age).split('-')
                
        if(ages[0] and ages[1]):
            age1= (float(ages[0]))
            age2= (float(ages[1]))
            mean = (age1 + age2) / 2
        else:
            mean= (float(ages[0]))  
        
        age_category.append(bucket_age(mean))
    
    elif('months' in row.age or 'month' in row.age or 'weeks' in row.age):
        age_category.append(bucket_age(0))
        
    elif(len(row.age) == 4):
        age_category.append(bucket_age(-1))
        
    else:
        age_category.append(bucket_age(int(row.age)))

df['age_category'] = age_category


# In[64]:


df['age_category'].value_counts()


# In[65]:


df = pd.get_dummies(df, columns=['age_category'])


# In[66]:


print(df.shape)
print(df.head())


# # Visualisation

# In[67]:


#  plot correlation's matrix to explore dependency between features 

def plot_correlation(data):

    # init figure size
#     rcParams['figure.figsize'] = 15, 20
    fig = plt.figure(figsize=(15, 15))
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()


# plot correlation & densities
plot_correlation(df[['outcome','sex','con_to_Wuhan','age_category_adult','age_category_adolescent','age_category_child','age_category_senior',
                    'Number of chronic diseases','Hypertension','Diabetes','Heart', 'Kidney']])


# In[68]:


# Check correlation between death and sex

#Get all data where outcome was death
deaths_critical = df.query('outcome == "0.0"')
recovered_stable = df.query('outcome == "1.0"')
sns.countplot(data=deaths_critical, x = 'sex')


# In[69]:


sns.countplot(data=deaths_critical, x = 'travel_history_binary')


# In[70]:


sns.countplot(data=deaths_critical, x = 'con_to_Wuhan')


# In[71]:


chronic_diesease = sns.countplot(data=df, x = 'chronic_disease_binary', hue='outcome')

plt.legend(title='Chronic diseases vs utcome', labels=['death/critical', 'recovered/stable'])
plt.show(chronic_diesease)


# In[72]:


# Different chronic diseases correlation to patient death or critical situation

Hypertension = len((deaths_critical.query('Hypertension == "1"')).axes[0])
Diabetes =len((deaths_critical.query('Diabetes == "1"')).axes[0])
Heart = len((deaths_critical.query('Heart == "1"')).axes[0])
Kidney = len((deaths_critical.query('Kidney == "1" ')).axes[0])

Hypertension = (Hypertension)
Diabetes = (Diabetes)
Heart = (Heart)
Kidney = (Kidney)


fig, ax = plt.subplots()
index = np.arange(1)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index - bar_width, Hypertension, bar_width,
alpha=opacity,
color='peru',
label='Hypertension')

rects2 = plt.bar(index, Diabetes, bar_width,
alpha=opacity,
color='peachpuff',
label='Diabetes')

rects3 = plt.bar(index + bar_width, Heart, bar_width,
alpha=opacity,
color='sandybrown',
label='Heart')

rects4 = plt.bar(index + (2*bar_width) , Kidney, bar_width,
alpha=opacity,
color='saddlebrown',
label='Kidney')

plt.title('Chronic diseases related to Deaths or Critical situation')
plt.xticks([])
plt.legend()
plt.xlabel('Chronic diseases')
plt.tight_layout()
plt.show()


# In[73]:



male_death_critical = len((deaths_critical.query('sex == "0"')).axes[0])
female_deaths_critical =len((deaths_critical.query('sex == "1"')).axes[0])

male_recovered_stable = len((recovered_stable.query('sex == "0"')).axes[0])
female_recovered_stable =len((recovered_stable.query('sex == "1"')).axes[0])


objects = ('Male', 'Female')
deaths_critical = (male_death_critical, female_deaths_critical)
recovered_stable = (male_recovered_stable, female_recovered_stable)

fig, ax = plt.subplots()
index = np.arange(2)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index - bar_width, deaths_critical, bar_width,
alpha=opacity,
color='maroon',
label='Deaths/Critical')

rects2 = plt.bar(index, recovered_stable, bar_width,
alpha=opacity,
color='olivedrab',
label='Recovered/Stable')


plt.title('Health Status by gender')
plt.xticks(index , objects)
plt.legend()

plt.tight_layout()
plt.show()


# In[74]:



adolescent_death_critical = len((deaths_critical.query('age_category_adolescent == "1"')).axes[0])
adult_deaths_critical =len((deaths_critical.query('age_category_adult == "1"')).axes[0])
child_deaths_critical = len((deaths_critical.query('age_category_child == "1"')).axes[0])
senior_deaths_critical =len((deaths_critical.query('age_category_senior == "1"')).axes[0])

adolescent_recoverd_stable = len((recovered_stable.query('age_category_adolescent == "1"')).axes[0])
adult_recoverd_stable =len((recovered_stable.query('age_category_adult == "1"')).axes[0])
child_recoverd_stable= len((recovered_stable.query('age_category_child == "1"')).axes[0])
senior_recoverd_stable =len((recovered_stable.query('age_category_senior == "1"')).axes[0])

deaths_critical = (child_deaths_critical, adolescent_death_critical,adult_deaths_critical,senior_deaths_critical)
recovered_stable = (child_recoverd_stable, adolescent_recoverd_stable, adult_recoverd_stable, senior_recoverd_stable)
objects = ('child', 'adolescent', 'adults', 'senior')

fig, ax = plt.subplots()
index = np.arange(4)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index - bar_width, deaths_critical, bar_width,
alpha=opacity,
color='maroon',
label='death/critical')

rects2 = plt.bar(index, recovered_stable, bar_width,
alpha=opacity,
color='olivedrab',
label='recovered/stable')

plt.title(' Outcomes vs Age')
plt.xticks(index , objects)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:



male_adolescent = len((deaths_critical.query('age_category_adolescent == "1" and sex == "0"')).axes[0])
male_adult = len((deaths_critical.query('age_category_adult == "1" and sex == "0"')).axes[0])
male_child = len((deaths_critical.query('age_category_child == "1" and sex == "0"')).axes[0])
male_senior =len((deaths_critical.query('age_category_senior == "1" and sex == "0"')).axes[0])

female_adolescent = len((deaths_critical.query('age_category_adolescent == "1" and sex == "1"')).axes[0])
female_adult = len((deaths_critical.query('age_category_adult == "1" and sex == "1"')).axes[0])
female_child = len((deaths_critical.query('age_category_child == "1" and sex == "1"')).axes[0])
female_senior =len((deaths_critical.query('age_category_senior == "1" and sex == "1"')).axes[0])


objects = ('child', 'adolescent', 'adult', 'senior')

male = (male_child, male_adolescent, male_adult, male_senior)
female = (female_child, female_adolescent, female_adult, female_senior)


fig, ax = plt.subplots()
index = np.arange(4)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index - bar_width, male, bar_width,
alpha=opacity,
color='saddlebrown',
label='male')

rects2 = plt.bar(index, female, bar_width,
alpha=opacity,
color='peachpuff',
label='female')


plt.title('Outcome vs gender along with age')
plt.xticks(index , objects)
plt.legend()

plt.tight_layout()
plt.show()


# # Normalise Feature Set

# In[ ]:


def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y


# # XGBoost Classifier
# With Nan

# In[37]:


X = df.drop(['outcome'], axis = 1)
X = X.select_dtypes(include=['int', 'float64', 'uint8'])
y = df['outcome']

x = X.apply(pd.to_numeric)
X = X.applymap(lambda x: np.log(x+1) if x >= 0 else -np.log(-x+1))
X = normalize(X)


# In[65]:


import sklearn
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    shuffle = True,
                                                    random_state=0)
params = {'random_state': [0],
                 'eta': [0.02, 0.03, 0.05],
                 'gamma': [2, 4, 6],
                 'max_depth': [2, 3, 4],
                 'scale_pos_weight': [0.5, 0.9, 1],
                 'n_estimators': [200, 300, 400]}

clf = GridSearchCV(xgb.XGBClassifier(), params, cv = 5)
clf.fit(X_train, y_train)


# In[ ]:


best_grid #Best combination of the parameters


# In[67]:


# fit the model with the best grid otained from the gridsearch - the reason why i have duplicated this 
# is so i can plot the feature importance easily. The grid search does not have this option to do it easily 
# (at least from what i could find)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    shuffle = True,
                                                    random_state=0)

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, eta=0.02, gamma=2, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=200, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

clf.fit(X_train, y_train)
       
from xgboost import plot_importance
plot_importance(clf._Booster)
plt.show()

y_train_pred =    clf.predict(X_train)
y_test_pred  =    clf.predict(X_test)
y_train_pred_pr = clf.predict_proba(X_train)[:,-1]
y_test_pred_pr =  clf.predict_proba(X_test)[:,-1]

auc_score, accuracy, precision, recall, f1 = test_results(y_test, y_test_pred, y_test_pred_pr)


# # KNN Classifier
# Fillna(-1)

# In[75]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

X = df.drop(['outcome'], axis = 1)
X = X.select_dtypes(include=['int', 'float64', 'uint8'])
y = df['outcome']

X = X.apply(pd.to_numeric)
X = X.applymap(lambda x: np.log(x+1) if x >= 0 else -np.log(-x+1))
X = normalize(X)
X = X.fillna(-1)


# In[76]:


import sklearn

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.245,
                                                    shuffle = True,
                                                    random_state=4)
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train,y_train)
y_train_pred =    clf.predict(X_train)
y_test_pred  =    clf.predict(X_test)
y_train_pred_pr = clf.predict_proba(X_train)[:,-1]
y_test_pred_pr =  clf.predict_proba(X_test)[:,-1]

auc_score, accuracy, precision, recall, f1 = test_results(y_test, y_test_pred, y_test_pred_pr)

#Cross validation

#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(X, y)



print('*****************************CROSS VALIDATION****************************** \n')
#check top performing n_neighbors value
print('Top performing n_neighbors value:', knn_gscv.best_params_)

#check mean score for the top performing value of n_neighbors
print('mean score for the top performing value of n_neighbors:', knn_gscv.best_score_)


# # KNN using fillna() Visualization 

# In[77]:


from matplotlib.colors import ListedColormap
import pylab as pl

# Create color maps for 5-class classification problem
label_class = ['Critical/Died', 'Recovered/Stable']
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'  ])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

X_train = np.array(X_train)
x_min, x_max = X_train[:, 1].min() - .1, X_train[:, 1].max() + .1
y_min, y_max = X_train[:, 4].min() - .1, X_train[:, 4].max() + .1

xx, yy = np.meshgrid(np.linspace(x_min, x_max,19),
                     np.linspace(y_min, y_max,19))

Z= y_test_pred
# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure()
pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
scatter = pl.scatter(X_train[:, 1], X_train[:, 4], c=y_train_pred, cmap=cmap_bold)
# pl.legend(handles=scatter.legend_elements()[0], labels=label_class, frameon=False)
pl.colorbar(label=label_class)
# pl.title('Age and connection to Wuhan effecting health')
pl.xlabel('connection to wuhan')
pl.ylabel('No of chronic diseases')
pl.axis('tight')


# In[78]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure(1, figsize=(20, 15))
ax = Axes3D(fig, elev=48, azim=134)

# surf= ax.plot_surface(xx, yy, Z, cmap=cmap_light,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)


ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred,
           cmap=cmap_bold, edgecolor='k', s = X_train[:, 3]*50)

for name, label in [('Died/Critical', 0), ('stable/Recovered', 1)]:
    ax.text3D(X_train[y_train_pred == label, 0].mean(),
              X_train[y_train_pred == label, 1].mean(),
              X_train[y_train_pred == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)

ax.set_title("3D KNN", fontsize=40)
ax.set_xlabel("sex ", fontsize=25)
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("connection to wuhan", fontsize=25)
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("chronic diseases", fontsize=25)
ax.w_zaxis.set_ticklabels([])

plt.show()


# # Using KNN Imputer
# 

# In[79]:


from fancyimpute import KNN

X = df.drop(['outcome'], axis = 1)
X = X.select_dtypes(include=['int', 'float64', 'uint8'])
y = df['outcome']

X = X.apply(pd.to_numeric)
X = X.applymap(lambda x: np.log(x+1) if x >= 0 else -np.log(-x+1))
X = normalize(X)

df_knn_impute = KNN(k=2).fit_transform(X)


# # KNN using fancy impute

# In[80]:


#Fancy Impute

X = df_knn_impute
y = df['outcome']


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.245,
                                                    shuffle = True,
                                                    random_state=4)
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train,y_train)
y_train_pred =    clf.predict(X_train)
y_test_pred  =    clf.predict(X_test)
y_train_pred_pr = clf.predict_proba(X_train)[:,-1]
y_test_pred_pr =  clf.predict_proba(X_test)[:,-1]

auc_score, accuracy, precision, recall, f1 = test_results(y_test, y_test_pred, y_test_pred_pr)

#Cross validation

#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(X, y)



print('*****************************CROSS VALIDATION****************************** \n')
#check top performing n_neighbors value
print('Top performing n_neighbors value:', knn_gscv.best_params_)

#check mean score for the top performing value of n_neighbors
print('mean score for the top performing value of n_neighbors:', knn_gscv.best_score_)


# In[81]:



#KNN using fancy impute Visualization

X_train = np.array(X_train)
x_min, x_max = X_train[:, 1].min() - .1, X_train[:, 1].max() + .1
y_min, y_max = X_train[:, 4].min() - .1, X_train[:, 4].max() + .1

xx, yy = np.meshgrid(np.linspace(x_min, x_max,19),
                     np.linspace(y_min, y_max,19))

Z= y_test_pred
# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure()
pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
scatter = pl.scatter(X_train[:, 1], X_train[:, 4], c=y_train_pred, cmap=cmap_bold)
# pl.legend(handles=scatter.legend_elements()[0], labels=label_class, frameon=False)
pl.colorbar(label=label_class)
# pl.title('Age and connection to Wuhan effecting health')
pl.xlabel('connection to wuhan')
pl.ylabel('No of chronic diseases')
pl.axis('tight')


# In[ ]:




