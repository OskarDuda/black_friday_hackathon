import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

df_raw= pd.read_csv('train.csv')
df=df_raw.copy()

n = 20
df['Purchase_bins'] = pd.qcut(df['Purchase'], n, labels=range(n))

var_mod = ['User_ID','Product_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
print('\n')


#Feature Engineering
var_mod_cat = ['Gender_and_City_Category', #0
                 'Gender_and_Age_and_City_Category', #1
                 'Age_and_City_Category', #2
                 'Gender_and_Age_and_Marital', #3
                 'Group'] #4

df[var_mod_cat[0]] = df['Gender']+3*df['City_Category']
df[var_mod_cat[1]] = len(df['Age'].value_counts())*len(df['City_Category'].value_counts())*df['Gender']+len(df['Age'].value_counts())*df['City_Category']+df['Age']
df[var_mod_cat[2]] = len(df['City_Category'].value_counts())*df['Age']+df['City_Category']
df[var_mod_cat[3]] = len(df['Age'].value_counts())*len(df['Marital_Status'].value_counts())*df['Gender']+len(df['Age'].value_counts())*df['Marital_Status']+df['Age']


#df[var_mod_num[4]] = np.zeros(len(df))
n = 8820
df[var_mod_cat[4]] = n/2*df['Marital_Status'] + n/(2*5) * df['Stay_In_Current_City_Years'] + n/(2*5*3) * df['City_Category'] + n/(2*5*3*21) * df['Occupation'] + n/(2*5*3*21*2) * df['Gender'] + df['Age']     


for i in var_mod_cat:
    df[i] = le.fit_transform(df[i])
print('\n')

var_mod_num = ['Age_Lower', #0
                 'Age_Upper', #1
                 'Age_to_Stay', #2
                 'Group_freq'] #3


df[var_mod_num[0]] = int(df_raw['Age'][5].split('-')[0])
df[var_mod_num[1]] = int(df_raw['Age'][5].split('-')[1])

df[var_mod_num[2]] = df[var_mod_num[1]]/df['Stay_In_Current_City_Years']
df[var_mod_num[2]] = pd.to_numeric(np.int64(df[var_mod_num[2]] ))

tmp = df[var_mod_cat[4]].value_counts()
df[var_mod_num[3]] = tmp[df[var_mod_cat[4]]].values/sum(tmp)




#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome, k):
    model.fit(data[predictors], data[outcome])
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    
    #K-fold validation
    kf = KFold(data.shape[0], n_folds=k)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
        
    
    return([accuracy, error])
    model.fit(data[predictors],data[outcome])

##Selecting clients who bought the three most popular products
#df = df[(df['Product_Category_1'] == 1 ) | (df['Product_Category_1'] == 5 ) | (df['Product_Category_1'] == 8 ) ]

#Select k for k-fold cross validation
k=5
outcome_var = 'Purchase_bins'
#bool_index = (df['Product_Category_1'].apply(lambda x: x in [1]))
bool_index = (df==df)

##Logistic regression    
#outcome_var = 'Product_Category_1'
#model = LogisticRegression()
#predictor_var = ['Gender','Age']
#print('Logistic regression predicting {1} based on {2}'.format('',outcome_var,predictor_var[:]))
#classification_model(model,df, predictor_var, outcome_var, k)
#print('\n')
# 
##Decision tree   
#outcome_var = 'Product_ID'
#model = DecisionTreeClassifier()
#bool_index = (df['Product_Category_1'].apply(lambda x: x in [5,1,8]))
#predictor_var = ['Occupation','Group_freq','Product_Category_1','Age_to_Stay','Stay_In_Current_City_Years']
#print('Random forest predicting {0} based on {1}'.format(outcome_var,predictor_var[:]))
#
#[accuracy, error] = classification_model(model, df[bool_index], predictor_var, outcome_var, k)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy))
#print ("Cross-Validation Score : %s" %"{0:.3%}".format(np.mean(error)))
#      
#featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print('{0}\n'.format(featimp))
 

##Random forest   
#model = RandomForestClassifier()
#predictor_var = var_mod+var_mod_cat+var_mod_num
#print('Random forest predicting {0} based on {1}'.format(outcome_var,predictor_var[:]))
#
#[accuracy, error] = classification_model(model, df[bool_index], predictor_var, outcome_var, k)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy))
#print ("Cross-Validation Score : %s" %"{0:.3%}".format(np.mean(error)))
#      
#featimp1 = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print('{0}\n'.format(featimp1))

##Random forest with improved predictor variables
#model = RandomForestClassifier()
#predictor_var = featimp1.index[:6]
#print('Improve Random forest predicting {0} based on {1}'.format(outcome_var,predictor_var[:]))
#[accuracy, error] = classification_model(model,df[bool_index], predictor_var, outcome_var, k)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy))
#print ("Cross-Validation Score : %s" %"{0:.3%}".format(np.mean(error)))     
#
#featimp2 = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print('{0}\n'.format(featimp2))

model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
predictor_var = ['Occupation','Group_freq','Product_Category_3','Product_Category_2','Product_Category_1']
print('Improve Random forest predicting {0} based on {1}'.format(outcome_var,predictor_var[:]))
[accuracy, error] = classification_model(model,df[bool_index], predictor_var, outcome_var, k)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))
print ("Cross-Validation Score : %s" %"{0:.3%}".format(np.mean(error)))     

#featimp2 = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print('{0}\n'.format(featimp2))
