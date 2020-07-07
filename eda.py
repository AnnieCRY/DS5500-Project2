import plotly
import json
import plotly.graph_objects as go
import pandas
import seaborn as sns
import io
import matplotlib.pyplot as plt
import base64
from sklearn.model_selection import StratifiedKFold,cross_validate,train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
    
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def classimbalance(x):
    x_val = ['Action Denied','Action Accepted']
    colors = ['lightslategray','crimson'] 

    x_0 = len(x[x['ACTION'] == 0])
    x_1 = len(x[x['ACTION'] == 1])
    fig = go.Figure([go.Bar(x=x_val, y = [x_0,x_1], text=[x_0,x_1],textposition='auto',  marker_color=colors)])
    fig.update_layout(title_text='Count of Action Denied and Action Accepted!')
    a = fig.to_json()
    return json.loads(json.dumps(a,cls=plotly.utils.PlotlyJSONEncoder))

def uniqueCategories(x):
    unique = pandas.DataFrame([(col,x[col].nunique()) for col in x.columns], 
                           columns=['Columns', 'Count of Category'])
    colors = ['lightslategray'] 
    fig = plotly.graph_objects.Figure([go.Bar(x=unique.Columns, y = unique['Count of Category'], text=unique['Count of Category'],textposition='auto',  marker_color=colors)])
    fig.update_layout(title_text='Unique categories - Train Data Set!')
    a = fig.to_json()
    return json.loads(json.dumps(a,cls=plotly.utils.PlotlyJSONEncoder))


def correlation(x):
    plotlist = []
    fig = plt.figure(figsize=(30,20))
    fig = sns.heatmap(x.corr(),annot=True, cmap ='viridis', linewidth = 1)
    buf = io.BytesIO()
    fig.figure.savefig(buf, format = "png", dpi = 600, box_inches = "tight", pad_inches = 0)
    plotlist.append(buf.getvalue())

    return plotlist

def columHistograms(df):
    plotlist = []
    fig = plt.figure()
    fig = plt.figure(figsize=(30,20))
    for i in range(1, len(df.columns)):
        fig =plt.subplot(5,2,i)
        fig = plt.hist(df[df.columns[i]])
        fig = plt.xlabel(df.columns[i])
        fig = plt.ylabel("Frequency")
    buf = io.BytesIO()
    fig.figure.savefig(buf, format = "png", dpi = 600, box_inches = "tight", pad_inches = 0)
    plotlist.append(buf.getvalue())

    return plotlist

def featurimportance(train_data):

    X = train_data.iloc[:,1:]
    y = train_data.iloc[:,:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    data1 = pandas.DataFrame(importance, columns = ['Importance'])
    data1['Columns'] = train_data.iloc[:,1:].columns
    data1 = data1.sort_values('Importance')

    fig = go.Figure([go.Bar(x=data1['Columns'], y=data1['Importance'])])
    fig.update_layout(title_text='Feature Importance using Random Forest')
    a = fig.to_json()
    
    return json.loads(json.dumps(a,cls=plotly.utils.PlotlyJSONEncoder))







def eda_train(train_data):

    graphs = {}
    class_imbalance = classimbalance(train_data)
    uniqueCategory = uniqueCategories(train_data)
    correlationMatrix = correlation(train_data)
    columnHisto = columHistograms(train_data)
    featureImport = featurimportance(train_data)

    graphs['Class Imbalance'] = class_imbalance
    graphs['Unique Categories'] = uniqueCategory
    graphs['Feature Importance'] = featureImport
    graphs['Correlation Matrix'] = base64.b64encode(correlationMatrix[0]).decode('utf-8')
    graphs['Histograms'] = base64.b64encode(columnHisto[0]).decode('utf-8')

    return graphs

def eda_over(train_data, graphs):

    class_imbalance = classimbalance(train_data)
    uniqueCategory = uniqueCategories(train_data)
    correlationMatrix = correlation(train_data)
    columnHisto = columHistograms(train_data)

    graphs['Class Imbalance Solved - OverSampling'] = class_imbalance
    graphs['Unique Categories - OverSampling'] = uniqueCategory
    graphs['Histograms-OverSampling'] = base64.b64encode(columnHisto[0]).decode('utf-8')

    return graphs

