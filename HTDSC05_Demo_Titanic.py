import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv('train.csv')

st.title('Data Science')
st.subheader('Titanic project app')

# Data pre-processing
data['Sex'] = data['Sex'].map(lambda x: 0 if x=='male' else 1)
data = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']]
data = data.dropna()

X = data.drop(['Survived'], axis=1)
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale data
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# Build model
model = LogisticRegression()
model.fit(train_features, y_train)

# Evaluation
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_predict = model.predict(test_features)
confusion = metrics.confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]
metrics.classification_report(y_test, y_predict)

# Calc ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_predict)

# Calc AUC
auc = metrics.roc_auc_score(y_test, y_predict)

# PART 2:
menu = ['Overview', 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)
if choice == menu[0]:
    st.subheader(menu[0])

    st.write('training set (train.csv)')
    st.write('testing set (test.csv)')
elif choice == menu[1]:
    st.subheader(menu[1])

    st.write('#### Data Preprocssing')
    st.table(data.head(5))
    st.write('#### Build model and Evaluation')
    st.write('Train set score: {}'.format(round(train_score, 2)))
    st.write('Test set score: {}'.format(round(test_score, 2)))
    st.write('#### Confusion Matrix')
    st.table(confusion)
    st.write(metrics.classification_report(y_test, y_predict))
    st.write('##### AUC: %.3f' %auc)

    st.write('#### Visualization')
    fig, ax = plt.subplots()
    ax.bar(['False Negaive', 'True Negative', 'True Positive', 'False Positive'], [FN, TN, TP, FP])
    st.pyplot(fig)

    st.write('ROC curve')
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(fpr, tpr, marker='.')
    ax1.set_title('ROC curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    st.pyplot(fig1)
else:
    st.subheader(menu[2])
    
    st.write('#### New Prediction')

    name = st.text_input('Name:')
    sex_ = st.radio('Sex', options=['Male', 'Female'])
    sex = 0 if sex_ == 'Male' else 1

    age = st.slider('Age', 1, 100, 1)

    p_class_list = np.sort(data['Pclass'].unique())
    pclass = st.selectbox('Pclass', options=p_class_list)

    max_sibsp = max(data['SibSp'])
    sibsp = st.slider('Siblings', 0, max_sibsp, 1)

    max_parch = max(data['Parch'])
    parch = st.slider('Parch', 0, max_parch, 1)

    max_fare = round(max(data['Fare'])+10, 2)
    fare = st.slider('Fare', 0.0, max_fare, 0.1)

    submit = st.button('Submit')

    if submit:
        # Make new prediction
        new_data = scaler.transform([[sex, age, pclass, sibsp, parch, fare]])
        prediction = model.predict(new_data)
        predict_probability = model.predict_proba(new_data)

        if prediction[0] == 1:
            st.subheader('Passenger {} would have survived with a probability of {}%'
            .format(name, round(predict_probability[0][1]*100, 2)))
        else:
            st.subheader('Passenger {} would NOT have survived with a probability of {}%'
            .format(name, round(predict_probability[0][0]*100, 2)))