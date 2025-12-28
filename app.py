import streamlit as st
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from plotly import express as px
import duckdb

import joblib
import json

st.set_page_config(layout='wide')

MODEL_PATH = 'KNeighborsClassifier.joblib'
SCALER_PATH = 'scaler.joblib'
RAW_DATA_PATH = 'insurance_us.csv'
X_DATA_PATH = 'x_test.pkl'
Y_DATA_PATH = 'y_test.pkl'
METADATA_PATH = 'metadata.json'
FULL_DATA_PATH = 'insurance_us.csv'

EMPTY_DF = {
    'gender':['Male'],
    'age':[18],
    'income':[1],
    'family_members':[0]
}

if 'df' not in st.session_state:
    st.session_state.df = EMPTY_DF

@st.cache_data
def load_model_and_scaler(model_path=MODEL_PATH,scaler_path=SCALER_PATH):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

@st.cache_data
def load_raw_data(raw_data_path=RAW_DATA_PATH):
    df = pd.read_csv(raw_data_path)
    return df

@st.cache_data
def data_for_testing(x_path=X_DATA_PATH,y_path=Y_DATA_PATH):
    X = pd.read_pickle(x_path)
    Y = pd.read_pickle(y_path)

    return X, Y

@st.cache_data
def load_metadata(path=METADATA_PATH):
    with open(METADATA_PATH,'r',encoding='utf-8') as f:
        data = json.load(f)

    return data

@st.cache_data
def plot_scores(y_true, y_pred):
    score_f1 = f1_score(y_true=y_true,y_pred=y_pred)
    score_precision = precision_score(y_true=y_true,y_pred=y_pred,zero_division=0)
    score_recall = recall_score(y_true=y_true,y_pred=y_pred)
    cm = pd.DataFrame(data=confusion_matrix(y_true=y_true,y_pred=y_pred))

    scores = pd.DataFrame(data=[[score_f1,score_precision,score_recall]],index=['Scores'],columns=['F1','Precision','Recall'])
    st.dataframe(scores.style.format("{:.2%}"))

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x='Predictions',y='Real Values'),x=['No Insurance (0)','Insurance (1)'],y=['No Insurance (0)','Insurance (1)'],
        color_continuous_scale='Blues'
        )
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig)

def model_performance(model,x,y,metadata,title):
    st.subheader(f'{title}')
    st.markdown(f'Hyperparameters: `n_neighbors={metadata}`')
    prediction = model.predict(x)
    plot_scores(y_true=y,y_pred=prediction)

def entry_form():
    st.subheader('Enter your data to check insurance eligibility at Sure Tomorrow')
    st.info("**Note:** The model prioritizes age as a risk factor, based on the 65% historical correlation detected in the exploratory analysis.")
    st.info("**Annual Income:** Please provide your yearly gross income (e.g., 50000).")
    data = st.data_editor(
        data=st.session_state.df,
        hide_index=True,
        num_rows='fixed',
        column_config={
            'gender':st.column_config.SelectboxColumn(options=['Male','Female']),
            'age':st.column_config.NumberColumn(min_value=18,max_value=65,step=1,help='Range (18 - 65)'),
            'income':st.column_config.NumberColumn(min_value=0,max_value=100000,step=1,format='dollar',help='Please enter your total gross income per year.'),
            'family_members':st.column_config.NumberColumn(min_value=0,max_value=6,step=1,help='Range (0 - 6)')
        })

    predict = st.button(label='Can I Get An Insurance?',width='stretch',type='primary')

    if predict:
        df = pd.DataFrame(data=data)

        if df['gender'].isnull().sum() > 0:
            
            st.info('Data Is Missing')
        
        elif df.empty:
            
            st.info('Data Is Missing')
        
        else:
            
            df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
            return df

def scalate_data(data,scaler):
    labels = ['age','income','family_members']
    data[labels] = scaler.transform(data[labels])
    return data

def predict(model,x_matrix):
    prediction = model.predict(x_matrix)
    
    data = pd.DataFrame(data=prediction,columns=['Prediction'])
    data['Prediction'] = data['Prediction'].apply(lambda x: 'Posible Insurance Available' if x == 1 else 'Insurance not available for this profile')

    st.subheader(f'Result')
    st.dataframe(data=data,hide_index=True)

def features_correlation(full_data_path = FULL_DATA_PATH):

    data = pd.read_csv(full_data_path)

    # Consulta Condicional SQL, creamos la columna objetivo "Insurance"
    query = duckdb.query(f"""SELECT *, ("Insurance benefits" > 0)::INTEGER AS Insurance FROM data""").df()
    
    # Creacion de una matriz de correlacion incluyendo solo columnas numericas para mas seguridad:
    correlation_matrix = query.select_dtypes(include=['number']).corr()
    # Seleccion de la correlacion "unicamente" con respecto a "Insurance"
    insurance_correlation = correlation_matrix[['Insurance']].sort_values(by='Insurance',ascending=False)
    insurance_correlation = insurance_correlation.drop(labels='Insurance benefits',axis=0)

    cm = pd.DataFrame(data=[insurance_correlation['Insurance']],columns=insurance_correlation.index)

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x='Variables Correlation to Insurance Acquisition',y='Scale'),x=insurance_correlation.index,y=['Correlation'],
        color_continuous_scale='Greens'
        )
    fig.update_xaxes(side="bottom")

    st.plotly_chart(fig)

# -------- LOADED FUNCTIONS
model, scaler = load_model_and_scaler()
data = load_raw_data()
x, y = data_for_testing()
metadata = load_metadata()

# -------- INTERFACE
st.title('Insurance Application Model Classifier')
with st.expander('Raw Data'):
    st.dataframe(data=data,hide_index=True)

col1, col2 = st.columns([1.5,2.5])
with col1:
    model_performance(model=model,x=x,y=y,metadata=metadata,title='KNeighborsClassifier')

with col2:
    df = entry_form()
    if isinstance(df, (pd.Series,pd.DataFrame)):
        x_matrix = scalate_data(data=df,scaler=scaler)
        with st.expander('Scaled Data'):
            st.dataframe(x_matrix)
        predict(model=model,x_matrix=x_matrix)

    features_correlation()