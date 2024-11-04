
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import keras
import numpy as np
import tensorflow as tf
import joblib
from dotenv import load_dotenv 
import os
import psycopg2

env_path="/Users/laurasofiahurtadourrego/Downloads/proyecto.env"
load_dotenv(dotenv_path=env_path)

# extract env variables
USER=os.getenv('USUARIO')
PASSWORD=os.getenv('PASSWORD')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')

engine = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)
cursor = engine.cursor()

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# cargar archivo de disco
model = keras.models.load_model('/Users/laurasofiahurtadourrego/Downloads/modelo_pls_lda.keras')

file_data = '/Users/laurasofiahurtadourrego/Downloads/bank-full.csv'
df = pd.read_csv(file_data, sep=';')
data = pd.read_csv("/Users/laurasofiahurtadourrego/Downloads/data_dummies.csv", sep=';')
scaler = joblib.load("/Users/laurasofiahurtadourrego/Downloads/escalador.pkl")

valores_validos = {
    "job": ["admin", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
            "blue-collar", "self-employed", "retired", "technician", "services"],
    "marital": ["married", "divorced", "single"],
    "education": ["unknown", "secondary", "primary", "tertiary"],
    "default": ["yes", "no"],
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["unknown", "telephone", "cellular"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["unknown", "other", "failure", "success"]
}

# Convertir a valores binarios
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = data.drop(columns=['y_yes'])
columnas_modelo = X.columns


# Definir un diccionario de mapeo para los trabajos en español
etiquetas_trabajo = {
    'management': 'gestión',
    'technician': 'técnico',
    'entrepreneur': 'emprendedor',
    'blue-collar': 'trabajador azul',
    'unknown': 'desconocido',
    'retired': 'jubilado',
    'admin.': 'administrativo',
    'services': 'servicios',
    'self-employed': 'autónomo',
    'unemployed': 'desempleado',
    'housemaid': 'empleada del hogar',
    'student': 'estudiante'
}

# Reemplazar las etiquetas en el DataFrame
df['job'] = df['job'].replace(etiquetas_trabajo)

# Etiquetas en español para el nivel de educación
etiquetas_educacion = {
    'primary': 'primario',
    'secondary': 'secundario',
    'tertiary': 'terciario',
    'unknown': 'desconocido'
}

# Reemplazar las etiquetas en el DataFrame
df['education'] = df['education'].replace(etiquetas_educacion)

# Definir las columnas continuas
columnas_continuas = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

def predecir_nuevo_cliente(model, scaler, nuevo_cliente, columnas_modelo):
    """
    Predice si un nuevo cliente se suscribirá a un depósito a plazo.

    Args:
        model: El modelo entrenado.
        scaler: El objeto StandardScaler utilizado para escalar los datos.
        nuevo_cliente (dict): Un diccionario con los datos del nuevo cliente.
        columnas_modelo (list): Una lista con los nombres de las columnas del modelo.

    Returns:
        int: 1 si se predice que el cliente se suscribirá, 0 en caso contrario.
    """
    # Convertir el diccionario a un DataFrame de Pandas
    if isinstance(nuevo_cliente, dict):
        nuevo_cliente_df = pd.DataFrame([nuevo_cliente])
    else: # Assume it's already a DataFrame
        nuevo_cliente_df = nuevo_cliente

    # Reorder and align columns of nuevo_cliente_df to match columnas_modelo
    nuevo_cliente_df = nuevo_cliente_df.reindex(columns=columnas_modelo, fill_value=0)

    # Now, proceed with scaling.
    datos_escalados = scaler.transform(nuevo_cliente_df)

    # Realizar la predicción
    prediccion = model.predict(datos_escalados)

    # Devolver 1 si la probabilidad es mayor que 0.5, 0 en caso contrario
    return (prediccion > 0.5).astype(int)

# Diseño de la aplicación
app.layout = html.Div([
    html.H1("Dashboard de Estadísticas Descriptivas", style={'textAlign': 'center'}),

    # Sección de selección de estado civil
    html.H6("Seleccione un Estado Civil"),
    dcc.Dropdown(
        id='marital_status',
        options=[
            {'label': 'Married', 'value': 'Married'},
            {'label': 'Single', 'value': 'Single'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='Married'  # Estado civil predeterminado
    ),
    html.Br(),
    html.H6("Estadísticas:"),
    html.Div(["Edad Promedio:", html.Div(id='output-age')]),
    html.Div(["Saldo Promedio:", html.Div(id='output-balance')]),

    # Resto del dashboard
    html.Div([
        html.Label("Seleccione una variable continua para visualizar su distribución:"),
        dcc.Dropdown(
            id='variable-selector',
            options=[{'label': col, 'value': col} for col in columnas_continuas],
            placeholder="Seleccione una variable continua"
        ),
    ], style={'width': '50%', 'margin': 'auto'}),

    dcc.Graph(id='variable-violin-plot'),

    html.Div([
        html.Label("Gráfico de Suscripciones por Nivel de Educación y Tipo de Trabajo"),
        dcc.Graph(id='education-job-plot')
    ]),

    # Sección de clasificación de usuarios
    html.Div([
        html.H2("Clasificación de Usuario", style={'textAlign': 'center'}),
        html.Div([
            # Inputs para clasificación
            html.Label("Trabajo"),
            dcc.Dropdown(id='job-dropdown', options=[{'label': job, 'value': job} for job in valores_validos['job']], value='admin.'),
            html.Label("Estado Civil"),
            dcc.Dropdown(id='marital-dropdown', options=[{'label': status, 'value': status} for status in valores_validos['marital']], value='single'),
            html.Label("Nivel de Educación"),
            dcc.Dropdown(id='education-dropdown', options=[{'label': edu, 'value': edu} for edu in valores_validos['education']], value='primary'),
            html.Label("¿Tiene crédito previo?"),
            dcc.Dropdown(id='default-dropdown', options=[{'label': option, 'value': option} for option in valores_validos['default']], value='no'),
            html.Label("¿Tiene hipoteca?"),
            dcc.Dropdown(id='housing-dropdown', options=[{'label': option, 'value': option} for option in valores_validos['housing']], value='no'),
            html.Label("¿Tiene préstamo personal?"),
            dcc.Dropdown(id='loan-dropdown', options=[{'label': option, 'value': option} for option in valores_validos['loan']], value='no'),
            html.Label("Método de contacto"),
            dcc.Dropdown(id='contact-dropdown', options=[{'label': contact, 'value': contact} for contact in valores_validos['contact']], value='unknown'),
            html.Label("Mes"),
            dcc.Dropdown(id='month-dropdown', options=[{'label': month, 'value': month} for month in valores_validos['month']], value='jan'),
            html.Label("Resultado previo"),
            dcc.Dropdown(id='poutcome-dropdown', options=[{'label': outcome, 'value': outcome} for outcome in valores_validos['poutcome']], value='unknown'),
            html.Label("Edad"),
            dcc.Input(id='age-input', type='number', value=30, min=18, max=100),
            html.Label("Saldo"),
            dcc.Input(id='balance-input', type='number', value=1000, min=-10000, max=100000),
            html.Label("Día del mes"),
            dcc.Input(id='day-input', type='number', value=15, min=1, max=31),
            html.Button('Clasificar', id='classify-button', n_clicks=0),
        ]),
        html.Div(id='classification-output', style={'margin-top': '20px'})
    ])
])

# Función para crear el gráfico de violín de la variable seleccionada
def update_violin_plot(selected_variable):
    if selected_variable and selected_variable in columnas_continuas:
        # Crear gráfico de violín para la variable seleccionada
        fig = px.violin(
            df,
            y=selected_variable,
            x='y',
            box=True,
            points='all',
            title=f'Distribución de {selected_variable} por Suscripción de Depósito',
            labels={selected_variable: selected_variable, 'y': 'Depósito Suscrito (y)'}
        )

        # Ajustes adicionales para el diseño
        fig.update_layout(
            title_x=0.5,
            xaxis_title='Depósito Suscrito (y)',
            yaxis_title=selected_variable
        )
        return fig
    else:
        # Figura vacía o mensaje de error
        fig = go.Figure()
        fig.add_annotation(
            text="Seleccione una variable continua válida",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig

# Función para crear el gráfico de suscripciones por educación y trabajo
def update_education_job_plot():
    fig = px.histogram(
        df,
        x='education',
        color='job',
        barmode='stack',
        title='Suscripción a Depósito por Nivel de Educación y Tipo de Trabajo',
        category_orders={"education": ['primario', 'secundario', 'terciario', 'desconocido']}
    )

    fig.update_layout(
        xaxis_title='Nivel de Educación',
        yaxis_title='Cantidad de Suscripciones',
        legend_title='Tipo de Trabajo',
        title_x=0.5
    )

    return fig

# Callback para actualizar el gráfico según la variable seleccionada
@app.callback(
    Output('variable-violin-plot', 'figure'),
    [Input('variable-selector', 'value')]
)
def update_output(selected_variable):
    print(f'Selected variable: {selected_variable}')  # Imprimir el valor seleccionado
    try:
        return update_violin_plot(selected_variable)
    except Exception as e:
        print(f'Error: {e}')
        return go.Figure().add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )

# Callback para actualizar el gráfico de educación y trabajo
@app.callback(
    Output('output-age', 'children'),
    Output('output-balance', 'children'),
    Input('marital_status', 'value')
)
def update_output_div(marital_status):
    cursor = engine.cursor()
    query = f"""
    SELECT 
        ROUND(AVG(age), 2) AS average_age,
        ROUND(AVG(balance), 2) AS average_balance
    FROM banco
    WHERE 
        CASE 
            WHEN marital_married = 1 THEN 'Married'
            WHEN marital_single = 1 THEN 'Single'
            ELSE 'Other'
        END = '{marital_status}';
    """
    cursor.execute(query)
    result = cursor.fetchone()
    average_age = result[0] if result else "N/A"
    average_balance = result[1] if result else "N/A"
    return f"{average_age} años", f"{average_balance} unidades monetarias"


@app.callback(
    Output('education-job-plot', 'figure'),
    [Input('variable-selector', 'value')]  # Aunque no se use en este callback, puedes dejarlo si necesitas alguna interacción futura
)
def update_job_plot(selected_variable):
    return update_education_job_plot()

# Callback para la clasificación del usuario
# Callback para la clasificación del usuario
@app.callback(
    Output('classification-output', 'children'),
    [Input('classify-button', 'n_clicks')],
    [
        Input('job-dropdown', 'value'),
        Input('marital-dropdown', 'value'),
        Input('education-dropdown', 'value'),
        Input('default-dropdown', 'value'),
        Input('housing-dropdown', 'value'),
        Input('loan-dropdown', 'value'),
        Input('contact-dropdown', 'value'),
        Input('month-dropdown', 'value'),
        Input('poutcome-dropdown', 'value'),
        Input('age-input', 'value'),
        Input('balance-input', 'value'),
        Input('day-input', 'value')
    ]
)
def classify_user(n_clicks, job, marital, education, default, housing, loan, contact, month, poutcome, age, balance, day):
    if n_clicks is None or n_clicks == 0:
        return "Presiona 'Clasificar' para ver el resultado."

    # Creación del DataFrame con un diccionario de columnas
    nuevo_cliente_df = pd.DataFrame({
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'poutcome': [poutcome],
        'age': [age],
        'balance': [balance],
        'day': [day],
        # Agregar cualquier otra variable que el modelo espere
    })

    # Realizar predicción
    resultado = predecir_nuevo_cliente(model, scaler, nuevo_cliente_df, columnas_modelo)
    return f"Resultado de la clasificación: {'Suscripción exitosa' if resultado[0] == 1 else 'No se suscribirá'}"

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, port=8040)