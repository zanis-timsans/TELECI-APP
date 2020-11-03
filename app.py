# -*- coding: utf-8 -*-

# visit http://127.0.0.1:8050/ in your web browser.

# Load libraries
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

# Styles - CERULEAN (!), COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA (!),
# MINTY, PULSE (!), SANDSTONE (!), SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB (!), SUPERHERO, UNITED (!), YETI (!)
external_stylesheets = [dbc.themes.PULSE]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,  # , external_stylesheets=external_stylesheets
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
server = app.server

# PREPROCESSING Load dataframe
camera = dict(
        eye=dict(x=2, y=2, z=0.8)
    )


# APPLICATION LAYOUT
app.layout = html.Div([
    dbc.Row(dbc.Col([
        html.H1('TELECI DATA VISUALIZATION', className="text-center my-5"),
        html.Div('''
                Welcome to data visualization web application for TELECI project. 
                Visualization method is called telecides and it reflects quantitative visualization of the 
                appropriateness of an e-content unit for the needs of 
                the specific learner or learners target group.
                ''', className="text-center my-5"),
    ], width=7,
    ), justify='center',
    ),

    dbc.Row([
        dbc.Col([
            html.Div([html.H5('About visualization'),
                      '''
                The student performance data before and after learning the e-content were used for knowledge acquisition 
                model design. This model is based on the assumption that knowledge acquisition of real e-content can be 
                quantified by superposition of the impact of learning “perfect” content, too easy content, and too 
                complicated content. The data of real course learner knowledge acquisition are located on this surface. 
                These points are performing curves called telecides. Telecides are quantitative visualization of the 
                appropriateness of an e-content unit for the needs of the specific learner or learners target group.
                '''], className='ml-3 my-3 border p-3'
                     ),
            html.Div("Choose course", className='ml-3'),
            dcc.Dropdown(options=[
                {'label': 'Business Planning for Open Markets and Entrepreneurship',
                 'value': 'https://bpome.mii.lv/webservice/rest/server.php?wstoken=abc77eef5247b3aa21488df6a394a3d4'
                          '&wsfunction=local_getstudentactivitydata_functiongetstudentactivitydata&date_from=0'},
                {'label': 'Other course',
                 'value': 'https://artss.mii.lv/webservice/rest/server.php?courseid=12&wstoken'
                          '=a78e76c2570f41a3f180d0979914c7dc&wsfunction'
                          '=local_notifyemailsignup_functiongetstudentactivitydata&moodlewsrestformat=json',
                 'disabled': True},
            ],
                value='https://bpome.mii.lv/webservice/rest/server.php?wstoken=abc77eef5247b3aa21488df6a394a3d4'
                      '&wsfunction=local_getstudentactivitydata_functiongetstudentactivitydata&date_from=0',
                searchable=True,
                placeholder='Izvēlies vai ievadi kursa nosaukumu',
                persistence=True,
                persistence_type='local',  # memory-tab refresh, session-tab is closed, local-cookies
                id='courses_dropdown',
                className='ml-3 mb-5',
                clearable=False
            ),
            html.Div(
                [
                    html.Div([
                        html.H6('Content suitability areas'),
                        html.Img(title='example', src='assets/landscape-segments.jpg', width='100%'),
                        dbc.Alert("Ideally matching content", color="success", className='mb-0'),
                        dbc.Alert("Too complicated content", color="danger", className='my-1'),
                        dbc.Alert("Too easy content ", color="primary", className='mt-0'),
                    ], className='mb-3 border p-3'),
                ], className='ml-3'
            )
        ], width=3),
        dbc.Col([
            dbc.CardGroup([
                dbc.Card(
                    dbc.CardBody([
                        html.Div('Unit', className="card-title text-center"),
                        html.H1(id='tema', className='text-center clearfix')
                    ]), className=""
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.Div('Students', className="card-title text-center"),
                        html.H1(id='number_of_students', className='text-center')
                    ]), className=""
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.Div('Question pairs', className="card-title text-center"),
                        html.H1(id='pari', className='text-center')
                    ]), className=""
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.Div('Analyzed pairs', className="card-title text-center"),
                        html.H1(id='summa', className='text-center clearfix')
                    ]), className=""
                )
            ], className='mt-3 mb-3'),
            dbc.Col([
                html.H5('Tēmas', className='ac'),
                dcc.Loading(id="loading-2", children=[
                    html.Div(id="loading-output-2", style={'display': 'none'}),
                    dcc.Graph(
                        id='telecides-unit',
                        figure={},
                        config={'displaylogo': False, 'showTips': True},
                    ),
                ], type="default"),
            ], className='border p-3'),
        ], width=7),
    ], justify="center"),
])


# ------------------- CALLBACKS -------------------------------------------------------------------------------------
@app.callback(
    [Output(component_id='telecides-unit', component_property='figure'),
     Output(component_id='tema', component_property='children'),
     Output(component_id='number_of_students', component_property='children'),
     Output(component_id='pari', component_property='children'),
     Output(component_id='summa', component_property='children'),
     Output(component_id="loading-output-2", component_property='children')],
    [Input(component_id='courses_dropdown', component_property='value')]
)
def update_telecides(value):
    if value is None:
        # PreventUpdate prevents ALL outputs updating
        raise dash.exceptions.PreventUpdate
    api = value
    df = pd.read_json(api, orient="split")

    df1 = pd.DataFrame(df['user'].values.tolist())
    df1.columns = 'user_' + df1.columns
    col = df.columns.difference(['user'])
    df = pd.concat([df[col], df1], axis=1)

    # Number of unique users including admins and managers
    number_of_users = len(df["user_id"].unique())

    rez = df[df["type"] == "result"]
    # Filter out only first item in 'user_role'

    rez = rez.loc[np.array(list(map(len, rez.user_roles.values))) == 1]

    # Convert column from list to string
    rez['user_roles'] = rez['user_roles'].apply(lambda x: ','.join(map(str, x)))

    # Select only 'student' roles and make copy of original dataframe to avoid slicing view
    rez = rez.loc[rez.user_roles == 'student'].copy()

    # Transform time feature to datetime object
    rez["datetime"] = pd.to_datetime(rez["datetime"])

    # Keeping only first occurrence and dropping all other
    rez = rez.drop_duplicates(subset=['user_id', 'itemid'], keep="first")

    # Drop nevajadzīgās colonas
    rez.drop(['id', 'answer', 'answer_submitted', 'question', 'timestamp', 'type', 'user_roles'], axis=1, inplace=True)

    # Convert everything to uppercase in case there are some in lowercase
    rez['title'] = rez['title'].str.upper()

    # Take last
    rez['letter'] = rez['title'].str.strip().str[-1]

    # Remove rows that are not a or b
    rez = rez[(rez['title'].str.contains('A', case=False)) | (rez['title'].str.contains('B', case=False))].copy()

    # Remove rows with '.' for 'letter' value
    rez = rez[rez['letter'] != '.'].copy()

    # Sakartoti pēc Lietotāja un priekšmeta ID *apskatei*
    rez.sort_values(by=['section', 'lessonid', 'user_id'], inplace=True)

    # mapping true/false to p/n (pareizi/nepareizi)
    di = {'true': 'p', 'false': 'n'}
    rez.replace({'correct_answer': di}, inplace=True)

    # Convert everything to uppercase in case there are some in lowercase
    rez['letter'] = rez['letter'].str.upper()

    # All necessary 'A'
    a = rez[(rez['letter'].shift(-1) == 'B')].copy()

    # All necessary 'B'
    b = rez[(rez['letter'] == 'B')].copy()

    # Combine two filtered dataframes
    rez = a.append(b, sort=True)

    # Sort values and reset index
    rez.sort_values(by=['section', 'lessonid', 'user_id'], inplace=True)
    rez.reset_index(inplace=True)

    # Number of question pairs
    pari = rez["lessonid"].nunique()

    # Number of students only
    number_of_students = rez["user_id"].nunique()

    # Join 'p' and 'n' results into one column based on 'a' and 'b' questions. Keep section number. Convert to dataframe
    # using to_frame
    final_df = rez.groupby(by=[rez.index // 2, 'section'])['correct_answer'].agg('-'.join).to_frame()

    # Reset index
    final_df.reset_index(level=['section'], inplace=True)

    # Create final dataframe containing sum of all question pairs for each Unit (section)
    telecides = pd.crosstab(index=final_df['section'], columns=final_df['correct_answer'])

    # Create necessary column if they do not exist
    if 'n-n' not in telecides.columns:
        telecides['n-n'] = 0
    if 'n-p' not in telecides.columns:
        telecides['n-p'] = 0
    if 'p-p' not in telecides.columns:
        telecides['p-p'] = 0
    if 'p-n' not in telecides.columns:
        telecides['p-n'] = 0

    # Create column 'x-n' that is sum of 'n-n' and 'p-n' columns
    telecides['x-n'] = telecides['n-n'] + telecides['p-n']
    telecides.drop(columns=['n-n', 'p-n'], inplace=True)

    # Create column sum of all pairs per Unit
    telecides['sum'] = telecides['n-p'] + telecides['p-p'] + telecides['x-n']

    # Sum of all question pairs
    summa = telecides['sum'].sum()

    # Number of units
    temas = len(telecides.index)

    # Create final values using average probability
    telecides['n-p'] = telecides['n-p'] / telecides['sum']
    telecides['p-p'] = telecides['p-p'] / telecides['sum']
    telecides['x-n'] = telecides['x-n'] / telecides['sum']
    telecides.drop(columns=['sum'], inplace=True)

    # Create dataframe for visualising 'Complete learning acquisition landscape'
    df_tele = pd.DataFrame(data=[['too complicated content', 0.222, 0.111, 0.666],
                                 ['too easy content', 0, 1, 0, ],
                                 ['ideally matching content', 0.667, 0.333, 0]],
                           columns=['content', 'N-P', 'P-P', 'X-N'], index=None)
    # print(df_tele)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=df_tele['N-P'],
            y=df_tele['P-P'],
            z=df_tele['X-N'],
            color='steelblue',
            opacity=0.3,
            hoverinfo='skip',
        ),
        go.Scatter3d(
            x=telecides['n-p'],
            y=telecides['p-p'],
            z=telecides['x-n'],
            mode="markers",
            text=telecides.index-1,
            hovertemplate='Matching: %{x:.2f}<br>Easy: %{y:.2f}<br>Complicated: %{z:.2f}<extra>Unit: %{text}</extra>',
            marker=dict(size=7, symbol="circle", color=telecides)  # color=telecides.index, colorscale='balance'
        ),
    ])
    fig.update_layout(
        template='plotly',
        scene_camera=camera,
        scene=dict(
            xaxis_title="Matching",
            yaxis_title="Easy",
            zaxis_title="Complicated",
            xaxis=dict(
                nticks=6, range=[0, 1],
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showspikes=False),
            yaxis=dict(
                nticks=6, range=[0, 1],
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showspikes=False),
            zaxis=dict(
                nticks=6, range=[0, 1],
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showspikes=False),
        ),
        height=600,
        margin=dict(
            r=0, l=0,
            b=0, t=0),
    )
    return fig, temas, number_of_students, pari, summa, value


# -------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
