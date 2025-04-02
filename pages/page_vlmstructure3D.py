import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import subprocess
from SOLVEUR_COUPLE import solveur_couple 

from euler2d import euler_solver

dash.register_page(__name__, path="/pages-vlmstructure3D")

# ======================================================================
# Data Processing Functions
# ======================================================================
def write_vlm_file(Mach, alpha, Pinf, Tinf, it_max, datapath, wingmesh, nodes, quatercord):

    input_content = f"""Mach = {Mach}
alpha = {alpha}
p_inf = {Pinf}
T_inf = {Tinf}
couplage_Euler = 1
y_0_profils = {" ".join(row['col1_euler'] for row in datapath)}
profils_paths = SOLVEUR_COUPLE/database/{" SOLVEUR_COUPLE/database/".join(row['col2_euler'] for row in datapath)}
it_max = {it_max}
wing_mesh_path = SOLVEUR_COUPLE/{wingmesh}
nb_nodes = {nodes+1}
quarter_chord_ratio = {quatercord}"""

    with open("SOLVEUR_COUPLE/input_main.txt", "w") as f:
        f.write(input_content)

def write_struct_file(data_struct):

    col_mapping = {
    'col1_struct': 'Aire de section',
    'col2_struct': 'Module de Young',
    'col3_struct': 'Coefficient de Poisson',
    'col4_struct': 'Iz',
    'col5_struct': 'Iy',
    'col6_struct': 'J',
    'col7_struct': 'Vecteur V 0',
    'col8_struct': 'Vecteur V 1',
    'col9_struct': 'Vecteur V 2'
    }

    # Écriture dans un fichier
    with open("Calcul_structure/Inputs.txt", "w") as f:
        for idx, row in enumerate(data_struct, start=1):
            f.write(f"ID={idx}\n")
            for col, label in col_mapping.items():
                f.write(f"{label}={row[col]}\n")



# ======================================================================
# Solver Page
# ======================================================================

layout = html.Div([
    html.H1("Fluid-Structure Solver Configuration", className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                html.H1(html.Strong("Fluid Settings")),
                dbc.Col([
                    html.Label("Mach Number"),
                    dcc.Input(id='vlm_Mach', type='number', value=0.5, className="mb-2"),
                ], width=2),
                dbc.Col([
                    html.Label("Angle of Attack (deg)"),
                    dcc.Input(id='vlm_alpha', type='number', value=5.0, className="mb-2"),
                ], width=2),
                 dbc.Col([
                    html.Label("Pressure (p_inf) [Pa]"),
                    dcc.Input(id='vlm_p_inf', type='number', value=1e5, className="mb-2"),
                ], width=2),
                dbc.Col([
                    html.Label("Temperature (T_inf) [K]"),
                    dcc.Input(id='vlm_T_inf', type='number', value=300.0, className="mb-2"),
                ], width=2),
                dbc.Col([
                    html.Label("Wing Mesh Path"),
                    dcc.Input(id='vlm_wingmesh', type='text', value="../mesh3d.x", className="mb-2"),
                ], width=2),
            ], className="mb-3", justify="center"),

            dbc.Row([
                dbc.Col([
                    html.Label("Number of Euler Profils :"),
                    dcc.Input(id='num_profil', type='number', value=22, min=1, step=1, className="mb-2"),
                ], width=2),
            ], className="mb-3", justify="center"),

            dbc.Table([
                dash_table.DataTable(
                    id='eulerprofil_table',
                    columns=[
                        {'name': 'Y-axis Coordinate', 'id': 'col1_euler', 'editable': True},
                        {'name': 'Profil file path', 'id': 'col2_euler', 'editable': True}
                    ],
                    data=[{'col1_euler': '-0.7933', 'col2_euler': 'Coupey0,7933.xyz'},
                          {'col1_euler': '-0.7536', 'col2_euler': 'Coupey0,7536.xyz'},
                          {'col1_euler': '-0.714', 'col2_euler': 'Coupey0,714.xyz'},
                          {'col1_euler': '-0.6346', 'col2_euler': 'Coupey0,6346.xyz'},
                          {'col1_euler': '-0.5553', 'col2_euler': 'Coupey0,5553.xyz'},
                          {'col1_euler': '-0.476', 'col2_euler': 'Coupey0,476.xyz'},
                          {'col1_euler': '-0.3967', 'col2_euler': 'Coupey0,3967.xyz'},
                          {'col1_euler': '-0.3173', 'col2_euler': 'Coupey0,3173.xyz'},
                          {'col1_euler': '-0.238', 'col2_euler': 'Coupey0,238.xyz'},
                          {'col1_euler': '-0.1587', 'col2_euler': 'Coupey0,1587.xyz'},
                          {'col1_euler': '-0.0873', 'col2_euler': 'Coupey0,0873.xyz'},
                          {'col1_euler': '0.0873', 'col2_euler': 'Coupey0,0873.xyz'},
                          {'col1_euler': '0.1587', 'col2_euler': 'Coupey0,1587.xyz'},
                          {'col1_euler': '0.238', 'col2_euler': 'Coupey0,238.xyz'},
                          {'col1_euler': '0.3173', 'col2_euler': 'Coupey0,3173.xyz'},
                          {'col1_euler': '0.3967', 'col2_euler': 'Coupey0,3967.xyz'},
                          {'col1_euler': '0.476', 'col2_euler': 'Coupey0,476.xyz'},
                          {'col1_euler': '0.5553', 'col2_euler': 'Coupey0,5553.xyz'},
                          {'col1_euler': '0.6346', 'col2_euler': 'Coupey0,6346.xyz'},
                          {'col1_euler': '0.714', 'col2_euler': 'Coupey0,714.xyz'},
                          {'col1_euler': '0.7536', 'col2_euler': 'Coupey0,7536.xyz'},
                          {'col1_euler': '0.7933', 'col2_euler': 'Coupey0,7933.xyz'}],
                    row_deletable=True,  # Permet de supprimer des lignes
                    style_table={'overflowX': 'auto', 'border': '1px solid #dee2e6'},
                    style_header={'backgroundColor': 'primary', 'textAlign': 'center'},
                    style_data={'border': '1px solid #dee2e6', 'textAlign': 'center'}
                )
            ], bordered=True, responsive=True, className="table-hover"),

            html.H1(html.Strong("Structure Settings")),

            dbc.Row([
                dbc.Col([
                    html.Label("Number of elements"),
                    dcc.Input(id='vlm_nodes', type='number', value=10, className="mb-2"),
                ], width=2),
                dbc.Col([
                    html.Label("Quater Cord Ratio"),
                    dcc.Input(id='vlm_qquatercord', type='number', value=0.25, className="mb-2"),
                ], width=2),
            ], className="mb-3", justify="center"),

            dbc.Table([
                dash_table.DataTable(
                    id='structure_table',
                    columns=[
                        {'name': 'ID', 'id': 'col0_struct', 'editable': False},
                        {'name': 'Aire de section', 'id': 'col1_struct', 'editable': True},
                        {'name': 'Module de Young', 'id': 'col2_struct', 'editable': True},
                        {'name': 'Coefficient de Poisson', 'id': 'col3_struct', 'editable': True},
                        {'name': 'Iz', 'id': 'col4_struct', 'editable': True},
                        {'name': 'Iy', 'id': 'col5_struct', 'editable': True},
                        {'name': 'J', 'id': 'col6_struct', 'editable': True},
                        {'name': 'V0', 'id': 'col7_struct', 'editable': True},
                        {'name': 'V1', 'id': 'col8_struct', 'editable': True},
                        {'name': 'V2', 'id': 'col9_struct', 'editable': True}
                    ], style_cell={'width': 'auto', 'minWidth': '100px', 'maxWidth': '200px'},
        
                    data=[{'col0_struct':1, 'col1_struct': '108201.33', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.57e12', 'col5_struct': '8.07e10', 'col6_struct': '4.11e10', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':2,'col1_struct': '91660.88', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '9.67e11', 'col5_struct': '3.56e10', 'col6_struct': '1.55e10', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':3,'col1_struct': '75028.56', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '5.35e11', 'col5_struct': '1.53e10', 'col6_struct': '5.90e09', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':4,'col1_struct': '62055.53', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '3.03e11', 'col5_struct': '7.69e09', 'col6_struct': '2.79e09', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':5,'col1_struct': '55725.77', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '2.20e11', 'col5_struct': '5.39e09', 'col6_struct': '1.92e09', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':6,'col1_struct': '49264.39', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.52e11', 'col5_struct': '3.70e09', 'col6_struct': '1.32e09', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':7,'col1_struct': '42984.77', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.01e11', 'col5_struct': '2.48e09', 'col6_struct': '8.85e08', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':8,'col1_struct': '36710.73', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '6.28e10', 'col5_struct': '1.61e09', 'col6_struct': '5.85e08', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':9,'col1_struct': '30467.23', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '3.59e10', 'col5_struct': '9.71e08', 'col6_struct': '3.64e08', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'},
                          {'col0_struct':10,'col1_struct': '27328.40', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '2.59e10', 'col5_struct': '7.19e08', 'col6_struct': '2.73e08', 'col7_struct': '0', 'col8_struct': '1', 'col9_struct': '0'}],

                    row_deletable=True,  # Permet de supprimer des lignes
                    style_table={'overflowX': 'auto', 'border': '1px solid #dee2e6'},
                    style_header={'backgroundColor': 'primary', 'textAlign': 'center'},
                    style_data={'border': '1px solid #dee2e6', 'textAlign': 'center'}
                )
            ], bordered=True, responsive=True, className="table-hover"),

            dbc.Row([
                dbc.Col([
                    html.Label("Max Iterations"),
                    dcc.Input(id='vlm_it_max', type='number', value=100, className="mb-2"),
                ], width=2),
            ], className="mb-3", justify="center"),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Run Simulation", id='run_solvervlm', color="primary", className="mt-2"),
                ], width=2),
            ], className="mb-3", justify="center"),

            
            
        ])
    ], className="mb-2"),

    dcc.Loading(
        id="solver-loading-vlm",
        type="default",
        children=html.Div(id='solver-status-vlm', className="mt-3")
    ),

    html.Div(id='visualization-redirect-vlm', style={'display': 'none'})
])


# ======================================================================
# Callbacks

@dash.callback(
    #[
    Output('solver-status-vlm', 'children'),
    #Output('visualization-redirect', 'children')],
    [Input('run_solvervlm', 'n_clicks')],
    [State('vlm_Mach', 'value'),
     State('vlm_alpha', 'value'),
     State('vlm_p_inf', 'value'),
     State('vlm_T_inf', 'value'),
     State('vlm_wingmesh', 'value'),
     State('vlm_nodes', 'value'),
     State('vlm_qquatercord', 'value'),
     State('eulerprofil_table', 'data'),
     State('structure_table', 'data'),
     State('vlm_it_max', 'value')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, Mach, alpha, Pinf, Tinf, wingmesh, nodes, quatercord, data_euler, data_structure, it_max):
    if not n_clicks:
        return dash.no_update, dash.no_update
    # Generate input file
    write_vlm_file(Mach, alpha, Pinf, Tinf, it_max, data_euler, wingmesh, nodes, quatercord)
    write_struct_file(data_structure)

    # Run solver
    try:
        solveur_couple.solve("SOLVEUR_COUPLE/input_main.txt")
        subprocess.run(["python", "SOLVEUR_COUPLE/write_vtu.py"], check=True)
        status = dbc.Alert("✅ Simulation completed successfully!", color="success")
        #redirect = dcc.Location(pathname="/page-euler2d-results", id="redirect")
    except Exception as e:
        status = dbc.Alert(f" Error: {str(e)}", color="danger")
        #redirect = dash.no_update

    return status #, redirect 

@dash.callback(
    Output('eulerprofil_table', 'data'),
    Input('num_profil', 'value'),
    State('eulerprofil_table', 'data')
)
def update_table(n, existing_data):
    # Si le tableau existe déjà, on garde les anciennes valeurs
    if existing_data:
        df = pd.DataFrame(existing_data)
    else:
        df = pd.DataFrame(columns=['col1_euler', 'col2_euler'])
    
    # Adapter la taille du tableau en fonction de `n`
    df = df.reindex(range(n)).fillna('')
    return df.to_dict('records')


#------------------------


@dash.callback(
    Output('structure_table', 'data'),
    Input('vlm_nodes', 'value'),
    State('structure_table', 'data')
)
def update_table(n, existing_data):
    # If the table already exists, keep the existing data
    if existing_data:
        df = pd.DataFrame(existing_data)
    else:
        df = pd.DataFrame(columns=['col0_struct', 'col1_struct', 'col2_struct', 'col3_struct', 'col4_struct', 
                                   'col5_struct', 'col6_struct', 'col7_struct', 'col8_struct', 'col9_struct'])

    # Adjust the size of the table to match `n`
    df = df.reindex(range(n)).fillna('')

    # Automatically fill the ID column with sequential numbers
    df['col0_struct'] = range(1, len(df) + 1)

    return df.to_dict('records')
