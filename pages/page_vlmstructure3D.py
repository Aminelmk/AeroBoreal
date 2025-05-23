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

import os
import time

import io
import subprocess
import threading
import queue
import sys
from contextlib import redirect_stdout
import re

output_queue_VLM = queue.Queue()

simulation_process_VLM = None

solver_start_time_VLM = None
solver_end_time_VLM = None


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
wing_mesh_path = {wingmesh}
nb_nodes = {nodes+1}
quarter_chord_ratio = {quatercord}"""

    with open("SOLVEUR_COUPLE/input_main.txt", "w") as f:
        f.write(input_content)

def write_struct_file(data_struct):

    col_mapping = {
    'col1_struct': 'Aire de section',
    'col2_struct': 'Module de Young',
    'col3_struct': 'Coefficient de Poisson',
    'col4_struct': 'Iy',
    'col5_struct': 'Iz',
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
    dbc.Container([
    html.H1("Fluid-Structure Solver Configuration", className="mb-4 text-center"),

    dbc.Row([
    dbc.Col([
    html.H4("Fluid Properties"),

    html.Br(),
    html.Div([
        html.Label("Mach Number: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_Mach", type="number", min=0, max=10, value=0.85, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    html.Div([
        html.Label("Angle of Attack (°): ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_alpha", type="number", value=1.0, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    html.Div([
        html.Label("Pressure (p_inf) [Pa]: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_p_inf", type="number", value=1e5, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    html.Div([
        html.Label("Temperature (T_inf) [K]: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_T_inf", type="number", value=300.0, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    html.Br(),

    html.Div([
        html.Label("Number of Euler Profils : ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="num_profil", type="number", min = 1, value=22, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    ], width = 5)
    ], justify="center"),

    dbc.Table([
        dash_table.DataTable(
            id='eulerprofil_table',
            columns=[
                {'name': 'Y-axis Coordinate', 'id': 'col1_euler', 'editable': True},
                {'name': 'Profil File Path', 'id': 'col2_euler', 'editable': True}
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

    html.Br(),
    dbc.Row([
    dbc.Col([
    html.H4("Structure Properties"),

    html.Br(),

    html.Div([
        html.Label("Quater Cord Ratio: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_qquatercord", type="number", min=0, max=1, value=0.5, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    html.Br(),

    html.Div([
        html.Label("Number of elements: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_nodes", type="number", min= 1, value=10, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    ], width = 5)
    ], justify="center"),

    dbc.Table([
        dash_table.DataTable(
            id='structure_table',
            columns=[
                {'name': 'ID', 'id': 'col0_struct', 'editable': False},
                {'name': 'Sectional Area', 'id': 'col1_struct', 'editable': True},
                {'name': 'Young\'s Modulus', 'id': 'col2_struct', 'editable': True},
                {'name': 'Poisson\'s ratio', 'id': 'col3_struct', 'editable': True},
                {'name': 'Iy', 'id': 'col4_struct', 'editable': True},
                {'name': 'Iz', 'id': 'col5_struct', 'editable': True},
                {'name': 'J', 'id': 'col6_struct', 'editable': True},
                {'name': 'V0', 'id': 'col7_struct', 'editable': True},
                {'name': 'V1', 'id': 'col8_struct', 'editable': True},
                {'name': 'V2', 'id': 'col9_struct', 'editable': True}
            ], style_cell={'width': 'auto', 'minWidth': '90px', 'maxWidth': '200px'},


            data =[{'col0_struct': 1, 'col1_struct': '2859.16', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.42E+06', 'col5_struct': '5.60E+07', 'col6_struct': '6.37e+05', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 2, 'col1_struct': '2412.57', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '4.59E+05', 'col5_struct': '2.62E+07', 'col6_struct': '2.16e+05', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 3, 'col1_struct': '1963.50', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '2.09E+05', 'col5_struct': '1.36E+07', 'col6_struct': '7.13e+04', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 4, 'col1_struct': '1613.23', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '8.50E+04', 'col5_struct': '6.19E+06', 'col6_struct': '2.88e+04', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 5, 'col1_struct': '1442.32', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '3.42E+04', 'col5_struct': '2.81E+06', 'col6_struct': '1.80e+04', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 6, 'col1_struct': '1267.87', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.86E+04', 'col5_struct': '1.78E+06', 'col6_struct': '1.10e+04', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 7, 'col1_struct': '1098.32', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.19E+04', 'col5_struct': '1.03E+06', 'col6_struct': '6.38e+03', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 8, 'col1_struct': '928.92', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '6.36E+03', 'col5_struct': '5.68E+05', 'col6_struct': '3.51e+03', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 9, 'col1_struct': '760.34', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '3.21E+03', 'col5_struct': '2.90E+05', 'col6_struct': '1.68e+03', 'col7_struct': '0.00', 'col8_struct': '1.00', 'col9_struct': '0.00'},
                    {'col0_struct': 10, 'col1_struct': '675.59', 'col2_struct': '1.87e+05', 'col3_struct': '0.31', 'col4_struct': '1.44E+03', 'col5_struct': '1.29E+05', 	'col6_struct':	'1.04e+03','col7_struct':'0.00','col8_struct':'1.00','col9_struct':'0.00'}],
            

            row_deletable=True,  # Permet de supprimer des lignes
            style_table={'overflowX': 'auto', 'border': '1px solid #dee2e6'},
            style_header={'backgroundColor': 'primary', 'textAlign': 'center'},
            style_data={'border': '1px solid #dee2e6', 'textAlign': 'center'}
        )
    ], bordered=True, responsive=True, className="table-hover"),

    html.Br(),
    dbc.Row([
    dbc.Col([
    html.H4("Solver Property"),
    html.Br(),

    html.Div([
        html.Label("Iterations: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
        dcc.Input(id="vlm_it_max", type="number", min=1, value=100, style={"flex": "1"}),
    ], className="row d-flex align-items-center mt-2"),

    
    ], width = 3)
    ], justify="center"),
    html.Br(),

    dbc.Row([
    dbc.Col([
    dbc.Button("Run Simulation", id='run_solvervlm', color="primary", className="mt-2"),
        ], width=5),
        ], justify="center"),



    html.Div(id="solver-realtime-convergence-VLM", hidden=True, children=[
        html.Hr(),
        html.Div(id="solver-status-VLM", className="mt-3"),
        html.H5("Convergence history: "),
        dcc.Graph(id="live-graph-VLM"),
        html.H5("Raw console log: "),
        dcc.Textarea(id='solver-console-VLM', disabled=True, style={'width': '100%', 'height': 300}),
    ]),



    dbc.Row([
    dcc.Loading(
        id="solver-loading-vlm",
        type="default",
        children=html.Div(id='solver-status-vlm', className="mt-3")
    ),

    html.Div(children=[
        html.Hr(),

        dbc.Row([
            dbc.Col(
                dbc.Button(
                    "Back to Configuration",
                    href="/page-mesh3d",
                    color="secondary",
                    className="mt-4"
                ),
                width="auto"
            ),

            dbc.Col(
                dbc.Button(
                    "See results",
                    href="/pages-pressionVLM",
                    id="button-see-results3D",
                    disabled=True,
                    color="primary",
                    className="mt-4"
                ),
                width="auto"
            )],

            justify="center",
            className="mb-4"
        ),
    ]),
    
    # ======= Residual data =======
    dcc.Store(id="convergence-store-VLM", data=[]),
    dcc.Store(id="console-store-VLM", data=[]),

    dcc.Interval(id="log-poll-VLM", interval=500, disabled=True, n_intervals=0),




    html.Div(id='visualization-redirect-vlm', style={'display': 'none'})
    ])
])
])


# ======================================================================
# Callbacks

@dash.callback(
    [Output('solver-realtime-convergence-VLM', 'hidden'),
     Output('log-poll-VLM', 'disabled', allow_duplicate=True),
     Output('convergence-store-VLM', 'data', allow_duplicate=True),
     Output('solver-console-VLM', 'value', allow_duplicate=True)],
    [Input('run_solvervlm', 'n_clicks')],
    [State('vlm_Mach', 'value'),
     State('vlm_alpha', 'value'),
     State('vlm_p_inf', 'value'),
     State('vlm_T_inf', 'value'),
     State('vlm_nodes', 'value'),
     State('vlm_qquatercord', 'value'),
     State('eulerprofil_table', 'data'),
     State('structure_table', 'data'),
     State('vlm_it_max', 'value')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, Mach, alpha, Pinf, Tinf, nodes, quatercord, data_euler, data_structure, it_max):
    wingmesh = 'mesh3d.x'
    if not n_clicks:
        return dash.no_update, dash.no_update
    # Generate input file
    write_vlm_file(Mach, alpha, Pinf, Tinf, it_max, data_euler, wingmesh, nodes, quatercord)
    write_struct_file(data_structure)

    with output_queue_VLM.mutex:
        output_queue_VLM.queue.clear()

    
    threading.Thread(target=run_solver_with_capture_VLM, daemon=True).start()

    global solver_start_time_VLM, solver_end_time_VLM

    solver_start_time_VLM = time.perf_counter()
    solver_end_time_VLM = 0.0

    # Run solver
    # try:
    #     solveur_couple.solve("SOLVEUR_COUPLE/input_main.txt")
    #     subprocess.run(["python", "SOLVEUR_COUPLE/write_vtu.py"], check=True)
    #     status = dbc.Alert("✅ Simulation completed successfully!", color="success")
    #     redirect = dcc.Location(pathname="/pages-pressionVLM", id="redirect-vlm")
    # except Exception as e:
    #     status = dbc.Alert(f" Error: {str(e)}", color="danger")
    #     redirect = dash.no_update

    # status = dbc.Alert("Running simulation...", color="info")
    redirect = dash.no_update

    return False, False, [], ""




def run_solver_with_capture_VLM():

    global simulation_process_VLM

    simulation_process_VLM = subprocess.Popen(
        ["python", "SOLVEUR_COUPLE/launch_vlm.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in simulation_process_VLM.stdout:
        output_queue_VLM.put(line)
        print(line)
    output_queue_VLM.put("__DONE__")

    print("Simulation process finished.")




@dash.callback(
    [Output("convergence-store-VLM", "data"),
     Output("live-graph-VLM", "figure"),
    #  Output("stop_solver", "disabled", allow_duplicate=True),
     Output("solver-console-VLM", "value"),
     Output("solver-status-VLM", "children", allow_duplicate=True),
     Output("log-poll-VLM", "disabled", allow_duplicate=True),
    Output("button-see-results3D", "disabled", allow_duplicate=True),
     ],
    Input("log-poll-VLM", "n_intervals"),
    [State("convergence-store-VLM", "data"),
     State("solver-console-VLM", "value"),],
    prevent_initial_call=True,
)
def update_convergence_graph(n, data, console_data):

    disable_stop_button = dash.no_update
    solver_status = dash.no_update
    log_poll = dash.no_update
    button_see_results3D = dash.no_update
    global solver_start_time_VLM, solver_end_time_VLM
    if data is None:
        data = []

    if console_data is None:
        console_data = ""

    while not output_queue_VLM.empty():
        line = output_queue_VLM.get()
        console_data += f"{line}"

        if line.startswith("__ERROR__"):
            print("ERROR in simulation output")
            continue

        if line == "__DONE__":
            print("Simulation finished streaming")
            disable_stop_button = True
            log_poll = True
            button_see_results3D = False
            solver_end_time_VLM = time.perf_counter()

            os.makedirs("temp_vlm/", exist_ok=True)

            f = open("temp_vlm/vlm_log.txt", "w")
            f.write(console_data)
            f.close()

            solver_status = dbc.Alert(f"Simulation completed after: {solver_end_time_VLM - solver_start_time_VLM:.1f} s.", color="success")
            break
        else:
            parsed = parse_line(line)
            if parsed is not None:
                data.append(parsed)
            solver_status = dbc.Alert(f"Solver is running. Up time: {time.perf_counter() - solver_start_time_VLM:.1f} s.", color="warning")

    # Plotting the convergence of residuals
    fig = go.Figure()

    if data:
        fig.add_trace(go.Scatter(x=[i for i in range(1,len(data)+1)], 
                                 y=[np.log10(d["erreur"]) for d in data],
                                 mode="lines+markers", name="Error",
                                 marker=dict(size=8, symbol="square",)
                                 ))

    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=-12,
        y1=-12,
        line=dict(
            color="black",
            width=1,
            dash="dash"
        )
    )

    fig.update_layout(xaxis_title="Iteration",
                      yaxis_title="Log10(displacement L2 norm)",
                      xaxis=dict(tickmode="linear", tick0=1, dtick=1))

    fig.update_xaxes(range=[1, len(data)])

    # return data, fig, disable_stop_button, console_data, solver_status, log_poll, button_see_results
    return data, fig, console_data, solver_status, log_poll, button_see_results3D



def parse_line(line):
    pattern = r"Erreur\s*=\s*([\d.eE+-]+)"
    match = re.search(pattern, line)
    if match:
        return {
            "erreur": float(match.group(1)),
        }
    return None







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
