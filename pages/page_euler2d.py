import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

output_queue = queue.Queue()

class StreamToQueue(io.TextIOBase):
    def __init__(self, queue):
        self.queue = queue
    def write(self, msg):
        self.queue.put(msg)
    def flush(self):
        pass


dash.register_page(__name__, path="/page-euler2d")


simulation_process = None

solver_start_time = None
solver_end_time = None

# ======================================================================
# Data Processing Functions
# =====================================================================

def parse_mesh(file_path):
    print("trying to parse mesh")

    with open(file_path, "r") as f:
        lines = f.readlines()
    nx, ny = map(int, lines[0].split())
    x_grid = np.array([float(value) for value in lines[1: 1 + nx * ny]])
    y_grid = np.array([float(value) for value in lines[1 + nx * ny: 1 + 2 * nx * ny]])
    return nx, ny, -y_grid.reshape(ny, nx), -x_grid.reshape(ny, nx)


def parse_test_q(file_path, nx, ny):
    with open(file_path, "r") as f:
        lines = f.readlines()
    numerical_lines = lines[2:]
    numerical_data = np.array([float(value) for line in numerical_lines for value in line.split()])
    data = numerical_data.reshape(4, ny, nx)
    return data[0], data[1], data[2], data[3]


def create_surface_plot(variable, title, colorbar_title, x_2d, y_2d):
    z = np.zeros_like(variable)
    min_val = np.min(variable)
    max_val = np.max(variable)

    # Create a single subplot (remove the second column)
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "surface"}]],
        subplot_titles=[title]
    )

    # Main surface plot with integrated colorbar
    fig.add_trace(
        go.Surface(
            x=x_2d,
            y=y_2d,
            z=z,
            surfacecolor=variable,
            colorscale='Magma',
            lighting=dict(ambient=1.0, diffuse=0.0),
            showscale=True,
            colorbar=dict(
                title=f"{colorbar_title}<br>Min: {min_val:.2f}<br>Max: {max_val:.2f}",
                tickvals=[min_val, max_val],
                ticktext=[f"{min_val:.2f}", f"{max_val:.2f}"],
                tickfont=dict(size=10),
                thickness=50,
                len=0.75
            )
        )
    )

    # Layout adjustments
    fig.update_layout(
        width=1200,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                range=[-0.3, 0.3],
                showgrid=False,
                showline=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                range=[-1.1, 0.1],
                showgrid=False,
                showline=False,
                zeroline=False,
                visible=False
            ),
            zaxis=dict(
                showticklabels=False,
                title='',
                visible=False,
                showgrid=False,
                showline=False,
                zeroline=False
            ),
            camera=dict(
                eye=dict(x=0, y=0, z=1.2),
                # projection=dict(type='orthographic')
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0)
        )
    )

    return fig


# ======================================================================
# Solver Page
# ======================================================================

layout = html.Div([
    dbc.Container([
    html.H1("Euler Solver Configuration", className="mb-4 text-center"),

    dbc.Row([
        dbc.Col([
        html.H4("Fluid Properties"),
        html.Br(),
        html.Div([
            html.Label("Mach Number: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="Mach", type="number", min=0.3, max=1.5, value=0.5, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("Angle of Attack (α): ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="alpha", type="number", min=0, value=0.0, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("Pressure (p_inf) [Pa]: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="p_inf", type="number", value=1e5, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("Temperature (T_inf) [K]: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="T_inf", type="number", value=300, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),


        ],width=5),

        dbc.Col([], width=1),

        dbc.Col([
        html.H4("Solver Properties"),
        html.Br(),
        html.Div([
            html.Label("CFL Number: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="CFL_number", type="number", value=3.0, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("Max Iterations: ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="it_max", type="number", value=10_000, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("k2 (2nd Order Dissipation): ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="k2", type="number", value=2, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("k4 (4th Order Dissipation): ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="k4", type="number", value=2, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        ],width=5),
    ], justify="center"),

    html.Br(),
    
    # Ligne 3 : Dropdowns
    dbc.Row([
        dbc.Col([
            html.Label("Multigrid"),
            dcc.Dropdown(
                id='multigrid',
                options=[{'label': 'Disabled', 'value': 0}, {'label': 'Enabled', 'value': 1}],
                value=0,
                clearable=False
            ),
        ], width=2),
        dbc.Col([
            html.Label("Residual Smoothing"),
            dcc.Dropdown(
                id='residual_smoothing',
                options=[{'label': 'Disabled', 'value': 0}, {'label': 'Enabled', 'value': 1}],
                value=0,
                clearable=False
            ),
        ], width=2),
    ], className="mb-4", justify="center"),

    

    # Ligne 5 : Upload Mesh
    dbc.Row([
        dbc.Col([
            html.Label("Upload Mesh (.xyz)", className="text-center"),
            dcc.Upload(
                id='upload-mesh',
                children=html.Div(['Drag and Drop or ', html.A('Select a mesh file (.xyz)')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'marginTop': '10px'
                },
                multiple=False
            ),
            html.Div(id='upload-mesh-status', className='text-center mt-2')
        ], width=6)
    ], justify="center", className="mt-4"),

    html.Br(),

    # Ligne 4 : Bouton
    dbc.Row([
        dbc.Col(
            dbc.Button("Run Simulation", id='run_solver', color="primary", className="mt-2"),
            width="auto"
        ),

        dbc.Col(
            dbc.Button("Stop simulation", id='stop_solver', color="danger", disabled=True, className="mt-2", ),
            width="auto"
        ),

    ], justify="center"),


    html.Div(id="solver-realtime-convergence", hidden=True, children=[
        html.Hr(),
        html.Div(id="solver-status", className="mt-3"),
        html.H5("Convergence history: "),
        dcc.Graph(id="live-graph"),
        html.H5("Raw console log: "),
        dcc.Textarea(id='solver-console', disabled=True, style={'width': '100%', 'height': 300}),
    ]),

    html.Div(children=[
        html.Hr(),

        dbc.Row([
            dbc.Col(
                dbc.Button(
                    "Back to Configuration",
                    href="/page-mesh2d",
                    color="secondary",
                    className="mt-4"
                ),
                width="auto"
            ),

            dbc.Col(
                dbc.Button(
                    "See results",
                    href="/page-euler2d-results",
                    id="button-see-results",
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

    dcc.Store(id="convergence-store", data=[]),
    dcc.Store(id="console-store", data=[]),

    dcc.Interval(id="log-poll", interval=500, disabled=True, n_intervals=0),

    # html.Div("Counter: ", id="counter"),


    # dcc.Loading(
    #     id="solver-loading",
    #     type="default",
    #     children=html.Div(id='solver-status', className="mt-3")
    # ),

    html.Div(id='visualization-redirect', style={'display': 'none'}),
    ])
])

#########DEBUT CHANGEMENT################## 
import os 
import base64
UPLOAD_FOLDER = "temp"

@dash.callback(
    Output('upload-mesh-status', 'children'),  
    Output('upload-mesh', 'filename', allow_duplicate=True),  
    Input('upload-mesh', 'contents'), 
    State('upload-mesh', 'filename'),  
    prevent_initial_call='initial_duplicate'  
)

def save_uploaded_mesh(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update

    
    filepath = os.path.join(UPLOAD_FOLDER, 'user_mesh.xyz')

    
    content_type, content_string = contents.split(',')
    new_file = base64.b64decode(content_string)

    
    with open(filepath, 'wb') as f:
        f.write(new_file)

    return dbc.Alert(f"Mesh '{filename}' uploaded successfully!", color="success"), filename


# ======================================================================
# Callbacks
# ======================================================================

@dash.callback(
    [Output('solver-status', 'children', allow_duplicate=True),
     Output('visualization-redirect', 'children'),
     Output("convergence-store", "data", allow_duplicate=True),
     Output("stop_solver", "disabled", allow_duplicate=True),
     Output("log-poll", "disabled", allow_duplicate=True),
     Output("solver-console", "value", allow_duplicate=True),
     Output("solver-realtime-convergence", "hidden"),],
    [Input('run_solver', 'n_clicks'),],
    [State('Mach', 'value'),
     State('alpha', 'value'),
     State('CFL_number', 'value'),
     State('p_inf', 'value'),
     State('T_inf', 'value'),
     State('multigrid', 'value'),
     State('residual_smoothing', 'value'),
     State('k2', 'value'),
     State('k4', 'value'),
     State('it_max', 'value')],
    prevent_initial_call=True,
)
def run_simulation(n_clicks, Mach, alpha, CFL_number, p_inf, T_inf,
                   multigrid, residual_smoothing, k2, k4, it_max):

    # Clears the queue for the next simulation
    with output_queue.mutex:
        output_queue.queue.clear()

    log_poll = dash.no_update

     #########DEBUT CHANGEMENT##################      
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update

    # Choix du maillage : uploadé ou par défaut
    uploaded_mesh_path = "temp/user_mesh.xyz"
    default_mesh_path = "temp/mesh.xyz"
    mesh_file = uploaded_mesh_path if os.path.exists(uploaded_mesh_path) else default_mesh_path
     #########FIN CHANGEMENT##################       
    # Génération du fichier d'entrée
    input_content = f"""num_threads = 1
mesh_file = {mesh_file}
Mach = {Mach}
alpha = {alpha}
p_inf = {p_inf}
T_inf = {T_inf}
multigrid = {multigrid}
CFL_number = {CFL_number}
residual_smoothing = {residual_smoothing}
k2 = {k2}
k4 = {k4}
it_max = {it_max}
output_file = temp/results.q
checkpoint_file = temp/checkpoint_euler.txt"""

    os.makedirs("temp/", exist_ok=True)

    with open("temp/input.txt", "w") as f:
        f.write(input_content)

    threading.Thread(target=run_solver_with_capture, daemon=True).start()

    status = dbc.Alert("Simulation started!", color="success")
    redirect = dcc.Location(pathname="/page-euler2d-results", id="redirect")

    global solver_start_time, solver_end_time

    solver_start_time = time.perf_counter()
    solver_end_time = 0.0


    # Exécution du solveur
    # try:
    #     euler_solver.solve("input.txt")
    #     status = dbc.Alert("Simulation completed successfully!", color="success")
    #     redirect = dcc.Location(pathname="/page-euler2d-results", id="redirect")
    # except Exception as e:
    #     status = dbc.Alert(f" Error: {str(e)}", color="danger")
    #     redirect = dash.no_update

    log_poll = False

    convergence_panel_hidden = False

    return status, dash.no_update, [], False, log_poll, "", convergence_panel_hidden


def run_solver_with_capture():

    global simulation_process

    simulation_process = subprocess.Popen(
        ["python", "euler2d/launch_euler2d.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in simulation_process.stdout:
        output_queue.put(line)
    output_queue.put("__DONE__")

@dash.callback(
    [Output("convergence-store", "data"),
    Output("live-graph", "figure"),
    Output("stop_solver", "disabled", allow_duplicate=True),
    Output("solver-console", "value"),
     Output("solver-status", "children", allow_duplicate=True),
     Output("log-poll", "disabled", allow_duplicate=True),
     Output("button-see-results", "disabled", allow_duplicate=True),],
    Input("log-poll", "n_intervals"),
    [State("convergence-store", "data"),
    State("solver-console", "value"),],
    prevent_initial_call=True,
)
def update_convergence_graph(n, data, console_data):

    disable_stop_button = dash.no_update

    solver_status = dash.no_update

    log_poll = dash.no_update

    button_see_results = dash.no_update

    global solver_start_time, solver_end_time

    if data is None:
        data = []

    if console_data is None:
        console_data = ""

    while not output_queue.empty():
        line = output_queue.get()
        console_data += f"{line}"

        if line.startswith("__ERROR__"):
            print("ERROR in simulation output")
            continue

        if line == "__DONE__":
            print("Simulation finished streaming")
            disable_stop_button = True
            log_poll = True
            button_see_results = False
            solver_end_time = time.perf_counter()

            solver_status = dbc.Alert(f"Simulation completed after: {solver_end_time - solver_start_time:.1f} s.", color="success")
            break
        else:
            parsed = parse_line(line)
            if parsed:
                data.append(parsed)

            solver_status = dbc.Alert(f"Solver is running. Up time: {time.perf_counter() - solver_start_time:.1f} s.", color="warning")

    # Plotting the convergence of residuals
    fig = go.Figure()
    if data:
        fig.add_trace(go.Scatter(x=[d["iter"] for d in data], y=[np.log10(d["res1"]) for d in data],
                                 mode="lines", name="rho"))

        # fig.add_trace(go.Scatter(x=[d["iter"] for d in data], y=[d["res2"] for d in data],
        #                          mode="lines+markers", name="rho_u"))
        # fig.add_trace(go.Scatter(x=[d["iter"] for d in data], y=[d["res3"] for d in data],
        #                          mode="lines+markers", name="rho_v"))
        # fig.add_trace(go.Scatter(x=[d["iter"] for d in data], y=[d["res4"] for d in data],
        #                          mode="lines+markers", name="rho_E"))

    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=-11,
        y1=-11,
        line=dict(
            color="black",
            width=1,
            dash="dash"
        )
    )

    fig.update_layout(xaxis_title="Iteration",
                      yaxis_title="Log10(L2 norm of RHO)",
                      legend_title="Variable")

    # fig.update_yaxes(title_text="Residual", type="log")

    return data, fig, disable_stop_button, console_data, solver_status, log_poll, button_see_results

def parse_line(line):
    pattern = r"Iteration:\s*(\d+)\s*:\s*L2_norms:\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*C_l:\s*([\d.eE+-]+)\s*C_d:\s*([\d.eE+-]+)\s*C_m:\s*([\d.eE+-]+)"
    match = re.search(pattern, line)
    if match:
        return {
            "iter": int(match.group(1)),
            "res1": float(match.group(2)),
            "res2": float(match.group(3)),
            "res3": float(match.group(4)),
            "res4": float(match.group(5)),
            "C_l": float(match.group(6)),
            "C_d": float(match.group(7)),
            "C_m": float(match.group(8)),
        }
    return None

@dash.callback(
    [Output("solver-status", "children"),
    Output("stop_solver", "disabled", allow_duplicate=True),
     Output("log-poll", "disabled"),],
    Input("stop_solver", "n_clicks"),
    prevent_initial_call=True,
)
def stop_simulation(n_clicks):
    global simulation_process

    if simulation_process and simulation_process.poll() is None:
        simulation_process.terminate()  # or .kill() if needed
        return dbc.Alert("Simulation stopped.", color="warning"), True, True
    else:
        return dbc.Alert("No running simulation to stop.", color="secondary"), True, dash.no_update