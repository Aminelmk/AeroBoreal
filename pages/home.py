import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import csv
import os

from mesh2D.naca4digits import Naca4Digits
from mesh2D.cst_class import CstAirfoil
from mesh2D.conformal_mapping import ConformalMapping

dash.register_page(__name__, path="/")

CSV_FILE = "user_inputs.csv"

INIT_CAMBER = 0
INIT_CAMBER_POS = 0
INIT_THICKNESS = 12

airfoil = None

# def naca_inputs(camber, camber_pos, thickness):
#     camber, camber_pos, thickness = int(camber), int(camber_pos), int(thickness)
#     airfoil = Naca4Digits(camber, camber_pos, thickness)
#     x, y = airfoil.get_all_surface(1000)
#
#     airfoil_fig = go.Figure()
#     airfoil_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Airfoil", line=dict(color="red")))
#
#     airfoil_fig.update_layout(title=f"NACA {camber}{camber_pos}{thickness}",
#                               xaxis_title="x/c", yaxis_title="y/c",
#                               xaxis_range=[-0.1, 1.1],
#                               yaxis_scaleanchor="x",
#                               showlegend=False, width=800, height=600)
#
#     return airfoil_fig #, mesh_fig,

# airfoil_graph = naca_inputs(INIT_CAMBER, INIT_CAMBER_POS, INIT_THICKNESS)


def display_airfoil(airfoil):

    x, y = airfoil.get_all_surface(1000)

    airfoil_fig = go.Figure()
    airfoil_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Airfoil", line=dict(color="red")))
    airfoil_fig.update_layout(title=f"Airfoil Profile",
                              xaxis_title="x/c", yaxis_title="y/c",
                              xaxis_range=[-0.1, 1.1],
                              yaxis_scaleanchor="x",
                              showlegend=False, width=800, height=600)

    return airfoil_fig


def update_naca_airfoil(camber, camber_pos, thickness):
    camber, camber_pos, thickness = int(camber), int(camber_pos), int(thickness)
    return Naca4Digits(camber, camber_pos, thickness)

def update_cst_airfoil(n_order, N1=0.5, N2=1.0):
    return CstAirfoil(n_order, N1=0.5, N2=1.0)


airfoil = update_naca_airfoil(INIT_CAMBER, INIT_CAMBER_POS, INIT_THICKNESS)
airfoil_fig = display_airfoil(airfoil)


mesh_controls = {
    "NACA": html.Div([
        html.Label("Max Camber (1/100)"),
        dcc.Slider(id="max_camber_slider", min=0, max=9, step=1, value=INIT_CAMBER,
                   marks={i: str(i) for i in range(0, 10)}),

        html.Label("Camber Position (1/10)"),
        dcc.Slider(id="max_camber_pos_slider", min=0, max=9, step=1, value=INIT_CAMBER_POS,
                   marks={i: str(i) for i in range(0, 10)}),

        html.Label("Max Thickness (%)"),
        dcc.Slider(id="max_thickness_slider", min=1, max=20, step=1, value=INIT_THICKNESS,
                   marks={i: str(i) for i in range(0, 100, 1)}),

        html.Label("Number of Points in Mesh"),
        dcc.Slider(id="nc_slider", min=8, max=128, step=8, value=32,
                   marks={i: str(i) for i in range(8, 129, 16)})
    ], className="controls"),

    "CST": html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'textAlign': 'center',
                                'height': '100%',  # Match button height
                                'display': 'flex',  # Align content
                                'alignItems': 'center',  # Center content vertically
                                'justifyContent': 'center'  # Center content horizontally
                            },
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
            ], width=8, style={'display': 'flex', 'alignItems': 'center'}),

            dbc.Col([
                html.Button('Fit Airfoil', id='fit-airfoil', n_clicks=0)
            ], width=4),
        ]),

        dbc.Row([
            dcc.Checklist(['Show Airfoil Points'], ['Show Airfoil Points'])
        ]),

        dbc.Row([
            dbc.Col([
                html.H5("N Order: ")
            ], width=8),
            dbc.Col([
                daq.NumericInput(
                    min=1,
                    max=7,
                    value=3,
                    id='n-order-input'
                )
            ], width=4)
        ]),

        dbc.Row(html.Div(id="coefficients-inputs")),

    ], className="controls"),
}

# 🔹 Layout for the Home Page
layout = html.Div([

    # ===== Geometry Generation =====
    html.Div([
        dbc.Container([
            dbc.Row([
                html.H2("Airfoil Geometry")
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="airfoil-plot", figure=airfoil_fig),

                ], width=8),

                dbc.Col([
                    dbc.Row([
                        dbc.Label("Select Method:", width="auto"),
                        dcc.Dropdown(
                            id="mesh-dropdown",
                            options=[{"label": key, "value": key} for key in mesh_controls.keys()],
                            value='NACA',
                            style={"flex": 1}  # Ensures it takes up remaining space
                        ),
                    ]),

                    html.Div(id="mesh-control-panel"),
                    html.Div([
                        html.H3("Naca Controls"),
                    html.Div([
                            html.Label("Max Camber (1/100)"),
                            dcc.Slider(id="max_camber_slider", min=0, max=9, step=1, value=INIT_CAMBER,
                                       marks={i: str(i) for i in range(0, 10)}),

                            html.Label("Camber Position (1/10)"),
                            dcc.Slider(id="max_camber_pos_slider", min=0, max=9, step=1, value=INIT_CAMBER_POS,
                                       marks={i: str(i) for i in range(0, 10)}),

                            html.Label("Max Thickness (%)"),
                            dcc.Slider(id="max_thickness_slider", min=1, max=20, step=1, value=INIT_THICKNESS,
                                       marks={i: str(i) for i in range(0, 100, 1)}),

                            html.Label("Number of Points in Mesh"),
                            dcc.Slider(id="nc_slider", min=8, max=128, step=8, value=32,
                                       marks={i: str(i) for i in range(8, 129, 16)})
                        ], className="controls"),

                    ], id="naca-controls", style={"display": "block"}),
                    html.Div([
                        html.H3("CST Controls"),
                    html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Upload(
                                                id='upload-data',
                                                children=html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select Files')
                                                ]),
                                                style={
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'textAlign': 'center',
                                                    'height': '100%',  # Match button height
                                                    'display': 'flex',  # Align content
                                                    'alignItems': 'center',  # Center content vertically
                                                    'justifyContent': 'center'  # Center content horizontally
                                                },
                                                # Allow multiple files to be uploaded
                                                multiple=False
                                            ),
                                ], width=8, style={'display': 'flex', 'alignItems': 'center'}),

                                dbc.Col([
                                    html.Button('Fit Airfoil', id='fit-airfoil', n_clicks=0)
                                ], width=4),
                            ]),

                            dbc.Row([
                                dcc.Checklist(['Show Airfoil Points'], ['Show Airfoil Points'])
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    html.H5("N Order: ")
                                ], width=8),
                                dbc.Col([
                                    daq.NumericInput(
                                        min=1,
                                        max=7,
                                        value=3,
                                        id='n-order-input'
                                    )
                                ], width=4)
                            ]),

                            dbc.Row(html.Div(id="coefficients-inputs")),

                        ], className="controls"),
                    ], id="cst-controls", style={"display": "none"}),
                ], width=4),
            ]),
        ]),
    ]),

    html.Div([
        html.Div([
            html.H3("Conformal Mesh"),
            dcc.Graph(id="mesh-plot")
        ], className="graph-container")
    ], className="visualization"),

    # Euler Solver Inputs
    html.H2("Euler Solver Inputs"),
    html.Div([
        dbc.Row([
            dbc.Col(dcc.Input(id="num_threads", type="number", placeholder="Threads")),
            dbc.Col(dcc.Input(id="mesh_file", type="text", placeholder="Mesh File", value="../mesh/lovell_elliptic_64.xyz")),
            dbc.Col(dcc.Input(id="multigrid", type="number", placeholder="multigrid")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Input(id="Mach", type="number", placeholder="Mach Number")),
            dbc.Col(dcc.Input(id="alpha", type="number", placeholder="Angle of Attack")),
            dbc.Col(dcc.Input(id="CFL_number", type="number", placeholder="CFL_number")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Input(id="p_inf", type="number", placeholder="Pressure (Pa)")),
            dbc.Col(dcc.Input(id="T_inf", type="number", placeholder="Temperature (K)")),
            dbc.Col(dcc.Input(id="residual_smoothing", type="number", placeholder="residual_smoothing")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Input(id="it_max", type="number", placeholder="Max Iterations")),
            dbc.Col(dcc.Input(id="output_file", type="text", placeholder="Output File")),
            dbc.Col(dcc.Input(id="k2", type="number", placeholder="k2")),
            dbc.Col(dcc.Input(id="k4", type="text", placeholder="k4")),
            
        ]),
        dbc.Row([dbc.Col(dcc.Input(id="checkpoint_file", type="text", placeholder="checkpoint_file")),
            
        ]),    
    ]),
    html.Button("Save Euler Inputs", id="save_euler", n_clicks=0),

    # VLM Solver Inputs
    html.H2("Vortex Lattice Method (VLM) Inputs"),
    html.Div([
        dbc.Row([
            dbc.Col(dcc.Input(id="nx", type="number", placeholder="Panels (nx)", )),
            dbc.Col(dcc.Input(id="ny", type="number", placeholder="Panels (ny)", )),
        ]),
        dbc.Row([
            dbc.Col(dcc.Input(id="AR", type="number", placeholder="Aspect Ratio", )),
            dbc.Col(dcc.Input(id="alpha_input", type="number", placeholder="Angle of Attack (°)", )),
        ]),
        dbc.Row([
            dbc.Col(dcc.Input(id="vlm_p_inf", type="number", placeholder="Pressure (Pa)")),
            dbc.Col(dcc.Input(id="vlm_T_inf", type="number", placeholder="Temperature (K)")),
            dbc.Col(dcc.Input(id="vlm_Mach", type="number", placeholder="Mach Number")),
        ]),
    ]),
    html.Button("Save VLM Inputs", id="save_vlm", n_clicks=0),

    # Display Previous Entries
    html.Div([
        html.H3("Previous Entries"),
        html.Ul(id="history-list")
    ]),
])

# ===== Controls which parameters to display =====
@dash.callback(
     [Output("naca-controls", "style"),
     Output("cst-controls", "style")],
    Input("mesh-dropdown", "value"),
)
def update_control_panel(selected_option):

    global airfoil

    if selected_option is None:
        return html.Div("Select an option to display geometry controls.")

    print(f"selected option is: {selected_option}")

    if selected_option == "NACA":
        airfoil = update_naca_airfoil(INIT_CAMBER, INIT_CAMBER_POS, INIT_THICKNESS)
        return {"display": "block"}, {"display": "none"}
    elif selected_option == "CST":
        airfoil = update_cst_airfoil(3, N1=0.5, N2=1.0)
        return {"display": "none"}, {"display": "block"}

@dash.callback(
    Output("coefficients-inputs", "children"),
    Input("n-order-input", "value")
)
def update_coefficients_inputs(n_order):

    # if type(airfoil) != CstAirfoil:
    #     airfoil = CstAirfoil(3, N1=0.5, N2=1.0)

    global airfoil

    print(f"Airfoil Type: {type(airfoil)}")
    if type(airfoil) is CstAirfoil:
        print("works")
        airfoil.set_n_order(n_order)

        return [
            dbc.Row([
                dbc.Col(html.Label(f"A_upper[{i}]: "), width=2),
                dbc.Col(dcc.Input(
                    id={"type": "A_upper", "index": i},
                    type="number",
                    min=-3,
                    max=3,
                    value=airfoil.A_upper[i],
                    step=0.1,
                ), width=4),
                dbc.Col(html.Label(f"A_lower[{i}]: "), width=2),
                dbc.Col(dcc.Input(
                    id={"type": "A_lower", "index": i},
                    type="number",
                    min=-3,
                    max=3,
                    value=airfoil.A_lower[i],
                    step=0.1,
                ), width=4)
            ]) for i in range(n_order + 1)
        ]
    else:
        return dash.no_update

# ========== CALLBACK TO UPDATE AIRFOIL ==========
@dash.callback(
    Output("airfoil-plot", "figure"),
    [
        Input("mesh-dropdown", "value"),
        Input("max_camber_slider", "value"),
        Input("max_camber_pos_slider", "value"),
        Input("max_thickness_slider", "value"),
        Input({"type": "A_upper", "index": dash.ALL}, "value"),
        Input({"type": "A_lower", "index": dash.ALL}, "value"),
        State("n-order-input", "value"),  # ✅ Use State to avoid errors
    ],
    prevent_initial_call=True
)
def update_airfoil(selected_option, camber, camber_pos, thickness, A_upper_values, A_lower_values, n_order):
    """Handles both NACA and CST airfoil updates in a single callback."""

    triggered_id = ctx.triggered_id

    if selected_option == "NACA" and triggered_id in ["max_camber_slider", "max_camber_pos_slider",
                                                      "max_thickness_slider"]:
        airfoil = update_naca_airfoil(camber, camber_pos, thickness)
    elif selected_option == "CST" and n_order is not None:
        airfoil = update_cst_airfoil(n_order, N1=0.5, N2=1.0)
        if A_upper_values is not None and A_lower_values is not None:
            airfoil.A_upper = np.array(A_upper_values)
            airfoil.A_lower = np.array(A_lower_values)
    else:
        return dash.no_update

    return display_airfoil(airfoil)


# 🔹 Callback to Update Graphs Dynamically
@dash.callback(
    [Output("mesh-plot", "figure"),
        Output("history-list", "children")],
    [Input("nc_slider", "value"),
        Input("save_euler", "n_clicks"),
        Input("save_vlm", "n_clicks")],
    [
        State("num_threads", "value"),
        State("mesh_file", "value"),
        State("Mach", "value"),
        State("alpha", "value"),
        State("p_inf", "value"),
        State("T_inf", "value"),
        State("it_max", "value"),
        State("output_file", "value"),
        
        State("multigrid", "value"),
        State("CFL_number", "value"),
        State("residual_smoothing", "value"),
        State("k2", "value"),
        State("k4", "value"),
        State("checkpoint_file", "value"),

        State("nx", "value"),
        State("ny", "value"),
        State("AR", "value"),
        State("alpha_input", "value"),
        State("vlm_p_inf", "value"),
        State("vlm_T_inf", "value"),
        State("vlm_Mach", "value"),
    ],
    prevent_initial_call=True
)
def update_visualizations_and_save(nc,
                                   save_euler_clicks, save_vlm_clicks,
                                   num_threads, mesh_file, Mach, alpha, p_inf, T_inf, it_max, output_file
                                   ,multigrid, CFL_number, residual_smoothing, k2, k4, checkpoint_file, nx, ny, AR, alpha_input, vlm_p_inf, vlm_T_inf, vlm_Mach):
    """Update airfoil, mesh plots, and save inputs dynamically"""
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Update airfoil and mesh
    nc = int(nc)

    # cm = ConformalMapping(airfoil, nc)
    # cm.generate_mesh_nodes()
    # mesh_fig = mesh_to_plotly(cm.all_nodes)
    #
    # save_to_csv([camber, camber_pos, thickness, nc])
    history_entries = load_csv_entries()

    # Handle input saving
    if triggered_id == "save_euler":
        euler_params = {
            "num_threads": num_threads, "mesh_file": mesh_file, "Mach": Mach,
            "multigrid": multigrid, "CFL_number": CFL_number, "residual_smoothing": residual_smoothing,
            "k2": k2, "k4": k4, "checkpoint_file": checkpoint_file,
            "alpha": alpha, "p_inf": p_inf, "T_inf": T_inf, "it_max": it_max, "output_file": output_file
        }
        save_to_file("euler_input.txt", euler_params)
        history_entries += [f"{k} = {v}" for k, v in euler_params.items()]

    elif triggered_id == "save_vlm":
        vlm_params = {
            "nx": nx, "ny": ny, "AR": AR, "alpha": alpha_input,
            "p_inf": vlm_p_inf, "T_inf": vlm_T_inf, "Mach": vlm_Mach
        }
        save_to_file("vlm_input.txt", vlm_params)
        history_entries += [f"{k} = {v}" for k, v in vlm_params.items()]

    return [html.Li(entry) for entry in history_entries]


# 🔹 Functions to Manage CSV History
def save_to_csv(values):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Camber", "Pos Camber", "Thickness", "NC"])
        writer.writerow(values)

def load_csv_entries():
    if not os.path.isfile(CSV_FILE):
        return []
    with open(CSV_FILE, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header
        return [", ".join(row) for row in reader]


# 🔹 Function to Convert Mesh Data to Plotly
def mesh_to_plotly(all_nodes):
    nodes = np.array(all_nodes)
    n_i, n_j, _ = nodes.shape
    mesh_fig = go.Figure()

    # Draw horizontal lines
    for i in range(n_i):
        x_line, y_line = nodes[i, :, 0], nodes[i, :, 1]
        mesh_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="blue")))

    # Draw vertical lines
    for j in range(n_j):
        x_line, y_line = nodes[:, j, 0], nodes[:, j, 1]
        mesh_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="blue")))

    # Add mesh nodes as red dots
    mesh_fig.add_trace(go.Scatter(x=nodes[:,:,0].flatten(), y=nodes[:,:,1].flatten(), mode="markers", marker=dict(size=4, color="red")))

    mesh_fig.update_layout(
        title="Conformal Mesh",
        xaxis_title="X", yaxis_title="Y",
        showlegend=False, width=600, height=600
    )
    return mesh_fig


# Obtenir le répertoire du script Dash (là où se trouve le fichier Python)
SAVE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Ensure the directory exists
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

def save_to_file(filename, data_dict):
    """Save parameters to a specified folder."""
    file_path = os.path.join(SAVE_DIRECTORY, filename)
    
    with open(file_path, "w") as file:
        for key, value in data_dict.items():
            file.write(f"{key} = {value}\n")
    
    print(f"File saved at: {os.path.abspath(file_path)}")  # Debugging info


def save_inputs(save_euler_clicks, save_vlm_clicks,
                num_threads, mesh_file, Mach, alpha, p_inf, T_inf, it_max, output_file,
                nx, ny, AR, alpha_input, vlm_p_inf, vlm_T_inf, vlm_Mach):
    """Save inputs to separate files depending on which button was clicked."""
    
    ctx = dash.callback_context  # Get the button that triggered the callback
    if not ctx.triggered:
        return dash.no_update  # If no button was clicked, do nothing

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]  # Identify the button

    if button_id == "save_euler":
        euler_params = {
            "num_threads": num_threads,
            "mesh_file": mesh_file,
            "Mach": Mach,
            "alpha": alpha,
            "p_inf": p_inf,
            "T_inf": T_inf,
            "it_max": it_max,
            "output_file": output_file,
            "multigrid": multigrip,
            "CFL_number": CFL_number,
            "residual_smoothing": residual_smoothing,
            "k2": k2, 
            "k4": k4, 
            "checkpoint_file": checkpoint_file,
        }
        save_to_file("euler_input.txt", euler_params)
        return [html.Li(f"{k} = {v}") for k, v in euler_params.items()]

    elif button_id == "save_vlm":
        vlm_params = {
            "nx": nx,
            "ny": ny,
            "AR": AR,
            "alpha": alpha_input,
            "p_inf": vlm_p_inf,
            "T_inf": vlm_T_inf,
            "Mach": vlm_Mach
        }
        save_to_file("vlm_input.txt", vlm_params)
        return [html.Li(f"{k} = {v}") for k, v in vlm_params.items()]

    return dash.no_update   