import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import csv
import os

from mesh2D.naca4digits import Naca4Digits
from mesh2D.conformal_mapping import ConformalMapping

dash.register_page(__name__, path="/home")

CSV_FILE = "user_inputs.csv"

# 🔹 Layout for the Home Page
layout = html.Div([
    html.H1("NACA Airfoil Generator with Conformal Mesh"),
    
    # Sliders for Airfoil Parameters
    html.Div([
        html.Label("Max Camber (%)"),
        dcc.Slider(id="max_camber_slider", min=0, max=9, step=1, value=2,
                   marks={i: str(i) for i in range(0, 10)}),

        html.Label("Camber Position (%)"),
        dcc.Slider(id="max_camber_pos_slider", min=0, max=9, step=1, value=4,
                   marks={i: str(i) for i in range(0, 10)}),

        html.Label("Max Thickness (%)"),
        dcc.Slider(id="max_thickness_slider", min=0, max=99, step=1, value=12,
                   marks={i: str(i) for i in range(0, 100, 10)}),

        html.Label("Number of Points in Mesh"),
        dcc.Slider(id="nc_slider", min=8, max=128, step=8, value=32,
                   marks={i: str(i) for i in range(8, 129, 16)})
    ], className="controls"),

    html.Div([
        html.Div([
            html.H3("NACA Airfoil Profile"),
            dcc.Graph(id="airfoil-plot")
        ], className="graph-container"),

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


# 🔹 Callback to Update Graphs Dynamically
@dash.callback(
    [Output("airfoil-plot", "figure"),
     Output("mesh-plot", "figure"),
     Output("history-list", "children")],
    [Input("max_camber_slider", "value"),
     Input("max_camber_pos_slider", "value"),
     Input("max_thickness_slider", "value"),
     Input("nc_slider", "value"),
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
def update_visualizations_and_save(camber, camber_pos, thickness, nc,
                                   save_euler_clicks, save_vlm_clicks,
                                   num_threads, mesh_file, Mach, alpha, p_inf, T_inf, it_max, output_file
                                   ,multigrid, CFL_number, residual_smoothing, k2, k4, checkpoint_file, nx, ny, AR, alpha_input, vlm_p_inf, vlm_T_inf, vlm_Mach):
    """Update airfoil, mesh plots, and save inputs dynamically"""
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Update airfoil and mesh
    camber, camber_pos, thickness, nc = int(camber), int(camber_pos), int(thickness), int(nc)
    airfoil = Naca4Digits(camber, camber_pos, thickness)
    x, y = airfoil.get_all_surface(1000)

    airfoil_fig = go.Figure()
    airfoil_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Airfoil", line=dict(color="red")))
    airfoil_fig.update_layout(title=f"NACA {camber}{camber_pos}{thickness}",
                              xaxis_title="x/c", yaxis_title="y/c",
                              showlegend=False, width=800, height=600)

    cm = ConformalMapping(airfoil, nc)
    cm.generate_mesh_nodes()
    mesh_fig = mesh_to_plotly(cm.all_nodes)

    save_to_csv([camber, camber_pos, thickness, nc])
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

    return airfoil_fig, mesh_fig, [html.Li(entry) for entry in history_entries]


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