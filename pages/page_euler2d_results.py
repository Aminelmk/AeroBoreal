

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import datetime


dash.register_page(__name__, path="/page-euler2d-results")

# ======================================================================
# Fonctions de traitement
# ======================================================================

def parse_mesh(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    nx, ny = map(int, lines[0].split())
    x_grid = np.array([float(value) for value in lines[1: 1 + nx * ny]])
    y_grid = np.array([float(value) for value in lines[1 + nx * ny: 1 + 2 * nx * ny]])
    return nx, ny, y_grid.reshape(ny, nx), -x_grid.reshape(ny, nx)

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

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "surface"}]],
        subplot_titles=[title]
    )

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

    fig.update_layout(
        width=900,
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
                eye=dict(x=0, y=0.0, z=1.3),  # Vue de dessus, bien centrée
                up=dict(x=1, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0)
        )
    )

    return fig


# ======================================================================
# Layout 
# ======================================================================

layout = html.Div([
    html.Div(style={
        'position': 'fixed',
        'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
        'backgroundImage': 'url("https://blog.spatial.com/hubfs/AdobeStock_88961670.jpeg")',
        'backgroundSize': 'cover',
        'backgroundRepeat': 'no-repeat',
        'backgroundPosition': 'center',
        'opacity': '0.75',
        'zIndex': -1
    }),

    dbc.Container([
        html.H1("Simulation Results", className="mb-4 text-center"),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='graph-selector',
                    options=[],
                    value='Density',
                    clearable=False,
                    className="mb-3"
                )
            ], width=4),
            dbc.Col([
                html.Div(id="mesh-filename", className="text-white text-center fw-bold")
            ], width=4)
        ], justify="center"),

        dbc.Row([
            dbc.Col([
                html.Div(
                    dcc.Graph(id='result-plot', style={'height': '90vh'}),
                    className="d-flex justify-content-center"
                )
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Button("⬇ Télécharger test.q", id="download-button", color="success", className="mt-3"),
                dcc.Download(id="download-result")
            ], width="auto")
        ], justify="center"),

        dbc.Row(
            dbc.Col(
                dbc.Button("Retour à la configuration", href="/", color="secondary", className="mt-4"),
                width="auto"
            ),
            justify="center",
            className="mb-4"
        ),

        dcc.Location(id='url', refresh=False),
        dcc.Store(id='selected-mesh-file', storage_type='session'),
        dcc.Interval(id='load-results-trigger', interval=1, max_intervals=1),
        html.Div(id='page-content')
    ], fluid=True, style={'height': '100vh', 'textAlign': 'center'})
])

# ======================================================================
# Callbacks Dash 
# ======================================================================

@dash.callback(
    [Output('graph-selector', 'options'),
     Output('graph-selector', 'value')],
    [Input('url', 'pathname')]
)
def initialize_results(pathname):
    return [
        {'label': 'Density', 'value': 'Density'},
        {'label': 'Momentum X', 'value': 'Momentum X'},
        {'label': 'Momentum Y', 'value': 'Momentum Y'},
        {'label': 'Energy', 'value': 'Energy'},
        {'label': 'Mach Number', 'value': 'Mach Number'}
    ], "Density"

@dash.callback(
    [Output('result-plot', 'figure'),
     Output('mesh-filename', 'children')],
    [Input('graph-selector', 'value'),
     Input('load-results-trigger', 'n_intervals')],
    [State('selected-mesh-file', 'data'),
    ]
)
def update_graph(selected_graph, trigger, mesh_file):
    try:
        mesh_file = mesh_file or "x.6"
        nx, ny, x_2d, y_2d = parse_mesh(mesh_file)
        rho, rho_u, rho_v, rho_E = parse_test_q("test.q", nx, ny)

        gamma = 1.4
        u = rho_u / rho
        v = rho_v / rho
        p = (gamma - 1) * (rho_E - 0.5 * rho * (u**2 + v**2))
        a = np.sqrt(gamma * p / rho)
        Mach = np.sqrt(u**2 + v**2) / a

        data_map = {
            "Density": (rho, "Density", "kg/m³"),
            "Momentum X": (rho_u, "Momentum X", "kg/(m²s)"),
            "Momentum Y": (rho_v, "Momentum Y", "kg/(m²s)"),
            "Energy": (rho_E, "Energy", "J/m³"),
            "Mach Number": (Mach, "Mach Number", "Mach")
        }

        variable, title, unit = data_map.get(selected_graph, (rho, "Density", "kg/m³"))
        fig = create_surface_plot(variable, title, unit, x_2d, y_2d)

        mesh_label = f"📂 Fichier maillage : {mesh_file}"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} → {selected_graph} sur {os.path.basename(mesh_file)}"

        return fig, mesh_label

    except Exception as e:
        print(f"Erreur update_graph : {e}")
        return go.Figure()

@dash.callback(
    Output("download-result", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_q_file(n):
    return dcc.send_file("test.q")
