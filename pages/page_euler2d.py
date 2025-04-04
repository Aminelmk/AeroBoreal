import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from euler2d import euler_solver

dash.register_page(__name__, path="/page-euler2d")

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
            dcc.Input(id="Mach", type="number", min=0, max=10, value=0.5, style={"flex": "1"}),
        ], className="row d-flex align-items-center mt-2"),

        html.Div([
            html.Label("Angle of Attack (α): ", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
            dcc.Input(id="alpha", type="number", min=0, value=5.0, style={"flex": "1"}),
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
            dcc.Input(id="it_max", type="number", value=1000, style={"flex": "1"}),
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
        )
    ], justify="center"),

    dcc.Loading(
        id="solver-loading",
        type="default",
        children=html.Div(id='solver-status', className="mt-3")
    ),

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

 #########FIN CHANGEMENT##################       
# ======================================================================
# Visualization Page
# ======================================================================

viz_layout = dbc.Container([
    html.H1("Simulation Results", className="mb-4 text-center"),

    dbc.Row(
        dbc.Col(
            dcc.Dropdown(
                id='graph-selector',
                options=[],
                value='Density',
                clearable=False,
                className="mb-4",
                style={'minWidth': '300px'}
            ),
            width=8,
            className="text-center"
        ),
        justify="center"
    ),

    dbc.Row(
        dbc.Col(
            dcc.Graph(id='result-plot', style={'height': '70vh', 'width': '70%', 'margin': 'auto'}),
            width=100,
            className="d-flex justify-content-center"
        )
    ),

    dbc.Row(
        dbc.Col(
            dbc.Button(
                "Back to Configuration",
                href="/",
                color="secondary",
                className="mt-4"
            ),
            width="auto"
        ),
        justify="center",
        className="mb-4"
    )
], fluid=True, style={'height': '100vh'})


# ======================================================================
# Callbacks
# ======================================================================

@dash.callback(
    [Output('solver-status', 'children'),
     Output('visualization-redirect', 'children')],
    [Input('run_solver', 'n_clicks')],
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
    prevent_initial_call=True
)
def run_simulation(n_clicks, Mach, alpha, CFL_number, p_inf, T_inf,
                   multigrid, residual_smoothing, k2, k4, it_max):
     #########DEBUT CHANGEMENT##################      
    if not n_clicks:
        return dash.no_update, dash.no_update

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
output_file = test.q
checkpoint_file = checkpoint_test.txt"""

    with open("input.txt", "w") as f:
        f.write(input_content)

    # Exécution du solveur
    try:
        euler_solver.solve("input.txt")
        status = dbc.Alert("Simulation completed successfully!", color="success")
        redirect = dcc.Location(pathname="/page-euler2d-results", id="redirect")
    except Exception as e:
        status = dbc.Alert(f" Error: {str(e)}", color="danger")
        redirect = dash.no_update

    return status, redirect
