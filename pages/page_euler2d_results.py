import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from scipy.interpolate import griddata

dash.register_page(__name__, path="/page-euler2d-results")

# ======================================================================
# Data Processing Functions
# ======================================================================

def parse_mesh(file_path):
    print("trying to parse mesh")

    with open(file_path, "r") as f:
        lines = f.readlines()
    nx, ny = map(int, lines[0].split())
    x_grid = np.array([float(value) for value in lines[1: 1 + nx * ny]])
    y_grid = np.array([float(value) for value in lines[1 + nx * ny: 1 + 2 * nx * ny]])

    return nx, ny, x_grid.reshape(ny, nx), y_grid.reshape(ny, nx)


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
    # fig = make_subplots(
    #     rows=1, cols=1,
    #     specs=[[{"type": "surface"}]],
    #     subplot_titles=[title]
    # )

    fig = go.Figure()

    # Main surface plot with integrated colorbar
    # fig.add_trace(
    #     go.Surface(
    #         x=x_2d,
    #         y=y_2d,
    #         z=z,
    #         surfacecolor=variable,
    #         colorscale='Magma',
    #         lighting=dict(ambient=1.0, diffuse=0.0),
    #         showscale=True,
    #         colorbar=dict(
    #             title=f"{colorbar_title}<br>Min: {min_val:.2f}<br>Max: {max_val:.2f}",
    #             tickvals=[min_val, max_val],
    #             ticktext=[f"{min_val:.2f}", f"{max_val:.2f}"],
    #             tickfont=dict(size=10),
    #             thickness=50,
    #             len=0.75
    #         )
    #     )
    # )

    grid_x = np.linspace(-5, 5, 1_000)
    grid_z = np.linspace(-5, 5, 1_000)
    grid_X, grid_Z = np.meshgrid(grid_x, grid_z)

    airfoil_len = 2*x_2d.shape[1]

    x_flat = x_2d.flatten()
    y_flat = y_2d.flatten()
    variable = variable.flatten()

    points = np.column_stack((x_flat, y_flat))
    grid_v = griddata(points, variable, (grid_X, grid_Z), method='cubic', rescale=False)

    fig.add_trace(go.Contour(x=grid_x,
                             y=grid_z,
                             z=grid_v,
                             line=dict(width=0),
                             # line_smoothing=0.85,
                             contours_coloring='heatmap',
                             ),
                  )

    # fig.add_trace(go.Scatter(x=x_flat,y=y_flat, mode="markers", marker=dict(size=4, color="black")))
    # fig.add_trace(go.Scatter(x=x_2d[0,:airfoil_len+1], y=y_2d[0,:airfoil_len+1], fill='toself', fillcolor='white', mode="lines+markers", marker=dict(size=4, color="black")))
    fig.add_trace(go.Scatter(x=x_2d[0,:airfoil_len+1], y=y_2d[0,:airfoil_len+1], fill='toself', fillcolor='white', mode='none'))

    # Layout adjustments
    # fig.update_layout(
    #     width=1200,
    #     height=600,
    #     margin=dict(l=0, r=0, b=0, t=0),
    #     scene=dict(
    #         xaxis=dict(
    #             range=[-0.3, 0.3],
    #             showgrid=False,
    #             showline=False,
    #             zeroline=False,
    #             visible=False
    #         ),
    #         yaxis=dict(
    #             range=[-1.1, 0.1],
    #             showgrid=False,
    #             showline=False,
    #             zeroline=False,
    #             visible=False
    #         ),
    #         zaxis=dict(
    #             showticklabels=False,
    #             title='',
    #             visible=False,
    #             showgrid=False,
    #             showline=False,
    #             zeroline=False
    #         ),
    #         camera=dict(
    #             eye=dict(x=0, y=0, z=1.2),
    #             # projection=dict(type='orthographic')
    #         ),
    #         aspectmode='manual',
    #         aspectratio=dict(x=1, y=1, z=0)
    #     ),
    #     # dragmode="pan",
    # )
    # TODO: Review figure is upside down

    # fig.update_layout(modebar_remove=['orbitRotation', 'tableRotation'])

    fig.update_layout(
        width=1200,
        height=600,
        xaxis_title="x/c",
        yaxis_title="y/c",
        xaxis_range=[-0.1, 1.1],
        yaxis_range=[-0.4, 0.4],
        yaxis_scaleanchor="x",
        dragmode="pan",
    )

    return fig


# ======================================================================
# Visualization Page
# ======================================================================

layout = dbc.Container([
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
            dcc.Graph(id='result-plot', config={'scrollZoom':True}, style={'height': '70vh', 'width': '70%', 'margin': 'auto'}),
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
    ),

    dcc.Location(id='url', refresh=False),
    dcc.Store(id="simulation-status", data={"complete": False}),  # Store to track progress
    html.Div(id='page-content')

    ], fluid=True, style={'height': '100vh'})


# ======================================================================
# Callbacks
# ======================================================================


@dash.callback(
    [Output('graph-selector', 'options'),
     Output('graph-selector', 'value')],
    [Input('url', 'pathname')]
)
def initialize_results(pathname):

    try:
        return [
            {'label': 'Density', 'value': 'Density'},
            {'label': 'Momentum X', 'value': 'Momentum X'},
            {'label': 'Momentum Y', 'value': 'Momentum Y'},
            {'label': 'Energy', 'value': 'Energy'},
            {'label': 'Mach Number', 'value': 'Mach Number'}
        ], "Density"
    except Exception as e:
        return [], None

 #####DEBUT CHANGEMENT########
def maillage_depuis_input():
    with open("input.txt", "r") as f:
        for line in f:
            if line.startswith("mesh_file"):
                return line.split("=")[1].strip()
    return "temp/mesh.xyz"
@dash.callback(
    Output('result-plot', 'figure'),
    [Input('graph-selector', 'value')],
    [State('url', 'pathname')]
)

def update_selected_graph(selected_graph, pathname):
    print(f"Graph update triggered - pathname: {pathname}, selected_graph: {selected_graph}")

    try:

        mesh_path = maillage_depuis_input()
        nx, ny, x_2d, y_2d = parse_mesh(mesh_path)
        #####FIN CHANGEMENT########
        rho, rho_u, rho_v, rho_E = parse_test_q("test.q", nx, ny)

        print(f"Mesh loaded - nx: {nx}, ny: {ny}")  # Debugging
        print(f"len(x_2d): {x_2d.shape}")

        gamma = 1.4
        u = rho_u / rho
        v = rho_v / rho
        p = (gamma - 1) * (rho_E - 0.5 * rho * (u ** 2 + v ** 2))
        a = np.sqrt(gamma * p / rho)
        Mach = np.sqrt(u ** 2 + v ** 2) / a

        figures = {
            "Density": create_surface_plot(rho, "Density", "kg/m³", x_2d, y_2d),
            "Momentum X": create_surface_plot(rho_u, "Momentum X", "kg/(m²s)", x_2d, y_2d),
            "Momentum Y": create_surface_plot(rho_v, "Momentum Y", "kg/(m²s)", x_2d, y_2d),
            "Energy": create_surface_plot(rho_E, "Energy", "J/m³", x_2d, y_2d),
            "Mach Number": create_surface_plot(Mach, "Mach Number", "Mach", x_2d, y_2d)
        }

        print(f"Returning graph for {selected_graph}")  # Debugging
        return figures.get(selected_graph, go.Figure())

    except Exception as e:
        print(f"Error in graph update: {e}")
        return go.Figure()