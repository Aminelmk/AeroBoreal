import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os
import dash_daq as daq
import plotly.figure_factory as ff

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "euler2d")))
from scipy.interpolate import griddata

from post_process import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff

dash.register_page(__name__, path="/page-euler2d-results")





nx, ny = None, None
x, y = None, None

grid_x, grid_y = None, None

dep_var = {
    "Density": None,
    "Momentum X": None,
    "Momentum Y": None,
    "Energy": None,
    "Mach": None,
    "u": None,
    "v": None,
}

interp_var = {
    "Density": None,
    "Momentum X": None,
    "Momentum Y": None,
    "Energy": None,
    "Mach": None,
    "u": None,
    "v": None,
}

coeff = {
    "CL": None,
    "CD": None,
    "CM": None,
}

airfoil_x = None
airfoil_y = None


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

    grid_x = np.linspace(-5, 5, 100)
    grid_z = np.linspace(-5, 5, 100)
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

    html.Div([
    
        html.H1("Simulation Results", className="mb-4 text-center"),

        dbc.Row(
            [
                dbc.Col(dcc.Input(id='mesh-file', type='text', value="../temp/mesh.xyz", className="me-2"),
                        width="auto"),
                dbc.Col(dcc.Input(id='result', type='text', value="/test.q", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Load results", id="load-results", n_clicks=0), width="auto"),
            ],
            className="d-flex align-items-center justify-content-center",
        ),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Aerodynamic Coefficients", className="card-title"),
                        html.Div(id='coefficients-output', className="card-text")
                    ]),
                    className="mt-4 mb-4",
                    style={"width": "50%", "marginTop": "0px", "marginBottom": "40px", "marginLeft": "auto",
                           "marginRight": "auto", "textAlign": "center"}
                ))
        ]),

        dbc.Card(
            dbc.CardBody([

                html.Div([
                    html.Label("Show mesh: ", className="me-2",
                               style={"text-align": "left", "white-space": "nowrap", "display": "flex",
                                      "align-items": "center"}),
                    daq.BooleanSwitch(id="show-mesh", on=False, style={"margin-top": "-15px"})
                ], className="d-flex align-items-center justify-content-between w-90", style={"width": "80%", "margin": "0 auto"}),

                html.Div([
                    html.Label("Show streamlines:", className="me-2",
                               style={"text-align": "left", "white-space": "nowrap", "display": "flex",
                                      "align-items": "center"}),
                    daq.BooleanSwitch(id="show-streamlines", on=False, style={"margin-top": "-15px"})
                ], className="d-flex align-items-center justify-content-between align-items-center",
                    style={"width": "80%", "justify-content": "center", "margin": "0 auto"}),

                html.Br(),

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
                        width=10,
                        className="text-center"
                    ),
                    justify="center"
                ),

            ]),
            className="mt-4 mb-4",
            style={"width": "50%", "marginTop": "0px", "marginBottom": "40px", "marginLeft": "auto",
                "marginRight": "auto", "textAlign": "center", "justify-content": "center"}
        ),

        dbc.Row(
            dcc.Graph(id='result-plot', config={'scrollZoom':True}, style={'width': '95%', 'height': 'auto'}),
            # dcc.Graph(id='result-plot', config={'scrollZoom':True}, style={'height': '70vh', 'width': '70%', "marginTop": "10px", "marginBottom": "40px", "marginLeft": "70px",
            #    "marginRight": "auto",'display': 'block' }),
            # width=100,
            className="d-flex justify-content-center"
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
        html.Hr(style={"width": "60%", "margin": "60px auto 20px auto", "borderTop": "1px solid #ccc"}),

        dcc.Location(id='url', refresh=False),
        dcc.Store(id="simulation-status", data={"complete": False}),
        html.Div(id='page-content')

    ]),
], fluid=False)





def interpolate(x, y, var):

    grid_x = np.linspace(-5, 5, 1_000)
    grid_y = np.linspace(-5, 5, 1_000)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    x_flat = x.flatten()
    y_flat = y.flatten()
    var_flat = var.flatten()

    points = np.column_stack((x_flat, y_flat))
    grid_var = griddata(points, var_flat, (grid_X, grid_Y), method='linear', rescale=False)

    return grid_x, grid_y, grid_var


# ======================================================================
# Callbacks
# ======================================================================

@dash.callback(
    [Output("coefficients-output", "children"),
    Output("graph-selector", "options"),
     Output("graph-selector", "value")],
    Input("load-results", "n_clicks"),

)
def load_results(load_button):

    global nx, ny, x, y, grid_x, grid_y, dep_var, interp_var, coeff, airfoil_x, airfoil_y

    mesh_path = maillage_depuis_input()
    nx, ny, x, y = parse_mesh(mesh_path)
    rho, rho_u, rho_v, rho_E = parse_test_q("test.q", nx, ny)

    gamma = 1.4
    u = rho_u / rho
    v = rho_v / rho
    pressure = (gamma - 1) * (rho_E - 0.5 * rho * (u ** 2 + v ** 2))
    a = np.sqrt(gamma * pressure / rho)
    mach = np.sqrt(u ** 2 + v ** 2) / a
    energy = rho_E / rho

    dep_var = {
        "Density": rho,
        "Momentum X": rho_u,
        "Momentum Y": rho_v,
        "Energy": energy,
        "Mach": mach,
        "u": u,
        "v": v,
    }

    for key, var in dep_var.items():

        grid_x, grid_y, grid_var = interpolate(x, y, var)
        interp_var[key] = grid_var

    dropdown_options = []

    for key in interp_var.keys():
        dropdown_options.append(
            {'label': key, 'value': key}
        )

    C_L, C_D, C_M = calculer_coefficients()

    coeff = {
        "CL": C_L,
        "CD": C_D,
        "CM": C_M,
    }

    if C_L is not None:
        coeff_txt = f"CL = {C_L}   |   CD = {C_D}   |   CM = {C_M}"
    else:
        coeff_txt = f"Please load results."

    airfoil_len = 2*x.shape[1]
    airfoil_x = x[0, :airfoil_len+1]
    airfoil_y = y[0,:airfoil_len+1]

    return coeff_txt, dropdown_options, "Density"

# @dash.callback(
#     [Output('graph-selector', 'options'),
#      Output('graph-selector', 'value')],
#     [Input('url', 'pathname')]
# )
# def initialize_results(pathname):
#
#     try:
#         return [
#             {'label': 'Density', 'value': 'Density'},
#             {'label': 'Momentum X', 'value': 'Momentum X'},
#             {'label': 'Momentum Y', 'value': 'Momentum Y'},
#             {'label': 'Energy', 'value': 'Energy'},
#             {'label': 'Mach Number', 'value': 'Mach Number'}
#         ], "Density"
#     except Exception as e:
#         return [], None
    
def maillage_depuis_input():
    with open("input.txt", "r") as f:
        for line in f:
            if line.startswith("mesh_file"):
                return line.split("=")[1].strip()
    return "temp/mesh.xyz"

def calculer_coefficients():
    try:
        mesh_path = maillage_depuis_input()
        x, y = read_PLOT3D_mesh(mesh_path)
        _, _, mach, alpha, _, _, q_vertex = read_plot3d_2d("test.q")
        _, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1e5)
        return round(C_L, 4), round(C_D, 4), round(C_M, 4)
    except Exception as e:
        print(f"Erreur calcul coefficients : {e}")
        return None, None, None


@dash.callback(
    Output('result-plot', 'figure'),
    [Input('graph-selector', 'value'),
    Input('show-mesh', 'on'),
    Input('show-streamlines', 'on')],
)
def update_fig(selected_graph, show_mesh, show_streamlines):

    global grid_x, grid_y, interp_var, airfoil_x, airfoil_y

    fig = go.Figure()

    if show_streamlines:

        u = interp_var["u"]
        v = interp_var["v"]

        fig = ff.create_streamline(grid_x, grid_y, u, v,
                                   arrow_scale=.05,
                                   density=5,
                                   line_color="black",)

    else:
        pass


    if selected_graph is not None:
        fig.add_trace(go.Contour(x=grid_x,
                                 y=grid_y,
                                 z=interp_var[selected_graph],
                                 line=dict(width=0),
                                 # line_smoothing=0.85,
                                 contours_coloring='heatmap',
                                 colorscale='Viridis',
                                 ),
                      )


    fig.add_trace(go.Scatter(x=airfoil_x, y=airfoil_y,
                             fill='toself',
                             fillcolor='white',
                             mode='none'))

    fig.update_layout(
        height=800,
        xaxis_title="x/c",
        yaxis_title="y/c",
        xaxis=dict(
            range=[-0.2, 1.2],
            autorange=False,  # Disables automatic axis adjustment
        ),
        # xaxis_range=[-0.1, 1.1],
        yaxis_range=[-0.6, 0.6],
        yaxis_scaleanchor="x",
        dragmode="pan",
    )

    return fig

    # return dash.no_update


# @dash.callback(
#     [Output('result-plot', 'figure'),
#      Output('coefficients-output', 'children')],
#     [Input('graph-selector', 'value')],
#     [State('url', 'pathname')]
# )
#
# def update_selected_graph(selected_graph, pathname):
#     print(f"Graph update triggered - pathname: {pathname}, selected_graph: {selected_graph}")
#
#     try:
#         mesh_path = maillage_depuis_input()
#         nx, ny, x_2d, y_2d = parse_mesh(mesh_path)
#         rho, rho_u, rho_v, rho_E = parse_test_q("test.q", nx, ny)
#
#         gamma = 1.4
#         u = rho_u / rho
#         v = rho_v / rho
#         p = (gamma - 1) * (rho_E - 0.5 * rho * (u ** 2 + v ** 2))
#         a = np.sqrt(gamma * p / rho)
#         Mach = np.sqrt(u ** 2 + v ** 2) / a
#
#         figures = {
#             "Density": create_surface_plot(rho, "Density", "kg/m³", x_2d, y_2d),
#             "Momentum X": create_surface_plot(rho_u, "Momentum X", "kg/(m²s)", x_2d, y_2d),
#             "Momentum Y": create_surface_plot(rho_v, "Momentum Y", "kg/(m²s)", x_2d, y_2d),
#             "Energy": create_surface_plot(rho_E, "Energy", "J/m³", x_2d, y_2d),
#             "Mach Number": create_surface_plot(Mach, "Mach Number", "Mach", x_2d, y_2d)
#         }
#
#         C_L, C_D, C_M = calculer_coefficients()
#         if C_L is not None:
#             coeffs_text = f"CL = {C_L}   |   CD = {C_D}   |   CM = {C_M}"
#         else:
#             coeffs_text = "Erreur lors du calcul des coefficients."
#
#         return figures.get(selected_graph, go.Figure()), coeffs_text
#
#     except Exception as e:
#         print(f"Error in graph update: {e}")
#         return go.Figure(), "Erreur lors du chargement."
