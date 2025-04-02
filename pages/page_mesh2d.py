
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import base64
import io
import os

# from mesh2d.conformal_mapping import ConformalMapping
from mesh2d.naca4digits import Naca4Digits
from mesh2d.cst_class import CstAirfoil
from mesh2d.elliptic_grid import PoissonMesh
from mesh2d.bspline_wrapper import BSplineWrapper

# Converted CPP
from B_Spline_Solver import BSpline_solver

dash.register_page(__name__, path="/page-mesh2d")

INIT_CAMBER = 0
INIT_CAMBER_POS = 0
INIT_THICKNESS = 12


config = {'scrollZoom': True}

airfoil = Naca4Digits(INIT_CAMBER, INIT_CAMBER_POS, INIT_THICKNESS)
points_data = None
mesh = None
bspline_data = None
bspline_input = None

def update_naca_airfoil(camber, camber_pos, thickness):
    return Naca4Digits(int(camber), int(camber_pos), int(thickness))


def update_cst_airfoil(n_order, N1=0.5, N2=1.0):
    return CstAirfoil(n_order, N1=N1, N2=N2)

# Not in use
def group_nodes(grid):
    nodes_per_cell = []
    next_index = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    for i in range(grid.n_nodes - 1):
        for j in range(grid.n_nodes - 1):
            x = np.empty(4)
            y = np.empty(4)
            nodes_per_cell.append([])
            for k in range(4):
                pi = i + next_index[k][0]
                pj = j + next_index[k][1]
                x[k] = grid.X[pi][pj]
                y[k] = grid.Y[pi][pj]
            nodes_per_cell[-1].append((x, y))
    return nodes_per_cell

def single_trace(grid):
    # nodes_per_cell = []

    x_trace = np.array([np.nan])
    z_trace = np.array([np.nan])

    next_index = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    for i in range(grid.n_nodes - 1):
        for j in range(grid.n_nodes - 1):
            x = np.empty(5)
            y = np.empty(5)
            # nodes_per_cell.append([])

            for k in range(4):
                pi = i + next_index[k][0]
                pj = j + next_index[k][1]
                x[k] = grid.X[pi][pj]
                y[k] = grid.Y[pi][pj]

            x[-1] = np.nan
            y[-1] = np.nan

            x_trace = np.append(x_trace, x)
            z_trace = np.append(z_trace, y)

            # nodes_per_cell[-1].append((x, y))
    return x_trace, z_trace

# Lecture fichier .txt B-spline
def read_bspline_content(content):
    decoded = base64.b64decode(content.split(',')[1]).decode('utf-8')
    x, y = [], []
    for line in decoded.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return np.array(x), np.array(y)

# Not in use
def read_bspline_curve(file_path):
    x, y = [], []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
        return np.array(x), np.array(y)
    except FileNotFoundError:
        print(f"Erreur : fichier {file_path} introuvable")
        return None, None


# Not in use
def downsample_curve(x, y, max_points=300):
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        return x[idx], y[idx]
    return x, y


def display_airfoil(airfoil, points=None, mesh=None, bspline_data=None, bspline_init=None):
    fig = go.Figure()

    if airfoil:
        x, y = airfoil.get_all_surface(1000)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Airfoil", line=dict(color="red")))

    # Not in use
    if bspline_data is not None:
        xb, yb = bspline_data
        xb, yb = downsample_curve(xb, yb, max_points=250)    
        fig.add_trace(go.Scatter(
            x=xb, y=yb,
            mode="lines+markers",
            name="B-Spline",
            line=dict(color="blue", width=2, shape='spline', smoothing=1.3),
            marker=dict(size=4, color="blue")
        ))

    # Not in use
    if bspline_init is not None:
        init_x = bspline_init[:, 0]
        init_y = bspline_init[:, 1]
        init_x, init_y = downsample_curve(init_x, init_y, max_points=250)
        fig.add_trace(go.Scatter(
            x=bspline_init[:, 0], y=bspline_init[:, 1],
            mode="markers",
            name="B-Spline Init",
            marker=dict(color="red", size=6, symbol="circle")
        ))

    if points is not None:
        fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="Points", marker=dict(color="black")))

    if mesh is not None:
        x_trace, z_trace = single_trace(mesh)
        fig.add_trace(go.Scatter(x=x_trace, y=z_trace, name="Grid", mode="lines", line=dict(color="black")))

    fig.update_layout(title="Airfoil Display", xaxis_title="x/c", yaxis_title="y/c",
                      xaxis_range=[-0.1, 1.1], yaxis_range=[-0.4, 0.4], yaxis_scaleanchor="x",
                      showlegend=True, width=800, height=600, dragmode="pan")
    return fig

# Ajout interface pour B-Spline avec upload, fit et knots
bspline_controls = html.Div(id="bspline-controls", children=[
    html.Label("Knot Value (15, 20, 25, 30):"),
    dcc.Dropdown(
        id="bspline-knot",
        options=[{"label": str(k), "value": k} for k in [15, 20, 25, 30]],
        value=25
    ),
    html.Label("Degré (0, 1, 2, 3):"),
    dcc.Dropdown(
       id="bspline-degree",
       options=[{"label": str(p), "value": p} for p in [0, 1, 2, 3]],
       value=3
     ),
    dcc.Upload(
        id='upload-bspline',
        children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'marginBottom': '10px'
        },
        multiple=False
    ),
    html.Button('Fit Airfoil', id='fit-bspline', n_clicks=0)
])


layout = html.Div([
    dbc.Container([
        dbc.Row([html.H2("Airfoil Geometry", className="text-center my-4")]),
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-graph",
                    type="circle",
                    children=dcc.Graph(id="airfoil-plot", figure=display_airfoil(airfoil), config={'scrollZoom':True})
                )
            ], width=8),
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Label("Méthode de profil"),
                        dcc.Dropdown(
                            id="mesh-dropdown",
                            options=[{"label": k, "value": k} for k in ["NACA", "CST", "B-Spline"]],
                            value="NACA"
                        ),
                        html.Div(id="naca-controls", children=[
                            html.Label("Max Camber (1/100)"),
                            dcc.Slider(id="max_camber_slider", min=0, max=9, step=1, value=INIT_CAMBER,
                                       marks={i: str(i) for i in range(0, 10)}),

                            html.Label("Camber Position (1/10)"),
                            dcc.Slider(id="max_camber_pos_slider", min=0, max=9, step=1, value=INIT_CAMBER_POS,
                                       marks={i: str(i) for i in range(0, 10)}),

                            html.Label("Max Thickness (%)"),
                            dcc.Slider(id="max_thickness_slider", min=1, max=20, step=1, value=INIT_THICKNESS,
                                       marks={i: str(i) for i in range(1, 21)})
                        ]),

                        html.Div(id="cst-controls", children=[
                            html.Label("N Order:"),
                            dcc.Input(id="n-order-input", type="number", value=3, min=1, max=10),
                            html.Div(id="coefficients-inputs"),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'marginBottom': '10px'
                                }
                            ),
                            html.Button('Fit Airfoil', id='fit-airfoil', n_clicks=0),
                            daq.ToggleSwitch(id="show-airfoil-points", label="Afficher les points", labelPosition="bottom")
                        ]),

                        bspline_controls
                    ], title="Géométrie"),

                    dbc.AccordionItem([
                        html.Label("Nombre de cellules (2^x):"),
                        dcc.Slider(3, 7, step=1, value=6, id="n-cell-slider"),
                        html.Div([
                            dbc.Button("Générer le maillage", id="button-generate-mesh", n_clicks=0),
                            dbc.Button("Télécharger", id="button-download-mesh", className="ms-2")
                        ], className="d-flex align-items-center mt-2"),
                        dcc.Download(id="download-mesh")
                    ], title="Maillage")
                ])
            ])
        ])
    ])
])


@dash.callback(
    [Output("naca-controls", "style"),
     Output("cst-controls", "style"),
     Output("bspline-controls", "style")],
    Input("mesh-dropdown", "value")
)
def toggle_visibility(method):
    return (
        {"display": "block"} if method == "NACA" else {"display": "none"},
        {"display": "block"} if method == "CST" else {"display": "none"},
        {"display": "block"} if method == "B-Spline" else {"display": "none"}
    )

@dash.callback(
    Output("coefficients-inputs", "children"),
    Input("n-order-input", "value")
)
def update_coeffs(n):
    if n is None:
        return []
    return [
        dbc.Row([
            dbc.Col(html.Label(f"A_upper[{i}]"), width=6),
            dbc.Col(html.Label(f"A_lower[{i}]"), width=6)
        ]) if i == 0 else
        dbc.Row([
            dbc.Col(dcc.Input(id={'type': 'A_upper', 'index': i}, type='number', step=0.01, value=0), width=6),
            dbc.Col(dcc.Input(id={'type': 'A_lower', 'index': i}, type='number', step=0.01, value=0), width=6)
        ])
        for i in range(n + 1)
    ]


# Callback principal unique pour toutes les méthodes (NACA, CST, B-Spline)

@dash.callback(
    Output("airfoil-plot", "figure"),
    [
        Input("fit-airfoil", "n_clicks"),
        Input("fit-bspline", "n_clicks"),
        Input("button-generate-mesh", "n_clicks"),
        Input("show-airfoil-points", "value"),
        Input("max_camber_slider", "value"),
        Input("max_camber_pos_slider", "value"),
        Input("max_thickness_slider", "value"),
        Input("mesh-dropdown", "value"),
        Input({"type": "A_upper", "index": dash.ALL}, "value"),
        Input({"type": "A_lower", "index": dash.ALL}, "value")        
    ],
    [
        State("upload-bspline", "contents"),
        State("bspline-knot", "value"),
        State("upload-data", "contents"),
        State("n-order-input", "value"),
        State("n-cell-slider", "value"),
        State("bspline-degree", "value")  
    ],
    prevent_initial_call=True
)



def update_fig(fit_cst_clicks, fit_bspline_clicks, gen_clicks, show_pts, camber, camber_pos, thickness, method,
               A_upper_values, A_lower_values,
               bspline_content, bspline_knot, cst_content, n_order, n_cell,  bspline_degree):
        
    global airfoil, mesh, points_data, bspline_data, bspline_input
    triggered = ctx.triggered_id

    if triggered == "button-generate-mesh":

        n_nodes = 2 ** n_cell + 1
        n_xc_nodes = int((2 ** n_cell / 2) + 1)
        beta = np.linspace(0, np.pi, n_xc_nodes)
        xc = 0.5 * (1 - np.cos(beta))

        if method == "CST":
            xs, ys = airfoil.get_surface_from_x(xc)
        elif method == "NACA":
            xs, ys = airfoil.get_surface_from_x(xc)
        elif method == "B-Spline":
            xs, ys = airfoil.get_surface_from_x(xc)
        else:
            return display_airfoil(airfoil)

        mesh = PoissonMesh(n_nodes, xs, ys)
        mesh.ff_radius = 150
        mesh.init_grid()
        mesh.grid_relaxation(tol=1e-6, max_iter=50_000)
        mesh.write_plot3d("./temp/mesh.xyz")

        # if method == "B-Spline":
        #     return display_airfoil(None, bspline_data=bspline_data, bspline_init=np.column_stack(bspline_input), mesh=mesh)
        # else:
        return display_airfoil(airfoil, mesh=mesh)

    else:

        if method == "NACA":
            airfoil = Naca4Digits(camber, camber_pos, thickness)
            # return display_airfoil(airfoil)

        if method == "CST" and triggered == "fit-airfoil" and cst_content:
            decoded = base64.b64decode(cst_content.split(',')[1]).decode('utf-8')
            input_stream = io.StringIO(decoded)
            airfoil = CstAirfoil(n_order)
            airfoil.import_points(input_stream)
            airfoil.fit_airfoil()
            points_data = airfoil.imported_airfoil
            return display_airfoil(airfoil, points=points_data if show_pts else None)

        if method == "CST" and (triggered == "n-order-input" or (isinstance(triggered, dict) and triggered.get("type") in ["A_upper", "A_lower"])):
            airfoil = update_cst_airfoil(n_order, N1=0.5, N2=1.0)
            if A_upper_values is not None and A_lower_values is not None:
                airfoil.A_upper = np.array(A_upper_values)
                airfoil.A_lower = np.array(A_lower_values)
            return display_airfoil(airfoil)


        if method == "B-Spline" and triggered == "fit-bspline" and bspline_content:
            with open("temp_bspline.dat", "w") as f:
                decoded = base64.b64decode(bspline_content.split(',')[1]).decode('utf-8')
                f.write(decoded)
            BSpline_solver.run_bspline_solver("temp_bspline.dat", bspline_knot, bspline_degree)

            # curve = np.loadtxt("BSpline_curve.txt", dtype=float, usecols=(0, 1))
            airfoil = BSplineWrapper()
            airfoil.read_bspline_curve("BSpline_curve.txt")
            # airfoil.

            # xb, yb = read_bspline_curve("BSpline_curve.txt")
            # bspline_data = (xb, yb)
            # bspline_input = read_bspline_content(bspline_content)
            # return display_airfoil(None, bspline_data=bspline_data, bspline_init=np.column_stack(bspline_input))

            return display_airfoil(airfoil)

    return display_airfoil(airfoil, points=points_data if show_pts else None)


@dash.callback(
    Output("download-mesh", "data"),
    Input("button-download-mesh", "n_clicks"),
    prevent_initial_call=True
)
def download_mesh(n):
    return dcc.send_file("./temp/mesh.xyz")
