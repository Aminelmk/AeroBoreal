
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

import base64
import io

from mesh2D.naca4digits import Naca4Digits
from mesh2D.cst_class import CstAirfoil
from mesh2D.poisson_grid_testing import PoissonMesh

dash.register_page(__name__, path="/page-mesh2d")


INIT_CAMBER = 0
INIT_CAMBER_POS = 0
INIT_THICKNESS = 12

airfoil = None
mesh = None

def display_airfoil(airfoil, points=None, mesh=None):

    x, y = airfoil.get_all_surface(1000)

    airfoil_fig = go.Figure()
    airfoil_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Airfoil", line=dict(color="red")))

    if points is not None:
        airfoil_fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="Points", marker=dict(symbol="square", color="black")))

    airfoil_fig.update_layout(title=f"Airfoil Profile",
                              xaxis_title="x/c", yaxis_title="y/c",
                              xaxis_range=[-0.1, 1.1],
                              yaxis_scaleanchor="x",
                              showlegend=False, width=800, height=600)


    if mesh is not None:
        for cell in group_nodes(mesh):
            x, y = cell[0]
            airfoil_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="black")))

        airfoil_fig.update_layout(title=f"Airfoil Mesh",
                                  xaxis_title="x/c", yaxis_title="y/c",
                                  xaxis_range=[-2.5, 3],
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
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Label("Select Method:", width="auto"),
                                dcc.Dropdown(
                                    id="mesh-dropdown",
                                    options=[{"label": key, "value": key} for key in ['NACA', 'CST']],
                                    value='NACA',
                                    style={"flex": 1}  # Ensures it takes up remaining space
                                ),
                            ]),

                            html.Div(id="mesh-control-panel"),
                            html.Div([
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
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Upload([
                                                'Drag and Drop or ',
                                                html.A('Select a File')
                                            ], id='upload-data',
                                                style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center'
                                            })
                                        ], width=8),

                                        dbc.Col([
                                            html.Button('Fit Airfoil', id='fit-airfoil', n_clicks=0)
                                        ], width=4)
                                    ], className="g-2"),

                                    html.Br(),

                                    dbc.Alert(id="upload-status", color="info", is_open=False),  # Display upload status

                                    # html.Div(id='uploaded-file-path'),  # Shows the stored file path

                                    dbc.Row([
                                        daq.ToggleSwitch(
                                            label='Show Airfoil Points',
                                            labelPosition='bottom'
                                        )
                                    ]),

                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("N Order: ", className="mb-0")  # Removes bottom margin
                                        ], width="auto", className="d-flex align-items-center"),
                                        # Align text vertically

                                        dbc.Col([
                                            dcc.Input(
                                                type="number",
                                                min=1,
                                                max=7,
                                                value=3,
                                                id='n-order-input',
                                                step=1,
                                                style={"width": "100px"}  # Adjust input width
                                            )
                                        ], width="auto")
                                    ], className="g-2"),  # Reduces gutter spacing

                                    dbc.Row(html.Div(id="coefficients-inputs")),

                                ], className="controls"),
                            ], id="cst-controls", style={"display": "none"}),

                        ], title="Geometry"),

                        dbc.AccordionItem([
                            html.Div([
                                html.P("Number of cells (2^x):"),
                                dcc.Slider(3, 7, step=1, value=6, id="n-cell-slider"),
                                dbc.Button("Generate Mesh", id="button-generate-mesh", className="me-2", n_clicks=0),
                                dbc.Button("Download Mesh", id="button-download-mesh", className="me-2"),
                                dcc.Download(id="download-mesh"),
                            ])
                        ], title="Mesh"),
                    ], id="accordion", flush=True),
                ]),
            ]),
        ]),
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
                dbc.Col([
                    html.P("Upper")
                ], width=6),
                dbc.Col([
                    html.P("Lower")
                ], width=6),
            ])] + [

            dbc.Row([
                dbc.Col(html.Label(f"A [{i}]: "), width=2),
                dbc.Col(dcc.Input(
                    id={"type": "A_upper", "index": i},
                    type="number",
                    min=-3,
                    max=3,
                    value=airfoil.A_upper[i],
                    step=0.01,
                ), width=4),
                dbc.Col(html.Label(f"A [{i}]: "), width=2),
                dbc.Col(dcc.Input(
                    id={"type": "A_lower", "index": i},
                    type="number",
                    min=-3,
                    max=3,
                    value=airfoil.A_lower[i],
                    step=0.01,
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
        State("n-order-input", "value"),
        Input("upload-data", "contents"),
        Input("fit-airfoil", "n_clicks"),
        Input("n-cell-slider", "value"),
        Input("button-generate-mesh", "n_clicks"),
    ],
    prevent_initial_call=True
)
def update_airfoil(selected_option, camber, camber_pos, thickness, A_upper_values, A_lower_values, n_order, contents, fit_airfoil, n_cell, generate_mesh):
    """Handles both NACA and CST airfoil updates in a single callback."""

    global airfoil

    triggered_id = ctx.triggered_id

    if selected_option == "NACA" and triggered_id in ["max_camber_slider", "max_camber_pos_slider",
                                                      "max_thickness_slider"]:
        airfoil = update_naca_airfoil(camber, camber_pos, thickness)

        return display_airfoil(airfoil, points=None)

    elif selected_option == "CST" and triggered_id in ["fit-airfoil"]:

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        input_stream = io.StringIO(decoded)

        airfoil = update_cst_airfoil(n_order, N1=0.5, N2=1.0)
        airfoil.import_points(input_stream)
        airfoil.fit_airfoil()

        return display_airfoil(airfoil, points=airfoil.imported_airfoil)


    elif selected_option == "CST" and triggered_id in ["n-order-input"] or (isinstance(triggered_id, dict) and triggered_id.get("type") in ["A_upper", "A_lower"]):

        print("Generating CST")

        airfoil = update_cst_airfoil(n_order, N1=0.5, N2=1.0)
        if A_upper_values is not None and A_lower_values is not None:
            airfoil.A_upper = np.array(A_upper_values)
            airfoil.A_lower = np.array(A_lower_values)

    elif triggered_id == "button-generate-mesh":
        print("Generating mesh")

        n_nodes = 2**n_cell + 1

        n_xc_nodes = int((2**n_cell / 2) + 1)

        beta = np.linspace(0, np.pi, n_xc_nodes)
        xc = 0.5 * (1 - np.cos(beta))
        xs, ys = airfoil.get_surface_from_x(xc)

        mesh = PoissonMesh(n_nodes, xs, ys)
        mesh.ff_radius = 100
        mesh.init_grid()
        mesh.grid_relaxation(tol=1e-8, max_iter=10_000)
        mesh.write_plot3d("./temp/mesh.xyz")

        return display_airfoil(airfoil, mesh=mesh)

    else:
        print("Retuning airfoil as is")
        return display_airfoil(airfoil, points=None)

    return display_airfoil(airfoil)



@dash.callback(
    Output("download-mesh", "data"),
    Input("button-download-mesh", "n_clicks"),
)
def download_mesh(n_clicks):
    return dcc.send_file("./temp/mesh.xyz")


@dash.callback(
    Input("accordion", "active_item")
)
def change_item(item):
    print(f"Item selected: {type(item)}")


# @dash.callback(
#     [Output('uploaded-file-path', 'children'),
#      Output('upload-status', 'children'),
#      Output('upload-status', 'is_open')],
#     [Input('upload-data', 'filename')]
# )
# def store_file_path(filename):
#     if filename is None:
#         return dash.no_update, dash.no_update, dash.no_update
#
#     # Assume the file already exists in the upload directory
#     file_path = os.path.join(filename)
#
#     return (
#         html.Div(f"File path: {file_path}"),
#         f"Path stored: {file_path}",
#         True  # Show alert
#
#     )