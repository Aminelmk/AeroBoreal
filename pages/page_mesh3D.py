
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_daq as daq
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

import base64
import io

from mesh2d.naca4digits import Naca4Digits
from mesh2d.cst_class import CstAirfoil
from mesh3d.generate_vlm_mesh import mesh_wing, lovell_mesh_wing, WingElliptic, mesh_wing_CRM, save_mesh

dash.register_page(__name__, path="/page-mesh3d")

config = {'scrollZoom': True}
mesh = None

def display_mesh(mesh, nx, ny, show_nodes, nodes_struc, show_struct, mesh_qquatercord):
    x_mesh = mesh[1]
    y_mesh = mesh[0]
    z_mesh= mesh[2]

    mesh_fig = go.Figure()

    # -------------------- STRUCTURE NODES -----------------
    if show_struct == True:
        a=0
        if len(y_mesh[1]) != 2*ny + 1:
            a = 1
        indices = [0, len(mesh[0][0]) // 2 -a, -2]
        filtered_mesh = tuple(arr[:, indices] for arr in mesh)

        nbnodes_mesh = []
        add_nodes = (nodes_struc - 3)/2
        for arr in filtered_mesh:
            first_col = arr[:, 0]  # Premier point
            second_col = arr[:, 1]  # Deuxième point
            last_col = arr[:, 2]  # Dernier point

            # Générer X points équidistants entre first_col et second_col
            interpolated_cols = np.linspace(first_col, second_col, int(add_nodes + 2), axis=1)[:, 1:-1]  # Exclut les bornes
            
            # Concaténer avec le dernier point
            new_arr = np.hstack((first_col[:, None], interpolated_cols, second_col[:, None]))
            nbnodes_mesh.append(new_arr)

        nbnodes_mesh = tuple(nbnodes_mesh)

        x_meshfiltered = nbnodes_mesh[1]
        y_meshfiltered = nbnodes_mesh[0]
        z_meshfiltered= nbnodes_mesh[2]

        # Tracés des lignes uniquement (en noir)
        if show_nodes == True:
            mesh_fig.add_trace(go.Scatter3d(
                            x=x_meshfiltered[0] + mesh_qquatercord*(x_meshfiltered[nx]-x_meshfiltered[0]),
                            y=y_meshfiltered[0] + mesh_qquatercord*(y_meshfiltered[nx]-y_meshfiltered[0]),
                            z=z_meshfiltered[0] + mesh_qquatercord*(z_meshfiltered[nx]-z_meshfiltered[0]),
                            mode="markers",
                            marker=dict(color="green", size=3),
                            name=f"Points structure"
                        ))
            mesh_fig.add_trace(go.Scatter3d(
                            x=-(x_meshfiltered[0] + mesh_qquatercord*(x_meshfiltered[nx]-x_meshfiltered[0])),
                            y=y_meshfiltered[0] + mesh_qquatercord*(y_meshfiltered[nx]-y_meshfiltered[0]),
                            z=z_meshfiltered[0] + mesh_qquatercord*(z_meshfiltered[nx]-z_meshfiltered[0]),
                            mode="markers",
                            marker=dict(color="green", size=3),
                            name=f"Points structure"
                        ))
        mesh_fig.add_trace(go.Scatter3d(
                        x=x_meshfiltered[0] + mesh_qquatercord*(x_meshfiltered[nx]-x_meshfiltered[0]),
                        y=y_meshfiltered[0] + mesh_qquatercord*(y_meshfiltered[nx]-y_meshfiltered[0]),
                        z=z_meshfiltered[0] + mesh_qquatercord*(z_meshfiltered[nx]-z_meshfiltered[0]),
                        mode="lines",
                        line=dict(color="black", dash="dash"),
                        name=f"Points structure"
                    ))
        
        mesh_fig.add_trace(go.Scatter3d(
                        x=-(x_meshfiltered[0] + mesh_qquatercord*(x_meshfiltered[nx]-x_meshfiltered[0])),
                        y=y_meshfiltered[0] + mesh_qquatercord*(y_meshfiltered[nx]-y_meshfiltered[0]),
                        z=z_meshfiltered[0] + mesh_qquatercord*(z_meshfiltered[nx]-z_meshfiltered[0]),
                        mode="lines",
                        line=dict(color="black", dash="dash"),
                        name=f"Points structure"
                    ))

    # ------------------- FLUID ------------------------
    for i in range(nx + 1):
        mesh_fig.add_trace(go.Scatter3d(
            x=x_mesh[i],
            y=y_mesh[i],
            z=z_mesh[i],
            mode="lines",
            line=dict(color="black"),
            name=f"Ligne x{i}"
        ))

    for i in range(len(y_mesh[1])):
        mesh_fig.add_trace(go.Scatter3d(
            x=x_mesh[:, i],
            y=y_mesh[:, i],
            z=z_mesh[:, i],
            mode="lines",
            line=dict(color="black"),
            name=f"Ligne y{i}"
        ))


    if show_nodes == True:
    # Tracés des points uniquement (en rouge)
        for i in range(nx + 1):
            mesh_fig.add_trace(go.Scatter3d(
                x=x_mesh[i],
                y=y_mesh[i],
                z=z_mesh[i],
                mode="markers",
                marker=dict(color="red", size=3),
                name=f"Points x{i}"
            ))

        '''for i in range(ny + 1):
            mesh_fig.add_trace(go.Scatter3d(
                x=x_mesh[:, i],
                y=y_mesh[:, i],
                z=z_mesh[:, i],
                mode="markers",
                marker=dict(color="red", size=3),
                name=f"Points y{i}"
            ))'''

    '''if sym : 
        for i in range(nx + 1):
            ax.plot(x[i], -y[i], z[i], "-", color=color)  # Chordwise lines
        for i in range(ny + 1):
            ax.plot(x[:, i], -y[:, i], z[:, i], "-", color=color)  # Spanwise lines

    mesh_fig.add_trace(go.Scatter(x=mesh[1], y=mesh[2], mode="markers", name="Airfoil", line=dict(color="red")))'''

    mesh_fig.update_layout(
    scene=dict(
        xaxis_title='Y',
        yaxis_title='X',
        zaxis_title='Z',
        aspectmode='data',
        camera=dict(
            eye=dict(x=0, y=3, z=3)  # <-- Position de la caméra
        )  # <-- C’est ce qui garde le bon ratio entre les axes
        # Optionnel : tu peux forcer des ranges ou ratios si tu veux
        # aspectratio=dict(x=1, y=10, z=0.5),  # <- si tu veux le forcer manuellement
    ),
    width=800,
    height=600,
    showlegend=False
)

    return mesh_fig

mesh = (np.array([[0. , 0. , 0. , 0. , 0. ],
       [0.5, 0.5, 0.5, 0.5, 0.5],
       [1. , 1. , 1. , 1. , 1. ]]), 
       np.array([[-4.50000000e+00, -3.18198052e+00, -2.75545530e-16,
         3.18198052e+00,  4.50000000e+00],
       [-4.50000000e+00, -3.18198052e+00, -2.75545530e-16,
         3.18198052e+00,  4.50000000e+00],
       [-4.50000000e+00, -3.18198052e+00, -2.75545530e-16,
         3.18198052e+00,  4.50000000e+00]]), 
         np.array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -0.,  0.,  0.],
       [ 0.,  0., -0.,  0.,  0.]]))

mesh_fig = display_mesh(mesh, 2, 2, True, 2, True, 0.25)

layout = html.Div([
    # ===== Geometry Generation =====
    html.Div([
        dbc.Container([
            dbc.Row([
                html.H2("Wings Geometry", className="text-center my-4")]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="mesh-plot", figure=mesh_fig, config={'scrollZoom':True}),
                     html.Div(id="download-message", style={"display": "none"}),

                ], width=8),

                dbc.Col([
                    html.Div([
                        daq.BooleanSwitch(
                                    id="show-nodes-button",
                                    on=True,
                                    label="Show mesh points",
                                    labelPosition="bottom",
                                    style={"marginRight": "20px"}  # Space between the switch and button
                                ),
                    ]),

                    dbc.Accordion([
                        dbc.AccordionItem([
                            html.Div([
                                html.H6("Half-wing Caracteristics"),
                                html.Div([
                                html.Label("Number of panels in X: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="panelsx", type="number", min=1, value=2,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Number of panels in Y: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="panelsy", type="number", min=1, value=2,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),
                            
                            html.Br(),

                            html.H6("Leading Edge Position"),

                            html.Div([
                                html.Label("X-axis (m): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="lepos", type="number", value=0,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Z-axis (m): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="z0", type="number", value=0,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            ])
                        ], title="General Settings"),

                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Label("Select Wing Type:", className="col-6", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Dropdown(
                                    id="type-dropdown",
                                    options=[{"label": key, "value": key} for key in ['Custom', 'Lovell', 'Elliptic', 'CRM']],
                                    value='Custom',
                                    style={"flex": 1}  # Ensures it takes up remaining space
                                ),
                            ], className="row d-flex align-items-center mt-2"),
                            html.Br(),

                            html.Div([

                                html.Div([
                                html.Label("Aspect Ratio: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="custom_ar", type="number", min=1e-15, value=9,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                                html.Div([
                                html.Label("Cord (m): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="custom_cord", type="number", min=1e-15, value=1,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Taper Ratio: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="custom_lam", type="number", min=1e-15, max = 1, value=1,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Sweep Angle (°): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="custom_sweep", type="number", min=-89, max = 89, value=0,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Dihedral Angle (°): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="custom_diedre", type="number", min=-89, max = 89, value=0,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Twist Angle (°): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="custom_twist", type="number", min=-89, max = 89, value=0,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            ], id="custom-controls", style={"display": "none"}),
                            
                            html.Div([
                                html.Div([


                                ], className="controls"),

                            ], id="lovell-controls", style={"display": "none"}),

                            html.Div([

                                html.Div([
                                html.Label("Aspect Ratio: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="ell_ar", type="number", min=1e-15, value=9,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                                html.Div([
                                html.Label("Cord (m): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="ell_cord", type="number", min=1e-15, value=1,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Taper Ratio: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="ell_lam", type="number", min=1e-15, max = 1, value=1,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Sweep Angle (°): ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="ell_sweep", type="number", min=-89, max = 89, value=0,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            ], id="elliptic-controls", style={"display": "none"}),


                        ], title="Wing Type"),

                        dbc.AccordionItem([
                            daq.BooleanSwitch(
                                    id="show-structure-button",
                                    on=True,
                                    label="Show structure",
                                    labelPosition="bottom",
                                    style={"marginRight": "20px"}  # Space between the switch and button
                                ),
                            
                            html.Div([
                                html.Label("Number of elements: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="n-nodes-slider", type="number", min=2, value=2,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                            html.Div([
                                html.Label("Quater Cord Ratio: ", className="col-8", style={"text-align": "left", "white-space": "nowrap"}),
                                dcc.Input(id="mesh_qquatercord", type="number", value=0.25,
                                          style={"flex": "1"}),
                            ], className="row d-flex align-items-center mt-2"),

                        ], title="Structure (for visualization only)"),
                    ], id="accordion", flush=True),
                    html.Div([
                                dbc.Button("Download Mesh", id="button-download-mesh3d", className="me-2"),
                                dcc.Download(id="download-mesh")
                            ]),
                ]),

                html.Div(children=[
            html.Hr(),

            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Configure the NL-VLM-Structure Solver",
                        href="/pages-vlmstructure3D",
                        color="primary",
                        className="mt-4"
                    ),
                    width="auto"
                )],

                justify="center",
                className="mb-4"
            ),
        ]),

            ]),
        ]),
    ]),
])

# ===== Controls which parameters to display =====
@dash.callback(
     [Output("custom-controls", "style"),
     Output("lovell-controls", "style"),
     Output("elliptic-controls", "style")],
    Input("type-dropdown", "value"),
)
def update_control_panel(selected_option):


    if selected_option is None:
        return html.Div("Select an option to display geometry controls.")

    #print(f"selected option is: {selected_option}")
    if selected_option == "Custom":
        #vlm_mesh = update_naca_airfoil(INIT_CAMBER, INIT_CAMBER_POS, INIT_THICKNESS)
        return {"display": "block"}, {"display": "none"},  {"display": "none"}
    elif selected_option == "Lovell":
        #airfoil = update_cst_airfoil(3, N1=0.5, N2=1.0)
        return {"display": "none"}, {"display": "block"},  {"display": "none"}
    elif selected_option == "Elliptic":
        #airfoil = update_cst_airfoil(3, N1=0.5, N2=1.0)
        return {"display": "none"}, {"display": "none"},  {"display": "block"}
    elif selected_option == "CRM":
        #airfoil = update_cst_airfoil(3, N1=0.5, N2=1.0)
        return {"display": "none"}, {"display": "none"},  {"display": "none"}


@dash.callback(
    Output("mesh-plot", "figure"),
    [
        Input("type-dropdown", "value"),
        Input("panelsx", "value"),
        Input("panelsy", "value"),
        Input("lepos", "value"),
        Input("custom_ar", "value"),
        Input("custom_cord", "value"),
        Input("custom_sweep", "value"),
        Input("custom_lam", "value"),
        Input("custom_diedre", "value"),
        Input("custom_twist", "value"),
        Input("z0", "value"),
        Input("ell_ar", "value"),
        Input("ell_cord", "value"),
        Input("ell_sweep", "value"),
        Input("ell_lam", "value"),
        #Input("ell_twist", "value"),
        Input("n-nodes-slider", "value"),
        #Input("button-generate-mesh3d", "n_clicks"),
        Input("button-download-mesh3d", "n_clicks"),
        Input("show-structure-button", "on"),
        Input("show-nodes-button", "on"),
        Input("mesh_qquatercord", "value"),  
    ],
    prevent_initial_call=True
)
def update_mesh(selected_type, nx, ny, LE, cust_AR, cust_cord, cust_sweep, cust_lam, cust_diedre, cust_twist, z0, ell_AR, ell_cord, ell_sweep, ell_lam, nodes_struc, download, show_structure, show_nodes, mesh_qquatercord):
    #print(selected_type)
    global mesh
    y0=0

    nodes_struc = nodes_struc + 1
    ell_sweep = ell_sweep/45

    triggered_id = ctx.triggered_id

    if triggered_id in [
    "type-dropdown",
    "panelsx",
    "panelsy",
    "y0",
    "lepos",
    "custom_ar",
    "custom_cord",
    "custom_sweep",
    "custom_lam",
    "custom_diedre",
    "custom_twist",
    "z0",
    "ell_ar",
    "ell_cord",
    "ell_sweep",
    "ell_lam",
    #"ell_twist",
    "n-nodes-slider",
    #"button-generate-mesh3d",
    "show-structure-button",
    "show-nodes-button",
    "mesh_qquatercord",
]:  
        if selected_type == "Custom":
            span = cust_AR*(cust_cord+cust_lam)/4
            mesh = mesh_wing(ny, nx, y0, z0, span, cust_cord, 0, cust_sweep, cust_lam, cust_diedre, cust_twist, LE, 1, False)
            #print(vlm_mesh)
            return display_mesh(mesh, nx, ny, show_nodes, nodes_struc, show_structure, mesh_qquatercord)
        elif selected_type == "Lovell":
            mesh = lovell_mesh_wing(nx, ny, LE, 1, False, y0, z0)
            #print(vlm_mesh)
            return display_mesh(mesh, nx, ny, show_nodes, nodes_struc, show_structure, mesh_qquatercord)
        elif selected_type == "Elliptic":
            span = ell_AR*(ell_cord+ell_lam)/4
            mesh = WingElliptic(ny, nx, y0, z0, span, ell_cord, 0, ell_sweep, 0, 0, 0, LE, 1, False)
            #print(vlm_mesh)
            return display_mesh(mesh, nx, ny, show_nodes, nodes_struc, show_structure, mesh_qquatercord)
        elif selected_type == "CRM":
            mesh = mesh_wing_CRM(ny, nx)
            #print(vlm_mesh)
            return display_mesh(mesh, nx, ny, show_nodes, nodes_struc, show_structure, mesh_qquatercord)
 
        
@dash.callback(
    [Output("download-message", "children"),  # Mettre à jour le message
     Output("download-message", "style")],  # Changer le style (par exemple, visibilité)
    Input("panelsx", "value"),
    Input("panelsy", "value"),
    Input("button-download-mesh3d", "n_clicks"),
    Input("show-nodes-button", "on"),
    prevent_initial_call=True,
)
def save_mesh3d(nx, ny, download_clicks, show_nodes):
    global mesh
    triggered_id = ctx.triggered_id

    if triggered_id == "button-download-mesh3d":
        save_mesh(nx, 2*ny, mesh[0], mesh[1], mesh[2], './mesh3d.x')

        # Afficher le message "Mesh downloaded"
        return "Mesh downloaded to ./mesh3d.x !", {"color": "green", "fontWeight": "bold"}
    

    if triggered_id in [
    "panelsx",
    "panelsy",
    "lepos",
    "custom_ar",
    "custom_cord",
    "custom_sweep",
    "custom_lam",
    "custom_diedre",
    "custom_twist",
    "z0",
    "ell_ar",
    "ell_cord",
    "ell_sweep",
    "ell_lam",
    "ell_twist",
    "n-nodes-slider",
    "button-generate-mesh3d",
    "show-nodes-button"
]:  
        return "", {"color": "transparent"}
    

    return "", {"color": "transparent"}