import sys
import os
import dash
import dash_vtk
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
import vtk
from vtk.util import numpy_support
import numpy as np
import csv
import plotly.graph_objects as go

# Ajouter le dossier parent au sys.path pour que mesh2D soit reconnu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les fonctions de création du modèle 3D de l'avion
from Dash_3D.planevtk import create_airplane
from mesh2D.naca4digits import Naca4Digits
from mesh2D.conformal_mapping import ConformalMapping

# Fonction pour extraire les cellules de PolyData
def extract_cells(polydata):
    vtk_polys = polydata.GetPolys()
    if vtk_polys.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_polys.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()

    vtk_lines = polydata.GetLines()
    if vtk_lines.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_lines.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()

    vtk_verts = polydata.GetVerts()
    if vtk_verts.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_verts.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()

    raise ValueError("Le PolyData ne contient pas de polygones, lignes ou vertices.")

# Convertir PolyData VTK en format compatible Dash
def vtk_to_dash_vtk(polydata):
    vtk_points = polydata.GetPoints()
    np_points = numpy_support.vtk_to_numpy(vtk_points.GetData())
    polys = extract_cells(polydata)

    return {
        'points': np_points.flatten().tolist(),
        'polys': polys,
    }

# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Fichier CSV pour l'historique
CSV_FILE = "user_inputs.csv"

# Disposition de l'application
app.layout = dbc.Container(
    fluid=True,
    style={"marginTop": "15px", "height": "calc(100vh - 30px)"},
    children=[
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Accueil", href="#")),
                dbc.NavItem(dbc.NavLink("A propos", href="#")),
            ],
            brand="Générateur d'Airfoil et Maillage",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Row(
            [
                dbc.Col(width=3, children=dbc.Card(
                    [
                        dbc.CardHeader("Contrôles"),
                        dbc.CardBody(
                            [
                                html.H5('Ajuster les paramètres de l\'avion'),
                                html.Label('Rayon du fuselage'),
                                dcc.Slider(
                                    id='fuselage-radius-slider',
                                    min=0.5,
                                    max=2.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: f'{i}' for i in np.arange(0.5, 2.1, 0.5)},
                                ),
                                html.Br(),
                                html.Label('Longueur du fuselage'),
                                dcc.Slider(
                                    id='fuselage-length-slider',
                                    min=5.0,
                                    max=15.0,
                                    step=0.5,
                                    value=10.0,
                                    marks={i: f'{i}' for i in np.arange(5.0, 15.1, 5.0)}),
                                # Ajout des sliders pour NACA
                                html.Label("Max Camber (%)"),
                                dcc.Slider(id="max_camber_slider", min=0, max=9, step=1, value=2,
                                           marks={i: str(i) for i in range(0, 10)}),
                                html.Label("Camber Position (%)"),
                                dcc.Slider(id="max_camber_pos_slider", min=0, max=9, step=1, value=4,
                                           marks={i: str(i) for i in range(0, 10)}),
                                html.Label("Max Thickness (%)"),
                                dcc.Slider(id="max_thickness_slider", min=0, max=99, step=1, value=12,
                                           marks={i: str(i) for i in range(0, 100, 10)}),
                                html.Label("Nombre de points dans le maillage"),
                                dcc.Slider(id="nc_slider", min=8, max=128, step=8, value=32,
                                           marks={i: str(i) for i in range(8, 129, 16)}),
                            ]
                        ),
                    ]
                )),
                dbc.Col(
                    width=9,
                    children=[
                        # Visualisation des graphiques 2D avant le graphique 3D
                        html.Div([
                            dcc.Graph(id="airfoil-plot", style={"width": "100%", "height": "400px"}),
                            dcc.Graph(id="mesh-plot", style={"width": "100%", "height": "400px"})
                        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "gap": "20px"}),
                    ],
                ),
            ],
            style={"height": "100%"},
        ),
        html.Div([
            # Visualisation 3D - Assurez-vous que les données sont passées
            dash_vtk.View(
                id="vtk-view-3d",
                cameraPosition=[0, 0, 50],
                cameraViewUp=[0, 1, 0],
                background=[1, 1, 1],
                children=[
                    dash_vtk.GeometryRepresentation(
                        dash_vtk.PolyData(
                            id="vtk-polydata", points=[], polys=[]
                        ),
                        property={
                            "edgeVisibility": False,
                            "color": [0.8, 0.8, 0.8],
                            "backfaceCulling": False,
                            "frontfaceCulling": False
                        }
                    )
                ],
                style={"height": "500px", "width": "100%"}
            ),
        ], style={"marginTop": "30px"}),
        html.Div(id="history-list"),
    ],
)

# Mise à jour de la visualisation du modèle 3D et du maillage
@app.callback(
    [Output("vtk-polydata", "points"),
     Output("vtk-polydata", "polys"),
     Output("airfoil-plot", "figure"),
     Output("mesh-plot", "figure"),
     Output("history-list", "children")],
    [Input("fuselage-radius-slider", "value"),
     Input("fuselage-length-slider", "value"),
     Input("max_camber_slider", "value"),
     Input("max_camber_pos_slider", "value"),
     Input("max_thickness_slider", "value"),
     Input("nc_slider", "value")]
)
def update_visualizations(fuselage_radius, fuselage_length, camber, camber_pos, thickness, nc):
    # Création du modèle 3D
    polydata = create_airplane(
        fuselage_radius=fuselage_radius,
        fuselage_length=fuselage_length,
        tail_size=1.0,
        wing_points_file='message.txt',
        wing_connectivity="strips"
    )

    vtk_data = vtk_to_dash_vtk(polydata)

    # Génération de l'airfoil
    airfoil = Naca4Digits(camber, camber_pos, thickness)
    x, y = airfoil.get_all_surface(1000)
    airfoil_fig = go.Figure()
    airfoil_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Airfoil", line=dict(color="red")))
    airfoil_fig.update_layout(
        title=f"NACA {camber}{camber_pos}{thickness}",
        xaxis_title="x/c",
        yaxis_title="y/c",
        showlegend=False,
        width=800,  # Largeur fixe pour éviter l'agrandissement
        height=400, # Hauteur fixe
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Génération du maillage
    cm = ConformalMapping(airfoil, nc)
    cm.generate_mesh_nodes()
    mesh_fig = mesh_to_plotly(cm.all_nodes)

    # Sauvegarde de l'historique
    save_to_csv([camber, camber_pos, thickness, nc])
    history_entries = load_csv_entries()

    return vtk_data["points"], vtk_data["polys"], airfoil_fig, mesh_fig, [html.Li(entry) for entry in history_entries]

# Fonction pour enregistrer les données dans un fichier CSV
def save_to_csv(values):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Camber", "Pos Camber", "Thickness", "Number of points"])
        writer.writerow(values)

def load_csv_entries():
    if not os.path.isfile(CSV_FILE):
        return []
    with open(CSV_FILE, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)
        return [", ".join(row) for row in reader]

def mesh_to_plotly(all_nodes):
    nodes = np.array(all_nodes)
    n_i, n_j, _ = nodes.shape
    mesh_fig = go.Figure()

    for i in range(n_i):
        x_line = nodes[i, :, 0]
        y_line = nodes[i, :, 1]
        mesh_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="blue")))

    for j in range(n_j):
        x_line = nodes[:, j, 0]
        y_line = nodes[:, j, 1]
        mesh_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="blue")))

    x_vals = nodes[:,:,0].flatten()
    y_vals = nodes[:,:,1].flatten()
    mesh_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers", marker=dict(size=4, color="red")))

    mesh_fig.update_layout(
        title="Conformal Mesh",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=False,
        width=800,  # Largeur fixe pour éviter l'agrandissement
        height=400, # Hauteur fixe
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return mesh_fig

if __name__ == "__main__":
    app.run_server(debug=True)
