import dash
from dash import html, dcc, Input, Output, ctx
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
import plotly.colors as pc
import numpy as np
import plotly.graph_objs as go
import os
import pandas as pd
from urllib.parse import parse_qs, urlparse
from courbes_pression.Plot_cp import *

dash.register_page(__name__, path="/pages-pressionVLM")





def load_vtu_file(file_path):
    """
    Load a VTU file from disk using VTK and convert it to vtkPolyData.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()
    polydata = geometry_filter.GetOutput()

    # Perform cell-to-point data conversion.
    cell_to_point_filter = vtk.vtkCellDataToPointData()
    cell_to_point_filter.SetInputData(polydata)
    cell_to_point_filter.Update()
    converted_polydata = cell_to_point_filter.GetOutput()

    return converted_polydata

def polydata_to_plotly_quads(polydata, scalar_values=None, colorscale="Viridis"):
    """
    Extract quads (or other polygons) from vtkPolyData and render them as filled quads in Plotly.
    """
    # Extract points
    vtk_points = polydata.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(vtk_points)

    # Retrieve quads (or other polygons)
    polys = polydata.GetPolys()
    polys.InitTraversal()
    quads = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 4:  # Only process quads
            quad = [idList.GetId(i) for i in range(4)]
            quads.append(quad)

    # Create a list of Scatter3d traces for each quad
    traces = []
    for quad in quads:
        x = [points[quad[i], 0] for i in range(4)] + [points[quad[0], 0]]  # Close the quad
        y = [points[quad[i], 1] for i in range(4)] + [points[quad[0], 1]]
        z = [points[quad[i], 2] for i in range(4)] + [points[quad[0], 2]]

        # Create a Scatter3d trace for the quad
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",  # Use "lines" to render edges, "lines+markers" for vertices
            line=dict(color="black", width=2),  # Customize edge color and width
            marker=dict(size=2, color="blue"),  # Customize vertex markers
            hoverinfo="none",
            showlegend=False,
        )
        traces.append(trace)

    return traces

def triangulate_polydata(polydata):
    """
    Triangulate the polydata using vtkTriangleFilter for Plotly's Mesh3d
    (which expects triangular cells).
    """
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()
    return triangle_filter.GetOutput()

def extract_cell_data(polydata, scalar_name):
    """
    Extract cell-based scalar values (e.g., pressure) from vtkPolyData.
    """
    data_array = polydata.GetCellData().GetArray(scalar_name)
    if data_array:
        return numpy_support.vtk_to_numpy(data_array)
    else:
        raise ValueError(f"Scalar field '{scalar_name}' not found in VTU file.")

def polydata_to_plotly_mesh_with_scalar(polydata, pressure_values, scalar_name="pressure", colorscale="Viridis"):
    """
    Convert triangulated vtkPolyData to a Plotly Mesh3d object with scalar coloring.
    """
    # Extract points.
    vtk_points = polydata.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(vtk_points)

    # Retrieve triangles.
    polys = polydata.GetPolys()
    polys.InitTraversal()
    faces = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() != 3:
            continue  # Skip non-triangles.
        face = [idList.GetId(i) for i in range(3)]
        faces.append(face)

    faces_flat = [item for sublist in faces for item in sublist]

    # Extract scalar values for coloring.
    try:
        scalar_values = extract_cell_data(polydata, scalar_name)
    except ValueError as e:
        print(f"Warning: {e}")
        scalar_values = np.zeros(len(faces))  # Default to zeros if scalar not found.

    # Create a Mesh3d object with color mapping based on the scalar field.
    mesh = go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=faces_flat[0::3],
        j=faces_flat[1::3],
        k=faces_flat[2::3],
        intensity=pressure_values,  # Pass the point-based pressure values
        colorscale=colorscale,    # Use Viridis (or any other Plotly colorscale).
        showscale=True,           # Display the color scale bar.
        opacity=0.8,
        flatshading=True
    )
    return mesh


def polydata_to_plotly_mesh_with_panels(polydata, scalar_values=None, scalar_name="pressure", colorscale="Viridis"):
    """
    Convert vtkPolyData to a Plotly Mesh3d object with optional scalar coloring for panels.
    """
    # Extract points
    vtk_points = polydata.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(vtk_points)

    # Retrieve panels (triangles or quads)
    polys = polydata.GetPolys()
    polys.InitTraversal()
    faces = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:  # Only process triangles
            face = [idList.GetId(i) for i in range(3)]
            faces.append(face)

    faces_flat = [item for sublist in faces for item in sublist]

    

    # Create a Mesh3d object for panels
    mesh = go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=faces_flat[0::3],
        j=faces_flat[1::3],
        k=faces_flat[2::3],
        intensity=scalar_values if scalar_values is not None else np.zeros(len(faces)),
        colorscale=colorscale,
        showscale=True,
        opacity=0.8,
        flatshading=True
    )
    return mesh

def polydata_to_plotly_edges(polydata):
    """
    Extract edges from vtkPolyData and convert them to a Plotly Scatter3d object.
    """
    # Extract points
    vtk_points = polydata.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(vtk_points)

    # Retrieve edges from the triangles
    polys = polydata.GetPolys()
    polys.InitTraversal()
    edges = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:  # Only process triangles
            edges.append([idList.GetId(0), idList.GetId(1)])
            edges.append([idList.GetId(1), idList.GetId(2)])
            edges.append([idList.GetId(2), idList.GetId(0)])

    # Flatten the edges into x, y, z coordinates for Scatter3d
    x, y, z = [], [], []
    for edge in edges:
        x.extend([points[edge[0], 0], points[edge[1], 0], None])  # Add None for line breaks
        y.extend([points[edge[0], 1], points[edge[1], 1], None])
        z.extend([points[edge[0], 2], points[edge[1], 2], None])

    # Create a Scatter3d object for the edges
    edge_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="black", width=2),  # Customize edge color and width
        hoverinfo="none",
    )
    return edge_trace

def load_mesh(file_name):
        # Lire le fichier
        with open(file_name, "r") as f:
            # Lire la première ligne pour obtenir nx et ny
            nx, ny = map(int, f.readline().split())

            # Charger les coordonnées du maillage (les lignes suivantes)
            mesh_data = np.loadtxt(f, delimiter=',')
            point = int((len(mesh_data))/3)
            print(point)

            # Réorganiser les données dans x, y, z
            # Le maillage est stocké sous la forme x, y, z en une seule colonne, on les sépare en trois tableaux
            x = mesh_data[:point]
            y = mesh_data[point:2 * point]
            z = mesh_data[2 *point:]
            print(x,y,z)

        # Retourner les résultats
        return nx, ny, x, y, z



def fig_cp_curve():

    y = np.linspace(-0.7,0.7,10)

    x,cp = calculCp(y)
    mesh_fig = go.Figure()
    
    #y = np.linspace(-0.7,0.7,10)

    #x,cp = calculCp(y)
    colorscale = pc.sequential.Viridis
    num_curves = len(y)

    # Répéter ou interpoler les couleurs si le nombre de courbes est supérieur à la taille de l'échelle
    colors = (colorscale * ((num_curves // len(colorscale)) + 1))[:num_curves]  # Répéter les couleurs pour couvrir toutes les courbes

    for i in range(num_curves):
        # Ajouter la trace avec une couleur correspondant à l'échelle
        mesh_fig.add_trace(go.Scatter3d(
            x=x[i],
            y=y[i] * np.ones(len(x[i])),
            z=cp[i],
            mode="lines",
            line=dict(
                color=colors[i],  # Appliquer la couleur de la courbe
                width=4  # Ajuster l'épaisseur de la ligne
            ),
            name=f"Courbe {i+1}"
        ))

    
    x_outline, y_outline, z_outline = outline()

    mesh_fig.add_trace(go.Scatter3d(
                        x=x_outline,
                        y= y_outline ,
                        z= z_outline,
                        mode="lines",
                        line=dict(color="black"),
                        name=f"Wing Outline"
                    ))

    
    
    mesh_fig.update_layout(
    scene=dict(
        xaxis_title='Y',
        yaxis_title='X',
        zaxis_title='Cp',
        aspectmode='data',
        camera=dict(
            eye=dict(x=0, y=0, z=0)  # <-- Position de la caméra
        ), 
        aspectratio=dict(x=10, y=10, z=0),  # Réduire l'échelle de l'axe z par rapport aux autres
    ),
    showlegend=False
)
    return mesh_fig
# Layout with enhanced styling and usability
layout = dbc.Container(
    [
        # Header Section
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.Label("Select Wing Type:", className="form-label text-primary"),
                        dcc.Dropdown(
                            id="wing-selector",
                            options=[
                                {"label": "Deformed Wing", "value": "deformed"},
                                {"label": "Initial Wing (Non-Deformed)", "value": "initial"},
                            ],
                            value="deformed",  # Default selection.
                            placeholder="Select Wing Type",
                            className="dropdown"
                        )
                    ], className="mb-3"),
                    width=6
                ),
                dbc.Col(
                    html.Div([
                        html.Label("Select Value to Visualize:", className="form-label text-primary"),
                        dcc.Dropdown(
                            id="value-selector",
                            options=[
                                {"label": "Pressure", "value": "pressure"},
                                {"label": "Cp", "value": "cp"},
                                {"label": "Cp2D", "value": "cp2d"},
                                {"label": "Drag", "value": "drag"},
                                {"label": "Lift", "value": "lift"}
                            ],
                            value="pressure",  # Default selection.
                            placeholder="Select Value",
                            className="dropdown"
                        )
                    ], className="mb-3"),
                    width=6
                )
            ]
        ),
        # Row for "Show Panels" and "Selected Panel Value"
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.Label("Show Panels:", className="form-label text-primary", style={"font-size": "20px"}),
                        dcc.Checklist(
                            id="show-panels",
                            options=[{"label": "", "value": "show"}],  # Empty label for the checkbox
                            value=["show"],  # Default to showing panels
                            inline=True,
                            className="checkbox",
                            style={
                        "transform": "scale(1.5)",  # Scale up the checkbox size
                        "margin-left": "10px",  # Add spacing between the label and checkbox
                        }
                        )
                    ], className="d-flex align-items-center mb-3"),  # Align vertically with the panel value box
                    width=6
                ),
                dbc.Col(
                    html.Div(
                        id="clicked-panel-info",
                        style={
                            "font-size": "16px",
                            "color": "blue",
                            "padding": "10px",
                            "border": "1px solid #ccc",
                            "border-radius": "5px",
                            "background-color": "#f9f9f9",
                            "text-align": "center",
                        },
                    ),
                    width=6
                )
            ]
        ),
        dcc.Store(id="camera-state", data=None),  # Store for camera state
        # 3D Visualization Graph
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="3d-plot1", style={"width": "100%", "height": "600px"}, config={"scrollZoom": True}),
                    width=12,
                    className="mb-4"
                )
            ]
        ),

         # Tz vs. y and Ry vs. y plots
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="tz-vs-y-plot", style={"width": "100%", "height": "400px"}),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(id="ry-vs-y-plot", style={"width": "100%", "height": "400px"}),
                    width=6
                )
            ]
        ),

         # Cl and alpha discrete plots
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="Cl-discrete", style={"width": "100%", "height": "400px"}),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(id="alpha-discrete", style={"width": "100%", "height": "400px"}),
                    width=6
                )
            ]
        )
    ],
    fluid=True,
    style={"font-family": "Arial, sans-serif"}
)

@dash.callback(
    Output("3d-plot1", "figure"),
    [Input("wing-selector", "value"),
     Input("value-selector", "value"),
     Input("show-panels", "value"),
     Input("camera-state", "data")],  # Capture the current camera state
)
def update_plot(wing_type, scalar_name, show_panels, camera_state):
    # Path to the "temp" folder
    temp_folder = "../HTML_3D/temp"

    # Check for .vtu files in the "temp" folder
    vtu_files = [f for f in os.listdir(temp_folder) if f.endswith(".vtu")]
    if not vtu_files:
        # If no .vtu files are found, return an error message as a figure
        fig = go.Figure()
        fig.add_annotation(
            text="Error: No .vtu files found in the 'temp' folder.<br>Please ensure the folder contains valid VTU files.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="red"),
            align="center",
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=20, r=20, b=20, t=20),
        )
        return fig

    # VTU file paths
    wing_files = {
        "deformed": os.path.join(temp_folder, vtu_files[-1]),
        "initial": os.path.join(temp_folder, vtu_files[0]),
    }

    meshes = []

    # Load the selected wing
    try:
        wing_file = wing_files[wing_type]
        wing_polydata = load_vtu_file(wing_file)

        # Debug: Check the selected scalar field
        print(f"Selected Scalar Field: {scalar_name}")
        scalar_values = wing_polydata.GetPointData().GetArray(scalar_name)
        if scalar_values is not None:
            scalar_values = numpy_support.vtk_to_numpy(scalar_values)
        else:
            scalar_values = None

        # Add scalar visualization
        tri_wing_polydata = triangulate_polydata(wing_polydata)
        wing_mesh = polydata_to_plotly_mesh_with_scalar(
            tri_wing_polydata,
            scalar_values,
            scalar_name=scalar_name,
            colorscale="Viridis",
        )
        meshes.append(wing_mesh)

        # Add panel visualization if the checkbox is checked
        if "show" in show_panels:
            quad_traces = polydata_to_plotly_quads(
                wing_polydata,
                scalar_values=scalar_values,
                colorscale="Cividis",
            )
            meshes.extend(quad_traces)
    except Exception as e:
        print(f"Error loading wing file: {e}")

    fig = go.Figure(data=meshes)

    # Set the zoom level and center the view
    points = numpy_support.vtk_to_numpy(wing_polydata.GetPoints().GetData())
    x_range = [np.min(points[:, 0]), np.max(points[:, 0])]
    y_range = [-np.max(points[:, 1]), np.max(points[:, 1])]
    z_range = [-np.max(points[:, 1]), np.max(points[:, 1])]

    # Default camera settings
    scene_camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    )

    # Use the stored camera state if available
    if camera_state:
        scene_camera = camera_state

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=x_range),
            yaxis=dict(title="Y", range=y_range),
            zaxis=dict(title="Z", range=z_range),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2, z=1),  # Relative scaling of the axes
        ),
        scene_camera=scene_camera,  # Apply the preserved or default camera state
        margin=dict(l=20, r=20, b=20, t=20),
    )

    if scalar_name=='cp2d':
        fig = fig_cp_curve()
        fig.update_traces(line=dict(width=5))
        z_min = min(fig.data[0].z-0.5)  # On prend les valeurs de l'axe z du premier trace
        z_max = max(fig.data[0].z+0.5)
        fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=2, y=4, z=1.5),
            zaxis=dict(
                range=[z_max, z_min]  # Inversez les valeurs ici
            )
        )
    )

    return fig


@dash.callback(
    Output("clicked-panel-info", "children"),
    Input("3d-plot1", "clickData"),
)
def display_clicked_panel_info(click_data):
    if click_data is None:
        return html.Div(
            "Click on a node to see its value.",
            style={
                "font-size": "16px",
                "color": "gray",
                "padding": "10px",
                "border": "1px solid #ccc",
                "border-radius": "5px",
                "background-color": "#f9f9f9",
                "text-align": "center",
            },
        )

    # Extract the scalar value from the clicked data
    try:
        scalar_value = click_data["points"][0]["intensity"]
        return html.Div(
            f"Selected Panel Value: {scalar_value:.2f}",
            style={
                "font-size": "18px",
                "color": "white",
                "padding": "10px",
                "border": "1px solid #007bff",
                "border-radius": "5px",
                "background-color": "#007bff",
                "text-align": "center",
                "font-weight": "bold",
            },
        )
    except KeyError:
        return html.Div(
            "No scalar value available for the selected panel.",
            style={
                "font-size": "16px",
                "color": "red",
                "padding": "10px",
                "border": "1px solid #f5c6cb",
                "border-radius": "5px",
                "background-color": "#f8d7da",
                "text-align": "center",
            },
        )

import re  # For extracting numbers from filenames

@dash.callback(
    [Output("tz-vs-y-plot", "figure"), Output("ry-vs-y-plot", "figure"), Output("Cl-discrete", "figure"), Output("alpha-discrete", "figure")],
    Input("3d-plot1", "relayoutData"),  # Trigger when the 3D plot is updated
)
def update_displacement_plots(relayout_data):
    # Path to the "temp" folder
    temp_folder = "../HTML_3D/temp"

    # Find all displacement_i.csv files in the folder
    displacement_files = [f for f in os.listdir(temp_folder) if (f.startswith("displacement_") and f.endswith(".csv"))]
    if not displacement_files:
        # Return empty figures if no displacement files are found
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No displacement data available.")
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Identify the latest displacement file (highest i)
    latest_file_path = os.path.join(temp_folder, displacement_files[-1])

    # Load the latest displacement file
    data = pd.read_csv(latest_file_path)
    span = (data["y"].max() - data["y"].min())/2

    # Create the Tz vs. y plot
    tz_vs_y_fig = go.Figure()
    tz_vs_y_fig.add_trace(go.Scatter(x=data["y"]/span, y=data["Tz"]*1000, mode="lines+markers", name="Tz vs. y/span"))
    tz_vs_y_fig.update_layout(
        title="Tz vs. y/span",
        xaxis_title="y/span",
        yaxis_title="Tz (mm)",
        template="plotly_white",
    )

    # Create the Ry vs. y plot
    ry_vs_y_fig = go.Figure()
    ry_vs_y_fig.add_trace(go.Scatter(x=data["y"]/span, y=np.degrees(data["Ry"]), mode="lines+markers", name="Ry vs. y/span"))
    ry_vs_y_fig.update_layout(
        title="Ry vs. y/span",
        xaxis_title="y/span",
        yaxis_title="Ry (deg)",
        template="plotly_white",
    )



    it_max = []
    for filename in os.listdir("temp"):  
        if 'displacement' not in filename : 
            it_max.append(filename.split("_")[1])
    nx = int(filename.split("_")[3])
    ny = int(filename.split("_")[5])
    it_max = int(max(it_max))
    nomFichier = f"temp/output_{it_max}_nx_{nx}_ny_{ny}_Cl.csv"
    df = pd.read_csv(nomFichier, encoding="utf-8")
    y_aero = (df["y"].values[:ny] + df["y"].values[1:ny+1])/2
    Cl = df["Cl"].values[:ny]
    alpha_e = np.rad2deg(df["alpha_e"].values[:ny])


    # Create the Cl vs. y plot
    Cl_discrete = go.Figure()
    Cl_discrete.add_trace(go.Scatter(x=y_aero/span, y=Cl, mode="lines+markers", name="Cl vs span"))
    Cl_discrete.update_layout(
        title="Cl vs. y/span",
        xaxis_title="y/span",
        yaxis_title="Cl",
        template="plotly_white",
    )

    # Create the alpha vs. y plot
    alpha_discrete = go.Figure()
    alpha_discrete.add_trace(go.Scatter(x=y_aero/span, y=alpha_e, mode="lines+markers", name="alpha vs span"))
    alpha_discrete.update_layout(
        title="alpha_e vs. y/span",
        xaxis_title="y/span",
        yaxis_title="alpha_e (deg)",
        template="plotly_white",
    )




    return tz_vs_y_fig, ry_vs_y_fig, Cl_discrete, alpha_discrete


