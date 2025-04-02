import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
import numpy as np
import plotly.graph_objs as go
import os

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
        
        
        # 3D Visualization Graph
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="3d-plot1", style={"width": "100%", "height": "600px"}),
                    width=12,
                    className="mb-4"
                )
            ]
        )
    ],
    fluid=True,
    style={"font-family": "Arial, sans-serif"}
)

@dash.callback(
    Output("3d-plot1", "figure"),
    [Input("wing-selector", "value"), Input("value-selector", "value")]
)
def update_plot(wing_type, scalar_name):
        # Path to the "temp" folder
    temp_folder = "../HTML_3D/temp"

    # Check for .vtu files in the "temp" folder
    vtu_files = [f for f in os.listdir(temp_folder) if f.endswith(".vtu")]
    if not vtu_files:
        # If no .vtu files are found, return an error message as a figure
        fig = go.Figure()
        fig.update_layout(
            title="Error: No .vtu files found in the 'temp' folder.",
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
        # if scalar_values is None:
        #     raise ValueError(f"Scalar field '{scalar_name}' not found in the VTU file.")

        scalar_values = numpy_support.vtk_to_numpy(scalar_values)
        # print(f"{scalar_name} Values Shape:", scalar_values.shape)

        tri_wing_polydata = triangulate_polydata(wing_polydata)
        wing_mesh = polydata_to_plotly_mesh_with_scalar(
            tri_wing_polydata,
            scalar_values,
            scalar_name=scalar_name,
            colorscale="Viridis",
        )
        meshes.append(wing_mesh)
    except Exception as e:
        print(f"Error loading wing file: {e}")


    fig = go.Figure(data=meshes)
    #Set the zoom level and center the view
    points = numpy_support.vtk_to_numpy(wing_polydata.GetPoints().GetData())
    x_range = [np.min(points[:, 0]), np.max(points[:, 0])]
    y_range = [-np.max(points[:, 1]), np.max(points[:, 1])]
    z_range = [-np.max(points[:, 1]), np.max(points[:, 1])]
    # x_range = [-3, 3]
    # y_range = [-3, 3]
    # z_range = [-3, 3]
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=x_range),
            yaxis=dict(title="Y",range=y_range),
            zaxis=dict(title="Z", range=z_range),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2, z=1),  # Relative scaling of the axes
            # aspectmode="data",
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        ),
        margin=dict(l=20, r=20, b=20, t=20),
    )
    return fig

