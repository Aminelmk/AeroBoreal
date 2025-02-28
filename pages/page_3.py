import dash
from dash import html, Input, Output
import dash_vtk
import vtk
from vtk.util import numpy_support
import numpy as np
import os

dash.register_page(__name__, path="/page-3")  # Register the page




def read_wing_points(filename):
    """Read wing points from a comma-separated text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    
    points = []
    with open(filename, "r") as file:
        for line in file:
            print(f"Reading line: {line.strip()}")  # Debugging line
            try:
                # Split each line by commas and try to convert to float
                x, y, z = map(float, line.strip().split(','))
                points.append([x, y, z])
            except ValueError:
                # Handle lines that don't contain three values or cannot be converted to float
                print(f"Skipping invalid line: {line.strip()}")  # Debugging line
                continue
    
    if not points:
        raise ValueError("No valid points found in the file.")
    
    return points


def create_custom_wing_from_data(points_list, connectivity_type="strips"):
    """Create a wing using provided points and connectivity."""
    if len(points_list) < 3:  # Ensure there are enough points for a wing
        raise ValueError("Insufficient points for wing creation.")
    
    # Convert points list to numpy array
    points_array = np.array(points_list).reshape(-1, 3)
    
    # Ensure reshaping was successful (debugging line)
    print(f"Points reshaped: {points_array.shape}")
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points_array:
        vtk_points.InsertNextPoint(point)
    
    # Create the appropriate cell array based on connectivity
    if connectivity_type == "strips":
        # Assuming triangle strips
        strips = vtk.vtkCellArray()
        num_points = vtk_points.GetNumberOfPoints()
        strips.InsertNextCell(num_points)
        for i in range(num_points):
            strips.InsertCellPoint(i)
    else:
        # Default to polygons
        polygon = vtk.vtkPolygon()
        num_points = vtk_points.GetNumberOfPoints()
        polygon.GetPointIds().SetNumberOfIds(num_points)
        for i in range(num_points):
            polygon.GetPointIds().SetId(i, i)
        polys = vtk.vtkCellArray()
        polys.InsertNextCell(polygon)
    
    # Create PolyData
    wing_polydata = vtk.vtkPolyData()
    wing_polydata.SetPoints(vtk_points)
    if connectivity_type == "strips":
        wing_polydata.SetStrips(strips)
    else:
        wing_polydata.SetPolys(polys)
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(wing_polydata)
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.Update()

    wing_polydata_with_normals = normals.GetOutput()
    return wing_polydata_with_normals




def create_wing_points(wing_points_file):
    """Creates a VTK PolyData containing only points."""
    wing_points = read_wing_points(wing_points_file)
    if wing_points is None:
        return None
    points_array = np.array(wing_points).reshape(-1, 3)
    vtk_points = vtk.vtkPoints()
    for point in points_array:
        vtk_points.InsertNextPoint(point)
    wing_polydata = vtk.vtkPolyData()
    wing_polydata.SetPoints(vtk_points)
    return wing_polydata

def vtk_to_dash_vtk(polydata):
    """Convert VTK PolyData to Dash format."""
    if polydata is None:
        return {"points": [], "polys": []}

    # Convert VTK points to numpy
    vtk_points = polydata.GetPoints()
    np_points = numpy_support.vtk_to_numpy(vtk_points.GetData())

    # Get cells (polys) data
    vtk_cells = polydata.GetPolys()
    vtk_cells.InitTraversal()
    
    cells = []
    id_list = vtk.vtkIdList()
    while vtk_cells.GetNextCell(id_list):
        # Extract the point indices from vtkIdList and convert them to a list of integers
        cell_points = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
        cells.append(cell_points)

    # Convert cells to the format Dash VTK expects
    np_cells = np.array(cells, dtype=int)  # Ensure the cells are a numpy array of integers
    return {
        "points": np_points.flatten().tolist(),
        "polys": np_cells.flatten().tolist()
    }

# Define the Page Layout
layout = html.Div([
    html.H1("3D Wing Point Cloud"),
    html.Button("Load Wing Points", id="load-wing-btn", n_clicks=0),  # Button to load points
    html.Div(id="error-message", style={"color": "red"}),  # Error message placeholder

    # 3D Visualization
    html.Div([
        dash_vtk.View(
            id="vtk-view-3d",
            cameraPosition=[0, 0, 50],
            background=[1, 1, 1],
            children=[
                dash_vtk.GeometryRepresentation(
                    dash_vtk.PolyData(id="vtk-polydata-wing-points", points=[], polys=[]),
                    property={"color": [0, 0, 1], "representation": "points"}  # Blue points
                )
            ],
            style={"height": "200vh", "width": "200vw"}
        )
    ], style={"marginTop": "30px"})
])
def create_wing(wing_points_file, wing_connectivity="strips"):
    """Creates only the airplane wing without the fuselage and positions it in the XY plane rotated by 45 degrees."""
    if wing_points_file is None:
        raise ValueError("Wing points file is required.")
    
    wing_points = read_wing_points(wing_points_file)
    wing_polydata = create_custom_wing_from_data(wing_points, wing_connectivity)

    # Transform the wing to the correct position in the XY plane and rotate by 45 degrees
    wing_transform = vtk.vtkTransform()
    
    # Apply a rotation of 45 degrees around the Z-axis (XY plane rotation)
    wing_transform.RotateZ(45)  # Rotate 45 degrees around Z-axis
    
    wing_transform.Scale(5, 5, 5)  # Scale up the model for better visibility
    
    wing_transform_filter = vtk.vtkTransformPolyDataFilter()
    wing_transform_filter.SetInputData(wing_polydata)
    wing_transform_filter.SetTransform(wing_transform)
    wing_transform_filter.Update()

    return wing_transform_filter.GetOutput()




@dash.callback(
    [Output("vtk-polydata-wing-points", "points"),
     Output("vtk-polydata-wing-points", "polys"),
     Output("error-message", "children")],
    [Input("load-wing-btn", "n_clicks")]
)
def update_3d_visualization(n_clicks):  
    if n_clicks == 0:
        return [], [], ""  # Do nothing before the first click
    
    try:
        polydata = create_wing("wing_points.txt", "strips")
        vtk_data = vtk_to_dash_vtk(polydata)
        return vtk_data["points"], vtk_data["polys"], ""
    except Exception as e:
        return [], [], f"Error: {str(e)}"
