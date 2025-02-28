import dash
from dash import html, dcc, Input, Output
import dash_vtk
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
from Dash_3D.planevtk import read_wing_points
import numpy as np

dash.register_page(__name__, path="/page-2")

def create_custom_wing_from_data(points_list, connectivity_type="strips"):
    """Create a wing using provided points and connectivity."""
    # Convert points list to numpy array
    points_array = np.array(points_list).reshape(-1, 3)
    
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

def create_wing(wing_points_file, wing_connectivity="strips"):
    """Creates only the airplane wing without the fuselage and positions it in the XY plane."""
    if wing_points_file is None:
        raise ValueError("Wing points file is required.")
    
    wing_points = read_wing_points(wing_points_file)
    wing_polydata = create_custom_wing_from_data(wing_points, wing_connectivity)

    # Transform the wing to the correct position in the XY plane
    wing_transform = vtk.vtkTransform()
    wing_transform.RotateX(0)  # Ensures alignment with the XY plane
    wing_transform.Scale(5, 5, 5)  # Scale up the model for better visibility
    
    wing_transform_filter = vtk.vtkTransformPolyDataFilter()
    wing_transform_filter.SetInputData(wing_polydata)
    wing_transform_filter.SetTransform(wing_transform)
    wing_transform_filter.Update()

    return wing_transform_filter.GetOutput()

# Function to convert VTK PolyData to Dash-compatible format
def extract_cells(polydata):
    vtk_polys = polydata.GetPolys()
    if vtk_polys.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_polys.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()
    
    raise ValueError("Le PolyData ne contient pas de polygones.")

def vtk_to_dash_vtk(polydata):
    vtk_points = polydata.GetPoints()
    np_points = numpy_support.vtk_to_numpy(vtk_points.GetData())
    polys = extract_cells(polydata)

    return {
        'points': np_points.flatten().tolist(),
        'polys': polys,
    }

# Layout of Page 2
layout = html.Div([
    html.H1("3D Wing Visualization"),
    html.P("Adjust the parameters below to update the wing model."),
    
    # 3D Visualization with dash_vtk
    html.Div([
        dash_vtk.View(
            id="vtk-view-3d",
            cameraPosition=[0, 0, 100],  # Move camera further back
            cameraViewUp=[0, 1, 0],
            background=[1, 1, 1],
            children=[
                dash_vtk.GeometryRepresentation(
                    dash_vtk.PolyData(id="vtk-polydata-wing", points=[], polys=[]),
                    property={
                        "edgeVisibility": False,
                        "color": [0.8, 0.8, 0.8],
                        "backfaceCulling": False,
                        "frontfaceCulling": False
                    }
                )
            ],
            style={"height": "100vh", "width": "100vw"}  # Make visualization full page
        ),
    ], style={"marginTop": "30px"}),
])

# Callback to update the 3D visualization dynamically
@dash.callback(
    [Output("vtk-polydata-wing", "points"),
     Output("vtk-polydata-wing", "polys")],
    [Input("vtk-view-3d", "id")]  # Using a dummy input to trigger the callback
)
def update_3d_visualization(_):  
    polydata = create_wing("wing_points.txt", "strips")
    vtk_data = vtk_to_dash_vtk(polydata)
    return vtk_data["points"], vtk_data["polys"]
