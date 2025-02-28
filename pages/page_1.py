
import dash
from dash import html, dcc, Input, Output
import dash_vtk
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
from Dash_3D.planevtk import create_airplane 
from Dash_3D.planevtk import create_custom_wing_from_data

dash.register_page(__name__, path="/page-1")

# Function to convert VTK PolyData to Dash-compatible format
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

def vtk_to_dash_vtk(polydata):
    vtk_points = polydata.GetPoints()
    np_points = numpy_support.vtk_to_numpy(vtk_points.GetData())
    polys = extract_cells(polydata)

    return {
        'points': np_points.flatten().tolist(),
        'polys': polys,
    }

# Layout of Page 1
layout = html.Div([
    html.H1("3D Airplane Visualization"),
    html.P("Adjust the parameters below to update the airplane model."),
    
    # Sliders for controlling the 3D airplane
    html.Label("Fuselage Radius"),
    dcc.Slider(id="fuselage-radius-slider", min=0.5, max=2.0, step=0.1, value=1.0,
               marks={i: str(i) for i in range(1, 3)}),
    
    html.Label("Fuselage Length"),
    dcc.Slider(id="fuselage-length-slider", min=5.0, max=15.0, step=0.5, value=10.0,
               marks={i: str(i) for i in range(5, 16, 5)}),
    
    # 3D Visualization with dash_vtk
    html.Div([
        dash_vtk.View(
            id="vtk-view-3d",
            cameraPosition=[0, 0, 50],
            cameraViewUp=[0, 1, 0],
            background=[1, 1, 1],
            children=[
                dash_vtk.GeometryRepresentation(
                    dash_vtk.PolyData(id="vtk-polydata", points=[], polys=[]),
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
])

# Callback to update the 3D visualization dynamically
@dash.callback(
    [Output("vtk-polydata", "points"),
     Output("vtk-polydata", "polys")],
    [Input("fuselage-radius-slider", "value"),
     Input("fuselage-length-slider", "value")]
)
def update_3d_visualization(fuselage_radius, fuselage_length):
    polydata = create_airplane(
        fuselage_radius=fuselage_radius,
        fuselage_length=fuselage_length,
        tail_size=1.0,
        wing_points_file = "c:/Users/nadir/Desktop/C++/HTML_3D-NACA_2V/message.txt",
        wing_connectivity="strips"
    )

    vtk_data = vtk_to_dash_vtk(polydata)
    return vtk_data["points"], vtk_data["polys"]
