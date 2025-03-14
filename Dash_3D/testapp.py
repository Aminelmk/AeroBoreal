import dash
import dash_vtk
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
import os
import vtk
from vtk.util import numpy_support
import numpy as np

# Import the create_airplane function
from planevtk import create_airplane

# Function to extract cell data from polydata
def extract_cells(polydata):
    """Extract cell data (polys, lines, or vertices) from polydata."""
    # Try to extract polys (faces)
    vtk_polys = polydata.GetPolys()
    if vtk_polys.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_polys.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()

    # If no polys, try to extract lines
    vtk_lines = polydata.GetLines()
    if vtk_lines.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_lines.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()

    # If no lines, try to extract vertices
    vtk_verts = polydata.GetVerts()
    if vtk_verts.GetNumberOfCells() > 0:
        vtk_cell_array = vtk_verts.GetData()
        vtk_cell_array_np = numpy_support.vtk_to_numpy(vtk_cell_array)
        return vtk_cell_array_np.tolist()

    # If no drawable cells are found
    raise ValueError("The VTK object does not contain any polygons, lines, or vertices.")
    pass

# Function to convert VTK PolyData to dash_vtk format
def vtk_to_dash_vtk(polydata):
    vtk_points = polydata.GetPoints()
    np_points = numpy_support.vtk_to_numpy(vtk_points.GetData())
    polys = extract_cells(polydata)

    return {
        'points': np_points.flatten().tolist(),
        'polys': polys,
    }
    pass

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout components (you can include sliders if you want to adjust parameters)
controls = dbc.Card(
    [
        dbc.CardHeader("Controls"),
        dbc.CardBody(
            [
                html.H5('Adjust Airplane Parameters'),
                html.Label('Fuselage Radius'),
                dcc.Slider(
                    id='fuselage-radius-slider',
                    min=0.5,
                    max=2.0,
                    step=0.1,
                    value=1.0,
                    marks={i: f'{i}' for i in np.arange(0.5, 2.1, 0.5)},
                ),
                html.Br(),
                html.Label('Fuselage Length'),
                dcc.Slider(
                    id='fuselage-length-slider',
                    min=5.0,
                    max=15.0,
                    step=0.5,
                    value=10.0,
                    marks={i: f'{i}' for i in np.arange(5.0, 15.1, 5.0)},
                ),
                # Add more sliders as needed
            ]
        ),
    ],
    style={"height": "100%"},
)

# Initialize the VTK view without data; it will be set in the callback
vtk_view = dash_vtk.View(
    id="vtk-view",
    children=[
        dash_vtk.GeometryRepresentation(
            dash_vtk.PolyData(
                id="vtk-polydata",
                points=[],  # Points will be updated in the callback
                polys=[],
            ),
            property={
                "edgeVisibility": False,
                "color": [0.8, 0.8, 0.8],
                "backfaceCulling": False,  # Disable backface culling
                "frontfaceCulling": False,
            },
        )
    ],
    background=[1, 1, 1],  # White background
)

# Layout of the app
app.layout = dbc.Container(
    fluid=True,
    style={"marginTop": "15px", "height": "calc(100vh - 30px)"},
    children=[
        dbc.Row(
            [
                dbc.Col(width=3, children=controls),
                dbc.Col(
                    width=9,
                    children=[
                        html.Div(vtk_view, style={"height": "100%", "width": "100%"}),
                    ],
                ),
            ],
            style={"height": "100%"},
        ),
    ],
)

vtk_view = dash_vtk.View(
    id="vtk-view",
    cameraPosition=[0, 0, 50],         # Set camera position
    cameraViewUp=[0, 1, 0],            # Set the view up vector
    interactorSettings={"zoomStyle": "mouseWheel"},  # Optional: Configure zoom controls
    background=[1, 1, 1],              # White background
    children=[
        dash_vtk.GeometryRepresentation(
            dash_vtk.PolyData(
                id="vtk-polydata",
                points=[],     # Points will be updated in the callback
                polys=[],
            ),
            property={
                "edgeVisibility": False,
                "color": [0.8, 0.8, 0.8],  # Set the color of the model
            },
        )
    ]
)
# Callback to update the 3D model

@app.callback(
    Output("vtk-polydata", "points"),
    Output("vtk-polydata", "polys"),
    Input("fuselage-radius-slider", "value"),
    Input("fuselage-length-slider", "value"),
    # Add other inputs as necessary
)
def update_model(fuselage_radius, fuselage_length):
    # Create the airplane with updated parameters
    polydata = create_airplane(
        fuselage_radius=fuselage_radius,
        fuselage_length=fuselage_length,
        tail_size=1.0,  # You can add more parameters controlled by sliders
        wing_points_file='message.txt',
        wing_connectivity="strips",
    )

    vtk_data = vtk_to_dash_vtk(polydata)
    return vtk_data["points"], vtk_data["polys"]

if __name__ == "__main__":
    app.run_server(debug=True)
