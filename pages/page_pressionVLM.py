import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
import numpy as np
import plotly.graph_objs as go

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
            dbc.Col(
                html.H2("VTU Visualization Dashboard", className="text-center my-4 text-primary"),
                width=12
            )
        ),
        
        # Wing Type Selector
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.Label("Select Wing Type:", className="form-label text-primary"),
                        dcc.Dropdown(
                            id="wing-selector",
                            options=[
                                {"label": "Initial Wing (Non-Deformed)", "value": "initial"},
                                {"label": "Deformed Wing", "value": "deformed"}
                            ],
                            value="initial",  # Default selection.
                            placeholder="Select Wing Type",
                            className="dropdown"
                        )
                    ], className="mb-3"),
                    width={"size": 6, "offset": 3}
                )
            ]
        ),
        
        # Reload Button
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Reload VTU Files", id="reload-button", color="primary", n_clicks=0, className="my-3 w-50"),
                    width={"size": 6, "offset": 3},
                    className="d-flex justify-content-center"
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
    [Input("reload-button", "n_clicks"), Input("wing-selector", "value")]
)
def update_plot(n_clicks, wing_type):
    # VTU file paths
    wing_files = {
        "initial": "../HTML_3D/output_0_nx_3_ny_40.vtu",
        "deformed": "../HTML_3D/output_10_nx_3_ny_40.vtu",
    }
    other_components = [
        "../HTML_3D/mesh_fuse_vertical.vtu",
        "../HTML_3D/mesh_fuse_horizontal.vtu",
        "../HTML_3D/mesh_vstab.vtu",
        "../HTML_3D/mesh_hstab.vtu",
    ]

    meshes = []

    # Load the selected wing
    try:
        wing_file = wing_files[wing_type]
        wing_polydata = load_vtu_file(wing_file)
        
        # Debug: Check pressure field after conversion
        print("Pressure (after conversion):", wing_polydata.GetPointData().GetArray("pressure"))

        pressure_values = numpy_support.vtk_to_numpy(wing_polydata.GetPointData().GetArray("pressure"))
        print("Pressure Values Shape:", pressure_values.shape)

        tri_wing_polydata = triangulate_polydata(wing_polydata)
        wing_mesh = polydata_to_plotly_mesh_with_scalar(
            tri_wing_polydata,
            pressure_values,
            scalar_name="pressure",
            colorscale="Viridis",
        )
        meshes.append(wing_mesh)
    except Exception as e:
        print(f"Error loading wing file: {e}")

    # Load other components
    for i, file_path in enumerate(other_components):
        try:
            polydata = load_vtu_file(file_path)  # Correct polydata reference
            tri_polydata = triangulate_polydata(polydata)
            mesh = polydata_to_plotly_mesh_with_scalar(
                tri_polydata,
                scalar_name="pressure",
                colorscale="Cividis",
            )
            meshes.append(mesh)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    fig = go.Figure(data=meshes)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, b=20, t=20),
    )
    return fig

