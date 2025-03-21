import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
import numpy as np
import plotly.graph_objs as go

# Register the page with the route "/page-0"
dash.register_page(__name__, path="/page-0")

def load_vtu_file(file_path):
    """
    Load a VTU file from disk using VTK and convert it to vtkPolyData.
    The unstructured grid is converted to polydata using vtkGeometryFilter.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()
    polydata = geometry_filter.GetOutput()
    return polydata

def triangulate_polydata(polydata):
    """
    Convert polydata to a triangulated version using vtkTriangleFilter.
    This ensures that each cell is a triangle, which is required by Plotly's Mesh3d.
    """
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()
    return triangle_filter.GetOutput()

def polydata_to_plotly_mesh(polydata, color="blue"):
    """
    Convert triangulated vtkPolyData to a Plotly Mesh3d object.
    Assumes that polydata cells are triangles.
    """
    # Extract points array.
    points_vtk = polydata.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(points_vtk)

    # Retrieve triangle faces from polydata.
    polys = polydata.GetPolys()
    polys.InitTraversal()
    faces = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() != 3:
            continue  # skip non-triangles
        face = [idList.GetId(i) for i in range(3)]
        faces.append(face)
    
    faces_flat = [item for sublist in faces for item in sublist]

    mesh = go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=faces_flat[0::3],
        j=faces_flat[1::3],
        k=faces_flat[2::3],
        color=color,
        opacity=0.6,
        flatshading=True
    )
    return mesh

# Layout: includes a dropdown for switching wings, reload button, and a 3D plot.
layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H2("VTU Files Visualization Using Plotly (Interactive 3D)"),
            width={"size": 8, "offset": 2}
        ),
        className="mt-4"
    ),
    dbc.Row(
        dbc.Col(
            dcc.Dropdown(
                id="wing-selector",
                options=[
                    {"label": "Initial Wing (Non-Deformed)", "value": "initial"},
                    {"label": "Deformed Wing", "value": "deformed"}
                ],
                value="initial",  # Default selection
                placeholder="Select Wing Type"
            ),
            width={"size": 6, "offset": 3}
        ),
        className="mb-3"
    ),
    dbc.Row(
        dbc.Col(
            dbc.Button("Reload VTU Files", id="reload-button", color="primary", n_clicks=0),
            width={"size": 4, "offset": 4}
        ),
        className="mb-3"
    ),
    dbc.Row(
        dbc.Col(
            dcc.Graph(id="3d-plot", style={"width": "100%", "height": "800px"}),
            width=10
        )
    )
], fluid=True)

@dash.callback(
    Output("3d-plot", "figure"),
    [Input("reload-button", "n_clicks"),
     Input("wing-selector", "value")]
)
def update_plot(n_clicks, wing_type):
    """
    Callback triggered by the reload button or the wing type selector.
    Dynamically loads the appropriate VTU file based on the selected wing type.
    """
    # List your VTU file paths.
    vtu_files = {
        "initial": "../HTML_3D/output_0_nx_3_ny_40.vtu",  # Path for the initial (non-deformed) wing
        "deformed": "../HTML_3D/output_10_nx_3_ny_40.vtu"  # Path for the deformed wing
    }

    # Always load the other components (fuselage, stabilizers, etc.)
    other_components = [
        "../HTML_3D/mesh_fuse_vertical.vtu",
        "../HTML_3D/mesh_fuse_horizontal.vtu",
        "../HTML_3D/mesh_vstab.vtu",
        "../HTML_3D/mesh_hstab.vtu"
    ]
    
    colors = ["blue", "green", "red", "yellow", "purple", "orange"]
    meshes = []

    # Load the selected wing type.
    try:
        wing_file = vtu_files[wing_type]
        wing_polydata = load_vtu_file(wing_file)
        tri_wing_polydata = triangulate_polydata(wing_polydata)
        wing_mesh = polydata_to_plotly_mesh(tri_wing_polydata, color="blue")
        meshes.append(wing_mesh)
    except Exception as e:
        print(f"Error loading wing file: {e}")

    # Load other components.
    for i, file_path in enumerate(other_components):
        try:
            polydata = load_vtu_file(file_path)
            tri_polydata = triangulate_polydata(polydata)
            mesh = polydata_to_plotly_mesh(tri_polydata, color=colors[i % len(colors)])
            meshes.append(mesh)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    fig = go.Figure(data=meshes)

    # Update layout for better visualization and axis scaling.
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"  # Ensures scaling stays true to geometry.
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig
