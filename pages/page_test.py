import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support
import numpy as np
import plotly.graph_objs as go
import io, base64

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

def polydata_to_plotly_mesh(polydata, color="red"):
    """
    Convert triangulated vtkPolyData to a Plotly Mesh3d object for 3D visualization.
    Assumes that polydata cells are triangles.
    """
    # Extract points array.
    points_vtk = polydata.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(points_vtk)

    # Retrieve triangle faces from polydata.
    # Since we have triangulated the polydata, each cell is a triangle.
    polys = polydata.GetPolys()
    polys.InitTraversal()
    faces = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() != 3:
            continue  # skip non-triangles
        # Build a face (triangle) that is a list of 3 vertex indices.
        face = [idList.GetId(i) for i in range(3)]
        faces.append(face)
    
    # Flatten the list of faces.
    # Each face is a triangle, so faces_flat will be a multiple of 3.
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

# Layout: header, reload button, and a Plotly 3D graph.
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
            dbc.Button("Reload VTU Files", id="reload-button", color="primary", n_clicks=0),
            width={"size": 4, "offset": 4}
        ),
        className="mb-3"
    ),
    dbc.Row(
        dbc.Col(
            dcc.Graph(id="3d-plot1", style={"width": "100%", "height": "800px"}),
            width=10
        )
    )
], fluid=True)

@dash.callback(
    Output("3d-plot1", "figure"),
    Input("reload-button", "n_clicks")
)
def update_plot(n_clicks):
    """
    Callback triggered by the reload button that:
      1. Loads the VTU file(s) from disk.
      2. Triangulates each VTU's polydata.
      3. Converts each triangulated polydata to a Plotly mesh3d object.
      4. Creates a Plotly 3D figure with all the meshes.
    """
    # List the VTU file paths you want to load. Adjust these paths as necessary.
    vtu_files = [
        "../HTML_3D/mesh_fuse_vertical.vtu",
        "../HTML_3D/mesh_fuse_horizontal.vtu",
        "../HTML_3D/mesh_wing.vtu",
        "../HTML_3D/mesh_vstab.vtu",
        "../HTML_3D/mesh_hstab.vtu",
        # You can add more file paths if needed.
    ]

    # Define colors to differentiate each model.
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]

    meshes = []
    for i, file_path in enumerate(vtu_files):
        try:
            polydata = load_vtu_file(file_path)
            # Triangulate the polydata.
            tri_polydata = triangulate_polydata(polydata)
            mesh = polydata_to_plotly_mesh(tri_polydata, color=colors[i % len(colors)])
            meshes.append(mesh)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    fig = go.Figure(data=meshes)

    # Update layout for better visualization and equal aspect ratio.
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"  # Ensures uniform scaling on all axes.
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig
