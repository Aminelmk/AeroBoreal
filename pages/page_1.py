import dash
from dash import html, dcc, Input, Output
import dash_vtk
import dash_bootstrap_components as dbc
import vtk
from vtk.util import numpy_support

# Register the page with the route "/page-0"
dash.register_page(__name__, path="/page-1")

def load_vtu_file(file_path):
    """
    Load a VTU file from disk using VTK and convert it to vtkPolyData.
    We use vtkGeometryFilter to transform the unstructured grid into polydata
    that dash_vtk can display.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()
    
    polydata = geometry_filter.GetOutput()
    return polydata

def vtk_polydata_to_dash(polydata):
    """
    Convert a vtkPolyData object into a dictionary for dash_vtk.PolyData.
    The conversion returns:
      - "points": a flat list of coordinates (x0, y0, z0, x1, y1, z1, ...).
      - "polys": a flat list where each polygon is stored as:
          [n, i1, i2, ..., in]
        with n = number of vertices for that polygon.
    """
    # Extract points
    vtk_points = polydata.GetPoints().GetData()
    pts = numpy_support.vtk_to_numpy(vtk_points)
    points = pts.flatten().tolist()

    # Extract polygons and pack in a flat array
    polys = polydata.GetPolys()
    polys.InitTraversal()
    poly_flat = []
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        n = idList.GetNumberOfIds()
        poly_flat.append(n)  # first element is the number of vertices
        for i in range(n):
            poly_flat.append(idList.GetId(i))
    return {"points": points, "polys": poly_flat}

# Define the layout: a header, a reload button, and a VTK View component.
layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H2("Interactive VTU Visualization using dash_vtk"),
                width={"size": 8, "offset": 2}),
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
            dash_vtk.View(id="vtk-view", children=[], style={"height": "600px"}),
            width=12
        )
    )
], fluid=True)

@dash.callback(
    Output("vtk-view", "children"),
    Input("reload-button", "n_clicks")
)
def update_vtk_view(n_clicks):
    """
    Callback that loads the VTU file(s), converts the polydata for dash_vtk,
    and creates interactive GeometryRepresentation components.
    
    The returned dash_vtk.View automatically supports mouse interactions
    such as rotation, panning, and zooming.
    """
    # List the VTU file paths you wish to load. Adjust as needed.
    vtu_files = [
        "/home/vincent/pi4/GUI/HTML_3D/mesh_fuse_vertical.vtu"
    ]
    
    representations = []
    colors = [
        [1, 0, 0],   # Red
        [0, 1, 0],   # Green
        [0, 0, 1],   # Blue
        [1, 1, 0]    # Yellow
    ]
    
    for i, file_path in enumerate(vtu_files):
        try:
            polydata = load_vtu_file(file_path)
            dash_data = vtk_polydata_to_dash(polydata)
            poly_component = dash_vtk.PolyData(**dash_data)
            
            rep = dash_vtk.GeometryRepresentation(
                children=[poly_component],
                property={"color": colors[i % len(colors)], "opacity": 1.0}
            )
            representations.append(rep)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    return representations
