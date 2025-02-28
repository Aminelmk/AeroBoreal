import vtk
import dash_vtk
import numpy as np

def read_wing_points(filename):
    """Read wing points from a file and return them as a flat list."""
    points = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and trailing commas
            line = line.strip().rstrip(',')
            if not line:
                continue  # Skip empty lines
            # Split the line into components by commas
            parts = line.split(',')
            # Filter out any empty strings
            parts = [p for p in parts if p]
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                points.extend([x, y, z])
            else:
                raise ValueError(f"Invalid line in wing points file: {line}")
    return points

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

def create_airplane(
    fuselage_radius=1.0,
    fuselage_length=10.0,
    tail_size=1.0,
    wing_points=None,
    wing_points_file=None,
    wing_connectivity="strips"
):
    # Fuselage
    fuselage = vtk.vtkCylinderSource()
    fuselage.SetRadius(fuselage_radius)
    fuselage.SetHeight(fuselage_length)
    fuselage.SetResolution(50)
    fuselage.SetCenter(0.0, 0.0, 0.0)
    
    # Rotate the fuselage to align with the x-axis
    fuselage_transform = vtk.vtkTransform()
    fuselage_transform.RotateZ(90)
    fuselage_transform_filter = vtk.vtkTransformPolyDataFilter()
    fuselage_transform_filter.SetInputConnection(fuselage.GetOutputPort())
    fuselage_transform_filter.SetTransform(fuselage_transform)
    fuselage_transform_filter.Update()
    
    # Custom Wing
    if wing_points_file is not None:
        wing_points = read_wing_points(wing_points_file)
        wing_polydata = create_custom_wing_from_data(wing_points, wing_connectivity)
        
        # Transform the wing to the correct position
        wing_transform = vtk.vtkTransform()
        wing_transform.Translate(fuselage_length * 0.1, 0, 0)
        wing_transform_filter = vtk.vtkTransformPolyDataFilter()
        wing_transform_filter.SetInputData(wing_polydata)
        wing_transform_filter.SetTransform(wing_transform)
        wing_transform_filter.Update()
        wing_transform.RotateX(0)  # Rotate around X-axis if needed
        wing_transform.RotateY(90)  # Rotate around Y-axis to flip direction
        wing_transform.RotateZ(0)  # Adjust dihedral angle as needed
        
        # Mirror the wing to create the opposite wing
        mirror_transform = vtk.vtkTransform()
        mirror_transform.Scale(1, -1, 1)  # Mirror over the Y-axis
        mirror_transform_filter = vtk.vtkTransformPolyDataFilter()
        mirror_transform_filter.SetInputConnection(wing_transform_filter.GetOutputPort())
        mirror_transform_filter.SetTransform(mirror_transform)
        mirror_transform_filter.Update()
        wing_transform.RotateX(0)  # Rotate around X-axis if needed
        wing_transform.RotateY(90)  # Rotate around Y-axis to flip direction
        wing_transform.RotateZ(0)  # Adjust dihedral angle as needed
    else:
        raise ValueError("Wing points file is required.")
    
    # # Tail Wing (Horizontal Stabilizer)
    # tail_wing = vtk.vtkCubeSource()
    # tail_wing.SetXLength(0.7 * tail_size)
    # tail_wing.SetYLength(0.1 * tail_size)
    # tail_wing.SetZLength(3.0 * tail_size)
    # tail_wing.SetCenter(-fuselage_length * 0.4, fuselage_radius * 1.7, 0)
    
    # # Tail Fin (Vertical Stabilizer)
    # tail_fin = vtk.vtkCubeSource()
    # tail_fin.SetXLength(0.7 * tail_size)
    # tail_fin.SetYLength(1.5 * tail_size)
    # tail_fin.SetZLength(0.1 * tail_size)
    # tail_fin.SetCenter(-fuselage_length * 0.4, fuselage_radius * 1.0, 0.0)
    
    # Combine all parts
    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputConnection(fuselage_transform_filter.GetOutputPort())  # Fuselage
    append_filter.AddInputConnection(wing_transform_filter.GetOutputPort())  # Left Wing
    append_filter.AddInputConnection(mirror_transform_filter.GetOutputPort())  # Right Wing
    # append_filter.AddInputConnection(tail_wing.GetOutputPort())
    # append_filter.AddInputConnection(tail_fin.GetOutputPort())
    append_filter.Update()
    
    # Clean the combined data
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(append_filter.GetOutputPort())
    cleaner.Update()
    
    return cleaner.GetOutput()
    pass
