import vtk
import numpy as np

def extract_points_from_stl(stl_filename):
    """Extracts 3D points from an STL file using VTK"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_filename)
    reader.Update()

    polydata = reader.GetOutput()
    points_vtk = polydata.GetPoints()
    
    if points_vtk is None:
        raise ValueError("No points found in STL file.")

    num_points = points_vtk.GetNumberOfPoints()
    points_array = np.array([points_vtk.GetPoint(i) for i in range(num_points)])
    
    return points_array

# Example usage
stl_file = r"c:\Users\nadir\Desktop\C++\HTML_3D-NACA_2V\pages\Dash_3D\LH WING-AS BUILT MODEL SCALE.stl"

points = extract_points_from_stl(stl_file)

# Save points
np.savetxt(r"C:\Users\nadir\Desktop\C++\HTML_3D-NACA_2V\pages\Dash_3D\wing_points.txt", points, delimiter=",")
print(f"Extracted {len(points)} points and saved to 'wing_points.txt'.")
