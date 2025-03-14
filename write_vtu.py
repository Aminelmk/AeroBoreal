import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np

def save_vtu_mesh(x, y, z, file_name):
    """
    Saves structured mesh to a VTU file for visualization in ParaView.
    
    Parameters:
        x, y, z (numpy arrays): Mesh coordinates.
        file_name (str): Name of the output .vtu file.
    """
    points = vtk.vtkPoints()
    num_points = x.size
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            points.InsertNextPoint(x[i, j], y[i, j], z[i, j])
    
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    
    # Create cells (assume quadrilateral elements)
    num_cells = (x.shape[0] - 1) * (x.shape[1] - 1)
    cells = vtk.vtkCellArray()
    cell_types = vtk.vtkUnsignedCharArray()
    cell_types.SetNumberOfComponents(1)
    cell_types.SetNumberOfValues(num_cells)
    
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i * x.shape[1] + j)
            quad.GetPointIds().SetId(1, (i+1) * x.shape[1] + j)
            quad.GetPointIds().SetId(2, (i+1) * x.shape[1] + (j+1))
            quad.GetPointIds().SetId(3, i * x.shape[1] + (j+1))
            cells.InsertNextCell(quad)
            cell_types.SetValue(i * (x.shape[1] - 1) + j, vtk.VTK_QUAD)
    
    grid.SetCells(cell_types, cells)
    
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(grid)
    writer.Write()
    
    print(f"Mesh saved as {file_name}")

def write_vtu_with_data(file_name, x, y, z, pressure):
    """
    Writes a VTU file with structured mesh and pressure data.
    
    Parameters:
        file_name (str): Name of the output file.
        x, y, z (numpy arrays): Mesh coordinates.
        pressure (numpy array): Pressure values at each cell.
    """
    num_points = x.size
    num_cells = (x.shape[0] - 1) * (x.shape[1] - 1)  

    # Create VTK points
    points = vtk.vtkPoints()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            points.InsertNextPoint(x[i, j], y[i, j], z[i, j])

    # Create unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    # Define quadrilateral cells
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i * x.shape[1] + j)
            quad.GetPointIds().SetId(1, (i + 1) * x.shape[1] + j)
            quad.GetPointIds().SetId(2, (i + 1) * x.shape[1] + (j + 1))
            quad.GetPointIds().SetId(3, i * x.shape[1] + (j + 1))
            grid.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

    # Add pressure data
    pressure_array = vtk.vtkFloatArray()
    pressure_array.SetName("Pressure")
    for p in pressure:
        pressure_array.InsertNextValue(p)

    grid.GetCellData().AddArray(pressure_array)

    # Write the VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputDataObject(grid)
    writer.Write()
