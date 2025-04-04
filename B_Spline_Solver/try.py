import numpy as np
import plotly.graph_objects as go
import BSpline_solver
'''PLOTLY
def read_bspline_curve(file_path): 
    """Reads points from a text file containing x y coordinates"""
    x = []
    y = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x.append(float(parts[0]))

                    y.append(float(parts[1]))
        return np.array(x), np.array(y)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None

def display_bspline_result(initial_points_file, result_file="BSpline_curve.txt"):
    # Read both initial points and generated curve
    x_init, y_init = read_bspline_curve(initial_points_file)
    x_b, y_b = read_bspline_curve(result_file)
    
    if x_init is None or y_init is None or x_b is None or y_b is None:
        return
    
    # Create visualization with both datasets
    fig = go.Figure()
    
    # Add initial points (red markers)
    fig.add_trace(go.Scatter(
        x=x_init, 
        y=y_init, 
        mode='markers',
        name='Initial Points',
        marker=dict(color='red', size=6)
    ))
    
    # Add B-spline curve (blue line)
    fig.add_trace(go.Scatter(
        x=x_b, 
        y=y_b, 
        mode='lines+markers',
        name='BSpline Curve',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
      

    
    fig.update_layout(
        title="Airfoil B-Spline approximation",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        showlegend=True,
        width=1000,
        height=800,
        template='plotly_white',
        xaxis_range=[-0.1, 1.2],
        yaxis_range=[-0.5, 0.5]
    )
    
    fig.show()

# Usage example
output_file = "BSpline_curve.txt"  
input_file = "crm_test.dat"
n_knots = [15 ,20 ,25 ,30]
for knot in n_knots:

  if knot in [15, 20, 25, 30]:
    output_file = BSpline_solver.run_bspline_solver(input_file, knot)
    display_bspline_result(input_file, "BSpline_curve.txt")
  else:
    print("Erreur : Le nombre de knots doit être 15, 20 ou 25.")
    '''
    
import numpy as np
import matplotlib.pyplot as plt
import BSpline_solver

def read_bspline_curve(file_path): 
    x = []
    y = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
        return np.array(x), np.array(y)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None

def display_bspline_result(initial_points_file, result_file="BSpline_curve.txt"):
    x_init, y_init = read_bspline_curve(initial_points_file)
    x_b, y_b = read_bspline_curve(result_file)

    if x_init is None or x_b is None:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x_b, y_b, label="B-Spline Curve", color="blue", linewidth=2)
    plt.scatter(x_init, y_init, label="Initial Points", color="red", s=30)
    plt.title("Airfoil B-Spline Approximation")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.5, 0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Usage example ---
output_file = "BSpline_curve.txt"  
input_file = "crm_test.dat"
n_knots = [15 ,20 ,25 ,30]
for knot in n_knots:

  if knot in [15, 20, 25, 30]:
    output_file = BSpline_solver.run_bspline_solver(input_file, knot)
    display_bspline_result(input_file, "BSpline_curve.txt")
  else:
    print("Erreur : Le nombre de knots doit être 15, 20 ou 25.")
