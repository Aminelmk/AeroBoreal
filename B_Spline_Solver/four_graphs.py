import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import BSpline_solver

def read_bspline_curve(file_path):  
    x, y = [], []
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

def display_bspline_subplots(initial_points_file, knots_list):
    x_init, y_init = read_bspline_curve(initial_points_file)
    if x_init is None or y_init is None:
        return

    # Create 2x2 subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"B-Spline (knot={k})" for k in knots_list],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    for i, knot in enumerate(knots_list):
        row = i // 2 + 1
        col = i % 2 + 1

        BSpline_solver.run_bspline_solver(initial_points_file, knot)
        x_b, y_b = read_bspline_curve("BSpline_curve.txt")
        if x_b is None or y_b is None:
            print(f"[Erreur] Lecture échouée pour knot={knot}")
            continue

        # Add initial points
        fig.add_trace(go.Scatter(
            x=x_init,
            y=y_init,
            mode='markers',
            name='Initial Points',
            marker=dict(color='red', size=6),
            showlegend=(i == 0)
        ), row=row, col=col)

        # Add B-Spline curve
        fig.add_trace(go.Scatter(
            x=x_b,
            y=y_b,
            mode='lines+markers',
            name=f'B-Spline (knot={knot})',
            line=dict(width=2),
            marker=dict(size=4),
            showlegend=(i == 0)
        ), row=row, col=col)

    fig.update_layout(
        title="Airfoil B-Spline ",
        height=900,
        width=1000,
        template='plotly_white'
    )

    fig.show()

# --- Usage ---
input_file = "crm_test.dat"
n_knots = [15, 20, 25, 30]
display_bspline_subplots(input_file, n_knots)
