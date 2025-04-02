import euler_solver
import numpy as np
from read_PLOT3D import *
from post_process import *
from FVM import cell
import os
import csv
import shutil
import re 

# Configuration
base_input = "input.txt"
results_dir = "simulation_results"
mach_values = np.arange(0.2, 1.1, 0.1) 
alpha_values = np.arange(-5, 12, .5)     

def update_input_file(mach, alpha, input_path):
    """Update input file with new parameters using regex for robustness"""
    with open(base_input, 'r') as f:
        content = f.read()   
    
    # Use regex to handle different whitespace formats
    content = re.sub(r'Mach\s*=\s*[\d\.]+', f'Mach = {mach:.2f}', content)
    
    # Foxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxcxxxxxxxxxxxxxxxxxxxxxxxxcxcsxrmat alpha as integer when whole number, else 1 decimal
    alpha_str = f"{alpha:.2f}" 
    content = re.sub(r'alpha\s*=\s*[\d\.]+', f'alpha = {alpha_str}', content)
    
    with open(input_path, 'w') as f:
        f.write(content)
def extract_string_from_input(input_path, key):
    with open(input_path, 'r') as f:
        for line in f:
            if line.startswith(key):
                return line.split('=')[1].strip()
    raise ValueError(f"{key} not found in {input_path}")

def extract_value_from_input(input_path, key):
    with open(input_path, 'r') as f:
        for line in f:
            if line.startswith(key):
                return float(line.split('=')[1].strip())
    raise ValueError(f"{key} not found in {input_path}")


def run_simulation(mach, alpha, results_dir):
    """Run simulation with enhanced parameter handling"""
    # Format alpha to 1 decimal place for filename
    temp_input = f"temp_input_{mach:.2f}_{alpha:.2f}.txt"
    update_input_file(mach, alpha, temp_input)
    
    # Run solver
    print(f"✅ Running simulation: Mach={mach:.2f}, Alpha={alpha:.1f}")
    euler_solver.solve(temp_input)

    # Save and rename solution file
    mach_str = f"{mach:.2f}".replace('.', '')
    alpha_str = f"{alpha:.1f}".replace('.', '')
    output_filename = f"test_M{mach_str}_alpha{alpha_str}.q"
    output_path = os.path.join(results_dir, output_filename)
    
    if os.path.exists("test.q"):
        shutil.copy("test.q", output_path)
        print(f"📄 Saved solution to {output_path}")
    else:
        print("⚠️ Warning: test.q not found")
        return {'CL': np.nan, 'CD': np.nan, 'CM': np.nan}

    # Extract parameters from TEMPORARY input file
    try:
        T_inf = extract_value_from_input(temp_input, "T_inf")
        p_inf = extract_value_from_input(temp_input, "p_inf")
        mesh_file = extract_string_from_input(temp_input, "mesh_file")  # Fixed key name
    except ValueError as e:
        print(f"⚠️ Parameter error: {str(e)}")
        return {'CL': np.nan, 'CD': np.nan, 'CM': np.nan}

    # Read data using FULL PATH
    try:
        x, y = read_PLOT3D_mesh(mesh_file)
        ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(output_path)  # Use full path
        
        cp_airfoil, C_L, C_D, C_M = compute_coeff(
            x, y, q_vertex, mach, alpha, T_inf, p_inf
        )
    except Exception as e:
        print(f"⚠️ Processing error: {str(e)}")
        return {'CL': np.nan, 'CD': np.nan, 'CM': np.nan}
    
    os.remove(temp_input)
    return {
        'CL': float(C_L),
        'CD': float(C_D),
        'CM': float(C_M)
    }
    
    #os.remove(temp_input)
    return results

# Create results directory
os.makedirs(results_dir, exist_ok=True)

# Main simulation loop
for mach in mach_values:
    mach_str = f"{mach:.2f}".replace('.', '_')
    csv_filename = os.path.join(results_dir, f"mach_{mach_str}.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for alpha in alpha_values:
         results = run_simulation(mach, alpha,results_dir)
         writer.writerow([
            alpha,
            results['CL'],
            results['CD'],
            results['CM']
        ])
            
    print(f" Saved results for Mach {mach} to {csv_filename}")

