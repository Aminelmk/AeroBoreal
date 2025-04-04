import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import csv
import os

dash.register_page(__name__, path="/")

CSV_FILE = "user_inputs.csv"

# 🔹 Layout for the Home Page
layout = html.Div([

html.Div([
        html.H1("Welcome to the Aerospace Engineering Simulation Software", 
                style={
                    "textAlign": "center", 
                    "fontSize": "48px", 
                    "fontWeight": "bold", 
                    "color": "black",  
                    "padding": "20px"
                }),
        html.P(
            "This software suite is designed for subsonic and transonic flow simulations, incorporating structural deformations. "
            "It includes the essential steps for aero-structural analysis: CAD geometric processing, parameterization, mesh generation, "
            "solution of linear equations, and post-processing of results. The system is designed for ease of use, with both graphical and non-graphical interfaces.",
            style={"textAlign": "center", "fontSize": "18px", "color": "#7f8c8d", "maxWidth": "800px", "margin": "0 auto"}
        ),
    ], style={"paddingTop": "40px", "paddingBottom": "40px"}),


    html.Div([
        html.H3("Explore the Software Features", 
                style={"textAlign": "center", "fontSize": "30px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "30px"}),

  
        dbc.Row([
            dbc.Col([
                html.H4("Mesh 2D", style={"fontSize": "24px", "fontWeight": "bold", "color": "#2980b9"}),
                html.P(
                    "Create and visualize 2D meshes for aerodynamic simulations. Define geometry and grid resolution to generate meshes used for fluid dynamics analysis.",
                    style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}
                ),
            dbc.Row([
                dbc.Col(
                    dbc.Button("Mesh 2D", color="primary", href="/page-mesh2d", size="lg", className="w-100",
                            style={"marginBottom": "20px", "fontSize": "20px", "padding": "20px"}),
                    width=6
                )
            ], justify="center")

            ], width=6),
            
            
            dbc.Col([
                html.H4("Mesh 3D", style={"fontSize": "24px", "fontWeight": "bold", "color": "#2980b9"}),
                html.P(
                    "Generate and visualize 3D meshes for structural and fluid-structure simulations. Supports complex geometries, enhancing the accuracy of your models.",
                    style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}
                ),

            dbc.Row([
                dbc.Col(
                    dbc.Button("Mesh 3D", color="primary", href="/page-mesh3d", size="lg", className="w-100", style={"marginBottom": "20px", "fontSize": "20px", "padding": "20px"}),
                    width=6
                )
            ], justify="center"),
            ], width=6),

            
        ], style={"marginBottom": "30px"}),


        dbc.Row([
            dbc.Col([
                html.H4("Euler 2D", style={"fontSize": "24px", "fontWeight": "bold", "color": "#2980b9"}),
                html.P(
                    "Perform aerodynamic simulations for 2D airfoils using the Euler solver. Set Mach number, angle of attack, and other parameters to analyze airfoil performance.",
                    style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}
                ),

                dbc.Row([
                dbc.Col(
                    dbc.Button("Euler 2D", color="primary", href="/page-euler2d", size="lg", className="w-100", style={"marginBottom": "20px", "fontSize": "20px", "padding": "20px"}),
                    width=6
                            )
                        ], justify="center"),
                        ], width=6),

            dbc.Col([
                html.H4("VLM-Structure 3D", style={"fontSize": "24px", "fontWeight": "bold", "color": "#2980b9"}),
                html.P(
                    "Conduct aero-structural simulations for complex 3D configurations like wings and aircraft. This page integrates both fluid and structural models for comprehensive analysis.",
                    style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}
                ),

                dbc.Row([
                dbc.Col(
                    dbc.Button("VLM-Structure 3D", color="primary", href="/pages-vlmstructure3D", size="lg", className="w-100", style={"marginBottom": "20px", "fontSize": "20px", "padding": "20px"}),
                    width=6
                            )
                        ], justify="center"),
                        ], width=6),

        ], style={"marginBottom": "30px"}),

    html.Br(),
html.Div([
    html.H3("Development Contributors", 
            style={"textAlign": "center", "fontSize": "30px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "30px"}),

    dbc.Row([
        dbc.Col([
            html.P("Youcef Benouadah", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Nadir Bettahar", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Oihan Cordelier", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Cléo Delêtre", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Vincent Dubé", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Ouèpiya Chris David Fogue", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Cédric Gagné", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
        ], width=3, style={"textAlign": "center"}),  # Centrer le texte dans cette colonne

        dbc.Col([
            html.P("Rayan Hamza", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Shajeevan Kanapathippillai", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Amine Lamkinsi", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Marina Latendresse", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Hieu Nhan Tran", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Jacob Tremblay", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
            html.P("Brandon Velasquez", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
        ], width=3, style={"textAlign": "center"}),  # Centrer le texte dans cette colonne
    ], justify="center", style={"marginBottom": "30px", "textAlign": "center"}),  # Centrer les éléments dans le Row
], style={"padding": "20px", "textAlign": "center"}),

html.Div([
    html.H3("Development Supervisors", 
            style={"textAlign": "center", "fontSize": "30px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "30px"}),

    dbc.Row([
        dbc.Col([
            html.P("Éric Laurendeau, PhD", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
        ], width=3, style={"textAlign": "center"}),  # Centrer le texte dans cette colonne

        dbc.Col([
            html.P("Matthieu Parenteau, PhD", style={"fontSize": "18px", "color": "#7f8c8d", "marginBottom": "20px"}),
        ], width=3, style={"textAlign": "center"}),  # Centrer le texte dans cette colonne
    ], justify="center", style={"marginBottom": "30px", "textAlign": "center"}),  # Centrer les éléments dans le Row
], style={"padding": "20px", "textAlign": "center"})



    ], style={"padding": "20px"})
])

# # 🔹 Callback to Update Graphs Dynamically
# @dash.callback(
#     [Output("mesh-plot", "figure"),
#         Output("history-list", "children")],
#     [Input("nc_slider", "value"),
#         Input("save_euler", "n_clicks"),
#         Input("save_vlm", "n_clicks")],
#     [
#         State("num_threads", "value"),
#         State("mesh_file", "value"),
#         State("Mach", "value"),
#         State("alpha", "value"),
#         State("p_inf", "value"),
#         State("T_inf", "value"),
#         State("it_max", "value"),
#         State("output_file", "value"),
#
#         State("multigrid", "value"),
#         State("CFL_number", "value"),
#         State("residual_smoothing", "value"),
#         State("k2", "value"),
#         State("k4", "value"),
#         State("checkpoint_file", "value"),
#
#         State("nx", "value"),
#         State("ny", "value"),
#         State("AR", "value"),
#         State("alpha_input", "value"),
#         State("vlm_p_inf", "value"),
#         State("vlm_T_inf", "value"),
#         State("vlm_Mach", "value"),
#     ],
#     prevent_initial_call=True
# )
# def update_visualizations_and_save(nc,
#                                    save_euler_clicks, save_vlm_clicks,
#                                    num_threads, mesh_file, Mach, alpha, p_inf, T_inf, it_max, output_file
#                                    ,multigrid, CFL_number, residual_smoothing, k2, k4, checkpoint_file, nx, ny, AR, alpha_input, vlm_p_inf, vlm_T_inf, vlm_Mach):
#     """Update airfoil, mesh plots, and save inputs dynamically"""
#
#     ctx = dash.callback_context
#     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
#
#     # Update airfoil and mesh
#     nc = int(nc)
#
#     # cm = ConformalMapping(airfoil, nc)
#     # cm.generate_mesh_nodes()
#     # mesh_fig = mesh_to_plotly(cm.all_nodes)
#     #
#     # save_to_csv([camber, camber_pos, thickness, nc])
#     history_entries = load_csv_entries()
#
#     # Handle input saving
#     if triggered_id == "save_euler":
#         euler_params = {
#             "num_threads": num_threads, "mesh_file": mesh_file, "Mach": Mach,
#             "multigrid": multigrid, "CFL_number": CFL_number, "residual_smoothing": residual_smoothing,
#             "k2": k2, "k4": k4, "checkpoint_file": checkpoint_file,
#             "alpha": alpha, "p_inf": p_inf, "T_inf": T_inf, "it_max": it_max, "output_file": output_file
#         }
#         save_to_file("euler_input.txt", euler_params)
#         history_entries += [f"{k} = {v}" for k, v in euler_params.items()]
#
#     elif triggered_id == "save_vlm":
#         vlm_params = {
#             "nx": nx, "ny": ny, "AR": AR, "alpha": alpha_input,
#             "p_inf": vlm_p_inf, "T_inf": vlm_T_inf, "Mach": vlm_Mach
#         }
#         save_to_file("vlm_input.txt", vlm_params)
#         history_entries += [f"{k} = {v}" for k, v in vlm_params.items()]
#
#     return [html.Li(entry) for entry in history_entries]
#
#
# # 🔹 Functions to Manage CSV History
# def save_to_csv(values):
#     file_exists = os.path.isfile(CSV_FILE)
#     with open(CSV_FILE, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(["Camber", "Pos Camber", "Thickness", "NC"])
#         writer.writerow(values)
#
# def load_csv_entries():
#     if not os.path.isfile(CSV_FILE):
#         return []
#     with open(CSV_FILE, mode='r') as file:
#         reader = csv.reader(file)
#         next(reader, None)  # Skip the header
#         return [", ".join(row) for row in reader]
#
#
# # 🔹 Function to Convert Mesh Data to Plotly
# def mesh_to_plotly(all_nodes):
#     nodes = np.array(all_nodes)
#     n_i, n_j, _ = nodes.shape
#     mesh_fig = go.Figure()
#
#     # Draw horizontal lines
#     for i in range(n_i):
#         x_line, y_line = nodes[i, :, 0], nodes[i, :, 1]
#         mesh_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="blue")))
#
#     # Draw vertical lines
#     for j in range(n_j):
#         x_line, y_line = nodes[:, j, 0], nodes[:, j, 1]
#         mesh_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="blue")))
#
#     # Add mesh nodes as red dots
#     mesh_fig.add_trace(go.Scatter(x=nodes[:,:,0].flatten(), y=nodes[:,:,1].flatten(), mode="markers", marker=dict(size=4, color="red")))
#
#     mesh_fig.update_layout(
#         title="Conformal Mesh",
#         xaxis_title="X", yaxis_title="Y",
#         showlegend=False, width=600, height=600
#     )
#     return mesh_fig
#
#
# # Obtenir le répertoire du script Dash (là où se trouve le fichier Python)
# SAVE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
#
# # Ensure the directory exists
# os.makedirs(SAVE_DIRECTORY, exist_ok=True)
#
# def save_to_file(filename, data_dict):
#     """Save parameters to a specified folder."""
#     file_path = os.path.join(SAVE_DIRECTORY, filename)
#
#     with open(file_path, "w") as file:
#         for key, value in data_dict.items():
#             file.write(f"{key} = {value}\n")
#
#     print(f"File saved at: {os.path.abspath(file_path)}")  # Debugging info
#
#
# def save_inputs(save_euler_clicks, save_vlm_clicks,
#                 num_threads, mesh_file, Mach, alpha, p_inf, T_inf, it_max, output_file,
#                 nx, ny, AR, alpha_input, vlm_p_inf, vlm_T_inf, vlm_Mach):
#     """Save inputs to separate files depending on which button was clicked."""
#
#     ctx = dash.callback_context  # Get the button that triggered the callback
#     if not ctx.triggered:
#         return dash.no_update  # If no button was clicked, do nothing
#
#     button_id = ctx.triggered[0]["prop_id"].split(".")[0]  # Identify the button
#
#     if button_id == "save_euler":
#         euler_params = {
#             "num_threads": num_threads,
#             "mesh_file": mesh_file,
#             "Mach": Mach,
#             "alpha": alpha,
#             "p_inf": p_inf,
#             "T_inf": T_inf,
#             "it_max": it_max,
#             "output_file": output_file,
#             "multigrid": multigrip,
#             "CFL_number": CFL_number,
#             "residual_smoothing": residual_smoothing,
#             "k2": k2,
#             "k4": k4,
#             "checkpoint_file": checkpoint_file,
#         }
#         save_to_file("euler_input.txt", euler_params)
#         return [html.Li(f"{k} = {v}") for k, v in euler_params.items()]
#
#     elif button_id == "save_vlm":
#         vlm_params = {
#             "nx": nx,
#             "ny": ny,
#             "AR": AR,
#             "alpha": alpha_input,
#             "p_inf": vlm_p_inf,
#             "T_inf": vlm_T_inf,
#             "Mach": vlm_Mach
#         }
#         save_to_file("vlm_input.txt", vlm_params)
#         return [html.Li(f"{k} = {v}") for k, v in vlm_params.items()]
#
#     return dash.no_update
