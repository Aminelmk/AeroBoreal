import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pages

# Initialize Dash App with Pages Support
app = dash.Dash(
    __name__, 
    use_pages=True, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    suppress_callback_exceptions=True  
)
app.config.suppress_callback_exceptions = True
app.enable_dev_tools(dev_tools_props_check=False)


server = app.server

# Navigation Bar
# navbar = dbc.NavbarSimple(
#     children=[
#         dbc.NavItem(dcc.Link("Home", href="/", className="nav-link")),
#         dbc.NavItem(dcc.Link("Mesh", href="/page-mesh2d", className="nav-link")),
#         dbc.NavItem(dcc.Link("Page 1", href="/page-1", className="nav-link")),
#         dbc.NavItem(dcc.Link("Page 2", href="/page-2", className="nav-link")),
#         dbc.NavItem(dcc.Link("Page 3", href="/page-3", className="nav-link")),
#     ],
#     brand="NACA Airfoil Dashboard Test",
#     color="blue",
#     dark=True,
#     className="ml-auto",
# )

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Mesh 2D", href="/page-mesh2d"),
                dbc.DropdownMenuItem("Euler 2D", href="/page-euler2d"),
                dbc.DropdownMenuItem("Results 2D", href="/page-euler2d-results"),
            ],
            nav=True,
            in_navbar=True,
            label="2D",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Mesh 3D", href="/page-mesh3d"),
                dbc.DropdownMenuItem("NL-VLM-Structure 3D", href="/pages-vlmstructure3D"),
                dbc.DropdownMenuItem("Results 3D", href="/pages-pressionVLM"),
            ],
            nav=True,
            in_navbar=True,
            label="3D",
        ),
    ],
    brand="AeroBoréal",
    brand_href="#",
    brand_style={"color": "black"},
    color="primary",
    dark=False,
    className="compact-navbar",
    # style={
    #     "padding": "0.0rem 0rem",
    #     "minHeight": "10px",
    #     "fontSize": "0.9rem"
    # },
)

# App Layout (Navigation + Page Container)
app.layout = html.Div([
    navbar,
    dash.page_container  # This will display the selected page
])

if __name__ == "__main__":
    app.run(debug=False)
