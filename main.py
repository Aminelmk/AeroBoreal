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
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Mesh 2D", href="/page-mesh2d"),
                dbc.DropdownMenuItem("Mesh 3D", href="/page-mesh3d"),
                dbc.DropdownMenuItem("Euler 2D", href="/page-euler2d"),
                dbc.DropdownMenuItem("VLM-Structure 3D", href="/pages-vlmstructure3D"),
                dbc.DropdownMenuItem("VLM-Pression", href="/pages-pressionVLM"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Logiciel Aéro-élastique",
    brand_href="#",
    color="primary",
    dark=False,
)

# App Layout (Navigation + Page Container)
app.layout = html.Div([
    navbar,
    dash.page_container  # This will display the selected page
])

if __name__ == "__main__":
    app.run(debug=True)
