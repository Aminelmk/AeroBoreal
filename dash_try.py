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
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/home", active="exact")),
        dbc.NavItem(dbc.NavLink("Page test", href="/page-0", active="exact")),
        dbc.NavItem(dbc.NavLink("Page 1", href="/page-1", active="exact")),
        # dbc.NavItem(dbc.NavLink("Page 2", href="/page-2", active="exact")),
        dbc.NavItem(dbc.NavLink("Page 3", href="/page-3", active="exact")),
    ],
    brand="NACA Airfoil Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
)

# The layout includes a Location component (for URL tracking), your navbar, and the container where pages render.
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    dash.page_container
])

if __name__ == "__main__":
    app.run_server(debug=True)
