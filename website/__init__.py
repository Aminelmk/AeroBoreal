## inisialisation de lapplication FLask
from flask import Flask
import plotly.graph_objs as go
import plotly.io as pio


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'PI4'

    from .views import main
    app.register_blueprint(main)

    return app