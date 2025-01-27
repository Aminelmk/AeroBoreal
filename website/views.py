#logique routes et vues
from flask import Blueprint, render_template, request, redirect, url_for
import plotly.graph_objs as go
import plotly.io as pio
import sys
sys.path.append('/HTML/cpp_code/process_data.so')
#import process_data 


main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        
        value1 = request.form.get('value1')
        value2 = request.form.get('value2')
        value3 = request.form.get('value3')

        #result = process_data.process_data(value1, value2, value3)

        
        return redirect(url_for('main.graph', value1=value1, value2=value2, value3=value3))
    
    return render_template('form.html')
    
@main.route('/graph')

def graph():

    value1 = float(request.args.get('value1'))
    value2 = float(request.args.get('value2'))
    value3 = float(request.args.get('value3'))
    #result = float(request.args.get('result'))

    data = [go.Scatter3d(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9], mode='markers')]
    layout = go.Layout(title=f'Graphique 3D avec des valeurs: {value1}, {value2}, {value3}')
    fig = go.Figure(data=data, layout=layout)

    graph_html = pio.to_html(fig, full_html=False)
    return render_template('graph.html', graph_html=graph_html)
    