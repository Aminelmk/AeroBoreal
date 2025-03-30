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

    x = [0, 1, 2, 1, 0, 1, 1]
    y = [0, 0, 0, 1, 2, 1, 1]
    z = [0, 1, 1, 1, 1, 0, 2]
    fuselage_x = [0, 0, 0]  
    fuselage_y = [0, 0, 0]
    fuselage_z = [0, 3, 6]  

    ailes_x = [-3, 3, 0, -3, 3]  
    ailes_y = [0, 0, 0, 0, 0]
    ailes_z = [3, 3, 3, 3, 3]

    queue_x = [0, 0, -1, 1]  
    queue_y = [0, 0, 0, 0]
    queue_z = [6, 7, 6.5, 6.5]

    data = [
            go.Scatter3d(x=fuselage_x, y=fuselage_y, z=fuselage_z, mode='lines+markers', line=dict(color='red', width=5), marker=dict(size=5, color='red'), name="Fuselage"),
            go.Scatter3d(x=ailes_x, y=ailes_y, z=ailes_z, mode='lines+markers', line=dict(color='blue', width=5), marker=dict(size=5, color='blue'), name="Ailes"),
            go.Scatter3d(x=queue_x, y=queue_y, z=queue_z, mode='lines+markers', line=dict(color='green', width=5), marker=dict(size=5, color='green'), name="Queue")    
            ]
    layout = go.Layout(
        title=f'Graphique 3D avec des valeurs: {value1}, {value2}, {value3}',
                       width = 800,
                       height = 600,
                       scene=dict(
            xaxis=dict(showgrid=False),  
            yaxis=dict(showgrid=False),  
            zaxis=dict(showgrid=False),  
            aspectmode="cube"  
        ),
        showlegend=False  
                       )
    fig = go.Figure(data=data, layout=layout)

    graph_html = pio.to_html(fig, full_html=False)
    return render_template('graph.html', graph_html=graph_html)
    