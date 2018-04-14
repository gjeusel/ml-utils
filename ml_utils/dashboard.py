import sys
import logging

from flask import Flask
import dash
from dash.dependencies import Input, Output, Event, State
import dash_core_components as dcc
import dash_html_components as html

from .plot import corrmatrix_chart, boxplot_chart, missingvalues_chart

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_main_div():
    logger.debug('Loading main div ...')
    return html.Div([
        html.Div([
            dcc.Tabs(
                tabs=[
                    {'label': 'Correlation', 'value': 'correlation'},
                    {'label': 'MissingValues', 'value': 'missingvalues'},
                    {'label': 'BoxPlot', 'value': 'boxplot'},
                ],
                value='correlation',
                id='plot-tab',
            ), ],
        ),

        dcc.Graph(id='graph'),

    ], style={
        'fontFamily': 'Sans-Serif',
        'margin-left': 'auto',
        'margin-right': 'auto',
    })


def register_dashboard(df, server):
    app = dash.Dash(name='statistical_analysis', server=server,
                    url_base_pathname='/',
                    request_pathname_prefix='/statistical_analysis')

    app.config['suppress_callback_exceptions'] = True

    # Loading screen CSS (grey when loading figures)
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

    # Beautiful button
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    app.server.logger.addHandler(handler)
    app.server.logger.setLevel(logging.DEBUG)

    app_logger = app.server.logger

    app.layout = get_main_div()


    def get_figure(figtype):
        if figtype == 'correlation':
            return corrmatrix_chart(df, offline=False)
        elif figtype == 'missingvalues':
            return missingvalues_chart(df, offline=False)
        elif figtype == 'boxplot':
            return boxplot_chart(df, offline=False)
        else:
            pass

    app.callback(Output('graph', 'figure'),
                 [Input('plot-tab', 'value')])(get_figure)

    return app
