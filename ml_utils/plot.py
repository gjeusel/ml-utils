import logging
from pathlib import Path
import copy

import plotly
import plotly.graph_objs as go

from sklearn import preprocessing

import numpy as np
import pandas as pd
import cufflinks as cf  # bind ploty to pandas dataframes in IPython notebook
import colorlover

from .analytics import detect_frequency
from .colors import cl


logger = logging.getLogger(__name__)
cf.set_config_file(offline=True, world_readable=True, theme='white')


def offline2path(fig, offline):
    if offline:
        tmp_dir = Path('/tmp')
        if tmp_dir.exists():
            fpath = tmp_dir / 'output.html'
        else:
            fpath = Path('output.html')

        plotly.offline.plot(fig, filename=fpath.as_posix())
    else:
        pass  # TODO

    return fig


def add_shape_now_vertical_line(layout, ymin, ymax, xaxis='x1', yaxis='y1'):
    now = pd.Timestamp.now('CET').replace(microsecond=0).tz_localize(None)
    layout['shapes'].append({'x0': now, 'y0': ymin, 'x1': now, 'y1': ymax,
                             'xref': xaxis, 'yref': yaxis,
                             'opacity': 0.3,
                             'line': {'dash': 'dot'},
                             'type': 'line',
                             })
    return layout


def add_shape_now_rectangle_shape(layout, dt_max, xaxis='x1', yaxis='paper',
                                  ymin=0, ymax=1):
    now = pd.Timestamp.now('CET').replace(microsecond=0).tz_localize(None)
    dt_max = pd.Timestamp(dt_max).tz_convert('CET').tz_localize(None)
    layout['shapes'].append({
        'type': 'rect',
        'xref': xaxis, 'yref': yaxis,
        'x0': now,
        'x1': dt_max,
        'y0': ymin, 'y1': ymax,
        'fillcolor': '#d3d3d3',
        'opacity': 0.2,
        'line': {'width': 0},
    })
    return layout


def csv2dict(serie):
    """Translate CSV columns into dict for plotly plot purpose."""
    dct = dict(serie)

    # Pop not needed keys
    for e in ['col', 'plottype']:
        if e in dct.keys():
            dct.pop(e)

    result = copy.deepcopy(dct)
    # Recreate dict from keys with '.' in the name:
    for key in dct.keys():
        lst_keys = key.split('.')
        tree_dict = dct[key]
        for k in reversed(lst_keys):
            tree_dict = {k: tree_dict}
        result.pop(key)
        result.update(tree_dict)
    return result


def read_df_references(csv_spec_path):
    df = pd.read_csv(csv_spec_path.absolute().as_posix())
    # treat void lines in csv that help the reading
    df = df.dropna(how='all', axis=0)
    df['marker.color'] = [cl.__dict__[color] for color in df['marker.color']]
    return df


def get_plotly_traces(df, csv_spec_path=None):
    """Return a list of traces with specification coming from a dictionnary.

    :param df: the dataframe to be plotted.
    :param plottype: type of plot.
    :param spec: dictionnary with specification for each column of df.
        if is None, autoconfigure.
    :param kwargs: arguments added to all traces.
    :return: a list of traces
    """

    if csv_spec_path:
        specdf = read_df_references(csv_spec_path)
    else:
        specdf = pd.DataFrame({'col': df.columns,
                               'plottype': 'Scatter',
                               'name': df.columns})

    assert not df.columns.duplicated().any()

    traces = []
    for col in specdf['col']:
        spec = specdf[specdf['col'] == col].iloc[0].dropna()

        if col in df.columns:
            serie = df[col]
            serie = serie[serie != 0]  # drop zeros
            x_values = serie.index.tolist()
            y_values = serie.tolist()

            kwargs = csv2dict(spec)

            # Adaptative bar width if datetime index and bar plot
            if pd.api.types.is_datetime64_any_dtype(serie.index) and \
                    (spec['plottype'] == 'Bar') and (len(serie.index) >= 2):
                freq = detect_frequency(serie.index)
                kwargs['width'] = ([pd.Timedelta(freq).total_seconds() * 1000] *
                                   len(serie.index))  # witdh in milliseconds

            traces.append(
                go.__dict__[spec['plottype']](
                    x=x_values,
                    y=y_values,
                    legendgroup=spec['name'],
                    **kwargs
                ))

    return traces


def timeseriesplot(df, csv_spec_path=None, offline=True):
    layout = go.Layout(
        height=365 * 2,  # plot's height in px
        legend={'x': 1.05, 'y': 0.5},
        showlegend=True,
        barmode='relative',
        # bargap=0.01,  # spaces between bar if not relative
        bargroupgap=0.01,  # space between bar groups
        # barnorm=0.1,
        xaxis={
            'gridcolor': '#bdbdbd',
            'type': 'date',
            'tickformat': '%d/%m %-H:%M',
            'autorange': True,
        },

        yaxis={
            'title': 'y',
            'titlefont': {'size': 12},
            'autorange': True,
            'scaleanchor': 'x',  # lock to x
            'gridcolor': '#bdbdbd',
            'ticklen': 2,
            'tickwidth': 2,
            'ticks': 'inside',
            'tickfont': {'size': 14},
            'showline': True,
            'tickmode': 'auto',
        },
    )

    traces = get_plotly_traces(df, csv_spec_path)

    fig = {'layout': layout, 'data': traces}

    return offline2path(fig, offline)


def corrmatrix_chart(df, colorscale='rdbu', offline=True):
    layout = {
        "margin": {
            "r": 60,
            "t": 60,
            "b": 100,
            "l": 400,
            "pad": 10,
        },
        # "title": "Correlation Matrix",
        "autosize": False,
        "height": 2 * 500,
        "width": 2 * 500 + 300,
        "yaxis": {"nticks": len(df.columns)},
    }

    fig = df.corr().iplot(kind='heatmap', colorscale=colorscale,
                          layout=layout, asFigure=True,
                          center_scale=0,
                          )

    fig['data'].update({'xgap': 3})
    fig['data'].update({'ygap': 3})

    return offline2path(fig, offline)


def missingvalues_chart(df, colorscale='reds', offline=True):
    layout = {
        "margin": {
            "r": 60,
            "t": 60,
            "b": 100,
            "l": 400,
            "pad": 10,
        },
        "yaxis": {"nticks": len(df.columns)},
    }

    df = df.isna().astype(int)
    # df = df.replace(0, np.nan)  # not better for plot perfs
    fig = df.iplot(kind='heatmap', colorscale=colorscale,
                   layout=layout, asFigure=True)

    fig['data'].update({'xgap': 3})
    fig['data'].update({'ygap': 3})
    return offline2path(fig, offline)


def filledline_chart(df, offline=True):
    fig = df.iplot(kind='scatter',
                   colorscale='spectral', fill=True, asFigure=True)
    return offline2path(fig, offline)


def boxplot_chart(df, normalize=True, colorscale='rdylbu', offline=True):
    layout = {"margin": {"b": 200},
              "autosize": True}

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        for col in df.columns:
            idx = df[col].dropna().index
            if idx.empty:
                df = df.drop(columns=col)
                continue

            x = df[col].dropna().values
            x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
            df[col].update(pd.Series(data=x_scaled[:, 0], index=idx))

    fig = df.iplot(kind='box', colorscale=colorscale,
                   layout=layout, asFigure=True)
    return offline2path(fig, offline)


def iplot2html(fig):
    return offline2path(fig, offline=True)
