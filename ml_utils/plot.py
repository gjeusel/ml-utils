from pathlib import Path
import copy
import colorlover

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go

import numpy as np
import seaborn as sns
import pandas as pd

from .analytics import detect_frequency


class cl:
    # cf for colors: https://plot.ly/ipython-notebooks/color-scales/
    brown = colorlover.scales['11']['div']['BrBG'][2]
    brown_lighter = colorlover.scales['11']['div']['BrBG'][3]
    dark_red = '#8B0000'
    red = colorlover.scales['11']['div']['RdYlBu'][0]
    red_lighter = colorlover.scales['11']['div']['RdYlGn'][1]
    orange = colorlover.scales['11']['div']['RdYlGn'][2]
    orange_lighter = colorlover.scales['11']['div']['RdYlGn'][3]
    yellow = colorlover.scales['11']['div']['Spectral'][4]
    green_darker = colorlover.scales['11']['div']['RdYlGn'][10]
    green = colorlover.scales['11']['div']['RdYlGn'][9]
    green_lighter = colorlover.scales['11']['div']['RdYlGn'][7]
    blue = colorlover.scales['11']['div']['RdYlBu'][9]
    blue_lighter = colorlover.scales['11']['div']['RdYlBu'][7]
    purple = colorlover.scales['11']['div']['PRGn'][2]
    grey = colorlover.scales['11']['div']['RdGy'][7]
    grey_darker = colorlover.scales['11']['div']['RdGy'][8]
    black = colorlover.scales['11']['div']['RdGy'][10]


cmap_redblue = ListedColormap(sns.color_palette("RdBu_r", 20).as_hex())


def get_dist_mat(df, target_col, metric='euclidean',
                 figsize=(20, 20), cmap=cmap_redblue,):
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Sorting according to clusters to make then apparent :
    M = np.concatenate((X, y[:, np.newaxis]), axis=1)
    # Sort according to last column :
    M = M[M[:, -1].argsort()]
    M = M[0: -1]  # remove last column

    from scipy.spatial.distance import pdist, squareform
    dist_mat = pdist(M, metric=metric)
    # translates this flattened form into a full matrix
    dist_mat = squareform(dist_mat)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_mat, cmap=cmap, interpolation='none')

    # get colorbar smaller than matrix
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # want a more natural, table-like display
    ax.invert_yaxis()

    # Move top xaxes :
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.axis('off')

    return dist_mat, fig, ax


def get_corr_mat(df, figsize=(20, 20), cmap=cmap_redblue):
    # Compute correlation matrix
    corrmat = df.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set ax & colormap with seaborn.
    ax = sns.heatmap(corrmat, vmin=-1, vmax=1, center=0,
                     square=True, linewidths=1, xticklabels=True,
                     yticklabels=True,
                     cmap=cmap)

    ax.set_xticklabels(df.columns, minor=False, rotation='vertical')
    ax.set_yticklabels(df.columns[df.shape[1]::-1],
                       minor=False, rotation='horizontal')

    return corrmat, fig, ax


def boxplot(df, normalize=True, figsize=(20, 20)):
    sns.set(style="ticks")
    # Initialize the figure with a logarithmic x axis
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        df = (df - df.mean()) / (df.max() - df.min())

    sns.boxplot(data=df,
                orient='h',
                # whis=whis # Proportion of the IQR past the low and high quartiles to extend the plot whiskers. Points outside this range will be identified as outliers.
                )

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)

    return fig, ax


# Plotly part:

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

    if offline:
        tmp_dir = Path('/tmp')
        if tmp_dir.exists():
            fpath = tmp_dir / 'output.html'
        else:
            fpath = Path('output.html')

        plotly.offline.plot(fig, filename=fpath.as_posix())

    return fig
