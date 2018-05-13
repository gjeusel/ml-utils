import colorlover


class CustomScales:
    _spectral = colorlover.scales['11']['div']['Spectral']
    spectral = _spectral[:4] + _spectral[7:]

    _rdylgn = colorlover.scales['11']['div']['RdYlGn']
    rdylgn = _rdylgn[:4] + _rdylgn[7:]


class cl:
    # cf for colors: https://plot.ly/ipython-notebooks/color-scales/
    brown          = colorlover.scales['11']['div']['BrBG'][2]
    brown_lighter  = colorlover.scales['11']['div']['BrBG'][3]
    dark_red       = '#8B0000'
    red            = colorlover.scales['11']['div']['RdYlBu'][0]
    red_lighter    = colorlover.scales['11']['div']['RdYlGn'][1]
    orange         = colorlover.scales['11']['div']['RdYlGn'][2]
    orange_lighter = colorlover.scales['11']['div']['RdYlGn'][3]
    yellow         = colorlover.scales['11']['div']['Spectral'][4]
    green_darker   = colorlover.scales['11']['div']['RdYlGn'][10]
    green          = colorlover.scales['11']['div']['RdYlGn'][9]
    green_lighter  = colorlover.scales['11']['div']['RdYlGn'][7]
    blue           = colorlover.scales['11']['div']['RdYlBu'][9]
    blue_lighter   = colorlover.scales['11']['div']['RdYlBu'][7]
    purple         = colorlover.scales['11']['div']['PRGn'][2]
    grey           = colorlover.scales['11']['div']['RdGy'][7]
    grey_darker    = colorlover.scales['11']['div']['RdGy'][8]
    black          = colorlover.scales['11']['div']['RdGy'][10]
    white          = colorlover.scales['11']['div']['RdGy'][5]


def str2scale(str_scale, n=7):
    import colorlover as cl
    scale = None
    for t in ['div', 'qual', 'seq']:
        for s in cl.scales[str(n)][t]:
            if str_scale.lower() == s.lower():
                scale = cl.scales[str(n)][t][s]
    return scale


def colorhandler(colors, n):
    if isinstance(colors, str):
        colors = str2scale(colors, n)

    return colors


def _heatmap_color_handler(colors, n=11):
    scale = colorhandler(colors, n)
    ncolors = len(scale)
    colorscale = [[float(i)/(ncolors-1), color] for i, color in enumerate(scale)]
    return colorscale

