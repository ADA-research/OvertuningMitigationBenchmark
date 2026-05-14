import matplotlib as mpl


_STYLE_APPLIED = False


def configure_plot_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    mpl.rcParams.update(
        {
            "font.size": 13,
            "font.family": "serif",
            "font.serif": [
                "Linux Libertine O",
                "Linux Libertine",
                "Libertinus Serif",
                "DejaVu Serif",
            ],
            "font.sans-serif": [
                "Linux Biolinum O",
                "Linux Biolinum",
                "DejaVu Sans",
            ],
            "font.monospace": [
                "Inconsolata",
                "Inconsolatazi4",
                "DejaVu Sans Mono",
            ],
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "axes.titlesize": 19,
            "axes.titleweight": "semibold",
            "axes.labelsize": 16,
            "axes.labelweight": "normal",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#2f2f2f",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": True,
            "legend.fontsize": 12,
            "legend.fancybox": False,
            "legend.edgecolor": "#d8d8d8",
            "legend.framealpha": 0.95,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "text.color": "#181818",
            "axes.labelcolor": "#181818",
            "axes.titlecolor": "#181818",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    _STYLE_APPLIED = True