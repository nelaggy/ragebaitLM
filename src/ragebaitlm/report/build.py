"""Build a self-contained HTML report from analysis output.

Mood is signed, so every chart uses a double-sided axis: rage at ``-1`` (red)
and joy at ``+1`` (green), with a zero line in the middle.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.colors import sample_colorscale

from ..analyze import JOY_THRESHOLD, RAGE_THRESHOLD

TEMPLATE_DIR = Path(__file__).parent / "templates"
RAGE = "#d62728"
JOY = "#2ca02c"
NEUTRAL = "#9aa0a6"
AXIS_TITLE = "rage ← mood → joy"
MOOD_SCALE = 100
RDGN = [[0.0, RAGE], [0.5, "#d9d9d9"], [1.0, JOY]]
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def _mood_colors(values, domain: tuple[float, float] | None = None) -> list[str]:
    lo, hi = domain if domain else (-1.0, 1.0)
    neg_span = abs(lo) if lo < 0 else (hi if hi > 0 else 1.0)
    pos_span = hi if hi > 0 else (abs(lo) if lo < 0 else 1.0)
    positions = []
    for v in values:
        if domain is None:
            pos = (v + 1) / 2
        elif v < 0:
            pos = 0.5 - 0.5 * min(1.0, -v / neg_span)
        else:
            pos = 0.5 + 0.5 * min(1.0, v / pos_span)
        positions.append(max(0.0, min(1.0, pos)))
    return sample_colorscale(RDGN, positions)


def _with_alpha(color: str, alpha: float = 0.6) -> str:
    rgb = color[color.index("(") + 1 : color.index(")")].split(",")
    r, g, b = (int(float(c)) for c in rgb)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _fig_html(fig: go.Figure) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _distribution(rows: list[dict], key: str = "key", title: str = "", limit: int = 15) -> str:
    rows = [r for r in rows if r.get("values")]
    rows = sorted(rows, key=lambda r: r.get("mean", 0) or 0)[:limit]
    if not rows:
        return "<p class='muted'>No data.</p>"
    means = [r["mean"] for r in rows]
    colors = _mood_colors(means, domain=(min(means), max(means)))

    labels = [str(r[key]) for r in rows]

    fig = go.Figure()
    for row, label, color in zip(rows, labels, colors):
        values = [v * MOOD_SCALE for v in row["values"]]
        fig.add_trace(
            go.Violin(
                x=values,
                y=[label] * len(values),
                orientation="h",
                name=label,
                points=False,
                box_visible=False,
                meanline_visible=False,
                fillcolor=_with_alpha(color),
                line=dict(color="#e8eaed", width=1),
                spanmode="hard",
                scalemode="width",
                width=0.9,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[r["median"] * MOOD_SCALE for r in rows],
            y=labels,
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=16,
                line=dict(color="#f5f5f5", width=2),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[r["mean"] * MOOD_SCALE for r in rows],
            y=labels,
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=9,
                color="#f5f5f5",
                line=dict(color="#0f1116", width=1),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_layout(
        title=title,
        height=max(240, 56 * len(rows) + 80),
        margin=dict(l=10, r=20, t=56, b=30),
        xaxis=dict(range=[-MOOD_SCALE, MOOD_SCALE], title=AXIS_TITLE, zeroline=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        annotations=[
            dict(
                text="◆ mean&nbsp;&nbsp;&nbsp;| median",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=10, color=NEUTRAL),
            )
        ],
        **DARK,
    )
    return _fig_html(fig)


def _hist(histogram: dict, title: str = "Mood distribution (all user turns)") -> str:
    centers = histogram.get("bin_centers") or []
    counts = histogram.get("counts") or []
    if not counts:
        return "<p class='muted'>No data.</p>"
    width = (centers[1] - centers[0]) if len(centers) > 1 else 0.1

    sections = (
        ("RAGE", RAGE, -1.0, RAGE_THRESHOLD,
         [(c, n) for c, n in zip(centers, counts) if c <= RAGE_THRESHOLD]),
        ("NEUTRAL", NEUTRAL, RAGE_THRESHOLD, JOY_THRESHOLD,
         [(c, n) for c, n in zip(centers, counts) if RAGE_THRESHOLD < c < JOY_THRESHOLD]),
        ("JOY", JOY, JOY_THRESHOLD, 1.0,
         [(c, n) for c, n in zip(centers, counts) if c >= JOY_THRESHOLD]),
    )
    total = sum(counts) or 1

    fig = go.Figure()
    for name, color, x0, x1, section in sections:
        xs = [c for c, _ in section]
        ys = [n for _, n in section]
        if not xs:
            continue
        fig.add_trace(
            go.Bar(
                x=[c * MOOD_SCALE for c in xs],
                y=ys,
                name=name,
                marker_color=_mood_colors(xs),
                width=width * MOOD_SCALE * 0.9,
                hovertemplate=f"{name}<br>mood=%{{x:.1f}}<br>turns=%{{y}}<extra></extra>",
            )
        )

    for boundary in (RAGE_THRESHOLD, JOY_THRESHOLD):
        fig.add_vline(
            x=boundary * MOOD_SCALE, line_color=NEUTRAL, line_width=1, line_dash="dot"
        )

    annotations = []
    for name, color, _, _, section in sections:
        count = sum(n for _, n in section)
        center = sum(c for c, _ in section) / len(section) if section else 0.0
        annotations.append(
            dict(
                x=center * MOOD_SCALE,
                y=1.0,
                xref="x",
                yref="paper",
                yanchor="bottom",
                showarrow=False,
                text=f"<b>{name}</b><br>{count} · {count / total * 100:.1f}%",
                font=dict(color=color, size=11),
            )
        )

    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=50, r=20, t=60, b=30),
        xaxis=dict(range=[-MOOD_SCALE, MOOD_SCALE], title=AXIS_TITLE, zeroline=False),
        yaxis=dict(title="user turns"),
        showlegend=False,
        annotations=annotations,
        **DARK,
    )
    return _fig_html(fig)


def _table(rows: list[dict], columns: list[tuple[str, str]], limit: int = 20) -> str:
    if not rows:
        return "<p class='muted'>No data.</p>"
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = []
    for row in rows[:limit]:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float) and key.startswith("pct_"):
                value = f"{value * 100:.1f}%"
            elif isinstance(value, float):
                value = f"{value * MOOD_SCALE:+.1f}"
            elif value is None:
                value = "–"
            cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def build_report(analysis: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    overall = analysis.get("overall", {})

    charts = {
        "by_model": _distribution(analysis.get("by_model", []), title="Mood by model"),
        "by_harness": _distribution(analysis.get("by_harness", []), title="Mood by harness"),
        "hist": _hist(analysis.get("mood_histogram", {})),
    }

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    from plotly.offline import get_plotlyjs

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        plotlyjs=get_plotlyjs(),
        overall=overall,
        charts=charts,
        by_model_table=_table(
            analysis.get("by_model", []),
            [
                ("key", "model"),
                ("n", "messages"),
                ("mean", "mean mood"),
                ("median", "median mood"),
                ("pct_rage", "% rage"),
                ("pct_joy", "% joy"),
            ],
        ),
        by_harness_table=_table(
            analysis.get("by_harness", []),
            [
                ("key", "harness"),
                ("n", "messages"),
                ("mean", "mean mood"),
                ("median", "median mood"),
                ("pct_rage", "% rage"),
                ("pct_joy", "% joy"),
            ],
        ),
    )
    out.write_text(html, encoding="utf-8")
    return out
