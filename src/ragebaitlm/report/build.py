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

TEMPLATE_DIR = Path(__file__).parent / "templates"
RAGE = "#d62728"
JOY = "#2ca02c"
NEUTRAL = "#9aa0a6"
AXIS_TITLE = "rage ← mood → joy"
MOOD_COLORSCALE = [[0.0, RAGE], [0.5, "#6b7280"], [1.0, JOY]]


def _fig_html(fig: go.Figure) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _bar(
    rows: list[dict],
    key: str = "key",
    value: str = "mean",
    title: str = "",
    ci: bool = True,
) -> str:
    rows = [r for r in rows if r.get("n", 0) >= 1]
    rows = sorted(rows, key=lambda r: r.get(value, 0) or 0)[:15]
    if not rows:
        return "<p class='muted'>No data.</p>"
    labels = [str(r[key]) for r in rows]
    values = [r.get(value) or 0 for r in rows]
    colors = [RAGE if v < 0 else JOY for v in values]
    error = None
    if ci and all(r.get("ci_low") is not None for r in rows):
        error = [
            [v - (r.get("ci_low") or v) for v, r in zip(values, rows)],
            [(r.get("ci_high") or v) - v for v, r in zip(values, rows)],
        ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data", array=error[1], arrayminus=error[0], thickness=1
            )
            if error
            else None,
            customdata=[[r.get("n", 0)] for r in rows],
            hovertemplate="%{y}<br>mood=%{x:.3f}<br>n=%{customdata[0]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_layout(
        title=title,
        height=max(220, 34 * len(rows) + 80),
        margin=dict(l=10, r=20, t=40, b=30),
        xaxis=dict(range=[-1, 1], title=AXIS_TITLE, zeroline=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return _fig_html(fig)


def _hist(histogram: dict, title: str = "Mood distribution (all user turns)") -> str:
    centers = histogram.get("bin_centers") or []
    counts = histogram.get("counts") or []
    if not counts:
        return "<p class='muted'>No data.</p>"
    width = (centers[1] - centers[0]) if len(centers) > 1 else 0.1
    colors = [RAGE if c < 0 else JOY for c in centers]
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors, width=width * 0.9))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=50, r=20, t=40, b=30),
        xaxis=dict(range=[-1, 1], title=AXIS_TITLE, zeroline=False),
        yaxis=dict(title="user turns"),
        annotations=[
            dict(x=-0.97, y=1.12, xref="paper", yref="paper", showarrow=False,
                 text="rage", font=dict(color=RAGE), xanchor="left"),
            dict(x=0.97, y=1.12, xref="paper", yref="paper", showarrow=False,
                 text="joy", font=dict(color=JOY), xanchor="right"),
        ],
    )
    return _fig_html(fig)


def _session_figure(session: dict) -> str:
    points = session["points"]
    if not points:
        return ""
    xs = [
        datetime.fromtimestamp(p["ts"] / 1000) if p["ts"] else None for p in points
    ]
    mood = [p["mood"] for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=mood,
            mode="lines+markers",
            name="mood",
            line=dict(color="#6b7280", width=1.5),
            marker=dict(
                size=8,
                color=mood,
                colorscale=MOOD_COLORSCALE,
                cmin=-1,
                cmax=1,
                line=dict(color="#0f1116", width=0.5),
            ),
            customdata=[[p["model"]] for p in points],
            hovertemplate="mood=%{y:.2f}<br>model=%{customdata[0]}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    for marker in session.get("markers", []):
        if marker.get("ts"):
            fig.add_vline(
                x=datetime.fromtimestamp(marker["ts"] / 1000),
                line_dash="dot",
                line_color="#999999",
                line_width=1,
                annotation_text=str(marker.get("model") or ""),
                annotation_position="top left",
                annotation_font_size=9,
            )
    fig.update_layout(
        title=f"{session['title']}  ·  {session['harness']}  ·  n={session['n_turns']}",
        height=300,
        margin=dict(l=40, r=40, t=50, b=30),
        yaxis=dict(range=[-1, 1], title=AXIS_TITLE, zeroline=False),
        legend=dict(orientation="h", y=1.12, x=0),
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
            if isinstance(value, float):
                value = f"{value:.3f}"
            elif value is None:
                value = "–"
            cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def build_report(analysis: dict, out_path: str | Path, max_sessions: int = 12) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    overall = analysis.get("overall", {})
    sessions = analysis.get("sessions", [])[:max_sessions]

    charts = {
        "by_model": _bar(analysis.get("by_model", []), title="Mood by model (blame)"),
        "by_harness": _bar(analysis.get("by_harness", []), title="Mood by harness"),
        "hist": _hist(analysis.get("mood_histogram", {})),
    }

    session_charts = [
        {"title": s["title"], "html": _session_figure(s)} for s in sessions
    ]

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
        session_charts=session_charts,
        revisions=analysis.get("revisions", {}),
        by_model_table=_table(
            analysis.get("by_model", []),
            [
                ("key", "model"),
                ("n", "n"),
                ("mean", "mean mood"),
                ("mean_delta", "mean Δ"),
                ("delta_ci_low", "Δ ci low"),
                ("delta_ci_high", "Δ ci high"),
                ("pct_rage", "% rage"),
                ("pct_joy", "% joy"),
            ],
        ),
        by_harness_table=_table(
            analysis.get("by_harness", []),
            [
                ("key", "harness"),
                ("n", "n"),
                ("mean", "mean mood"),
                ("mean_delta", "mean Δ"),
                ("pct_rage", "% rage"),
                ("pct_joy", "% joy"),
            ],
        ),
        top_sessions_table=_table(
            analysis.get("top_sessions", []),
            [
                ("title", "session"),
                ("harness", "harness"),
                ("n_turns", "turns"),
                ("mean_mood", "mean mood"),
                ("min_mood", "min mood"),
            ],
        ),
    )
    out.write_text(html, encoding="utf-8")
    return out
