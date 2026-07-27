from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def output_plot(bundle, proposed=None):
    names = [x.name for x in bundle.outputs[:6]]
    values = [x.score for x in bundle.outputs[:6]]
    errors = [x.uncertainty for x in bundle.outputs[:6]]
    fig = go.Figure(go.Bar(name="Current", x=values, y=names, orientation="h", error_x={"array": errors}, marker_color="#9B5DE5"))
    if proposed:
        fig.add_bar(name="Proposed", x=[proposed.get(x, 0) for x in names], y=names, orientation="h", marker_color="#00C2A8")
    fig.update_layout(template="plotly_dark", barmode="group", xaxis_range=[0, 100], height=430, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def radar(features):
    names = list(features)
    values = list(features.values())
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill="toself", line_color="#00C2A8"))
    fig.update_layout(template="plotly_dark", polar={"radialaxis": {"range": [0, 100]}}, height=400, margin=dict(l=30, r=30, t=30, b=30))
    return fig


def waveform(audio, bars=None):
    fig = go.Figure()
    if audio:
        sr, y = audio
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.mean(axis=1)
        stride = max(1, len(y) // 5000)
        t = np.arange(0, len(y), stride) / sr
        fig.add_scatter(x=t, y=y[::stride], mode="lines", line={"width": 1, "color": "#F15BB5"})
        for bar in bars or []:
            fig.add_vline(x=bar.start, line_dash="dot", line_color="#FEE440")
    fig.update_layout(template="plotly_dark", height=260, title="Waveform and estimated bar boundaries", margin=dict(l=20, r=20, t=45, b=20))
    return fig

