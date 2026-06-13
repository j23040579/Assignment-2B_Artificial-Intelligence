"""
predict_map.py
===============
Recovered / refactored version of your original predict_all.py.

Loads the trained model, runs predictions for EVERY location x EVERY
hour, then renders the results on a Folium map with edges coloured and
labelled by predicted flow at a specific hour (or all-day totals).

This version shares its site/connection/site-name data (config.py) and
its model-loading + prediction logic (model_utils.py) with A2B.py (the
A* route finder), so the map and the route finder are always
consistent with each other.

Usage:
    python predict_map.py                 # all-day totals, RNN model
    python predict_map.py --hour 13       # 1PM flow, RNN model
    python predict_map.py --hour 8        # 8AM flow, RNN model
    python predict_map.py --hour 17 --model LSTM

Hour mapping (0-23):
    0=12AM  1=1AM  2=2AM  ...  12=12PM  13=1PM  ...  17=5PM  ...  23=11PM
"""

import os
import argparse
import webbrowser

import folium

from config import COORDS, CONNECTIONS, SITE_NAMES, HOUR_LABELS
from model_utils import load_artefacts, load_long_data, predict_all, site_flow


# ── Parse command-line arguments ────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict traffic flow and render on map.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python predict_map.py                  -> all-day total on edges
  python predict_map.py --hour 13        -> 1PM flow on edges
  python predict_map.py --hour 8         -> 8AM flow on edges
  python predict_map.py --hour 17 --model LSTM

Hour reference:
  0=12AM  1=1AM  2=2AM  3=3AM  4=4AM  5=5AM
  6=6AM   7=7AM  8=8AM  9=9AM 10=10AM 11=11AM
 12=12PM 13=1PM 14=2PM 15=3PM 16=4PM  17=5PM
 18=6PM  19=7PM 20=8PM 21=9PM 22=10PM 23=11PM
        """
    )
    parser.add_argument(
        '--hour', type=int, default=None,
        help='Hour to predict (0-23). Omit to show all-day totals.'
    )
    parser.add_argument(
        '--model', type=str, default='RNN',
        help="Model to use: RNN, LSTM, or GRU (default: RNN)."
    )
    args = parser.parse_args()

    if args.hour is not None and not (0 <= args.hour <= 23):
        parser.error(f"--hour must be between 0 and 23, got {args.hour}")

    return args


# ── SVG bar chart for popup ───────────────────────────────────────────────────────
def build_bar_chart_html(predictions, location, highlight_hour=None):
    max_flow = max(predictions) if max(predictions) > 0 else 1
    peak_hour = predictions.index(max(predictions))
    total = sum(predictions)
    W, H = 380, 120
    bar_w = W / 24
    bar_gap = 1
    label_h = 15
    bars_svg = ""

    for i, flow in enumerate(predictions):
        bar_h = (flow / max_flow) * (H - label_h)
        x = i * bar_w + bar_gap / 2
        y = H - bar_h - label_h
        if i == highlight_hour:
            colour = "#8e44ad"   # purple = selected hour
        elif i == peak_hour:
            colour = "#e74c3c"   # red = peak
        elif i < 12:
            colour = "#3498db"   # blue = AM
        else:
            colour = "#e67e22"   # orange = PM
        bars_svg += (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - bar_gap:.1f}" '
            f'height="{bar_h:.1f}" fill="{colour}" opacity="0.85">'
            f'<title>{HOUR_LABELS[i]}: {flow} vehicles</title></rect>'
        )

    axis_labels = ""
    for i in [0, 6, 12, 18, 23]:
        x = i * bar_w + bar_w / 2
        axis_labels += (
            f'<text x="{x:.1f}" y="{H - 2}" font-size="9" '
            f'text-anchor="middle" fill="#555">{HOUR_LABELS[i]}</text>'
        )

    svg = f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">{bars_svg}{axis_labels}</svg>'
    legend_note = ""
    if highlight_hour is not None:
        legend_note = f'<span style="color:#8e44ad;">\u25a0</span> Selected ({HOUR_LABELS[highlight_hour]}) &nbsp;'

    return f"""
    <div style="font-family:Arial,sans-serif; font-size:12px; width:390px;">
      <b style="font-size:13px;">{location}</b><br>
      <span style="color:#888;">Predicted 24-hour traffic flow</span>
      {svg}
      <table style="width:100%; font-size:11px; margin-top:4px;">
        <tr><td>\U0001F534 <b>Peak hour</b></td><td><b>{HOUR_LABELS[peak_hour]}</b> \u2014 {max(predictions)} vehicles</td></tr>
        <tr><td>\U0001F4CA <b>Daily total</b></td><td><b>{total:,}</b> vehicles</td></tr>
      </table>
      <div style="margin-top:4px; font-size:10px; color:#aaa;">
        {legend_note}\U0001F535 AM &nbsp; \U0001F7E0 PM &nbsp; \U0001F534 Peak
      </div>
    </div>"""


# ── Build map ─────────────────────────────────────────────────────────────────────
def build_map(results, hour=None):
    """
    hour=None  -> edges show daily total flow
    hour=0-23  -> edges show flow at that specific hour
    """
    hour_label = HOUR_LABELS[hour] if hour is not None else "All Day"
    output_map = f'boroondara_predictions_{hour_label.replace(":", "")}.html'

    centre_lat = sum(lat for lat, _ in COORDS.values()) / len(COORDS)
    centre_lon = sum(lon for _, lon in COORDS.values()) / len(COORDS)
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    # Title banner
    title_html = f"""
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:1000; background:white; padding:8px 18px;
                border-radius:8px; border:2px solid #555;
                font-family:Arial,sans-serif; font-size:14px; font-weight:bold;">
        Traffic Prediction \u2014 {hour_label}
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Collect all edge flows so we can scale colours consistently ─────────────
    edge_flows = []
    for site_a, site_b, _ in CONNECTIONS:
        flow_a = site_flow(site_a, results, hour)
        flow_b = site_flow(site_b, results, hour)
        edge_flows.append((flow_a + flow_b) // 2)

    max_edge_flow = max(edge_flows) if edge_flows else 1

    # Thresholds: top 33% = red, middle 33% = orange, bottom = grey
    high_thresh = max_edge_flow * 0.66
    medium_thresh = max_edge_flow * 0.33

    # ── Draw edges ────────────────────────────────────────────────────────────
    for (site_a, site_b, distance), avg_flow in zip(CONNECTIONS, edge_flows):
        if site_a not in COORDS or site_b not in COORDS:
            continue

        if avg_flow >= high_thresh:
            edge_color = '#c0392b'   # red - high
        elif avg_flow >= medium_thresh:
            edge_color = '#e67e22'   # orange - medium
        else:
            edge_color = '#555555'   # grey - low

        # ── Speed & travel time (assignment spec formulas) ────────────────────
        # speed = (93.75 + sqrt(8791.015625 - 5.85935 * flow)) / 2.929675
        # capped at 60 km/h
        # travel time = distance / speed + 30 sec junction delay
        flow_a = site_flow(site_a, results, hour)
        discriminant = max(0, 8791.015625 - 5.85935 * flow_a)
        speed_kmh = min((93.75 + discriminant ** 0.5) / 2.929675, 60.0)
        travel_mins = (distance / speed_kmh) * 60 + 0.5   # +30 sec = +0.5 min

        tooltip_text = (
            f"Site {site_a} \u2194 Site {site_b} | {distance} km | "
            f"Flow ({hour_label}): {avg_flow:,} veh/hr | "
            f"Speed: {speed_kmh:.1f} km/h | "
            f"Travel time: {travel_mins:.1f} min (inc. 30s junction delay)"
        )

        folium.PolyLine(
            locations=[COORDS[site_a], COORDS[site_b]],
            color=edge_color,
            weight=5,
            opacity=0.9,
            tooltip=tooltip_text
        ).add_to(m)

        # ── Edge label at midpoint ───────────────────────────────────────────
        # Only show when a specific hour is given - daily totals clutter the map
        if hour is not None:
            lat_a, lon_a = COORDS[site_a]
            lat_b, lon_b = COORDS[site_b]
            mid_lat = (lat_a + lat_b) / 2
            mid_lon = (lon_a + lon_b) / 2

            label_html = f"""<div style="
                background-color: {edge_color};
                color: white;
                font-family: Arial, sans-serif;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 6px;
                border-radius: 4px;
                white-space: nowrap;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.4);
                line-height: 1.5;
                text-align: center;
            ">Vehicle: {avg_flow:,} <br>Speed: {speed_kmh:.1f} km/h<br>Travel time: {travel_mins:.1f} min</div>"""

            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=label_html,
                    icon_size=(90, 52),
                    icon_anchor=(45, 26),
                ),
                tooltip=(
                    f"Site {site_a} \u2194 Site {site_b} | "
                    f"Flow: {avg_flow:,} veh/hr | "
                    f"Speed: {speed_kmh:.1f} km/h | "
                    f"Time: {travel_mins:.1f} min"
                )
            ).add_to(m)

    # ── Draw site markers ─────────────────────────────────────────────────────
    for site_id, loc_preds in results.items():
        if site_id not in COORDS:
            continue

        lat, lon = COORDS[site_id]
        site_name = SITE_NAMES.get(site_id, f"Site {site_id}")
        all_dirs = list(loc_preds.values())
        if not all_dirs:
            continue

        combined = [sum(d[h] for d in all_dirs) for h in range(24)]
        peak_hour = combined.index(max(combined))
        daily_total = sum(combined)
        hour_flow = site_flow(site_id, results, hour)

        color = 'red' if daily_total > 15000 else ('orange' if daily_total > 8000 else 'green')

        tabs_html = ""
        for location, preds in loc_preds.items():
            chart = build_bar_chart_html(preds, location, highlight_hour=hour)
            tabs_html += f"<div style='margin-bottom:16px; border-bottom:1px solid #eee; padding-bottom:12px;'>{chart}</div>"

        # Show hour-specific flow in popup if hour is selected
        hour_line = ""
        if hour is not None:
            hour_line = f"<p style='margin:0 0 6px 0; font-size:11px; color:#8e44ad;'>\U0001F7E3 <b>{hour_label} flow: {hour_flow:,} vehicles</b></p>"

        popup_html = f"""
        <div style="max-height:420px; overflow-y:auto; padding:8px;">
          <h3 style="margin:0 0 4px 0; font-size:14px;">Site {site_id}</h3>
          <p style="margin:0 0 6px 0; color:#555; font-size:12px;">{site_name}</p>
          <p style="margin:0 0 6px 0; font-size:11px;">Peak: <b>{HOUR_LABELS[peak_hour]}</b> &nbsp;|&nbsp; Daily total: <b>{daily_total:,}</b></p>
          {hour_line}
          {tabs_html}
        </div>"""

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=420),
            tooltip=f"Site {site_id} \u2014 {site_name} | {hour_label}: {hour_flow:,} veh",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    flow_unit = "vehicles/hr" if hour is not None else "vehicles/day"
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px; border-radius:8px;
                border:2px solid grey; font-size:12px;">
      <b>Predicted Flow \u2014 {hour_label}</b><br>
      <span style="color:#c0392b;">\u2501\u2501</span> High flow (&gt;{int(high_thresh):,} {flow_unit})<br>
      <span style="color:#e67e22;">\u2501\u2501</span> Medium flow<br>
      <span style="color:#555555;">\u2501\u2501</span> Low flow<br>
      <br>
      <b>Site markers</b><br>
      <span style="color:red;">\u25cf</span> &gt;15,000 daily total<br>
      <span style="color:orange;">\u25cf</span> 8,000\u201315,000 daily total<br>
      <span style="color:green;">\u25cf</span> &lt;8,000 daily total<br>
      <br><i>Hover edges \u00b7 Click markers</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, output_map


# ── Main ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()

    if args.hour is not None:
        print(f"\nTarget hour : {args.hour} ({HOUR_LABELS[args.hour]})")
    else:
        print("\nNo hour specified - showing all-day totals on edges.")

    model, scaler, le = load_artefacts(args.model)
    df_long = load_long_data()
    results = predict_all(model, scaler, le, df_long)

    print("\nBuilding map...")
    m, output_path = build_map(results, hour=args.hour)
    m.save(output_path)

    abs_path = os.path.abspath(output_path)
    webbrowser.open(f'file://{abs_path}')
    print(f"\nDone! Map saved and opened: {output_path}")
