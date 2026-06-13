"""
route_map.py
============
Renders a Folium map for A2B.py: the full Boroondara road network is
drawn in grey/orange/red (coloured by predicted flow, same scheme as
predict_map.py), and the BEST PATH found by A* is highlighted as a
thick blue line with numbered markers showing the order of travel.

This module only needs the SINGLE-HOUR site flows already computed by
A2B.py (via model_utils.predict_site_flows) — it does NOT require the
full 24-hour predict_all() results, so it stays fast even though A2B.py
runs a much lighter prediction pass than predict_map.py.
"""

import os
import webbrowser

import folium

from config import COORDS, CONNECTIONS, SITE_NAMES, HOUR_LABELS
from route_finder import compute_speed_and_time


# ── Build the route map ──────────────────────────────────────────────────────────
def build_route_map(origin, destination, path, edges, site_flows, hour, total_time):
    """
    Build a Folium map showing:
      - The full road network, coloured by predicted flow at `hour`
        (grey = low, orange = medium, red = high — same thresholds as
        predict_map.py).
      - The best path (from A2B's A* search) highlighted as a thick
        blue line, with numbered markers at each stop showing the
        order of travel and the predicted flow/speed/travel time for
        that leg.
      - The origin marked in green, the destination marked in red.

    Parameters
    ----------
    origin, destination : int
        Site IDs for the start and end of the journey.
    path : list[int]
        Ordered list of site IDs from origin to destination
        (as returned by route_finder.find_best_route).
    edges : list[tuple]
        List of (from_site, to_site, flow, travel_min) for each leg of
        the path (as returned by route_finder.find_best_route).
    site_flows : dict[int, int]
        Predicted flow (vehicles/hour) for every site, at `hour`.
    hour : int
        Hour index (0-23) the prediction was made for.
    total_time : float
        Total travel time for the path, in minutes.

    Returns
    -------
    (m, output_path) : (folium.Map, str)
        The built map, and the filename it should be saved to.
    """
    hour_label = HOUR_LABELS[hour]
    output_path = f'route_{origin}_to_{destination}_{hour_label.replace(":", "")}.html'

    # ── Base map, centred on the network ────────────────────────────────────────
    centre_lat = sum(lat for lat, _ in COORDS.values()) / len(COORDS)
    centre_lon = sum(lon for _, lon in COORDS.values()) / len(COORDS)
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    # ── Title / summary banner ───────────────────────────────────────────────────
    route_str = " \u2192 ".join(str(s) for s in path)
    title_html = f"""
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:1000; background:white; padding:8px 18px;
                border-radius:8px; border:2px solid #555;
                font-family:Arial,sans-serif; font-size:13px; font-weight:bold;
                text-align:center; max-width:90%;">
        Best Route ({hour_label}): {route_str}<br>
        <span style="font-size:12px; font-weight:normal; color:#555;">
            Total Driving Time: {total_time:.1f} min
        </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Pre-compute flow-based colour thresholds for the NON-route edges ───────────
    # (same red/orange/grey scheme as predict_map.py, based on this hour's flow)
    edge_flows = []
    for site_a, site_b, _ in CONNECTIONS:
        flow_a = site_flows.get(site_a, 0)
        flow_b = site_flows.get(site_b, 0)
        edge_flows.append((flow_a + flow_b) // 2)

    max_edge_flow = max(edge_flows) if edge_flows else 1
    high_thresh = max_edge_flow * 0.66
    medium_thresh = max_edge_flow * 0.33

    # ── Build a lookup of path edges -> (flow, travel_min), in either direction ───
    path_edge_info = {}
    for from_site, to_site, flow, travel_min in edges:
        path_edge_info[(from_site, to_site)] = (flow, travel_min)
        path_edge_info[(to_site, from_site)] = (flow, travel_min)

    # ── Draw every road connection ───────────────────────────────────────────────
    for (site_a, site_b, distance), avg_flow in zip(CONNECTIONS, edge_flows):
        if site_a not in COORDS or site_b not in COORDS:
            continue

        on_route = (site_a, site_b) in path_edge_info

        if on_route:
            flow, travel_min = path_edge_info[(site_a, site_b)]
            speed_kmh, _ = compute_speed_and_time(distance, flow)
            color = '#2980b9'   # blue - on the chosen route
            weight = 7
            opacity = 0.95
            tooltip_text = (
                f"ON ROUTE: Site {site_a} \u2194 Site {site_b} | {distance} km | "
                f"Flow ({hour_label}): {flow:,} veh/hr | "
                f"Speed: {speed_kmh:.1f} km/h | "
                f"Travel time: {travel_min:.1f} min (inc. 0.5 min delay)"
            )
        else:
            if avg_flow >= high_thresh:
                color = '#c0392b'   # red - high flow
            elif avg_flow >= medium_thresh:
                color = '#e67e22'   # orange - medium flow
            else:
                color = '#999999'   # grey - low flow / not on route

            weight = 3
            opacity = 0.5

            flow_a = site_flows.get(site_a, 0)
            speed_kmh, travel_min = compute_speed_and_time(distance, flow_a)
            tooltip_text = (
                f"Site {site_a} \u2194 Site {site_b} | {distance} km | "
                f"Flow ({hour_label}): {avg_flow:,} veh/hr | "
                f"Speed: {speed_kmh:.1f} km/h | "
                f"Travel time: {travel_min:.1f} min"
            )

        folium.PolyLine(
            locations=[COORDS[site_a], COORDS[site_b]],
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=tooltip_text,
        ).add_to(m)

        # ── Edge label at midpoint (flow / speed / travel time) ─────────────────
        lat_a, lon_a = COORDS[site_a]
        lat_b, lon_b = COORDS[site_b]
        mid_lat = (lat_a + lat_b) / 2
        mid_lon = (lon_a + lon_b) / 2

        if on_route:
            label_flow = flow
        else:
            label_flow = avg_flow

        label_html = f"""<div style="
            background-color: {color};
            color: white;
            font-family: Arial, sans-serif;
            font-size: 10px;
            font-weight: bold;
            padding: 3px 5px;
            border-radius: 4px;
            width: 78px;
            white-space: normal;
            word-wrap: break-word;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.4);
            line-height: 1.3;
            text-align: center;
        ">Vehicle: {label_flow:,}<br>Speed: {speed_kmh:.1f} km/h<br>Travel time: {travel_min:.1f} min</div>"""

        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                html=label_html,
                icon_size=(86, 60),
                icon_anchor=(43, 30),
            ),
            tooltip=tooltip_text,
        ).add_to(m)

    # ── Markers for every site ───────────────────────────────────────────────────
    # path_order[site_id] = step number (1-based) if the site is on the route
    path_order = {site_id: i + 1 for i, site_id in enumerate(path)}

    for site_id, (lat, lon) in COORDS.items():
        site_name = SITE_NAMES.get(site_id, f"Site {site_id}")
        flow_here = site_flows.get(site_id, 0)

        if site_id == origin:
            icon = folium.Icon(color='green', icon='play', prefix='fa')
            label = "ORIGIN"
        elif site_id == destination:
            icon = folium.Icon(color='red', icon='flag-checkered', prefix='fa')
            label = "DESTINATION"
        elif site_id in path_order:
            icon = folium.Icon(color='blue', icon='arrow-right', prefix='fa')
            label = f"Stop {path_order[site_id]} of {len(path)}"
        else:
            icon = folium.Icon(color='lightgray', icon='circle', prefix='fa')
            label = None

        popup_lines = [
            f"<b>Site {site_id}</b> - {site_name}",
            f"Predicted flow ({hour_label}): <b>{flow_here:,}</b> veh/hr",
        ]
        if label:
            popup_lines.insert(0, f"<b>{label}</b>")

        popup_html = "<div style='font-family:Arial,sans-serif; font-size:12px;'>" \
                      + "<br>".join(popup_lines) + "</div>"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Site {site_id} \u2014 {site_name}"
                    + (f" | {label}" if label else ""),
            icon=icon,
        ).add_to(m)

        # Numbered DivIcon overlay for sites on the route, so the order of
        # travel is visible directly on the map without opening a popup.
        if site_id in path_order:
            number_html = f"""<div style="
                background-color:#2980b9; color:white; font-weight:bold;
                font-family:Arial,sans-serif; font-size:11px;
                border-radius:50%; width:20px; height:20px;
                display:flex; align-items:center; justify-content:center;
                border:2px solid white; box-shadow:1px 1px 3px rgba(0,0,0,0.5);
            ">{path_order[site_id]}</div>"""

            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html=number_html, icon_size=(24, 24), icon_anchor=(28, 28)),
            ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────────
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px; border-radius:8px;
                border:2px solid grey; font-size:12px;">
      <b>Route @ {hour_label}</b><br>
      <span style="color:#2980b9;">\u2501\u2501</span> Best path (A*)<br>
      <span style="color:#c0392b;">\u2501\u2501</span> High flow (other roads)<br>
      <span style="color:#e67e22;">\u2501\u2501</span> Medium flow<br>
      <span style="color:#999999;">\u2501\u2501</span> Low flow / unused<br>
      <br>
      \U0001F7E2 Origin &nbsp; \U0001F534 Destination &nbsp; \U0001F535 Stop on route<br>
      <br><i>Hover edges for flow/speed/time \u00b7 Click markers for details</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, output_path


# ── Save + open in browser ───────────────────────────────────────────────────────
def save_and_open_map(m, output_path):
    """
    Saves the given Folium map to `output_path` and opens it in the
    default web browser, mirroring the behaviour of predict_map.py.
    """
    m.save(output_path)
    abs_path = os.path.abspath(output_path)
    webbrowser.open(f'file://{abs_path}')
    return abs_path