"""
route_map.py
============
Renders a Folium map for A2B.py: the full Boroondara road network is
drawn in grey/orange/red (coloured by predicted flow, same scheme as
predict_map.py), and up to THREE candidate routes found by A* are
highlighted as thick coloured lines (blue / green / pink) with numbered
markers showing the order of travel.

Where two or more routes share the same physical road segment, each
route's line is offset sideways (perpendicular to the road) by a small
amount so all of them stay visible side-by-side instead of overlapping.

A button group in the title banner lets the viewer switch which route's
summary is shown up top; clicking a button also highlights that route's
line (full opacity / extra weight) while dimming the other two.

This module only needs the SINGLE-HOUR site flows already computed by
A2B.py (via model_utils.predict_site_flows) — it does NOT require the
full 24-hour predict_all() results, so it stays fast even though A2B.py
runs a much lighter prediction pass than predict_map.py.
"""

import json
import math
import os
import webbrowser

import folium

from config import COORDS, CONNECTIONS, SITE_NAMES, HOUR_LABELS
from route_finder import compute_speed_and_time

# Sideways offset applied to each route's polyline, in degrees of
# latitude/longitude, so that routes sharing a road segment are drawn
# side-by-side rather than on top of one another. Tuned for the
# Boroondara network's scale (~city-block distances); adjust if the
# offset looks too large/small at your default zoom level.
ROUTE_OFFSET_DEG = 0.00018

# Per-route visual identity: CSS colour, JS-side line id, sideways
# offset direction (in multiples of ROUTE_OFFSET_DEG), and the leaflet
# pane each route's lines live in (higher panes draw on top).
ROUTE_STYLES = [
    {"color": "#2980b9", "id": "route1", "offset_mult": 0, "label": "Best Route"},
    {"color": "#00b81c", "id": "route2", "offset_mult": -1, "label": "2nd Best Route"},
    {"color": "#ff009d", "id": "route3", "offset_mult": 1, "label": "3rd Best Route"},
]


# ── Geometry helpers ────────────────────────────────────────────────────────
def _offset_segment(coord_a, coord_b, offset_mult):
    """
    Shift the segment (coord_a -> coord_b) sideways, perpendicular to its
    own direction, by `offset_mult * ROUTE_OFFSET_DEG`. Returns the two
    shifted (lat, lon) endpoints.

    Used so that when two or more routes travel along the same physical
    road, their drawn lines fan out side-by-side instead of perfectly
    overlapping (where only the topmost colour would ever be visible).
    """
    if offset_mult == 0:
        return coord_a, coord_b

    lat_a, lon_a = coord_a
    lat_b, lon_b = coord_b

    d_lat = lat_b - lat_a
    d_lon = lon_b - lon_a
    length = math.hypot(d_lat, d_lon)
    if length == 0:
        return coord_a, coord_b

    # Unit vector perpendicular to the segment's direction.
    perp_lat = -d_lon / length
    perp_lon = d_lat / length

    shift = offset_mult * ROUTE_OFFSET_DEG
    shifted_a = (lat_a + perp_lat * shift, lon_a + perp_lon * shift)
    shifted_b = (lat_b + perp_lat * shift, lon_b + perp_lon * shift)
    return shifted_a, shifted_b


# ── Build the route map ──────────────────────────────────────────────────────────
def build_route_map(origin, destination, path, edges, site_flows, hour, total_time,
                     route_number=1, extra_routes=None):
    """
    Build a Folium map showing:
      - The full road network, coloured by predicted flow at `hour`
        (grey = low, orange = medium, red = high — same thresholds as
        predict_map.py).
      - Up to three candidate routes (best / 2nd best / 3rd best),
        each drawn as its own thick coloured line with numbered stop
        markers. Shared road segments are offset sideways so all
        routes stay visible.
      - A button group in the title banner to switch which route's
        summary is shown, and to highlight that route's line.
      - The origin marked in green, the destination marked in red.

    Parameters
    ----------
    origin, destination : int
        Site IDs for the start and end of the journey.
    path : list[int]
        Ordered list of site IDs from origin to destination for the
        BEST route (as returned by route_finder.find_best_route).
    edges : list[tuple]
        List of (from_site, to_site, flow, travel_min) for each leg of
        the best route (as returned by route_finder.find_best_route).
    site_flows : dict[int, int]
        Predicted flow (vehicles/hour) for every site, at `hour`.
    hour : int
        Hour index (0-23) the prediction was made for.
    total_time : float
        Total travel time for the best route, in minutes.
    extra_routes : list[tuple] | None
        Optional list of up to 2 additional (path, edges, total_time)
        tuples (route 2 and route 3) to overlay on the same map.

    Returns
    -------
    (m, output_path) : (folium.Map, str)
        The built map, and the filename it should be saved to.
    """
    if extra_routes is None:
        extra_routes = []

    hour_label = HOUR_LABELS[hour]

    output_path = (
        f'route{route_number}_'
        f'{origin}_to_{destination}_'
        f'{hour_label}.html'
    )

    # All candidate routes, in rank order: [(path, edges, total_time), ...]
    all_routes = [(path, edges, total_time)] + list(extra_routes)

    # ── Base map, centred on the network ────────────────────────────────────────
    centre_lat = sum(lat for lat, _ in COORDS.values()) / len(COORDS)
    centre_lon = sum(lon for _, lon in COORDS.values()) / len(COORDS)
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    # ── Pre-compute flow-based colour thresholds for background roads ─────────────
    # (same red/orange/grey scheme as predict_map.py, based on this hour's flow)
    edge_flows = []
    for site_a, site_b, _ in CONNECTIONS:
        flow_a = site_flows.get(site_a, 0)
        flow_b = site_flows.get(site_b, 0)
        edge_flows.append((flow_a + flow_b) // 2)

    max_edge_flow = max(edge_flows) if edge_flows else 1
    high_thresh = max_edge_flow * 0.66
    medium_thresh = max_edge_flow * 0.33

    # Distance lookup so route segments can recover their road length
    # without re-scanning CONNECTIONS for every edge of every route.
    distance_lookup = {}
    for site_a, site_b, distance in CONNECTIONS:
        distance_lookup[(site_a, site_b)] = distance
        distance_lookup[(site_b, site_a)] = distance

    # ── Layer 1: the full road network background (always faint/thin) ──────────────
    for (site_a, site_b, distance), avg_flow in zip(CONNECTIONS, edge_flows):
        if site_a not in COORDS or site_b not in COORDS:
            continue

        if avg_flow >= high_thresh:
            color = '#c0392b'   # red - high flow
        elif avg_flow >= medium_thresh:
            color = '#e67e22'   # orange - medium flow
        else:
            color = '#999999'   # grey - low flow / not on route

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
            weight=3,
            opacity=0.5,
            tooltip=tooltip_text,
        ).add_to(m)

        # ── Edge label at midpoint for background roads ──────────────────────────
        lat_a, lon_a = COORDS[site_a]
        lat_b, lon_b = COORDS[site_b]
        mid_lat = (lat_a + lat_b) / 2
        mid_lon = (lon_a + lon_b) / 2

        bg_label_html = f"""<div style="
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
            opacity: 0.75;
        ">Vehicle: {avg_flow:,}<br>Speed: {speed_kmh:.1f} km/h<br>Travel time: {travel_min:.1f} min</div>"""

        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                html=bg_label_html,
                icon_size=(86, 60),
                icon_anchor=(43, 30),
            ),
            tooltip=tooltip_text,
        ).add_to(m)

    # ── Layer 2: each candidate route, drawn with its own sideways offset ──────────
    # route_step_info[route_idx][site_id] = step number (1-based) on that route
    route_step_info = [
        {site_id: i + 1 for i, site_id in enumerate(r_path)}
        for r_path, _, _ in all_routes
    ]

    for route_idx, (r_path, r_edges, r_total_time) in enumerate(all_routes):
        if route_idx >= len(ROUTE_STYLES):
            break  # Only 3 visual styles are defined.

        style = ROUTE_STYLES[route_idx]
        color = style["color"]
        line_id = style["id"]
        offset_mult = style["offset_mult"]

        # Lookup of this route's own edges -> (flow, travel_min).
        r_edge_info = {}
        for from_site, to_site, flow, travel_min in r_edges:
            r_edge_info[(from_site, to_site)] = (flow, travel_min)
            r_edge_info[(to_site, from_site)] = (flow, travel_min)

        for i in range(len(r_path) - 1):
            site_a, site_b = r_path[i], r_path[i + 1]
            if site_a not in COORDS or site_b not in COORDS:
                continue

            flow, travel_min = r_edge_info.get((site_a, site_b), (0, 0))
            distance = distance_lookup.get((site_a, site_b), 0.0)
            speed_kmh, _ = compute_speed_and_time(distance, flow)

            coord_a, coord_b = _offset_segment(
                COORDS[site_a], COORDS[site_b], offset_mult
            )

            tooltip_text = (
                f"{style['label']}: Site {site_a} \u2194 Site {site_b} | "
                f"Flow ({hour_label}): {flow:,} veh/hr | "
                f"Speed: {speed_kmh:.1f} km/h | "
                f"Travel time: {travel_min:.1f} min (inc. 0.5 min delay)"
            )

            line = folium.PolyLine(
                locations=[coord_a, coord_b],
                color=color,
                weight=7,
                opacity=0.95,
                tooltip=tooltip_text,
                class_name=f"route-line {line_id}",
            )
            line.add_to(m)

            # ── Edge label at the (offset) midpoint ──────────────────────────────
            mid_lat = (coord_a[0] + coord_b[0]) / 2
            mid_lon = (coord_a[1] + coord_b[1]) / 2

            label_html = f"""<div class="route-label {line_id}" style="
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
            ">Vehicle: {flow:,}<br>Speed: {speed_kmh:.1f} km/h<br>Travel time: {travel_min:.1f} min</div>"""

            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=label_html,
                    icon_size=(86, 60),
                    icon_anchor=(43, 30),
                ),
                tooltip=tooltip_text,
            ).add_to(m)

        # ── Numbered stop markers for this route (also offset where the
        # site itself doesn't coincide with the origin/destination) ──────────────
        for i, site_id in enumerate(r_path):
            if site_id not in COORDS:
                continue
            if site_id == origin or site_id == destination:
                continue  # origin/destination get their own marker below

            lat, lon = COORDS[site_id]
            number_html = f"""
            <div class="route-stop {line_id}" style="
                background-color:{color}; 
                color:white; 
                font-weight:bold;
                font-family:Arial,sans-serif; 
                font-size:11px;
                border-radius:50%; 
                width:20px; 
                height:20px;
                display:flex; 
                align-items:center; 
                justify-content:center;
                border:2px solid white; 
                box-shadow:1px 1px 3px rgba(0,0,0,0.5);
            ">
            {i + 1}
            </div>"""

            site_name = SITE_NAMES.get(site_id, f"Site {site_id}")
            flow_here = site_flows.get(site_id, 0)
            popup_html = (
                "<div style='font-family:Arial,sans-serif; font-size:12px;'>"
                f"<b>Stop {i + 1} of {len(r_path)} ({style['label']})</b><br>"
                f"<b>Site {site_id}</b> - {site_name}<br>"
                f"Predicted flow ({hour_label}): <b>{flow_here:,}</b> veh/hr"
                "</div>"
            )

            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html=number_html, icon_size=(24, 24), icon_anchor=(28, 28)),
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"Site {site_id} \u2014 {site_name} | Stop {i + 1} ({style['label']})",
            ).add_to(m)

    # ── Layer 3: plain markers for every site not already covered above ────────────
    covered_site_ids = {origin, destination}
    for r_path, _, _ in all_routes:
        covered_site_ids.update(r_path)

    for site_id, (lat, lon) in COORDS.items():
        if site_id in covered_site_ids and site_id not in (origin, destination):
            continue  # already drawn as a numbered stop marker above

        site_name = SITE_NAMES.get(site_id, f"Site {site_id}")
        flow_here = site_flows.get(site_id, 0)

        if site_id == origin:
            icon = folium.Icon(color='green', icon='play', prefix='fa')
            label = "ORIGIN"
        elif site_id == destination:
            icon = folium.Icon(color='red', icon='flag-checkered', prefix='fa')
            label = "DESTINATION"
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

    # ── Title banner with route-switch buttons ──────────────────────────────────
    # Data for each route, passed to the front-end JS so buttons can swap the
    # summary text and highlight the matching line/markers.
    route_summaries = []
    for idx, (r_path, _, r_total_time) in enumerate(all_routes):
        if idx >= len(ROUTE_STYLES):
            break
        style = ROUTE_STYLES[idx]
        route_str = " \u2192 ".join(str(s) for s in r_path)
        eta_total_minutes = hour * 60 + int(r_total_time)
        eta_hour = (eta_total_minutes // 60) % 24
        eta_min = eta_total_minutes % 60
        eta_period = "AM" if eta_hour < 12 else "PM"
        eta_hour_12 = eta_hour % 12 or 12
        eta_str = f"{eta_hour_12}:{eta_min:02d} {eta_period}"

        route_summaries.append({
            "id": style["id"],
            "label": style["label"],
            "color": style["color"],
            "text": f"{style['label']} ({hour_label}): {route_str}",
            "time_text": f"Total Driving Time: {r_total_time:.1f} min | ETA: {eta_str}",
        })

    button_html_parts = []
    for i, summary in enumerate(route_summaries):
        active_class = "active" if i == 0 else ""
        button_html_parts.append(
            f"""<button class="route-switch-btn {active_class}"
                        data-route-id="{summary['id']}"
                        style="background:{summary['color']};"
                        onclick="window.__switchRoute('{summary['id']}')">
                {summary['label'].replace('Route', '').strip() or summary['label']}
            </button>"""
        )
    buttons_html = "".join(button_html_parts)

    title_html = f"""
    <div id="route-title-banner" style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:1000; background:white; padding:10px 18px;
                border-radius:8px; border:2px solid #555;
                font-family:Arial,sans-serif; font-size:13px; font-weight:bold;
                text-align:center; max-width:90%;">
        <div id="route-switch-buttons" style="margin-bottom:6px;">
            {buttons_html}
        </div>
        <div id="route-title-text">{route_summaries[0]['text']}</div>
        <span id="route-time-text" style="font-size:12px; font-weight:normal; color:#555;">
            {route_summaries[0]['time_text']}
        </span>
    </div>

    <style>
        .route-switch-btn {{
            color: white;
            border: none;
            border-radius: 5px;
            padding: 4px 10px;
            margin: 0 3px;
            font-family: Arial, sans-serif;
            font-size: 11px;
            font-weight: bold;
            cursor: pointer;
            opacity: 0.55;
        }}
        .route-switch-btn.active {{
            opacity: 1;
            box-shadow: 0 0 0 2px #333;
        }}
    </style>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # JS: clicking a button updates the banner text and dims/highlights the
    # matching route lines, edge labels, and stop markers by toggling CSS
    # opacity/weight on elements tagged with each route's class name.
    route_data_json = json.dumps(route_summaries)
    script_html = f"""
    <script>
    (function() {{
        var routeData = {route_data_json};

        window.__switchRoute = function(routeId) {{
            var titleEl = document.getElementById('route-title-text');
            var timeEl = document.getElementById('route-time-text');
            var match = routeData.find(function(r) {{ return r.id === routeId; }});
            if (match) {{
                titleEl.innerHTML = match.text;
                timeEl.innerHTML = match.time_text;
            }}

            document.querySelectorAll('.route-switch-btn').forEach(function(btn) {{
                btn.classList.toggle('active', btn.getAttribute('data-route-id') === routeId);
            }});

            // Highlight the chosen route's lines/labels/stops; dim the rest.
            // Leaflet draws each PolyLine as an SVG <path> with the className
            // we gave it (e.g. "route-line route1"). SVG elements need their
            // presentation attributes set via setAttribute (not el.style.*),
            // since stroke-width/opacity are SVG attributes, not CSS props
            // that map cleanly onto element.style in every browser.
            routeData.forEach(function(r) {{
                var isActive = (r.id === routeId);
                document.querySelectorAll('.route-line.' + r.id).forEach(function(el) {{
                    el.setAttribute('opacity', isActive ? 1 : 0.25);
                    el.setAttribute('stroke-width', isActive ? 9 : 4);
                }});
                document.querySelectorAll('.route-label.' + r.id).forEach(function(el) {{
                    el.style.opacity = isActive ? 1 : 0.3;
                }});
                document.querySelectorAll('.route-stop.' + r.id).forEach(function(el) {{
                    el.style.opacity = isActive ? 1 : 0.35;
                }});
            }});
        }};

        // Run once the map's SVG layer exists, so the initial state matches
        // the "Best Route" button being active by default.
        function initHighlight() {{
            if (document.querySelector('.route-line')) {{
                window.__switchRoute(routeData[0].id);
            }} else {{
                setTimeout(initHighlight, 150);
            }}
        }}
        initHighlight();
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(script_html))

    # ── Legend ────────────────────────────────────────────────────────────────────
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px; border-radius:8px;
                border:2px solid grey; font-size:12px;">
      <b>Route @ {hour_label}</b><br>
      <span style="color:#2980b9;">\u2501\u2501</span> Best path (A*)<br>
      <span style="color:#00b81c;">\u2501\u2501</span> 2nd Best path<br>
      <span style="color:#ff009d;">\u2501\u2501</span> 3rd Best path<br>
      <br>
      <span style="color:#c0392b;">\u2501\u2501</span> High flow (other roads)<br>
      <span style="color:#e67e22;">\u2501\u2501</span> Medium flow<br>
      <span style="color:#999999;">\u2501\u2501</span> Low flow / unused<br>
      <br>
      \U0001F7E2 Origin &nbsp; \U0001F534 Destination &nbsp; \U0001F535 Stop on route<br>
      <br><i>Hover lines for flow/speed/time \u00b7 Click buttons above to switch route \u00b7 Click markers for details</i>
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