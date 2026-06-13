import folium
import webbrowser
import os


# ── Hardcoded from map.txt ────────────────────────────────────────────────────
COORDS = {
    2000: (-37.851923, 145.094324),
    2820: (-37.794785, 145.030465),
    2825: (-37.786610, 145.062020),
    2827: (-37.781739, 145.077331),
    3002: (-37.815162, 145.026572),
    3120: (-37.822895, 145.057288),
    3127: (-37.825227, 145.077947),
    3180: (-37.796213, 145.083507),
    3662: (-37.808953, 145.027457),
    3682: (-37.837475, 145.096868),
    4032: (-37.802297, 145.061230),
    4043: (-37.847160, 145.052627),
    4051: (-37.794143, 145.069333),
    4057: (-37.804969, 145.081760),
    4263: (-37.823056, 145.024958),
    4266: (-37.825268, 145.043338),
    4270: (-37.830228, 145.032815),
    4321: (-37.800838, 145.049119),
}

CONNECTIONS = [
    (2827, 4051, 1.85),
    (4051, 3180, 1.57),
    (3180, 4057, 1.07),
    (4057, 4032, 2.47),
    (4057, 3127, 2.53),
    (3127, 3682, 3.57),
    (3120, 3127, 2.62),
    (2000, 3682, 1.75),
    (2000, 4043, 4.21),
    (4043, 3120, 2.86),
    (3120, 4032, 2.45),
    (3120, 4266, 1.87),
    (4043, 4270, 3.55),
    (4266, 4270, 1.67),
    (4270, 4263, 1.49),
    (4266, 4263, 2.14),
    (4266, 3002, 2.72),
    (4263, 3002, 1.15),
    (3002, 3662, 0.99),
    (3662, 4321, 2.38),
    (3662, 2820, 1.69),
    (2820, 4321, 3.54),
    (4321, 4032, 1.64),
    (4032, 2825, 1.96),
]

SCATS_NAMES = {
    2000: "WARRIGAL_RD / TOORAK_RD",
    2820: "TOORAK_RD / KOOYONG_RD",
    2825: "TOORAK_RD / BURKE_RD",
    2827: "TOORAK_RD / GLENFERRIE_RD",
    3002: "DENMARK_ST / BARKERS_RD",
    3120: "BURKE_RD / COTHAM_RD",
    3127: "CAMBERWELL_RD / RIVERSDALE_RD",
    3180: "BURKE_RD / BARKERS_RD",
    3662: "HIGH_ST / BURKE_RD",
    3682: "BURKE_RD / HIGH_ST",
    4032: "WHITEHORSE_RD / RATHMINES_RD",
    4043: "CANTERBURY_RD / MONT_ALBERT_RD",
    4051: "WHITEHORSE_RD / SPRINGVALE_RD",
    4057: "CANTERBURY_RD / BALWYN_RD",
    4263: "DONCASTER_RD / ELGAR_RD",
    4266: "AUBURN_RD / BURWOOD_RD",
    4270: "BURWOOD_RD / HIGHFIELD_RD",
    4321: "TOORAK_RD / AUBURN_RD",
}

ROUTE_COLOURS = {
    1: {'color': 'blue',    'label': 'Route 1 (Best)'},
    2: {'color': 'green',   'label': 'Route 2'},
    3: {'color': '#FFD700', 'label': 'Route 3'},
}


def build_map(coords, connections, routes=None, origin=None, destination=None):
    centre_lat = sum(lat for lat, lon in coords.values()) / len(coords)
    centre_lon = sum(lon for lat, lon in coords.values()) / len(coords)

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    # ── Draw road connections ─────────────────────────────────────────────────
    for site_a, site_b, distance in connections:
        if site_a in coords and site_b in coords:
            folium.PolyLine(
                locations=[coords[site_a], coords[site_b]],
                color='#555555',   # dark grey — visible over basemap
                weight=5,
                opacity=0.9,
                tooltip=f"{site_a} → {site_b}: {distance} km"
            ).add_to(m)

    # ── Draw routes ───────────────────────────────────────────────────────────
    if routes:
        for i, route in enumerate(routes, 1):
            if i > 3:
                break
            route_coords = [coords[site] for site in route if site in coords]
            if len(route_coords) >= 2:
                colour = ROUTE_COLOURS[i]['color']
                label  = ROUTE_COLOURS[i]['label']
                folium.PolyLine(
                    locations=route_coords,
                    color=colour,
                    weight=6,
                    opacity=0.9,
                    tooltip=label
                ).add_to(m)

    # ── Markers ───────────────────────────────────────────────────────────────
    route_sites = set()
    if routes:
        for route in routes:
            route_sites.update(route)

    for site_id, (lat, lon) in coords.items():
        name = SCATS_NAMES.get(site_id, f"Site {site_id}")

        if site_id == origin:
            color, icon = 'red',   'play'
        elif site_id == destination:
            color, icon = 'blue',  'stop'
        elif site_id in route_sites:
            color, icon = 'white', 'info-sign'
        else:
            color, icon = 'gray',  'map-marker'

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>Site {site_id}</b><br>{name}", max_width=200),
            tooltip=f"Site {site_id} — {name}",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background-color: white; padding: 12px; border-radius: 8px;
                border: 2px solid grey; font-size: 13px;">
        <b>Legend</b><br>
        <span style="color:red;">●</span> Origin<br>
        <span style="color:blue;">●</span> Destination<br>
        <span style="color:blue;">━━</span> Route 1 (Best)<br>
        <span style="color:green;">━━</span> Route 2<br>
        <span style="color:#FFD700;">━━</span> Route 3<br>
        <span style="color:#555555;">━━</span> Road connections
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def show_map(coords, connections, routes=None, origin=None, destination=None):
    m = build_map(coords, connections, routes, origin, destination)
    output_path = 'boroondara_map.html'
    m.save(output_path)
    abs_path = os.path.abspath(output_path)
    webbrowser.open(f'file://{abs_path}')
    print(f"Map saved and opened: {output_path}")


if __name__ == '__main__':
    print(f"Loaded {len(COORDS)} sites and {len(CONNECTIONS)} connections")
    print("Generating Boroondara traffic map...")

    dummy_routes = [
        [2000, 3682, 3127, 3002],
        [2000, 3682, 4032, 3120, 3002],
        [2000, 4043, 3120, 3002],
    ]

    show_map(
        COORDS,
        CONNECTIONS,
        routes=dummy_routes,
        origin=2000,
        destination=3002
    )