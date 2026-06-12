import folium
import webbrowser
import os
import re


def load_map_data(filepath='Dataset/boroondara_map.txt'):
    coords = {}
    connections = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match coordinate line: 2000:(-37.851923,145.094324)
            coord_match = re.match(r'(\d+):\((-?\d+\.\d+),(\d+\.\d+)\)', line)
            if coord_match:
                site_id = int(coord_match.group(1))
                lat = float(coord_match.group(2))
                lon = float(coord_match.group(3))
                coords[site_id] = (lat, lon)
                continue

            # Match connection line: 2827,4051,1.85
            conn_match = re.match(r'(\d+),(\d+),([\d.]+)', line)
            if conn_match:
                site_a = int(conn_match.group(1))
                site_b = int(conn_match.group(2))
                distance = float(conn_match.group(3))
                connections.append((site_a, site_b, distance))

    print(f"Loaded {len(coords)} sites and {len(connections)} connections")
    return coords, connections


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

# Route colours
ROUTE_COLOURS = {
    1: {'color': 'blue',    'label': 'Route 1 (Best)'},
    2: {'color': 'green',   'label': 'Route 2'},
    3: {'color': '#FFD700', 'label': 'Route 3'},  # gold yellow
}


def build_map(coords, connections, routes=None, origin=None, destination=None):
    """
    Build a Folium map with SCATS sites and connections.

    Parameters:
        coords      : dict {site_id: (lat, lon)}
        connections : list of (site_a, site_b, distance_km)
        routes      : list of routes, each route is a list of site_ids
                      e.g. [[2000, 3682, 3002], [2000, 4043, 3002]]
        origin      : origin site_id (optional)
        destination : destination site_id (optional)

    Returns:
        folium.Map object
    """
    # Centre map on Boroondara
    centre_lat = sum(lat for lat, lon in coords.values()) / len(coords)
    centre_lon = sum(lon for lat, lon in coords.values()) / len(coords)

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    # Draw road connections (grey, slightly bold)
    for site_a, site_b, distance in connections:
        if site_a in coords and site_b in coords:
            folium.PolyLine(
                locations=[coords[site_a], coords[site_b]],
                color='grey',
                weight=3,
                opacity=0.5,
                tooltip=f"{site_a} → {site_b}: {distance} km"
            ).add_to(m)

    # Draw routes with different colours
    if routes:
        for i, route in enumerate(routes, 1):
            if i > 3:
                break
            route_coords = [coords[site] for site in route if site in coords]
            if len(route_coords) >= 2:
                colour = ROUTE_COLOURS[i]['color']
                label = ROUTE_COLOURS[i]['label']
                folium.PolyLine(
                    locations=route_coords,
                    color=colour,
                    weight=6,
                    opacity=0.9,
                    tooltip=label
                ).add_to(m)

    # Collect all sites that are part of any route
    route_sites = set()
    if routes:
        for route in routes:
            route_sites.update(route)

    # Add markers for all SCATS sites
    for site_id, (lat, lon) in coords.items():
        name = SCATS_NAMES.get(site_id, f"Site {site_id}")

        # Choose marker colour
        if site_id == origin:
            color = 'red'
            icon = 'play'
        elif site_id == destination:
            color = 'blue'
            icon = 'stop'
        elif site_id in route_sites:
            color = 'white'
            icon = 'info-sign'
        else:
            color = 'gray'
            icon = 'map-marker'

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>Site {site_id}</b><br>{name}", max_width=200),
            tooltip=f"Site {site_id} — {name}",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)

    # Add legend
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
        <span style="color:grey;">━━</span> Road connections
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def show_map(coords, connections, routes=None, origin=None, destination=None):
    """Build map, save as HTML and open in browser."""
    m = build_map(coords, connections, routes, origin, destination)

    output_path = 'boroondara_map.html'
    m.save(output_path)

    abs_path = os.path.abspath(output_path)
    webbrowser.open(f'file://{abs_path}')
    print(f"Map saved and opened: {output_path}")


if __name__ == '__main__':
    coords, connections = load_map_data()

    print("Generating Boroondara traffic map...")

    # Dummy routes for testing
    # TODO: Replace with real A* routes from Member 4
    dummy_routes = [
        [2000, 3682, 3127, 3002],        # Route 1 (blue)
        [2000, 3682, 4032, 3120, 3002],  # Route 2 (green)
        [2000, 4043, 3120, 3002],        # Route 3 (gold)
    ]

    show_map(
        coords,
        connections,
        routes=dummy_routes,
        origin=2000,
        destination=3002
    )