"""
route_finder.py
================
Builds a directed graph from the road CONNECTIONS list, converts each
edge's predicted flow into a travel time (using the same speed/time
formula as the map renderer), and finds the fastest route between two
SCATS sites using the A* search algorithm.

Because predicted flow is a per-SITE quantity (not per-edge), each
undirected road connection is expanded into TWO directed edges — one
for travelling A -> B (using site A's predicted flow) and one for
travelling B -> A (using site B's predicted flow). This matches the
convention already used in build_map(), where an edge's colour/label is
based on the flow of its origin site.

A* extends Dijkstra by adding a heuristic estimate h(n) of the
remaining travel time from each node to the destination, guiding the
search towards the goal. The heuristic used here is the great-circle
(haversine) distance from a node to the destination, converted to
minutes assuming travel at the maximum possible speed (60 km/h). Since
no real road segment can be travelled faster than 60 km/h, this
heuristic never OVERESTIMATES the true remaining time — i.e. it is
admissible — which guarantees A* still finds the optimal (fastest)
route, while typically exploring fewer nodes than plain Dijkstra.
"""

import heapq
import math

from config import CONNECTIONS, COORDS

# Travel can never exceed this speed (km/h) under the assignment's
# speed/flow formula, so it's used as the basis for an admissible
# A* heuristic (the fastest theoretically possible travel time).
MAX_SPEED_KMH = 60.0


# ── 1. Speed & travel-time formula (from the assignment specification) ─────────
def compute_speed_and_time(distance_km, flow):
    """
    Given a road segment's distance (km) and the predicted flow
    (vehicles/hour) at its origin site, compute:
      - speed_kmh : travel speed in km/h, capped at 60 km/h
      - travel_min: travel time in minutes, including a fixed
                     0.5 minute (30 second) junction delay

    Formula:
        speed = (93.75 + sqrt(8791.015625 - 5.85935 * flow)) / 2.929675
        travel_time = distance / speed * 60 + 0.5
    """
    discriminant = max(0.0, 8791.015625 - 5.85935 * flow)
    speed_kmh = min((93.75 + discriminant ** 0.5) / 2.929675, 60.0)
    travel_min = (distance_km / speed_kmh) * 60.0 + 0.5
    return speed_kmh, travel_min


# ── 2. A* heuristic: estimated remaining travel time to the destination ────────
def haversine_km(coord_a, coord_b):
    """
    Great-circle distance (in km) between two (lat, lon) coordinates,
    using the haversine formula.
    """
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    earth_radius_km = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


def heuristic_time(site_id, destination):
    """
    Admissible A* heuristic h(n): a lower-bound estimate (in minutes)
    of the travel time remaining from `site_id` to `destination`.

    Computed as the straight-line (haversine) distance between the two
    sites, divided by the fastest possible travel speed (60 km/h).
    Because no real road segment can be faster than 60 km/h, this value
    can never exceed the true remaining travel time, so A* using this
    heuristic is guaranteed to find the optimal (minimum-time) route.
    """
    if site_id not in COORDS or destination not in COORDS:
        return 0.0

    straight_line_km = haversine_km(COORDS[site_id], COORDS[destination])
    return (straight_line_km / MAX_SPEED_KMH) * 60.0


# ── 3. Build a directed graph: node -> [(neighbour, distance_km, flow), ...] ───
def build_directed_graph(site_flows):
    """
    Expand the undirected CONNECTIONS list into a directed adjacency
    list. Each direction of travel carries the predicted flow of the
    site being LEFT (the "origin" of that directed edge), which is what
    determines the speed/travel time for that leg of the journey.

    site_flows: dict {site_id: predicted_flow_at_target_hour}
    """
    graph = {}
    for site_a, site_b, distance_km in CONNECTIONS:
        flow_a = site_flows.get(site_a, 0)
        flow_b = site_flows.get(site_b, 0)

        graph.setdefault(site_a, []).append((site_b, distance_km, flow_a))
        graph.setdefault(site_b, []).append((site_a, distance_km, flow_b))

    return graph

# ── 4. A* search: find the fastest route by estimated total travel time ────────
def find_best_route(origin, destination, site_flows):
    """
    Find the route from `origin` to `destination` that minimises total
    travel time (including the 0.5 min junction delay on every leg),
    using the A* search algorithm.

    For each node n, A* prioritises:
        f(n) = g(n) + h(n)
    where:
        g(n) = actual travel time so far, from origin to n
        h(n) = heuristic_time(n, destination) — admissible lower-bound
               estimate of the remaining travel time (see above)

    Returns a tuple (path, edges, total_time):
      - path      : list of site IDs from origin to destination
                     (inclusive), e.g. [2000, 3682, 3127, ...]
      - edges     : list of (from_site, to_site, flow, travel_min) for
                     each leg of the path, in order
      - total_time: total travel time in minutes (float)

    Returns (None, None, None) if no route exists between the two sites
    (e.g. if either site ID is not part of the road network).
    """
    if origin not in site_flows or destination not in site_flows:
        return None, None, None

    graph = build_directed_graph(site_flows)

    # g_score[n] = best known actual travel time from origin to n
    g_score = {origin: 0.0}
    previous_node = {}
    edge_used = {}            # (u, v) -> (flow, travel_min) for the edge taken
    visited = set()

    # Priority queue of (f_score, node). A tie-break counter isn't
    # needed here because node IDs (ints) are always comparable.
    start_f = heuristic_time(origin, destination)
    open_set = [(start_f, origin)]

    while open_set:
        _, u = heapq.heappop(open_set)

        if u in visited:
            continue
        visited.add(u)

        if u == destination:
            break

        for v, distance_km, flow in graph.get(u, []):
            _, travel_min = compute_speed_and_time(distance_km, flow)
            tentative_g = g_score[u] + travel_min

            if v not in g_score or tentative_g < g_score[v]:
                g_score[v] = tentative_g
                previous_node[v] = u
                edge_used[(u, v)] = (flow, travel_min)
                f_score = tentative_g + heuristic_time(v, destination)
                heapq.heappush(open_set, (f_score, v))

    if destination not in g_score:
        # No path connects origin and destination
        return None, None, None

    # ── Reconstruct the path by walking backwards from destination ─────────────
    path = [destination]
    node = destination
    while node != origin:
        node = previous_node[node]
        path.append(node)
    path.reverse()

    # ── Build the ordered list of edges (with flow + travel time) ──────────────
    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        flow, travel_min = edge_used[(u, v)]
        edges.append((u, v, flow, travel_min))

    return path, edges, g_score[destination]

def find_best_route_with_blocked_edges(origin, destination, site_flows, blocked_edges):
    if origin not in site_flows or destination not in site_flows:
        return None, None, None

    graph = build_directed_graph(site_flows)

    # remove blocked edges
    for u, v in blocked_edges:
        if u in graph:
            graph[u] = [
                edge for edge in graph[u]
                if edge[0] != v
            ]

    g_score = {origin: 0.0}
    previous_node = {}
    edge_used = {}
    visited = set()

    start_f = heuristic_time(origin, destination)
    open_set = [(start_f, origin)]

    while open_set:
        _, u = heapq.heappop(open_set)

        if u in visited:
            continue
        visited.add(u)

        if u == destination:
            break

        for v, distance_km, flow in graph.get(u, []):
            _, travel_min = compute_speed_and_time(distance_km, flow)
            tentative_g = g_score[u] + travel_min

            if v not in g_score or tentative_g < g_score[v]:
                g_score[v] = tentative_g
                previous_node[v] = u
                edge_used[(u, v)] = (flow, travel_min)
                f_score = tentative_g + heuristic_time(v, destination)
                heapq.heappush(open_set, (f_score, v))

    if destination not in g_score:
        return None, None, None

    path = [destination]
    node = destination
    while node != origin:
        node = previous_node[node]
        path.append(node)
    path.reverse()

    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        flow, travel_min = edge_used[(u, v)]
        edges.append((u, v, flow, travel_min))

    return path, edges, g_score[destination]


def find_top_3_routes(origin, destination, site_flows):
    """
    Find up to 3 fastest routes using repeated A* searches.

    Returns:
        [
            (path, edges, total_time),
            ...
        ]
    """

    routes = []

    # First route = best route
    path1, edges1, time1 = find_best_route(origin, destination, site_flows)

    if path1 is None:
        return routes

    routes.append((path1, edges1, time1))

    removed_edges = []

    # Generate Route 2 and Route 3
    while len(routes) < 3:

        best_candidate = None

        previous_path = routes[-1][0]

        # Try removing each edge of the previous route
        for i in range(len(previous_path) - 1):

            blocked_edge = (previous_path[i], previous_path[i + 1])

            candidate = find_best_route_with_blocked_edges(
                origin,
                destination,
                site_flows,
                removed_edges + [blocked_edge]
            )

            if candidate[0] is None:
                continue

            path, edges, total_time = candidate

            # Avoid duplicate paths
            duplicate = False
            for p, _, _ in routes:
                if p == path:
                    duplicate = True
                    break

            if duplicate:
                continue

            if best_candidate is None or total_time < best_candidate[2]:
                best_candidate = candidate
                best_blocked_edge = blocked_edge

        if best_candidate is None:
            break

        routes.append(best_candidate)
        removed_edges.append(best_blocked_edge)

    return routes