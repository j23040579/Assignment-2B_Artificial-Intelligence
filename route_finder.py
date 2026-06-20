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


# ── 4. Core A* search (supports excluding nodes/edges, for Yen's algorithm) ────
def a_star_search(graph, origin, destination, removed_edges=None, removed_nodes=None):
    """
    Run A* from `origin` to `destination` on a pre-built directed graph
    (see build_directed_graph).

    Optional exclusions (used by find_k_best_routes to find ALTERNATIVE
    routes via Yen's algorithm):
      - removed_edges : set of (u, v) directed edges that may not be used
      - removed_nodes : set of site IDs that may not be VISITED (the
                          origin itself is exempt, even if it appears here)

    Returns (path, edges, total_time) — same shape as find_best_route —
    or (None, None, None) if no route exists under these constraints.
    """
    removed_edges = removed_edges or set()
    removed_nodes = removed_nodes or set()

    if destination in removed_nodes:
        return None, None, None

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
            # Skip nodes that Yen's algorithm has excluded for this
            # spur search (but never skip the destination itself).
            if v in removed_nodes and v != destination:
                continue
            if (u, v) in removed_edges:
                continue

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


# ── 5. Single fastest route (thin wrapper around a_star_search) ────────────────
def find_best_route(origin, destination, site_flows):
    """
    Find the single fastest route from `origin` to `destination`,
    minimising total travel time (including the 0.5 min junction delay
    on every leg), using A* search.

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
    return a_star_search(graph, origin, destination)


# ── 6. Helpers used by Yen's algorithm to cost an arbitrary node sequence ──────
def edge_lookup(graph, u, v):
    """
    Returns (distance_km, flow) for the directed edge u -> v, or None if
    that edge doesn't exist in the graph.
    """
    for neighbour, distance_km, flow in graph.get(u, []):
        if neighbour == v:
            return distance_km, flow
    return None


def path_edges_and_cost(path, graph):
    """
    Given an arbitrary sequence of site IDs (`path`) that forms a valid
    walk through `graph`, compute the (edges, total_time) for it — the
    same shape returned by find_best_route's `edges` and `total_time`.

    Used by find_k_best_routes to cost the "root path" portion of each
    candidate route (the part shared with a previously-found route).
    """
    edges = []
    total_time = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        distance_km, flow = edge_lookup(graph, u, v)
        _, travel_min = compute_speed_and_time(distance_km, flow)
        edges.append((u, v, flow, travel_min))
        total_time += travel_min
    return edges, total_time


# ── 7. K best routes, via Yen's algorithm (built on top of A*) ─────────────────
def find_k_best_routes(origin, destination, site_flows, k=3):
    """
    Find up to `k` distinct fastest routes from `origin` to `destination`,
    ranked by total travel time, using Yen's k-shortest-paths algorithm
    with the A* search above as its underlying shortest-path subroutine.

    How it works (Yen's algorithm, in brief):
      1. Find the single best route A[0] with A* (as in find_best_route).
      2. To find the next-best route, take each node along the previous
         best route as a "spur node". Temporarily remove the edge(s)
         that previously-found routes used to leave that spur node
         (and remove the earlier nodes on the route, to avoid loops),
         then re-run A* from the spur node to the destination. Combine
         the unchanged "root path" up to the spur node with this new
         "spur path" to form a candidate route.
      3. Among all candidates generated this way, keep the single
         cheapest one as A[1], and repeat for A[2], etc.

    Returns a list of up to k tuples (path, edges, total_time), sorted
    from fastest (A[0], the same result as find_best_route) to slowest.
    Returns an empty list if no route exists at all.
    """
    if origin not in site_flows or destination not in site_flows:
        return []

    graph = build_directed_graph(site_flows)

    first = a_star_search(graph, origin, destination)
    if first[0] is None:
        return []

    A = [first]          # confirmed k-best routes so far, in order
    B = []                # candidate routes not yet confirmed

    for _ in range(1, k):
        prev_path = A[-1][0]

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]

            # Don't reuse the same edge out of the spur node that any
            # already-confirmed route with this same root path used.
            removed_edges = set()
            for confirmed_path, _, _ in A:
                if confirmed_path[:i + 1] == root_path and len(confirmed_path) > i + 1:
                    removed_edges.add((confirmed_path[i], confirmed_path[i + 1]))

            # Don't revisit any node already used earlier in the root
            # path (prevents the spur path looping back on itself).
            removed_nodes = set(root_path[:-1])

            spur_path, spur_edges, spur_time = a_star_search(
                graph, spur_node, destination,
                removed_edges=removed_edges,
                removed_nodes=removed_nodes,
            )

            if spur_path is None:
                continue

            total_path = root_path[:-1] + spur_path

            # Skip candidates that duplicate a route we already have.
            if total_path in [p for p, _, _ in A] or total_path in [p for p, _, _ in B]:
                continue

            root_edges, root_time = path_edges_and_cost(root_path, graph)
            total_edges = root_edges + spur_edges
            total_time = root_time + spur_time

            B.append((total_path, total_edges, total_time))

        if not B:
            break

        # Promote the cheapest remaining candidate to a confirmed route.
        B.sort(key=lambda candidate: candidate[2])
        A.append(B.pop(0))

    return A


# ── 8. Convenience wrapper: top 3 routes (used by A2B.py / route_map.py) ───────
def find_top_3_routes(origin, destination, site_flows):
    """
    Thin wrapper around find_k_best_routes() fixed at k=3.

    This is the name imported by A2B.py and route_map.py, since the
    console output and the interactive map both display exactly the
    top 3 candidate routes (Best / 2nd Best / 3rd Best).

    Returns a list of 1-3 tuples (path, edges, total_time), sorted from
    fastest to slowest. Returns fewer than 3 entries if the network
    doesn't have that many distinct loopless routes between the two
    sites. Returns an empty list if no route exists at all.
    """
    return find_k_best_routes(origin, destination, site_flows, k=3)