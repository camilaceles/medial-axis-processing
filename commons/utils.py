import random
import networkx as nx
import trimesh
from pygel3d import hmesh, graph
import numpy as np
from scipy.spatial import KDTree


def smooth(m, max_iter=1):
    pos = m.positions()
    for i in range(0, max_iter):
        new_pos = np.zeros(pos.shape)
        for vertex in m.vertices():
            adjacent_vertices = m.circulate_vertex(vertex, 'v')
            for adj in adjacent_vertices:
                new_pos[vertex] += pos[adj]
            new_pos[vertex] /= len(adjacent_vertices)
        pos[:] = new_pos[:]

    # remove zero area faces after smooth
    for fid in m.faces():
        if m.area(fid) < 1e-6:
            m.remove_face(fid)
    m.cleanup()


def __sample_point_in_face(m: hmesh.Manifold, fid: int):
    vertices = m.circulate_face(fid, mode='v')
    vertices_pos = m.positions()[vertices]
    s, t = sorted([np.random.random(), np.random.random()])
    return s * vertices_pos[0] + (t - s) * vertices_pos[1] + (1 - t) * vertices_pos[2]


def poisson_disk_sampling_on_mesh(m, n):
    face_areas = np.array([m.area(fid) for fid in m.faces()])
    face_probs = face_areas / face_areas.sum()
    total_area = face_areas.sum()
    threshold = np.sqrt(total_area / (np.pi * n))

    face_normals = np.array([m.face_normal(fid) for fid in m.faces()])

    sampled_points = np.zeros((0, 3))  # Initialize an empty array for points
    normals = np.zeros((0, 3))  # Initialize an empty array for normals

    # Optimization - build an array of random indices according to the face probabilities
    random_indices = np.random.choice(len(face_probs), size=10*n, p=face_probs)
    for drawn_fid in random_indices:
        new_point = __sample_point_in_face(m, drawn_fid)
        new_normal = face_normals[drawn_fid]

        # Check if this new point is far enough from all other points using broadcasting
        if sampled_points.size == 0 or np.all(np.linalg.norm(sampled_points - new_point, axis=1) >= threshold):
            sampled_points = np.vstack([sampled_points, new_point])
            normals = np.vstack([normals, new_normal])
            if len(sampled_points) == n:
                break

    return sampled_points, normals


def manifold_to_trimesh(m: hmesh.Manifold, process=False) -> trimesh.Trimesh:
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trim = trimesh.Trimesh(vertices=m.positions(), faces=faces, process=process)
    return trim


def trimesh_to_manifold(trim: trimesh.Trimesh) -> hmesh.Manifold:
    return hmesh.Manifold.from_triangles(trim.vertices, trim.faces)


def average_edge_length(m: hmesh.Manifold):
    edge_lengths = np.array([m.edge_length(hid) for hid in m.halfedges()])
    return np.mean(edge_lengths)


def compute_reconstruction_error(orig: hmesh.Manifold, recon: hmesh.Manifold):
    avg_edge_len = average_edge_length(orig)
    pos1 = orig.positions()
    pos2 = recon.positions()
    lens = np.linalg.norm(pos1 - pos2, axis=1)
    return np.mean(lens) / avg_edge_len


def barycentric_project(m: hmesh.Manifold, points: np.ndarray):
    trim = manifold_to_trimesh(m)
    prox_query = trimesh.proximity.ProximityQuery(trim)
    projected_points, _, face_ids = prox_query.on_surface(points)

    triangles = trim.triangles[face_ids]
    barycentrics = trimesh.triangles.points_to_barycentric(triangles, projected_points)

    return face_ids, barycentrics, projected_points


def project_points_to_curve(points, curve):
    # Prepare curve segments
    A = curve[:-1]  # Starting points of each segment
    B = curve[1:]   # Ending points of each segment

    # Vector from all starts to ends of segments (AB) and from all starts to each point (AP)
    AB = B - A
    A_expanded = A[np.newaxis, :, :]  # Broadcasting A over points
    AB_expanded = AB[np.newaxis, :, :]  # Broadcasting AB over points

    # Calculate AP for each point against each segment start
    AP = points[:, np.newaxis, :] - A_expanded

    # Calculate projection parameters
    AB_squared = np.sum(AB_expanded**2, axis=2)
    AP_dot_AB = np.einsum('ijk,ijk->ij', AP, AB_expanded)
    t = AP_dot_AB / AB_squared

    # Ensure t is within [0, 1] to stay within segment bounds
    t = np.clip(t, 0, 1)

    # Calculate distances from each point to these closest points
    distances = np.linalg.norm(AP - t[:, :, np.newaxis] * AB_expanded, axis=2)

    # Identify the closest segment for each point
    closest_segment_indices = np.argmin(distances, axis=1)
    t_values = t[np.arange(len(points)), closest_segment_indices]

    # Compute the actual closest points based on t_values and closest segments
    projected_points = A[closest_segment_indices] + t_values[:, np.newaxis] * (B[closest_segment_indices] - A[closest_segment_indices])

    return closest_segment_indices, t_values, projected_points


def flatten(xss):
    return [x for xs in xss for x in xs]


def read_obj(filename: str) -> np.ndarray:
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
    return np.array(vertices)


def read_correspondences(filename: str) -> list[list[int]]:
    with open(filename) as f:
        correspondences = [[int(x) for x in line.strip().split(',')] for line in f]
    return correspondences


def read_CA_MA(filename: str) -> hmesh.Manifold:
    mesh = hmesh.load(filename)
    trim = manifold_to_trimesh(mesh, process=True)
    mesh = trimesh_to_manifold(trim)
    mesh.cleanup()
    return trimesh_to_manifold(trim)


def find_minimum_gamma(mesh, inner_points, start, step):
    """Finds minimum addition to radii to cover all points in the mesh with given inner points"""
    tree = KDTree(mesh.positions())
    total_points = len(mesh.positions())
    addition = start
    covered_points = set()

    while len(covered_points) < total_points:
        # print("trying gamma: ", addition)
        R = tree.query(inner_points, k=1)[0] + addition
        covered_points = set(flatten(tree.query_ball_point(inner_points, R)))

        if len(covered_points) >= total_points:
            break
        addition += step

    return addition


def __build_opposite_dict(nested_lists):
    opposite_dict = {}
    for inner_index, outer_list in enumerate(nested_lists):
        for element in outer_list:
            if element in opposite_dict:
                opposite_dict[element].append(inner_index)
            else:
                opposite_dict[element] = [inner_index]
    return opposite_dict


def build_ball_correspondences(
        mesh: hmesh.Manifold,
        inner_points: np.ndarray,
        gamma: float = None,
        start: float = 0.01,
        step: float = 0.01
):
    # Find minimum gamma to cover all surface points and map correspondences from it
    if gamma is None:
        gamma = find_minimum_gamma(mesh, inner_points, start, step)
        print("Chosen minimum gamma:", gamma)

    tree = KDTree(mesh.positions())
    dist, _ = tree.query(inner_points, k=1)
    R = dist + gamma
    correspondences = tree.query_ball_point(inner_points, R)
    # return correspondences

    # Ensure each outer point is only associated to one inner point
    # Choose inner point where inner-outer connection is best aligned with surface normal
    pos = mesh.positions()

    for (outer, inners) in __build_opposite_dict(correspondences).items():
        if len(inners) == 1:
            continue
        best_inner = None
        best_angle = np.inf

        # find the inner point with the best normal alignment
        for inner in inners:
            # calculate normal alignment
            outer_normal = mesh.vertex_normal(outer)
            outer_normal /= np.linalg.norm(outer_normal)
            corr_normal = pos[outer] - inner_points[inner]
            corr_normal /= np.linalg.norm(corr_normal)
            angle = np.arccos(np.clip(np.dot(outer_normal, corr_normal), -1.0, 1.0))

            if angle < best_angle:
                best_angle = angle
                best_inner = inner

        # remove non optimal correspondences
        for inner, corr in enumerate(correspondences):
            if outer in corr and inner != best_inner:
                corr.remove(outer)

    return correspondences


def every_other_node(graph: graph.Graph):
    nodes = graph.nodes()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for node in nodes:
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    selected_nodes = set()
    for node in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True):
        if all(neighbor not in selected_nodes for neighbor in G.neighbors(node)):
            selected_nodes.add(node)

    return list(selected_nodes)


def select_nodes(g: graph.Graph, target_count):
    nodes = list(g.nodes())
    random.shuffle(nodes)  # Shuffle to avoid bias in node order

    selected_nodes = set()
    for node in nodes:
        if len(selected_nodes) >= target_count:
            break
        if all(neighbor not in selected_nodes for neighbor in g.neighbors(node)):
            selected_nodes.add(node)

    # If fewer nodes are selected than needed, adjust by adding nodes with minimal increase in adjacency conflicts
    if len(selected_nodes) < target_count:
        remaining_nodes = [node for node in nodes if node not in selected_nodes]
        remaining_nodes.sort(key=lambda x: len(set(g.neighbors(x)) & selected_nodes))  # Sort by fewest conflicts
        for node in remaining_nodes:
            if len(selected_nodes) < target_count:
                selected_nodes.add(node)
            else:
                break

    return selected_nodes
