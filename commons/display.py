import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pygel3d import hmesh
from numpy import array
import numpy as np
from commons.medial_axis import MedialAxis
from commons.utils import farthest_point_sampling

hand_camera = dict(
    up=dict(x=0, y=-1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=2.0)
)
hand_width, hand_height = 800, 800

x19_camera = dict(
    up=dict(x=0, y=-1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=2.0)
)
x19_width, x19_height = 850, 1200

leaf_camera = dict(
    up=dict(x=1, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.9, y=-1.8, z=1.1)
)
leaf_width, leaf_height = 800, 600

cam = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-2, y=-1.5, z=0.2)
)

camera = hand_camera
width, height = hand_width, hand_height


def __mesh_normal_data(m):
    pos = m.positions()
    vertex_normals = np.array([m.vertex_normal(v) for v in range(len(m.vertices()))])

    # Prepare data for vertex normals
    normals_data = []
    normal_length = 0.01  # Adjust the length of the normal vectors as needed
    for i, p in enumerate(pos):
        normal = vertex_normals[i]
        p_end = p + normal_length * normal
        normals_data.append(p)
        normals_data.append(p_end)
        normals_data.append(array([None, None, None]))
    normals_data = array(normals_data)

    normals_plot = go.Scatter3d(x=normals_data[:, 0], y=normals_data[:, 1], z=normals_data[:, 2],
                                mode='lines',
                                line=dict(color='rgb(0,0,255)', width=2),
                                hoverinfo='none',
                                name="vertex_normals")
    return normals_plot


def __wireframe_plot_data(m, color='#000000', width=1, opacity=1.0):
    m_tri = hmesh.Manifold(m)
    hmesh.triangulate(m_tri)
    pos = m.positions()
    xyze = []
    for h in m.halfedges():
        if h < m.opposite_halfedge(h):
            p0 = pos[m.incident_vertex(m.opposite_halfedge(h))]
            p1 = pos[m.incident_vertex(h)]
            xyze.append(array(p0))
            xyze.append(array(p1))
            xyze.append(array([None, None, None]))
    xyze = array(xyze)
    wireframe = go.Scatter3d(x=xyze[:, 0], y=xyze[:, 1], z=xyze[:, 2],
                          mode='lines',
                          opacity=opacity,
                          line=dict(color=color, width=width), hoverinfo='none')
    return wireframe


def __graph_plot_data(g, color='#000000', width=1, opacity=1.0):
    pos = g.positions()
    xyze = []
    for v in g.nodes():
        for w in g.neighbors(v):
            if v < w:
                p0 = pos[v]
                p1 = pos[w]
                p0[2] += 0.001
                p1[2] += 0.001
                xyze.append(array(p0))
                xyze.append(array(p1))
                xyze.append(array([None, None, None]))
    xyze = array(xyze)
    trace1 = go.Scatter3d(x=xyze[:, 0], y=xyze[:, 1], z=xyze[:, 2],
                          mode='lines',
                          opacity=opacity,
                          line=dict(color=color, width=width), hoverinfo='none')
    return trace1


def __mesh_plot_data(m, color, opacity=1.0, smooth=True, diffuse=False):
    xyz = array([p for p in m.positions()])
    ijk = array([[idx for idx in m.circulate_face(f, 'v')] for f in m.faces()])

    lighting = None
    if diffuse:
        lighting = dict(ambient=1,
                        diffuse=0,
                        fresnel=0,
                        specular=0,
                        roughness=1,
                        facenormalsepsilon=0,
                        vertexnormalsepsilon=0)

    return go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                 i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], color=color, flatshading=not smooth, opacity=opacity,
                 lighting=lighting)


def display_medial_mesh(ma: MedialAxis, save_path=None):
    surf = __mesh_plot_data(ma.surface, "#dddddd", 0.3, smooth=True)
    medial_sheet = __mesh_plot_data(ma.sheet, "#4d6aff", 1.0, diffuse=True)
    medial_mesh = __graph_plot_data(ma.graph, "#000000", 3)

    mesh_data = [surf, medial_sheet, medial_mesh]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height
    )
    if save_path is not None:
        fig.write_image(save_path + ".png")
    fig.show()


def display_mesh_pointset(m, points, show_normals=False):
    wireframe = __wireframe_plot_data(m)
    point_set = go.Scatter3d(x=points[:, 0],
                             y=points[:, 1],
                             z=points[:, 2],
                             mode='markers',
                             marker_size=3,
                             line=dict(color='rgb(125,0,0)', width=1),
                             hoverinfo='text', text=list(range(len(points))),
                             name="pointset")

    mesh_data = [wireframe, point_set]

    if show_normals:
        normals_plot = __mesh_normal_data(m)
        mesh_data += [normals_plot]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height
    )
    fig.show()


def display_medial_axis(ma: MedialAxis, save_path=None):
    wireframe = __wireframe_plot_data(ma.sheet)
    inner = go.Scatter3d(x=ma.inner_points[:, 0],
                         y=ma.inner_points[:, 1],
                         z=ma.inner_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,0,125)', width=1),
                         name="inner")
    outer = go.Scatter3d(x=ma.outer_points[:, 0],
                         y=ma.outer_points[:, 1],
                         z=ma.outer_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,125,0)', width=1),
                         name="outer")
    connections = []
    for i, q in enumerate(ma.inner_points):
        for p in ma.correspondences[i]:
            connections.append(q)
            connections.append(ma.outer_points[p])
            connections.append(array([None, None, None]))
    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='black', width=1),
                                    hoverinfo='none',
                                    name="connections")
    mesh_data = [inner, outer, wireframe, connecting_lines]
    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height
    )
    if save_path is not None:
        fig.write_image(save_path + ".png")
    fig.show()


def display_sheet_connections(ma: MedialAxis):
    inner = go.Scatter3d(x=ma.sheet.positions()[:, 0],
                         y=ma.sheet.positions()[:, 1],
                         z=ma.sheet.positions()[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,0,125)', width=1),
                         name="inner")

    outer = go.Scatter3d(x=ma.outer_points[:, 0],
                         y=ma.outer_points[:, 1],
                         z=ma.outer_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,125,0)', width=1),
                         name="outer")
    connections = []
    for i, vid in enumerate(ma.sheet.vertices()):
        for p in ma.sheet_correspondences[vid]:
            connections.append(ma.sheet.positions()[vid])
            connections.append(ma.outer_points[p])
            connections.append(array([None, None, None]))
    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='black', width=1),
                                    hoverinfo='none',
                                    name="connections")
    mesh_data = [inner, outer, connecting_lines]
    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height
    )
    fig.show()


def display_mesh(m, wireframe=True, color='#dddddd', save_path=None, save_html=None):
    mesh = __mesh_plot_data(m, color)

    mesh_data = [mesh]
    if wireframe:
        wireframe = __wireframe_plot_data(m, width=1.1)
        mesh_data += [wireframe]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height,
        showlegend=False
    )
    if save_path is not None:
        fig.write_image(save_path + ".png")
    if save_html is not None:
        fig.write_html(save_html + ".html")
    fig.show()


def display_two_meshes(m1, m2, wireframe=True, color='#dddddd', save_path=None):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.1
    )

    mesh_data1 = [__mesh_plot_data(m1, color)]
    mesh_data2 = [__mesh_plot_data(m2, color)]
    if wireframe:
        wireframe = __wireframe_plot_data(m1)
        mesh_data1 += [wireframe]
        wireframe = __wireframe_plot_data(m2)
        mesh_data2 += [wireframe]

    for data in mesh_data1:
        fig.add_trace(data, row=1, col=1)
    for data in mesh_data2:
        fig.add_trace(data, row=1, col=2)

    for i in range(1, 3):
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                camera=camera
            ),
            row=1, col=i
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        width=width, height=height/2
    )

    if save_path is not None:
        fig.write_image(save_path + ".png")
    fig.show()


def display_uv(m, uv):
    f = np.array([m.circulate_face(fid) for fid in m.faces()])

    fig = go.Figure(data=[
        go.Mesh3d(x=uv[:, 0], y=uv[:, 1], z=[0] * len(uv), i=f[:, 0], j=f[:, 1], k=f[:, 2], color='lightblue',
                  opacity=0.50)])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        width=width, height=height
    )

    fig.show()


def display_graph(g, show_points=False, save_path=None):
    trace1 = __graph_plot_data(g)
    mesh_data = [trace1]

    if show_points:
        pos = g.positions()
        point_set = go.Scatter3d(x=pos[:, 0],
                                 y=pos[:, 1],
                                 z=pos[:, 2],
                                 mode='markers',
                                 marker_size=3,
                                 line=dict(color='rgb(125,0,0)', width=1),
                                 text=list(range(len(g.nodes()))), hoverinfo='text')
        mesh_data += [point_set]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data",
            camera=camera,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        width=width, height=height,
        showlegend=False
    )
    if save_path is not None:
        fig.write_image(save_path + ".png")

    fig.show()


def display_graph_pointset(g, pointset, save_path=None):
    pos = g.positions()
    xyze = []
    for v in g.nodes():
        for w in g.neighbors(v):
            if v < w:
                p0 = pos[v]
                p1 = pos[w]
                xyze.append(array(p0))
                xyze.append(array(p1))
                xyze.append(array([None, None, None]))
    xyze = array(xyze)
    trace1 = go.Scatter3d(x=xyze[:, 0], y=xyze[:, 1], z=xyze[:, 2],
                          mode='lines',
                          line=dict(color='rgb(0,0,0)', width=1), hoverinfo='none')

    mesh_data = [trace1]
    point_set = go.Scatter3d(x=pointset[:, 0],
                             y=pointset[:, 1],
                             z=pointset[:, 2],
                             mode='markers',
                             marker_size=3,
                             line=dict(color='rgb(125,0,0)', width=1),
                             text=list(range(len(g.nodes()))), hoverinfo='text')
    mesh_data += [point_set]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data",
            camera=camera,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        width=width, height=height,
        showlegend=False
    )
    if save_path is not None:
        fig.write_image(save_path + ".png")

    fig.show()


def display_correspondences(outer_points, inner_points, correspondences):
    outer = go.Scatter3d(x=outer_points[:, 0],
                         y=outer_points[:, 1],
                         z=outer_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(125,0,0)', width=1),
                         name="surface")
    inner = go.Scatter3d(x=inner_points[:, 0],
                         y=inner_points[:, 1],
                         z=inner_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,0,125)', width=1),
                         name="inner")

    connections = []
    for i in range(len(correspondences)):
        start = inner_points[i]
        for j in range(len(correspondences[i])):
            end = correspondences[i][j]
            connections.append(start)
            connections.append(end)
            connections.append(array([None, None, None]))
    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='black', width=1),
                                    hoverinfo='none',
                                    name="connections")

    fig = go.Figure(data=[outer, inner, connecting_lines])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        width=width, height=height
    )
    fig.show()


def display_inner_projections(ma: MedialAxis, show_n=None, indices=None, save_path=None):
    if indices is None:
        indices = np.arange(len(ma.outer_points))
    if show_n is not None:
        indices = farthest_point_sampling(ma.outer_points, show_n)
        print(indices.tolist())

    surf = __mesh_plot_data(ma.surface, "rgb(0,125,0)", 0.2, smooth=True, diffuse=True)

    medial_sheet = __mesh_plot_data(ma.sheet, "#4d6aff", 0.3, diffuse=True)
    medial_mesh = __graph_plot_data(ma.graph, "#4d6aff", 3, 0.3)


    proj = go.Scatter3d(x=ma.inner_projections[indices, 0],
                         y=ma.inner_projections[indices, 1],
                         z=ma.inner_projections[indices, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,0,255)', width=5),
                         name="projection")
    outer = go.Scatter3d(x=ma.outer_points[indices, 0],
                         y=ma.outer_points[indices, 1],
                         z=ma.outer_points[indices, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(0,125,0)', width=5),
                         name="outer")

    inner = go.Scatter3d(x=ma.inner_points[:, 0],
                         y=ma.inner_points[:, 1],
                         z=ma.inner_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(255,0,0)', width=1),
                         name="inner")

    connections = []
    for i, q in enumerate(ma.inner_projections[indices]):
        connections.append(q)
        connections.append(ma.outer_points[indices][i])
        connections.append(array([None, None, None]))
    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='black', width=1),
                                    hoverinfo='none',
                                    name="connections")
    mesh_data = [surf, medial_sheet, medial_mesh, outer, proj, connecting_lines] # inner]
    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height,
        showlegend=False
    )
    if save_path is not None:
        fig.write_image(save_path + ".png")
    fig.show()

def display_mesh_difference(mesh1, mesh2):
    wireframe1 = __wireframe_plot_data(mesh1)
    # points1 = go.Scatter3d(x=mesh1.positions()[:, 0],
    #                          y=mesh1.positions()[:, 1],
    #                          z=mesh1.positions()[:, 2],
    #                          mode='markers',
    #                          marker_size=2,
    #                          line=dict(color='rgb(0,0,125)', width=1),
    #                          name="mesh1")
    points2 = go.Scatter3d(x=mesh2.positions()[:, 0],
                             y=mesh2.positions()[:, 1],
                             z=mesh2.positions()[:, 2],
                             mode='markers',
                             marker_size=2,
                             line=dict(color='rgb(125,0,0)', width=1),
                             name="mesh2")
    connections = []
    for i, q in enumerate(mesh1.positions()):
        connections.append(q)
        connections.append(mesh2.positions()[i])
        connections.append(array([None, None, None]))
    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='blue', width=3),
                                    hoverinfo='none',
                                    name="difference")
    mesh_data = [wireframe1, points2, connecting_lines]
    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height
    )
    fig.show()


def display_mesh_vertex_colors(m, vertex_colors=None, save_html=None):
    xyz = np.array([p for p in m.positions()])
    m_tri = hmesh.Manifold(m)
    hmesh.triangulate(m_tri)
    ijk = np.array([[idx for idx in m_tri.circulate_face(f, 'v')] for f in m_tri.faces()])
    mesh = go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                     i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], vertexcolor=vertex_colors, flatshading=False)

    mesh_data = [mesh]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=width, height=height
    )
    if save_html is not None:
        fig.write_html(save_html + ".html")
    else:
        fig.show()



def plot_frames(curve_positions, frames, sheet, title="Frames Visualization"):
    fig = go.Figure()

    xyz = np.array([p for p in sheet.positions()])
    ijk = np.array([[idx for idx in sheet.circulate_face(f, 'v')] for f in sheet.faces()])
    fig.add_trace(go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], opacity=0.5))

    # Add the curve itself
    fig.add_trace(go.Scatter3d(
        x=curve_positions[:, 0],
        y=curve_positions[:, 1],
        z=curve_positions[:, 2],
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(size=4),
        name='Curve'
    ))

    # Add the tangents, normals, and binormals
    for i in range(len(curve_positions)):
        fig.add_trace(go.Scatter3d(
            x=[curve_positions[i, 0], curve_positions[i, 0] + frames[i, 0, 0]],
            y=[curve_positions[i, 1], curve_positions[i, 1] + frames[i, 0, 1]],
            z=[curve_positions[i, 2], curve_positions[i, 2] + frames[i, 0, 2]],
            mode='lines',
            line=dict(color='red', width=2),
            name='Tangent' if i == 0 else None,
            showlegend=(i == 0)
        ))
        fig.add_trace(go.Scatter3d(
            x=[curve_positions[i, 0], curve_positions[i, 0] + frames[i, 1, 0]],
            y=[curve_positions[i, 1], curve_positions[i, 1] + frames[i, 1, 1]],
            z=[curve_positions[i, 2], curve_positions[i, 2] + frames[i, 1, 2]],
            mode='lines',
            line=dict(color='green', width=2),
            name='Normal' if i == 0 else None,
            showlegend=(i == 0)
        ))
        fig.add_trace(go.Scatter3d(
            x=[curve_positions[i, 0], curve_positions[i, 0] + frames[i, 2, 0]],
            y=[curve_positions[i, 1], curve_positions[i, 1] + frames[i, 2, 1]],
            z=[curve_positions[i, 2], curve_positions[i, 2] + frames[i, 2, 2]],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Binormal' if i == 0 else None,
            showlegend=(i == 0)
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,height=800
    )

    fig.show()