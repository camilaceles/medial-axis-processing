import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pygel3d import hmesh
from numpy import array
import numpy as np
from commons.medial_axis import MedialAxis

hand_camera = dict(
    up=dict(x=0, y=-1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=2.0)
)
hand_width, hand_height = 800, 800

leaf_camera = dict(
    up=dict(x=1, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.0, y=-0.8, z=1.0)
)
leaf_width, leaf_height = 800, 600


camera = hand_camera
width, height = hand_width, hand_height


def __wireframe_plot_data(m):
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
                             line=dict(color='rgb(75,75,75)', width=1),
                             hoverinfo='none',
                             name="wireframe")
    return wireframe


def __mesh_plot_data(m, color):
    xyz = array([p for p in m.positions()])
    ijk = array([[idx for idx in m.circulate_face(f, 'v')] for f in m.faces()])

    return go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                     i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], color=color, flatshading=False, opacity=1.0)


def display_mesh_pointset(m, points):
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
        wireframe = __wireframe_plot_data(m)
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
        width=2*width, height=height
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
    if show_points:
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


def display_inner_projections(ma: MedialAxis):
    proj = go.Scatter3d(x=ma.inner_projections[:, 0],
                         y=ma.inner_projections[:, 1],
                         z=ma.inner_projections[:, 2],
                         mode='markers',
                         marker_size=1,
                         line=dict(color='rgb(0,0,125)', width=1),
                         name="projection")
    outer = go.Scatter3d(x=ma.outer_points[:, 0],
                         y=ma.outer_points[:, 1],
                         z=ma.outer_points[:, 2],
                         mode='markers',
                         marker_size=1,
                         line=dict(color='rgb(0,125,0)', width=1),
                         name="outer")

    inner = go.Scatter3d(x=ma.inner_points[:, 0],
                         y=ma.inner_points[:, 1],
                         z=ma.inner_points[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(255,0,0)', width=1),
                         name="inner")

    connections = []
    for i, q in enumerate(ma.inner_projections):
        connections.append(q)
        connections.append(ma.outer_points[i])
        connections.append(array([None, None, None]))
    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='black', width=1),
                                    hoverinfo='none',
                                    name="connections")
    mesh_data = [proj, outer, connecting_lines, inner]
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

