import plotly.graph_objs as go
from pygel3d import hmesh
from numpy import array
import numpy as np
from commons.medial_axis import MedialAxis

camera = dict(
    up=dict(x=0, y=-1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=3)
)


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
                             line=dict(color='rgb(0,0,0)', width=1),
                             hoverinfo='none',
                             name="wireframe")
    return wireframe


def __mesh_plot_data(m, color):
    xyz = array([p for p in m.positions()])
    ijk = array([[idx for idx in m.circulate_face(f, 'v')] for f in m.faces()])

    return go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                     i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], color=color, flatshading=False, opacity=0.50)


def display_mesh_pointset(m, points):
    wireframe = __wireframe_plot_data(m)
    point_set = go.Scatter3d(x=points[:, 0],
                             y=points[:, 1],
                             z=points[:, 2],
                             mode='markers',
                             marker_size=3,
                             line=dict(color='rgb(125,0,0)', width=1),
                             # hoverinfo='text', text=list(range(len(points))),
                             name="pointset")

    mesh_data = [wireframe, point_set]
    lyt = go.Layout(width=850, height=800)
    lyt.scene.aspectmode = "data"

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=850, height=1200
    )
    fig.show()


def display_medial_axis(ma: MedialAxis):
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
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=850, height=1200
    )
    fig.show()


def display_mesh(m, wireframe=True, smooth=True, color='#dddddd'):
    xyz = array([p for p in m.positions()])
    m_tri = hmesh.Manifold(m)
    hmesh.triangulate(m_tri)
    ijk = array([[idx for idx in m_tri.circulate_face(f, 'v')] for f in m_tri.faces()])
    mesh = go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                     i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], color=color, flatshading=not smooth, hoverinfo='none')

    mesh_data = [mesh]
    if wireframe:
        wireframe = __wireframe_plot_data(m)
        mesh_data += [wireframe]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=850, height=1200
    )

    fig.show()


def display_uv(m, uv):
    f = np.array([m.circulate_face(fid) for fid in m.faces()])

    fig = go.Figure(data=[
        go.Mesh3d(x=uv[:, 0], y=uv[:, 1], z=[0] * len(uv), i=f[:, 0], j=f[:, 1], k=f[:, 2], color='lightblue',
                  opacity=0.50)])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        width=850, height=1200
    )

    fig.show()


def display_graph(g):
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

    point_set = go.Scatter3d(x=pos[:, 0],
                             y=pos[:, 1],
                             z=pos[:, 2],
                             mode='markers',
                             marker_size=3,
                             line=dict(color='rgb(125,0,0)', width=1),
                             name="pointset",
                             text=list(range(len(g.nodes()))), hoverinfo='text')
    mesh_data = [trace1, point_set]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=camera,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        width=850, height=1200
    )

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
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        width=850, height=1200
    )
    fig.show()
