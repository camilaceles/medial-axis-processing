import plotly.graph_objs as go
from pygel3d import hmesh
from numpy import array
import numpy as np
from medial_axis_processing.medial_axis import MedialAxis

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=1.5, z=-2)
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
                             line=dict(color='rgb(75,75,75)', width=1),
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


def display_result(m, outer_points, inner_points, show_wireframe=False, show_connections=True, debug=False, save_file=None):
    mesh_data = []
    if debug:
        s_fixed = inner_points.positions[inner_points.is_fixed]
        s_notfixed = inner_points.positions[~inner_points.is_fixed]
        medial_axis_fixed = go.Scatter3d(x=s_fixed[:, 0],
                                         y=s_fixed[:, 1],
                                         z=s_fixed[:, 2],
                                         mode='markers',
                                         marker_size=3,
                                         line=dict(color='rgb(0,0,125)', width=1),
                                         name="medial sheets")
        mesh_data += [medial_axis_fixed]

        if len(s_notfixed > 0):
            medial_axis_notfixed = go.Scatter3d(x=s_notfixed[:, 0],
                                                y=s_notfixed[:, 1],
                                                z=s_notfixed[:, 2],
                                                mode='markers',
                                                marker_size=3,
                                                line=dict(color='rgb(0,125,0)', width=1),
                                                name="medial curve")
            mesh_data += [medial_axis_notfixed]
    else:
        medial_axis = go.Scatter3d(x=inner_points.positions[:, 0],
                                   y=inner_points.positions[:, 1],
                                   z=inner_points.positions[:, 2],
                                   mode='markers',
                                   marker_size=3,
                                   line=dict(color='rgb(0,0,125)', width=1),
                                   name="inner")
        mesh_data = [medial_axis]

    outer = go.Scatter3d(x=outer_points.positions[:, 0],
                         y=outer_points.positions[:, 1],
                         z=outer_points.positions[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(125,0,0)', width=1),
                         name="outer")

    # Create the lines connecting outer to inner points.
    connections = []
    for start, end in zip(outer_points.positions, inner_points.positions):
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

    if show_wireframe:
        wireframe = __wireframe_plot_data(m)
        mesh_data += [wireframe]
    if show_connections:
        mesh_data += [connecting_lines, outer]

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

    if save_file is not None:
        fig.write_html(f"results/{save_file}.html")
    else:
        fig.show()


def display_medial_axis_result(ma: MedialAxis, show_connections=False, save_file=None):
    medial_axis = go.Scatter3d(x=ma.mesh.positions()[:, 0],
                               y=ma.mesh.positions()[:, 1],
                               z=ma.mesh.positions()[:, 2],
                               mode='markers',
                               marker_size=3,
                               line=dict(color='rgb(0,0,125)', width=1),
                               name="inner")

    outer = go.Scatter3d(x=ma.outer_points.positions[:, 0],
                         y=ma.outer_points.positions[:, 1],
                         z=ma.outer_points.positions[:, 2],
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(125,0,0)', width=1),
                         name="outer")

    # Create the lines connecting outer to inner points.
    connections = []
    for inner_pos, outer_list in ma.map.items():
        for outer in outer_list:
            connections.append(np.array(inner_pos))
            connections.append(outer.pos)
            connections.append(array([None, None, None]))

    connections = array(connections)

    connecting_lines = go.Scatter3d(x=connections[:, 0],
                                    y=connections[:, 1],
                                    z=connections[:, 2],
                                    mode='lines',
                                    line=dict(color='black', width=1),
                                    hoverinfo='none',
                                    name="connections")

    wireframe = __wireframe_plot_data(ma.mesh)
    mesh_data = [medial_axis, wireframe, outer]
    if show_connections:
        mesh_data += [connecting_lines]

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

    if save_file is not None:
        fig.write_html(f"results/{save_file}.html")
    else:
        fig.show()


def display_mesh(m, wireframe=True, smooth=True, color='#dddddd'):
    xyz = array([p for p in m.positions()])
    m_tri = hmesh.Manifold(m)
    hmesh.triangulate(m_tri)
    ijk = array([[idx for idx in m_tri.circulate_face(f, 'v')] for f in m_tri.faces()])
    mesh = go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                     i=ijk[:, 0], j=ijk[:, 1], k=ijk[:, 2], color=color, flatshading=not smooth)

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


def display_inner_mesh(m, inner_points, wireframe=True, save_file=None):
    inner_m = hmesh.Manifold(m)
    pos = inner_m.positions()
    pos[:] = inner_points.positions

    m = hmesh.Manifold(m)
    hmesh.quadric_simplify(m, 0.5)
    hmesh.triangulate(m)
    m.cleanup()

    inner_mesh = __mesh_plot_data(inner_m, color='#253AB7')
    mesh = __wireframe_plot_data(m)

    s_notfixed = inner_points.positions[~inner_points.is_fixed]
    medial_axis_notfixed = go.Scatter3d(x=s_notfixed[:, 0],
                                        y=s_notfixed[:, 1],
                                        z=s_notfixed[:, 2],
                                        mode='markers',
                                        marker_size=2,
                                        line=dict(color='#253AB7', width=1))

    mesh_data = [inner_mesh, medial_axis_notfixed]
    if wireframe:
        mesh_data += [mesh]
    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera
        ),
        width=850, height=1200,
    )
    if save_file is not None:
        fig.write_html(f"results/{save_file}.html")
    else:
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


def display_highlight(m, outer_points, inner_points, vid):
    mesh_data = []
    medial_axis = go.Scatter3d(x=np.array(inner_points.positions[vid, 0]),
                               y=np.array(inner_points.positions[vid, 1]),
                               z=np.array(inner_points.positions[vid, 2]),
                               mode='markers',
                               marker_size=3,
                               line=dict(color='rgb(0,0,125)', width=1),
                               name="inner")
    mesh_data = [medial_axis]

    outer = go.Scatter3d(x=np.array(outer_points.positions[vid, 0]),
                         y=np.array(outer_points.positions[vid, 1]),
                         z=np.array(outer_points.positions[vid, 2]),
                         mode='markers',
                         marker_size=3,
                         line=dict(color='rgb(125,0,0)', width=1),
                         name="outer")

    wireframe = __wireframe_plot_data(m)
    mesh_data += [wireframe, outer]

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=camera
        ),
        width=850, height=1200
    )

    fig.show()
