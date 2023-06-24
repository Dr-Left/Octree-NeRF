import numpy as np

from skimage import measure


# Save the mesh as an OBJ file
def save_as_obj(vertices, faces, normals, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for n in normals:
            f.write('vn {} {} {}\n'.format(n[0], n[1], n[2]))
        for face in faces:
            f.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))


def rebuild_3D(voxel):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    print(voxel)
    binarized_voxel = np.where(voxel > 0, 0, 1)
    verts, faces, normals, _ = measure.marching_cubes(binarized_voxel)
    save_as_obj(verts, faces, normals, 'mesh.obj')
    import logging
    logging.info('rebuild_3D_graph.py: rebuild_3D() finished.')
