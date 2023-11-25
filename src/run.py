import k3d, igl
import numpy as np 
import polyscope as ps



# load bowling ball from obj file place in field 
v, _, _, f, _, _ = igl.read_obj("../assets/simple_ball.obj")
v = v + np.array([-10, 0, 0])

# create global faces and vertices np.arrays 
vertices = np.array(v)
faces = np.array(f)

# add pins to the same mesh in this array 
v, _, _, f, _, _ = igl.read_obj("../assets/simple_pin.obj")
for i in range(2): 
    for j in range(-i, i+1): 
        faces = np.concatenate((faces, f + len(vertices)), axis=0)
        vertices = np.concatenate((vertices, (v+np.array([i*1.5, 0 ,0])+ np.array([0,0, j*1.5]))), axis=0)



# register complete mesh w/ polyscope
ps.init()
ps.register_surface_mesh("everything", vertices, faces)
ps.show()
