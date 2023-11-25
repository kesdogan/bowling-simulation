import k3d, igl
import numpy as np 
import openmesh as om
import polyscope as ps


class Object: 
    def __init__(self, v, f, velocities=None, forces=None):
        self.v = v
        self.f = f
        if not velocities: self.velocities = np.zeros_like(v)
        else: self.velocities = velocities
        if not forces: self.forces = np.zeros_like(v)
        else: self.forces = forces
    
    def update(self, dt):
        self.v += self.velocities * dt
        self.velocities += self.forces * dt
        self.forces = np.zeros_like(self.v)
    
dt = 0.1


# load pins from obj file and generate them in a bowling setting
v, _, _, f, _, _ = igl.read_obj("/Users/timurkesdogan/Desktop/simulation/project/bowling-simulation/assets/simple_pin.obj")
pins = []
for i in range(2): 
    for j in range(-i, i+1): 
        pins.append(Object(v + np.array([i*1.5, 0, 0]) + np.array([0, 0, j* 1.5]), f))


# load bowling ball from obj file place in field 
v, _, _, f, _, _ = igl.read_obj("/Users/timurkesdogan/Desktop/simulation/project/bowling-simulation/assets/simple_ball.obj")
v = v + np.array([-10, 0, 0])


ps.init()

# register the pins
for counter, pin in enumerate(pins):
    ps.register_surface_mesh("pin" + str(counter), pin.v, pin.f)

# register the ball
ps.register_surface_mesh("ball", v, f)
ps.show()


    
