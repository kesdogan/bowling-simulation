import polyscope as ps
import polyscope.imgui as psim

from src.caching import Cache

epoch = 0
running = False

cache = Cache.from_file()

ps.init()
mesh = ps.register_surface_mesh("everything", cache.vertices, cache.faces)


def callback():
    global running, epoch

    psim.PushItemWidth(100)
    psim.Separator()
    psim.TextUnformatted("Here we control the simulation")
    if psim.Button("Start / Pause"):
        running = not running

    psim.PopItemWidth()

    if running:
        mesh.update_vertex_positions(cache.get(epoch))
        epoch += 1


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
