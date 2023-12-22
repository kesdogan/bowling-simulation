"""This file allows to run a simulation from a cached file."""

from time import sleep

import polyscope as ps
import polyscope.imgui as psim

from src.caching import Cache

# the file name of the cache file to load
# if None, the latest cache file is loaded
FILE_NAME = "cache_nice_2.pkl"

epoch = 0
running = False

cache = Cache.from_file(FILE_NAME)

ps.init()
mesh = ps.register_surface_mesh("everything", cache.initial_vertices, cache.faces)


def callback():
    global running, epoch, mesh

    psim.PushItemWidth(100)
    psim.Separator()
    psim.TextUnformatted("Here we control the simulation")
    if psim.Button("Start / Pause"):
        running = not running

        if running:
            sleep(1)
            mesh = ps.register_surface_mesh(
                "everything", cache.initial_vertices, cache.faces
            )
            epoch = 0

    psim.PopItemWidth()

    if running:
        mesh.update_vertex_positions(cache.get_frame(epoch))
        epoch += 1


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
