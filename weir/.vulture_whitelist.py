# Vulture whitelist: names that are used structurally — gymnasium/API contracts,
# SB3 policy internals, MuJoCo struct fields — but never "called" by our code.
# Remove an entry only if the code it refers to is actually gone.

observation_space  # gymnasium Env interface (SpacesOnly, GymEnv)
action_space       # gymnasium Env interface (SpacesOnly, GymEnv)
metadata           # gymnasium Env class attribute
options            # gymnasium Env.reset(seed, options) API parameter
forward            # torch.nn.Module.forward override

# mujoco.MjvCamera / MjvLight struct fields set directly
lookat
distance
azimuth
elevation
headlight
intensity
pos
dir
diffuse
specular
ambient
