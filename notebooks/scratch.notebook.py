# %%
from lie_groups import su2

# %%
def comm(x, y):
    return x @ y - y @ x