#---modules
from src.tapas import Network
from src.bb import Leblanc

# from src import FS_NETS

DATADIR = "../data/"

net = 'SiouxFalls'
ins = 'SF_DNDP_10_1'

datadir = DATADIR + net + "/"

network = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
print(net, ins)


#---arbitrary y vector for testing
ytemp = {(7, 16): 0, (16, 7): 0, (19, 22): 1, (22, 19): 1, (11, 15): 1, (15, 11): 1, (9, 11): 0, (11, 9): 0, (13, 14): 0, (14, 13): 1}

y = {}
lbd = {}
for a in network.links2:
    y[a] = ytemp[(a.start.id, a.end.id)]
    # y[a] = 1
    lbd[a] = 0

print(y)

#---'L' mode solves penalized SO-TAP based on y and lbd
tstt = network.msa('UEL',y,lbd)
print(f"MSA L{max(lbd.values())} TSTT={tstt} cost={network.getCost('UEL')}")

# model = GBModel(network)
# model.solveOA(y)


#---solve SO-TAP based on y (will ignore lbd)
tstt = network.msa('SO', y, lbd)
print(f"MSA SO TSTT={tstt}  cost={network.getCost('SO')}")

#---solve UE-TAP based on y (will ignore lbd)
tstt = network.msa('UE', y, lbd)
print(f"MSA UE TSTT={tstt}  cost={network.getCost('UE')}")
print({a: a.x for a in network.links})

tstt = network.tapas('UE',y)
print(f"TAPAS UE TSTT={tstt} cost={network.getCost('UE')}")
print({a: a.x for a in network.links})

tstt = network.tapas('SO',y)
print(f"TAPAS SO TSTT={tstt} cost={network.getCost('SO')}")
print({a: a.x for a in network.links})

#---solve DNDP using Leblanc's BB algorithm
leblanc = Leblanc.Leblanc(network)
leblanc.BB()

'''
#---solve DNDP using FS NETS's BB algorithm
fs_nets = FS_NETS.FS_NETS(network)
fs_nets.BB()
'''

