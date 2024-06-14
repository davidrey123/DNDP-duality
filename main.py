#---modules
from src import Network
from src import Leblanc
from src import FS_NETS

net = 'SiouxFalls'
ins = 'SF_DNDP_10_1'

network = Network.Network(net,ins,0.5,1e-0,1e-3)
print(net,ins)

y = {}
for a network.links2
  y[a] = 0
  lbd[a] = 0

tstt = network.msa(y, lbd)
print(tstt)

#---solve DNDP using Leblanc's BB algorithm
leblanc = Leblanc.Leblanc(network)
leblanc.BB()

#---solve DNDP using FS NETS's BB algorithm
fs_nets = FS_NETS.FS_NETS(network)
fs_nets.BB()
