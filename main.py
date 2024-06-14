#---modules
from src import Network
from src import Leblanc
from src import FS_NETS

net = 'SiouxFalls'
ins = 'SF_DNDP_10_1'

network = Network.Network(net,ins,0.5,1e-0,1e-3)
print(net,ins)

#---arbitrary y vector for testing
#ytemp = {(7, 16): 0, (16, 7): 0, (19, 22): 1, (22, 19): 1, (11, 15): 1, (15, 11): 1, (9, 11): 0, (11, 9): 0, (13, 14): 0, (14, 13): 1}

y = {}
lbd = {}
for a in network.links2:
    #y[a] = ytemp[(a.start.id,a.end.id)]
    y[a] = 1
    lbd[a] = 0

#---'L' mode solves penalized SO-TAP based on y and lbd
tstt = network.msa('L',y,lbd)
print('TSTT',tstt)



'''
#---solve SO-TAP based on y (will ignore lbd)
tstt = network.msa('SO',y,lbd)
print('SO TSTT',tstt)

#---solve UE-TAP based on y (will ignore lbd)
tstt = network.msa('UE',y,lbd)
print('UE TSTT',tstt)
'''


'''
#---solve DNDP using Leblanc's BB algorithm
leblanc = Leblanc.Leblanc(network)
leblanc.BB()

#---solve DNDP using FS NETS's BB algorithm
fs_nets = FS_NETS.FS_NETS(network)
fs_nets.BB()
'''