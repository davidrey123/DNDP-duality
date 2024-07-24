#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:57:46 2024

@author: Sophie Demassey
"""

import time
import gurobipy as gp
from gurobipy import GRB
from src.tapas import Network
from src.tapas import Link


class GBModel:

    def __init__(self, network: Network):
        self.net = network
        self.minlp, self.yvar = self.build()
        self.minlp.write(ROOTDIR + "output/minlp.lp")

    def get_demand(self, node, zone):
            return - sum(r.getDemand(node) for r in self.net.zones) if node.id == zone.id else \
                node.getDemand(zone) if isinstance(node, type(zone)) else 0

    def get_traveltime_func(self, link: Link):
        """ t(x) = T + c.x^e """
        T = link.t_ff
        e = link.beta
        c = T * link.alpha / pow(link.C, e)
        return [T, e, c]

    def get_traveltime_poly(self, link: Link):
        """ t(x) = T + c.x^4 """
        [T, e, c] = self.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, T]

    def get_INT_poly(self, link: Link):
        """ int_[0,x] t(v) = T.x + (c/5).x^5 """
        [T, e, c] = self.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c/5, 0, 0, 0, T, 0]
        # """ x(t(x) + x.t'(x)) = Tx + c.x^5 + 4.c.x^5 = Tx  + 5.c.x^5 """
        # return [5 * c, 0, 0, 0, T, 0]

    def get_SO_poly(self, link: Link):
        """ xt(x) = T.x + c.x^5 """
        [T, e, c] = self.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, T, 0]

    def build(self):
        minlp = gp.Model('SODNDP_agg')

        yvar = minlp.addVars(self.net.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in self.net.links2) <= self.net.B, name="B")

        xvar = minlp.addVars(self.net.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=self.net.TD, name="x")
        minlp.addConstrs((xvar[a] <= yvar[a] * self.net.TD for a in self.net.links2), name="M")

        xsvar = minlp.addVars(self.net.links, self.net.zones, vtype=GRB.CONTINUOUS, lb=0.0, name="xs")
        minlp.addConstrs((sum(xsvar[a, s] for a in i.outgoing)
                        - sum(xsvar[a, s] for a in i.incoming) == self.get_demand(i, s)
                         for i in self.net.nodes for s in self.net.zones), name="D")

        minlp.addConstrs((xsvar.sum(a, '*') == xvar[a] for a in self.net.links), name="A")

        cvar = minlp.addVars(self.net.links, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
        for a in self.net.links:
            minlp.addGenConstrPoly(xvar[a], cvar[a], self.get_SO_poly(a))

        minlp.setObjective(cvar.sum(), GRB.MINIMIZE)

        return minlp, yvar

    def getYsolution(self):
        assert self.minlp.status == GRB.OPTIMAL
        return {a: int(y.x) for (a, y) in self.yvar.items()}

    def setYsolution(self, ysol: dict[Link, int]):
        for (a, y) in self.yvar.items():
            if ysol[a] == 1:
                y.lb = 1
            else:
                y.ub = 0

    def getXsolution(self):
        assert self.minlp.status == GRB.OPTIMAL
        return {a: model.minlp.getVarByName(f"x[{a}]").x for a in network.links}

    def setXsolution(self, flow: dict[Link, float]):
        for a in network.links:
            xvara = model.minlp.getVarByName(f"x[{a}]")
            xvara.lb = flow[a] - 1e-6
            xvara.ub = flow[a] + 1e-6

    def solve(self, pwl=False):
        """Solve the convex relaxation model cvxmodel."""
        cost = 0

        if pwl:
            self.minlp.setParam(GRB.Param.FuncNonlinear, 0)
            # self.minlp.setParam(GRB.Param.FuncPieceRatio, 0)
            # self.minlp.setParam(GRB.Param.FuncPieces, -1)
            # self.minlp.setParam(GRB.Param.FuncPieceError, 1e-2)
        else:
            self.minlp.setParam(GRB.Param.FuncNonlinear, 1)

        self.minlp.optimize()

        if self.minlp.status != GRB.OPTIMAL:
            print('Optimization was stopped with status %d' % self.minlp.status)
        else:
            cost = self.minlp.objval
            bctr = self.minlp.getConstrByName("B")
            print(f"solution cost={cost} slack={bctr.slack} ({bctr.rhs})")
        return cost

if __name__ == "__main__":

    ROOTDIR = "../../"

    net = 'SiouxFalls'
    ins = 'SF_DNDP_10_1'
    datadir = ROOTDIR + "data/" + net + "/"
    network = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
    print(net, ins)

    print("\n-- solve PWL approx --")
    model = GBModel(network)
    pwlcost = model.solve(pwl=True)
    pwlsol = model.getYsolution()

    print("\n-- simulate PWL approx solution --")
    tstt = network.msa('UE', pwlsol, {a: 0 for a in network.links2})
    flow = {a: a.x for a in network.links}
    print('SO TSTT', tstt)

    print("\n-- check solution in NLP --")
    model.setYsolution(pwlsol)
    model.setXsolution(flow)
    nlpcost = model.solve(pwl=False)

    print("\n----------------------")
    print(f"PWL solution: {pwlsol}")
    print(f"PWL cost: {pwlcost}")
    print(f"TAP cost {tstt}")
    print(f"NLP cost: {nlpcost}")
