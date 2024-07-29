#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:57:46 2024

@author: Sophie Demassey
"""

import gurobipy as gp
from gurobipy import GRB
from src.tapas import Network
from src.tapas import Link


class GBModel:

    def __init__(self, network: Network):
        self.net = network
        self.minlp, self.yvar = GBModel.build_model(network)
        self.minlp.write(ROOTDIR + "output/minlp.lp")

    @staticmethod
    def get_demand(network: Network, node, zone):
        return - sum(r.getDemand(node) for r in network.zones) if node.id == zone.id else \
            node.getDemand(zone) if isinstance(node, type(zone)) else 0

    @staticmethod
    def get_traveltime_func(link: Link):
        """ t(x) = t + c.x^e """
        t = link.t_ff
        e = link.beta
        c = t * link.alpha / pow(link.C, e)
        return [t, e, c]

    @staticmethod
    def get_traveltime_poly(link: Link):
        """ t(x) = t + c.x^4 """
        [t, e, c] = GBModel.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, t]

    @staticmethod
    def get_INT_poly(link: Link):
        """ int_[0,x] t(v) = t.x + (c/5).x^5 """
        [t, e, c] = GBModel.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c/5, 0, 0, 0, t, 0]
        # """ x(t(x) + x.t'(x)) = Tx + c.x^5 + 4.c.x^5 = Tx  + 5.c.x^5 """
        # return [5 * c, 0, 0, 0, T, 0]

    @staticmethod
    def get_SO_poly(link: Link):
        """ xt(x) = t.x + c.x^5 """
        [t, e, c] = GBModel.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, t, 0]

    @staticmethod
    def build_model(network: Network):
        minlp = gp.Model('SODNDP_agg')

        yvar = minlp.addVars(network.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")

        xvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=network.TD, name="x")
        minlp.addConstrs((xvar[a] <= yvar[a] * network.TD for a in network.links2), name="M")

        xsvar = minlp.addVars(network.links, network.zones, vtype=GRB.CONTINUOUS, lb=0.0, name="xs")
        minlp.addConstrs((sum(xsvar[a, s] for a in i.outgoing)
                          - sum(xsvar[a, s] for a in i.incoming) == GBModel.get_demand(network, i, s)
                          for i in network.nodes for s in network.zones), name="D")

        minlp.addConstrs((xsvar.sum(a, '*') == xvar[a] for a in network.links), name="A")

        cvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
        for a in network.links:
            minlp.addGenConstrPoly(xvar[a], cvar[a], GBModel.get_SO_poly(a))

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
        return {a: model.minlp.getVarByName(f"x[{a}]").x for a in self.net.links}

    def setXsolution(self, flow: dict[Link, float]):
        for a in self.net.links:
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
            print(f"solution cost={cost} slack={bctr.Slack} ({bctr.RHS})")
        return cost


if __name__ == "__main__":

    ROOTDIR = "../../"

    net = 'SiouxFalls'
    ins = 'SF_DNDP_10_1'
    datadir = ROOTDIR + "data/" + net + "/"
    ntk = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
    print(net, ins)

    print("\n-- solve PWL approx --")
    model = GBModel(ntk)
    pwlcost = model.solve(pwl=True)
    pwlsol = model.getYsolution()

    print("\n-- simulate PWL approx solution --")
    tstt = ntk.msa('UE', pwlsol, {a: 0 for a in ntk.links2})
    flowsol = {a: a.x for a in ntk.links}
    print('SO TSTT', tstt)

    print("\n-- check solution in NLP --")
    model.setYsolution(pwlsol)
    model.setXsolution(flowsol)
    nlpcost = model.solve(pwl=False)

    print("\n----------------------")
    print(f"PWL solution: {pwlsol}")
    print(f"PWL cost: {pwlcost}")
    print(f"TAP cost {tstt}")
    print(f"NLP cost: {nlpcost}")
