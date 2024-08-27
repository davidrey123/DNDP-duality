#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:57:46 2024

@author: Sophie Demassey
"""
import logging
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
from src.tapas import Network
from src.tapas import Link

ROOTDIR = "../../"
OUTDIR = Path(ROOTDIR, "output/")
IISFILE = Path(OUTDIR, "modeliis.ilp")

logging.basicConfig(
    handlers=[
        logging.FileHandler(Path(OUTDIR, "lag.log")),
        logging.StreamHandler()
    ],
    # format="%(levelname)s: %(message)s")
    # format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    level=logging.WARN
)


class GBModel:

    def __init__(self, network: Network):
        self.net = network
        self.allclose = {a: 0 for a in network.links2}
        self.minlp, self.yvar, self.xvar, self.cvar = GBModel.build_model(network)
        self.costmodel = None

    @staticmethod
    def get_demand(network: Network, node, zone):
        return - sum(r.getDemand(node) for r in network.zones) if node.id == zone.id else \
            node.getDemand(zone) if isinstance(node, type(zone)) else 0

    @staticmethod
    def get_traveltime_func(link: Link):
        """ f(x) = t + c.x^e """
        t = link.t_ff
        e = link.beta
        c = t * link.alpha / pow(link.C, e)
        return t, e, c

    @staticmethod
    def get_traveltime_poly_reverse(link: Link):
        """ f(x) = t + c.x^4 """
        t, e, c = GBModel.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, t]

    @staticmethod
    def get_INT_poly_reverse(link: Link):
        """ F(x) = int_[0,x] f(v) = t.x + (c/5).x^5 """
        t, e, c = GBModel.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c / 5, 0, 0, 0, t, 0]
        # """ x(t(x) + x.t'(x)) = Tx + c.x^5 + 4.c.x^5 = Tx  + 5.c.x^5 """
        # return [5 * c, 0, 0, 0, T, 0]

    @staticmethod
    def get_SO_poly_reverse(link: Link):
        """ c(x) = x.f(x) = t.x + c.x^5 """
        t, e, c = GBModel.get_traveltime_func(link)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, t, 0]

    @staticmethod
    def get_SOcut_poly(link: Link, x: float):
        """ c(x) >= c(X) + c'(X).(x-X) = c0 + c1.x with
        c(x) = t.x + c.x^(e+1), c'(x) = t + c.(e+1).x^e
        c1= c'(X) = t + c.(e+1).X^e
        c0 = c(X)-c'(X).X = t.X + c.X^(e+1) - (t.X + c.(e+1).X^(e+1)) =  - c.e.X^(e+1)
        """
        t, e, c = GBModel.get_traveltime_func(link)
        c1 = t + c * (e + 1) * pow(x, e)  # = a.getTravelTime(a.x, 'SO') =  t  + c.x^e + c.e.x^e
        c0 = - c * e * pow(x, e + 1)  # = -pow(a.x,2) * a.getDerivativeTravelTime(a.x) = - x^2.c.e.x^(e-1)
        return c0, c1

    def generateOAcuts(self, ysol: dict):
        """
        computes the optimal TAP flow X for configuration y
        min_{x} sum_a c_a: TAP(x), y=0 => x=0, c_a= x_a.t(x_a)
        then generate the OA cuts: c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) for X_a > 0
        and add them as constraints to the model
        """
        tstt = self.net.tapas('UE', ysol)  # @todo is it 'SO' or 'UE' ???
        logging.debug(f"TAP: UE={tstt}")
        # oacut = {a: GBModel.get_SOcut_poly(a, a.x) for a in self.net.links if a.y == 1}
        oacut = {}
        for a in self.net.links:
            if a.y == 1:
                c0, c1 = GBModel.get_SOcut_poly(a, a.x)
                oacut[a] = (self.cvar[a] >= c0 + c1 * self.xvar[a])
        return oacut

    def generateOAevenly(self, ncuts: int):
        """
        computes the optimal TAP flow for configuration y and cost perturbed with u
        f(y) = min_{x} x.(t(x) + u[0] + u[1]*x): TAP(x), y=0 => x=0

        Returns:
            tstt (float): the 'UE' flow cost
            f(y) (float): the 'costtype' flow cost
            sx (dict): the optimal flow solution
        """
        assert not self.costmodel, f"constraints c(x)=x.t(x) already generated: {self.costmodel}"
        delta = self.net.TD / ncuts
        x = 0
        for i in range(ncuts):
            x += delta
            for a in self.net.links:
                c0, c1 = GBModel.get_SOcut_poly(a, x)
                self.minlp.addConstr(self.cvar[a] >= c0 + c1 * self.xvar[a])
        logging.debug(f"generate {ncuts * len(self.net.links)} OA cuts")
        self.costmodel = f"OA{ncuts}"

    @staticmethod
    def build_model(network: Network):
        minlp = gp.Model('SODNDP_agg')

        yvar = minlp.addVars(network.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")

        xvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=network.TD, name="x")
        minlp.addConstrs((xvar[a] <= yvar[a] * network.TD for a in network.links2), name="M")

        xsvar = minlp.addVars(network.links, network.zones, vtype=GRB.CONTINUOUS, lb=0.0, ub=network.TD, name="xs")
        minlp.addConstrs((sum(xsvar[a, s] for a in i.outgoing)
                          - sum(xsvar[a, s] for a in i.incoming) == GBModel.get_demand(network, i, s)
                          for i in network.nodes for s in network.zones), name="D")

        minlp.addConstrs((xsvar.sum(a, '*') == xvar[a] for a in network.links), name="A")

        # xub = network.TD
        # cub = [xub * a.getTravelTime(xub, 'UE') for a in network.links]
        cvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
        minlp.setObjective(cvar.sum(), GRB.MINIMIZE)
        minlp.update()
        minlp.write('model.lp')
        return minlp, yvar, xvar, cvar

    def addNLcost(self):
        """ gurobi does not allow yet to add the convex polynomial constraint c >= x.t(x)...
        thus we add the nonconvex constraint c == x.t(x) instead """
        if self.costmodel:
            logging.debug(f"constraints c(x)=x.t(x) already generated: {self.costmodel}")
            return
        for a in self.net.links:
            self.minlp.addGenConstrPoly(self.xvar[a], self.cvar[a], GBModel.get_SO_poly_reverse(a))
        self.costmodel = "NL"

    def getsol(self, vardict: dict, isint=False):
        assert self.minlp.status == GRB.OPTIMAL
        return {a: int(v.x) if isint else v.x for (a, v) in vardict.items()}

    def getSolution(self, varname='y'):
        if varname == 'x':
            return self.getsol(self.xvar)
        if varname == 'c':
            return self.getsol(self.cvar)
        return self.getsol(self.yvar, isint=True)

    def setYsolution(self, ysol: dict[Link, int]):
        for (a, y) in self.yvar.items():
            if ysol[a] == 1:
                y.lb = 1
                y.ub = 1
            else:
                y.lb = 0
                y.ub = 0

    def setXsolution(self, flow: dict[Link, float], tol=1e-6):
        for a in self.net.links:
            self.xvar[a].lb = flow[a] - tol
            self.xvar[a].ub = flow[a] + tol

    def solve(self, otype: str, ncuts=10):
        """Solve the convex relaxation model cvxmodel."""
        cost = 0
        cback = None
        self.minlp.setParam(GRB.Param.OutputFlag, False)
        if otype == 'pwl':
            self.addNLcost()
            self.minlp.setParam(GRB.Param.FuncNonlinear, 0)
            # self.minlp.setParam(GRB.Param.FuncPieceRatio, 0)
            # self.minlp.setParam(GRB.Param.FuncPieces, -1)
            # self.minlp.setParam(GRB.Param.FuncPieceError, 1e-2)
        elif otype == 'nl':
            self.addNLcost()
            self.minlp.setParam(GRB.Param.FuncNonlinear, 1)
        elif otype == 'oa':
            self.generateOAevenly(ncuts)
        elif otype == 'oad':
            self.generateOAevenly(ncuts)
            self.minlp.Params.LazyConstraints = 1
            cback = DNDPOACallback(self)

        self.minlp.optimize(cback)

        if self.minlp.status == GRB.INFEASIBLE:
            print(f'no solution found write IIS file {str(IISFILE)}')
            self.minlp.computeIIS()
            self.minlp.write(str(IISFILE))

        if self.minlp.status != GRB.OPTIMAL:
            print('Optimization was stopped with status %d' % self.minlp.status)
        else:
            cost = self.minlp.objval
            bctr = self.minlp.getConstrByName("B")
            runtime = self.minlp.runtime
            print(f"{otype}: solution cost={cost} slack={bctr.Slack} ({bctr.RHS}) runtime={runtime:.2f}")
        return cost

    def simulate_n_checkNLP(self, ysol: dict, yname: str):
        print(f"-- simulate solution {yname}")
        stime = time.time()
        self.net.resetTapas()
        tstt = self.net.tapas('UE', ysol)
        xsol = {a: a.x for a in self.net.links}
        print(f"TAPAS: {xsol}")
        runtime = time.time() - stime
        print(f"solution cost= {tstt}  runtime={runtime:.2f}")

        print(f"-- check full solution (y, xTAP) in NLP")
        self.setYsolution(ysol)
        self.setXsolution(xsol)
        nlpcost = self.solve(otype='nl')
        return tstt, nlpcost


class DNDPOACallback:

    def __init__(self, gbm: GBModel):
        self.gbm = gbm

    # noinspection PyBroadException
    def __call__(self, m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                ncuts = self.add_oacuts_at_mipsol(m)
                costmip = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
                currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
                currentphase = int(m.cbGet(GRB.Callback.MIPSOL_PHASE))
                print(f"MIPSOL #{currentnode} P{currentphase}: obj={costmip} lb={currentlb:.2f} ncuts={ncuts}")

            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                m.terminate()

    def add_oacuts_at_mipsol(self, m):
        xsol = m.cbGetSolution(self.gbm.xvar)
        csol = m.cbGetSolution(self.gbm.cvar)
        oacuts = {a: GBModel.get_SOcut_poly(a, xsol[a]) for a in self.gbm.net.links}
        slackoa = {a: c0 + c1 * xsol[a] - csol[a] for a, (c0, c1) in oacuts.items()}
        # c0 + c1*x = x*a.getTravelTime(x, 'UE') = x*(t + c*x^e) with t, e, c = GBModel.get_traveltime_func(a)
        print(f"max slackoa ={max(slackoa.values())}")
        for a, (c0, c1) in oacuts.items():
            if csol[a] < c0 + c1 * xsol[a]:
                m.cbLazy(self.gbm.cvar[a] >= c0 + c1 * self.gbm.xvar[a])
        return len(oacuts)


if __name__ == "__main__":
    net = 'SiouxFalls'
    ins = 'SF_DNDP_10_1'
    datadir = ROOTDIR + "data/" + net + "/"
    ntk = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
    print(net, ins)

    otypes = {"nl": "nonlinear c(x)=x.t(x)",
              "pwl": "pwl approximation",
              "oa": "oa relaxation",
              "oad": "oa dynamic relaxation"}

    print("\n-- solve PWL approx --")
    model = GBModel(ntk)
    pwlcost = model.solve(otype='pwl')
    pwlsol = model.getSolution()
    pwltstt, pwlnlpcost = model.simulate_n_checkNLP(pwlsol, "yPWL")

    noacuts = 100
    print(f"\n-- solve OA relaxation {noacuts} ctrs/arc  --")
    modelOA = GBModel(ntk)
    oacost = modelOA.solve(otype='oa', ncuts=noacuts)
    oasol = modelOA.getSolution()
    oatstt, oanlpcost = model.simulate_n_checkNLP(oasol, "yOA")

    noacuts_init = 3
    print(f"\n-- solve OA dynamic relaxation {noacuts_init} ctrs/arc  --")
    modelOAD = GBModel(ntk)
    oadcost = modelOAD.solve(otype='oad', ncuts=noacuts_init)
    oadsol = modelOAD.getSolution()
    oadtstt, oadnlpcost = model.simulate_n_checkNLP(oadsol, "yOAD")
    oadflow = modelOAD.getSolution('x')
    nlpflow = model.getSolution('x')
    print(f"oad: {oadflow}")
    print(f"nlp: {nlpflow}")
    print("diff:")
    print({a: abs(oadflow[a] - nlpflow[a]) for a in oadflow})

    print(f"-- check full solution (yOAD, xOAD) in NLP")
    modelNLP = GBModel(ntk)
    modelNLP.setYsolution(oadsol)
    modelNLP.setXsolution(oadflow, tol=1e-12)
    oadnlpcost2 = modelNLP.solve(otype='nl')

    print(f"\n---------------------- PWL")
    print(f"PWL approx solution yPWL: {pwlsol}")
    print(f"cost= PWL approx: {pwlcost}, TAP(yPWL): {pwltstt}, NLP(yPWL): {pwlnlpcost}")

    print(f"\n---------------------- OA {noacuts} cuts/arc")
    print(f"OA relaxed solution yOA: {oasol}")
    print(f"cost= OA relax: {oacost}, TAP(yOA): {oatstt}, NLP(yOA): {oanlpcost}")

    print(f"\n---------------------- OAD {noacuts_init} cuts/arc")
    print(f"OA dynamic solution yOA: {oadsol}")
    print(f"cost= OAD relax: {oadcost}, TAP(yOAD): {oadtstt}, NLP(yOAD): {oadnlpcost}, NLP(yOAD,xOAD): {oadnlpcost2}")

# @todo why TAP(yOAD) != OAD* = OAD(yOAD) = NLP(yOAD, xOAD) ? Bad use of msa ?
