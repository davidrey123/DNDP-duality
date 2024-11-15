#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:57:46 2024
@author: Sophie Demassey

convex MINLP model within Gurobi API for the value function reformulation of SO-DNDP:
min_{sum_a c_a: c_a >= f_a(x_a), g.y <= B, x <= M.y, g(x) <= 0} with f_a(x_a)= x_a.t_a(x_a)
and g(x)=min v: sum_w h_w <= x + v.1, sum_pi h_pi = d^w is a proxy function for "x is feasible TAP flow"
this function is convex and a subgradient of g(x*) is -nu* with nu*_a= f'_a(x*_a)/F, F=sum_a f'_a(x*_a)
The model is solved by different methods to generate the points x* where to evaluate the OA cuts:
c_a >= x*_a.t(x*_a) + [t(x*_a) + x*_a.dt(x*_a)].(x_a - x*_a) and g(x*) - nu*.(x-x*) <= 0
Note that when x* is a feasible flow for TAP then g(x*)=0 and the second OA cut can be rewritten nu*.(x-x*) >= 0
- LP-NLP BB: progressive OA cut generation at each integer node y* for x* optimum of SO-TAP(y*)
- OA algorithm: iterative OA cut generation at each optimal relaxed solution y* for x* optimum of SO-TAP(y*)
"""
import logging
import time
from pathlib import Path
import matplotlib.pylab as plt

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


class GBInterModel:

    def __init__(self, network: Network):
        self.net = network
        self.allclose = {a: 0 for a in network.links2}
        self.bigm = network.TD
        self.minlp, self.yvar, self.cvar, self.xvar = GBInterModel.build_model(network)
        self.costmodel = None

    @staticmethod
    def build_model(network: Network):
        minlp = gp.Model('SODNDPval')
        yvar = minlp.addVars(network.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")
        xvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=network.TD, name="x")
        mctrs = minlp.addConstrs((xvar[a] <= yvar[a] * network.TD for a in network.links2), name="M")
        cvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
        minlp.setObjective(cvar.sum(), GRB.MINIMIZE)
        minlp.update()
        return minlp, yvar, cvar, xvar

    @staticmethod
    def get_traveltime_func(link: Link):
        """ f(x) = t + c.x^e """
        t = link.t_ff
        e = link.beta
        c = t * link.alpha / pow(link.C, e)
        return t, e, c

    @staticmethod
    def get_SOcut_poly(link: Link, x: float):
        """ c(x) >= c(X) + c'(X).(x-X) = c0 + c1.x with
        c(x) = t.x + c.x^(e+1), c'(x) = t + c.(e+1).x^e
        c1= c'(X) = t + c.(e+1).X^e
        c0 = c(X)-c'(X).X = t.X + c.X^(e+1) - (t.X + c.(e+1).X^(e+1)) =  - c.e.X^(e+1)
        """
        t, e, c = GBInterModel.get_traveltime_func(link)
        c1 = t + c * (e + 1) * pow(x, e)  # = a.getTravelTime(a.x, 'SO') =  t  + c.x^e + c.e.x^e
        c0 = - c * e * pow(x, e + 1)  # = -pow(a.x,2) * a.getDerivativeTravelTime(a.x) = - x^2.c.e.x^(e-1)
        return c0, c1

    def generate_OAcuts(self, ysol: dict):
        """
        generate the OA cut g(x*) - nu*.(x-x*) <= 0 for x* being the optimal flow for SO-TAP(y*)
        note that since x* is feasible flow for TAP, then g(x*)=0 and the cut reads nu*.(x-x*) >= 0
        with nu*_a= f'_a(x*_a)/F, F=sum_a f'_a(x*_a)
        solves SO-TAP(y*) to get the optimal primal and dual solutions x* and nu*
        """
        tstt = self.net.tapas('SO', ysol)
        logging.info(f"TAP: SO={tstt}")
        xsol = {a: a.x for a in self.net.links}
        nusol = self._generate_feas_OAcuts()
        nusum = sum(nusol.values())
        assert nusum > 0
        oacuts = self._generate_SOcost_OAcuts()
        return tstt, xsol, nusol, nusum, oacuts

    def _generate_feas_OAcuts(self):
        """
        generate the OA cut g(x*) - nu*.(x-x*) <= 0 for x* being the optimal flow for SO-TAP(y*)
        with nu*_a= f'_a(x*_a)/F, F=sum_a f'_a(x*_a)
        solves SO-TAP(y*) to get the optimal primal and dual solutions x* and nu*
        """
        return {a: a.gettt(a.x, 'SO') for a in self.net.links}

    def _generate_SOcost_OAcuts(self):
        """
        computes the optimal SO-TAP flow X for configuration y
        min_{x} sum_a c_a: TAP(x), y=0 => x=0, c_a= x_a.t(x_a)
        then generate the OA cuts: c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) for X_a > 0
        and add them as constraints to the model
        """
        oacut = {}
        for a in self.net.links:
            if a.y == 1:
                c0, c1 = GBInterModel.get_SOcut_poly(a, a.x)
                oacut[a] = (self.cvar[a] >= c0 + c1 * self.xvar[a])
        return oacut

    def getsol(self, vardict: dict, isint=False):
        assert self.minlp.status == GRB.OPTIMAL
        return {a: round(v.x) if isint else v.x for (a, v) in vardict.items()}

    def getSolution(self, varname='y'):
        return self.getsol(self.yvar, isint=True)

    def solveOA(self):
        """ run the OA algorithm on model min_{u: u >= v(y), g.y <= B}
        iterate on: solve the relaxed MILP at optimality; solve the restricted NLP; generate the new OA cuts. """
        runtime = time.perf_counter()
        milptimes = 0
        MAX_ITER = 1000
        OATOL = 1e-4
        lb = 0
        iters = {"lb": [], "tap": [], "ub": [], "mipt": [], "t": []}
        ub = GRB.INFINITY

        for i in range(MAX_ITER):
            self.minlp.optimize()
            milptimes += self.minlp.runtime

            if self.minlp.status != GRB.OPTIMAL:
                print('Optimization was stopped with status %d' % self.minlp.status)
                break

            lb = self.minlp.objval
            iters["lb"].append(lb)
            iters["mipt"].append(milptimes)

            if lb != 0 and (ub - lb)/lb < OATOL:
                print(f"OA STOP: abs gap = {ub}-{lb}={ub-lb}; rel gap < {OATOL}")
                break

            ysol = self.getSolution()
            tstt, xsol, nusol, nusum, oacuts = self.generate_OAcuts(ysol)
            iters["tap"].append(tstt)
            iters["ub"].append(min(ub, tstt))
            iters["t"].append(time.perf_counter()-runtime)
            if ub > tstt:
                ub = tstt
                self.cvar.ub = tstt
                print(f"OA new incumbent: {ub} y={ysol}")
                if lb != 0 and (ub - lb) / lb < OATOL:
                    print(f"OA STOP: abs gap = {ub}-{lb}={ub - lb}; rel gap < {OATOL}")
                    break
            self.minlp.addConstr(gp.quicksum(nusol[a] * (self.xvar[a] - xsol[a]) for a in self.net.links) >= 0)
            self.minlp.addConstrs(c for a, c in oacuts.items())
            self.minlp.write("model.lp")
            print(f"OA it {i}: ub={ub} lb={lb} tap={tstt} runtime={time.perf_counter()-runtime:.2f}")

        runtime = time.perf_counter() - runtime
        print(f"oa: solution ub={ub} lb={lb} gap={ub-lb} "
              f"milptime={milptimes:.2f} runtime={runtime:.2f} it={len(iters['lb'])}")
        # self.minlp.write("model.lp")
        self.show_iters(iters)
        return ub, runtime

    def solve(self, otype: str):
        """Solve SODNDP:  min_{u: u >= v(y), g.y <= B}
        by handling the NL constraints u >= v(y) in different ways according to otype:
        OAT: LP-NLP B&B algorithm, OA: OA algorithm. """
        cost = 0
        cback = None
        # self.minlp.setParam(GRB.Param.OutputFlag, False)
        if otype == 'oat':
            self.minlp.Params.LazyConstraints = 1
            cback = DNDPinterCallback(self)
        elif otype == 'oa':
            self.minlp.setParam(GRB.Param.OutputFlag, False)
            return self.solveOA()

        self.minlp.optimize(cback)

        if self.minlp.status == GRB.INFEASIBLE:
            print(f'no solution found write IIS file {str(IISFILE)}')
            self.minlp.computeIIS()
            self.minlp.write(str(IISFILE))

        runtime = self.minlp.runtime

        if self.minlp.status != GRB.OPTIMAL:
            print('Optimization was stopped with status %d' % self.minlp.status)
        else:
            cost = self.minlp.objval
            bctr = self.minlp.getConstrByName("B")
            print(f"{otype}: solution cost={cost} slack={bctr.Slack} ({bctr.RHS}) runtime={runtime:.2f}")
        return cost, runtime

    @staticmethod
    def show_iters(iters):
        dim = 2
        nits = len(iters['lb'])
        fig, axes = plt.subplots(nrows=1, ncols=dim, figsize=(20, 3))
        date = time.strftime("%y-%m-%d-%H:%M", time.gmtime())
        fig.suptitle(f"OA {date} - cpu={iters['t'][-1]:.1f} it={nits}", fontsize=10)
        colors = "rgbcmyrgbcmyrgbcmy"
        axes[0].plot(iters['lb'], color='b', label='lb')
        axes[0].plot(iters['tap'], color='g', label='tap')
        axes[0].plot(iters['ub'], color='r', label='ub')
        axes[1].plot(iters['t'], color='r', label='time')
        axes[1].plot(iters['mipt'], color='b', label='mip')
        fig.tight_layout()
        fig.legend()
        plt.savefig('iter_oa_y.png')


class DNDPinterCallback:
    """ a callback to generate OA cuts c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) forall a
    for X being the flow solution of TAP(Y) with Y the integer solution associated to a given MIPSOL node. """

    def __init__(self, gbm: GBInterModel):
        self.gbm = gbm
        self.ub = GRB.INFINITY

    # noinspection PyBroadException
    def __call__(self, m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                tstt, ysol, xsol, ncuts = self.add_SOcost_oacuts_at_mipsol(m)
                costmip = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
                currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
                currentphase = int(m.cbGet(GRB.Callback.MIPSOL_PHASE))
                print(f"MIPSOL #{currentnode} P{currentphase}: obj={costmip} lb={currentlb:.2f} ncuts={ncuts}")
                # if currentnode > 0:
                #    self.setTAPsol(m, ysol, xsol)

            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                m.terminate()

    def add_SOcost_oacuts_at_mipsol(self, m):
        ysol = {a: round(ya) for (a, ya) in m.cbGetSolution(self.gbm.yvar).items()}
        tstt, xsol, nusol, nusum, oacuts = self.gbm.generate_OAcuts(ysol)
        m.cbLazy(gp.quicksum(nusol[a] * (self.gbm.xvar[a] - xsol[a]) for a in self.gbm.net.links) >= 0)
        for a, c in oacuts.items():
            m.cbLazy(c)
            if self.ub > tstt:
                self.ub = tstt
                print(f"new ub={tstt} y={ysol}")
            #    m.cbLazy(self.gbm.cvar <= self.ub)

        return tstt, ysol, xsol, len(oacuts)+1


def solve(ntk, otype, noacuts=0):
    zestr = otypes[otype]
    if noacuts:
        zestr += " {noacuts} ctrs/arc"
    print(f"\n\n-- solve {zestr} --")
    model = GBInterModel(ntk)
    cost, time = model.solve(otype)
    ysol = model.getSolution()
    return {'c': cost, 't': time, 'y': ysol}


def printresults(results):
    for otype, res in results.items():
        zestr = f"cost: {res['c']} "
        if res.get('tap'):
            zestr += f"TAP(y{otype}): {res['tap']}"
        print(f"\n---------------------- {otype} {res['t']:.2f} s")
        print(f"y{otype}: {res['y']}")
        print(zestr)


if __name__ == "__main__":
    net = 'SiouxFalls'
    ins = 'SF_DNDP_10_1'
    datadir = ROOTDIR + "data/" + net + "/"
    netwk = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
    print(net, ins)

    otypes = {"oa": "oa algorithm",
              "oat": "lpnlp algorithm"}

    netwk.params.min_gap = 1e-4
    netwk.params.warmstart = True

    result = {}
    result['oa'] = solve(netwk, 'oa')  # @todo bug in the computation of the lower cost
    # result['oat'] = solve(netwk, 'oat')  # @todo misuse of cbLazy as it cutoffs the integer node

    printresults(result)
