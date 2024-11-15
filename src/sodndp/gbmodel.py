#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:57:46 2024
@author: Sophie Demassey

convex MINLP model within Gurobi API for SO-DNDP:
min_{x,y} c : TAP(x), g.y <= B, x <= M.y, c_a >= x_a.t_a(x_a)
solved with different methods for handling the NL constraints c_a >= x_a.t_a(x_a):
- Gurobi default nonconvex MINLP solver: c_a == x_a.t_a(x_a)
- Gurobi default PWL approximation: c_a == PWL(x_a.t_a(x_a))
- OA relaxation: c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) for X generated evenly
- progressive OA cut generation at each integer node Y for X the current relaxed MILP flow solution
- LP-NLP BB: progressive OA cut generation at each integer node Y for X optimum of SO-TAP(Y)
- OA algorithm: iterative OA cut generation at each optimal relaxed solution Y for X optimum of SO-TAP(Y)

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


class GBModel:

    def __init__(self, network: Network):
        self.net = network
        self.allclose = {a: 0 for a in network.links2}
        self.minlp, self.yvar, self.xvar, self.cvar, self.mctrs = GBModel.build_model(network)
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

    def generate_SOcost_OAcuts(self, ysol: dict):
        """
        computes the optimal SO-TAP flow X for configuration y
        min_{x} sum_a c_a: TAP(x), y=0 => x=0, c_a= x_a.t(x_a)
        then generate the OA cuts: c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) for X_a > 0
        and add them as constraints to the model
        """
        # tstt = self.net.msa('SO', ysol, self.allclose)
        tstt = self.net.tapas('SO', ysol)
        logging.info(f"TAP: SO={tstt}")
        # oacut = {a: GBModel.get_SOcut_poly(a, a.x) for a in self.net.links if a.y == 1}
        oacut = {}
        for a in self.net.links:
            if a.y == 1:
                c0, c1 = GBModel.get_SOcut_poly(a, a.x)
                oacut[a] = (self.cvar[a] >= c0 + c1 * self.xvar[a])
        flowsol = {a: a.x for a in self.net.links}
        return tstt, oacut, flowsol

    def generate_SOcost_OActrs_evenly(self, ncuts: int, small=False):
        """
        generate the supporting planes for functions x_a.t(x_a) for all a:
        c_a >= X^i_a.t(X^i_a) + [t(X^i_a) + x_a.dt(X^i_a)].(x_a - X^i_a)
        for X^1_a,...,X^N_a taken evenly in the interval [0,ub] with N=ncuts
        """
        assert not self.costmodel, f"constraints c(x)=x.t(x) already generated: {self.costmodel}"
        maxcut = self.net.TD/len(self.net.zones) if small else self.net.TD
        delta = maxcut / ncuts
        x = 0
        for i in range(ncuts):
            x += delta
            for a in self.net.links:
                c0, c1 = GBModel.get_SOcut_poly(a, x)
                self.minlp.addConstr(self.cvar[a] >= c0 + c1 * self.xvar[a])
        logging.debug(f"generate {ncuts * len(self.net.links)} OA cuts")
        self.costmodel = f"OA{ncuts}"

    def generate_SOcost_NLctrs(self):
        """
        generate the nonlinear constraints c_a = x_a.t(x_a) for all a:
        note that gurobi does not allow yet to add the convex polynomial constraint c >= x.t(x)...
        thus we add the nonconvex constraint c == x.t(x) instead
        """
        if self.costmodel:
            logging.debug(f"constraints c(x)=x.t(x) already generated: {self.costmodel}")
            return
        for a in self.net.links:
            self.minlp.addGenConstrPoly(self.xvar[a], self.cvar[a], GBModel.get_SO_poly_reverse(a))
        self.costmodel = "NL"

    @staticmethod
    def build_model(network: Network):
        minlp = gp.Model('SODNDP_agg')

        yvar = minlp.addVars(network.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")

        xvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=network.TD, name="x")
        mctrs = minlp.addConstrs((xvar[a] <= yvar[a] * network.TD for a in network.links2), name="M")

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
        return minlp, yvar, xvar, cvar, mctrs

    def getsol(self, vardict: dict, isint=False):
        assert self.minlp.status == GRB.OPTIMAL
        return {a: round(v.x) if isint else v.x for (a, v) in vardict.items()}

    def getSolution(self, varname='y'):
        if varname == 'x':
            return self.getsol(self.xvar)
        if varname == 'c':
            return self.getsol(self.cvar)
        return self.getsol(self.yvar, isint=True)

    def getFullSolution(self):
        return [v.x for v in self.minlp.getVars()]

    def setStartSolution(self, vals: list):
        for i, v in enumerate(self.minlp.getVars()):
            v.start = vals[i]

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

    def solveOA(self):
        """ run the OA algorithm on model min_{x} sum_a c_a: TAP(x), y=0 => x=0, OA(c_a= x_a.t(x_a)):
        iterate on: solve the relaxed MILP at optimality; solve the restricted NLP; generate the new OA cuts. """
        runtime = time.perf_counter()
        milptimes = 0
        MAX_ITER = 10
        OATOL = 1e-4
        ub = GRB.INFINITY
        lb = 0
        ncuts = 0
        iters = {"lb": [], "tap": [], "ub": [], "mipt": [], "t": []}

        for i in range(MAX_ITER):
            self.minlp.optimize()
            milptimes += self.minlp.runtime

            if self.minlp.status != GRB.OPTIMAL:
                print('Optimization was stopped with status %d' % self.minlp.status)
                break

            lb = self.minlp.objval
            iters["lb"].append(lb)
            iters["mipt"].append(milptimes)

            if (ub - lb)/lb < OATOL:
                print(f"OA STOP: abs gap = {ub}-{lb}={ub-lb}; rel gap < {OATOL}")
                break

            ysol = self.getSolution()
            tstt, oacuts, flowsol = self.generate_SOcost_OAcuts(ysol)
            iters["tap"].append(tstt)
            iters["ub"].append(min(ub, tstt))
            iters["t"].append(time.perf_counter()-runtime)
            if ub > tstt:
                ub = tstt
                print(f"OA new incumbent: {ub} y={ysol}")
                if (ub - lb) / lb < OATOL:
                    print(f"OA STOP: abs gap = {ub}-{lb}={ub - lb}; rel gap < {OATOL}")
                    break

            # @ todo restart with SO-TAP solution / cutoff the cost

            for a, c in oacuts.items():
                self.minlp.addConstr(c)
            ncuts += len(oacuts)
            print(f"OA it {i}: ub={ub} lb={lb} ncuts={ncuts} "
                  f"milptime={milptimes:.2f} runtime={time.perf_counter()-runtime:.2f}")

        runtime = time.perf_counter() - runtime
        print(f"oa: solution ub={ub} lb={lb} gap={ub-lb} milptime={milptimes:.2f} runtime={runtime:.2f}")
        self.show_iters(iters)
        return ub, runtime

    def solve(self, otype: str, ncuts=10):
        """Solve SO-DNDP: min_{x} sum_a c_a: TAP(x), y=0 => x=0, c_a= x_a.t(x_a)
        by handling the NL constraints c_a= x_a.t(x_a) in different ways according to otype:
        NL: nonconvex NL model, PWL: PWL approx model, OAR: OA relaxation, OAD: OA dynamic cut generation,
        OAT: LP-NLP B&B algorithm, OA: OA algorithm. """
        cost = 0
        cback = None
        self.minlp.setParam(GRB.Param.OutputFlag, False)
        if otype == 'pwl':
            self.generate_SOcost_NLctrs()
            self.minlp.setParam(GRB.Param.FuncNonlinear, 0)
            # self.minlp.setParam(GRB.Param.FuncPieceRatio, 0)
            # self.minlp.setParam(GRB.Param.FuncPieces, -1)
            # self.minlp.setParam(GRB.Param.FuncPieceError, 1e-2)
        elif otype == 'nl':
            self.generate_SOcost_NLctrs()
            self.minlp.setParam(GRB.Param.FuncNonlinear, 1)
        elif otype == 'oar':
            self.generate_SOcost_OActrs_evenly(ncuts)
        elif otype == 'oad':
            self.generate_SOcost_OActrs_evenly(ncuts, small=True)
            self.minlp.Params.LazyConstraints = 1
            cback = DNDPOACallback(self)
        elif otype == 'oat':
            self.generate_SOcost_OActrs_evenly(ncuts, small=True)
            self.minlp.Params.LazyConstraints = 1
            cback = DNDPTAPCallback(self)
        elif otype == 'oa':
            self.generate_SOcost_OActrs_evenly(ncuts, small=True)
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

    def simulate_n_checkNLP(self, ysol: dict, yname: str):
        """Optimize SO-TAP for a given design y: min_{x} sum_a c_a: TAP(x), y=0 => x=0, c_a= x_a.t(x_a)
        then check the cost of the solution in the MINLP model. """
        print(f"-- simulate solution {yname}: {ysol}")
        stime = time.time()
        tstttapas = self.net.tapas('SO', ysol)
        min_gap = self.net.params.min_gap
        self.net.params.min_gap = max(min_gap, 1e-2)
        tsttmsa = self.net.msa('SO', ysol, self.allclose)
        self.net.params.min_gap = min_gap
        xsol = {a: a.x for a in self.net.links}
        print(f"TAPAS: {tstttapas} MSA: {tsttmsa}")
        runtime = time.time() - stime
        tstt = tsttmsa
        print(f"solution cost= {tstt}  runtime={runtime:.2f}")

        print(f"-- check full solution ({yname}, xTAP) in NLP")
        self.setYsolution(ysol)
        self.setXsolution(xsol)
        nlpcost, runtime = self.solve(otype='nl')
        return tstt, nlpcost

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
        plt.savefig('iter_oa_xyz.png')


class DNDPOACallback:
    """ a callback to generate OA cuts c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) forall a
    for X being the flow solution associated to a given MIPSOL node. """
    def __init__(self, gbm: GBModel):
        self.gbm = gbm

    # noinspection PyBroadException
    def __call__(self, m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                ncuts = self.add_SOcost_oacuts_at_mipsol(m)
                costmip = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
                currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
                currentphase = int(m.cbGet(GRB.Callback.MIPSOL_PHASE))
                print(f"MIPSOL #{currentnode} P{currentphase}: obj={costmip} lb={currentlb:.2f} ncuts={ncuts}")

            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                m.terminate()

    def add_SOcost_oacuts_at_mipsol(self, m):
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


class DNDPTAPCallback:
    """ a callback to generate OA cuts c_a >= X_a.t(X_a) + [t(X_a) + x_a.dt(X_a)].(x_a - X_a) forall a
    for X being the flow solution of TAP(Y) with Y the integer solution associated to a given MIPSOL node. """

    def __init__(self, gbm: GBModel):
        self.gbm = gbm

    # noinspection PyBroadException
    def __call__(self, m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                ncuts, ysol, xsol = self.add_SOcost_oacuts_at_mipsol(m)
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
        ub, oacuts, flowsol = self.gbm.generate_SOcost_OAcuts(ysol)
        for a, c in oacuts.items():
            m.cbLazy(c)
        return len(oacuts), ysol, flowsol

    def setTAPsol(self, m, ysol: dict[Link, float], flow: dict[Link, float]):
        assert len(ysol) == len(self.gbm.yvar) and len(flow) == len(self.gbm.xvar)
        xyvars = [v for a, v in self.gbm.xvar.items()] + [v for a, v in self.gbm.yvar.items()]
        xyvals = [flow[a] for a, v in self.gbm.xvar.items()] + [ysol[a] for a, v in self.gbm.yvar.items()]
        m.cbSetSolution(xyvars, xyvals)
        # m.cbUseSolution()


def solve(ntk, otype, noacuts=0):
    zestr = otypes[otype]
    if noacuts:
        zestr += f" {noacuts} ctrs/arc"
    print(f"\n\n-- solve {zestr} --")
    model = GBModel(ntk)
    cost, time = model.solve(otype)
    ysol = model.getSolution()
    xsol = model.getSolution('x')
    return {'c': cost, 't': time, 'y': ysol, 'x': xsol}


def simulate(ntk, otype, ysol, xsol):
    print(f"-- check solution y{otype} in NLP")
    mnlp = GBModel(ntk)
    tstt, nlpcost = mnlp.simulate_n_checkNLP(ysol, f"y+{otype}")
    nxsol = mnlp.getSolution('x')
    res = {'tap': tstt, 'nlp': nlpcost, 'nlpx': nxsol}

    print("diff flow {otype}/nlp:")
    print(f"{otype}: {xsol} \nnlp: {nxsol}\n diff:")
    print({a: abs(xsol[a] - nxsol[a]) for a in xsol})
    return res


def fullsimulate(ntk, otype, ysol, xsol):
    print(f"-- check full solution (y{otype}, x{otype}) in NLP")
    mnlp = GBModel(ntk)
    mnlp.setYsolution(ysol)
    mnlp.setXsolution(xsol, tol=1e-12)
    cost, time = mnlp.solve(otype='nl')
    return {'nlpxy': cost}


def solvensim(ntk, otype, noacuts=0, sim=False, fsim=False):
    res = solve(ntk, otype, noacuts)
    if sim:
        res.update(simulate(ntk, otype, res['y'], res['x']))
    if fsim:
        res.update(fullsimulate(ntk, otype, res['y'], res['x']))
    return res


def printresults(results):
    for otype, res in results.items():
        zestr = f"cost: {res['c']} "
        if res.get('tap'):
            zestr += f"TAP(y{otype}): {res['tap']}, NLP(y{otype}): {res['nlp']} "
        if res.get('nlpxy'):
            zestr += f"NLP(y{otype},x{otype}): {res['nlpxy']} "
        print(f"\n---------------------- {otype} {res['t']:.2f} s")
        print(f"y{otype}: {res['y']}")
        print(zestr)


if __name__ == "__main__":
    net = 'SiouxFalls'
    ins = 'SF_DNDP_10_1'
    datadir = ROOTDIR + "data/" + net + "/"
    netwk = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
    print(net, ins)

    otypes = {"nl": "nonlinear c(x)=x.t(x)",
              "pwl": "pwl approximation",
              "oar": "oa relaxation",
              "oad": "oa dynamic relaxation",
              "oa": "oa algorithm",
              "oat": "lpnlp algorithm"}

    netwk.params.min_gap = 1e-4
    netwk.params.warmstart = True

    result = {}
    result['oa'] = solvensim(netwk, 'oa', noacuts=3, sim=True, fsim=False)
    result['pwl'] = solvensim(netwk, 'pwl', sim=True, fsim=False)
    result['oar'] = solvensim(netwk, 'oar', noacuts=100, sim=True, fsim=False)
    result['oad'] = solvensim(netwk, 'oad', noacuts=3, sim=True, fsim=True)
    result['oat'] = solvensim(netwk, 'oat', noacuts=3, sim=True, fsim=False)

    printresults(result)
