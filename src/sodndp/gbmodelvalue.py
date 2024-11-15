#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:57:46 2024
@author: Sophie Demassey

convex MINLP model within Gurobi API for the value function reformulation of SO-DNDP:
min_{u: u >= v(y), g.y <= B} with v(y):= min_{sum_a f_a(x_a): TAP(x), x <= M.y} and f_a(x_a) = x_a.t_a(x_a)
the value function v is convex:
v(y*)= l(y*,mu*) = max_mu l(y*,mu) = max_mu min { x.(t(x) + mu) - M.mu.y*: TAP(x) } then -M.mu* subgradient of v at y*
with mu*_a = max(0, max_w -f'_a(x*_a) + p*_{wj} - p*_{wi}) for a=(i,j) if y*_a=0, and mu*_a = 0 if y*_a = 1
where p* are the "node prices" of SO-TAP(y*): p*_{wj} is the min distance from origin(w) to j in DAG(V,A(y*),f_a(x_a*))
The model is solved by different methods to generate the points y* where to evaluate v and generate the OA cuts
u >= v(y*) - M.mu*.(y - y*) = v(y*) - M.mu*.y:
- GBD/OA: iterative OA cut generation at each optimal relaxed solution y*
- LP-NLP BB: progressive OA cut generation at each integer node y*

@todo the disaggregate variant
"""
import logging
import time
from pathlib import Path
import matplotlib.pylab as plt

import gurobipy as gp
from gurobipy import GRB

from src.cvxsolver.cvxsolver import Oracle
from src.cvxsolver.proximalbundle import ProximalBundle
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


class GBValModel:

    def __init__(self, network: Network):
        self.net = network
        self.allclose = {a: 0 for a in network.links2}
        self.bigm = network.TD
        self.minlp, self.yvar, self.cvar = GBValModel.build_model(network)
        self.costmodel = None
        self.memtapas = None

    def sotapas(self, ysol):
        if self.memtapas is None:
            tstt = self.net.tapas('SO', ysol)
        else:
            ystr = ''.join(ysol.values())
            tstt = self.memtapas.get(ystr)
            if tstt is None:
                tstt = self.net.tapas('SO', ysol)
                self.memtapas[ystr] = tstt
        return tstt

    @staticmethod
    def build_model(network: Network):
        minlp = gp.Model('SODNDPval')
        yvar = minlp.addVars(network.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")
        cvar = minlp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="u")
        minlp.setObjective(cvar, GRB.MINIMIZE)
        minlp.update()
        return minlp, yvar, cvar

    def compute_val_OAcut(self, ysol: dict):
        """
        generate the OA cut u >= v(y*) - M.mu*.(y - y*) = v(y*) - M.mu*.y:
        solves SO-TAP(y*) to get the optimal value v(y*):= min_{sum_a f_a(x_a): TAP(x), x <= M.y}
        and the "node prices": p*_{wj} = min distance from origin(w) to j in DAG(V,A(y*),f_a(x_a*))
        to compute the multipliers mu*:
        mu*_a = max(0, max_w -f'_a(x*_a) + p*_{wj} - p*_{wi}) for a=(i,j) if y*_a=0, and mu*_a = 0 if y*_a = 1
        note that the OA cut can be rewritten u >= v(y*) - M.mu*.y since mu*.y*=0
        """
        tstt = self.net.tapas('SO', ysol)
        logging.info(f"TAP: SO={tstt}")
        xsol = {a: a.x for a in self.net.links}
        musol = {a: 0 for a in self.net.links2}
        for r in self.net.origins:
            self.net.dijkstras(r, 'SO')
            for a in musol.keys():
                if ysol[a] == 0:
                    slack = a.gettt(a.x, 'SO') - (a.end.cost - a.start.cost)
                    musol[a] = max(musol[a], -slack)
        return tstt, musol, xsol

    def getsol(self, vardict: dict, isint=False):
        assert self.minlp.status == GRB.OPTIMAL
        return {a: round(v.x) if isint else v.x for (a, v) in vardict.items()}

    def getSolution(self, varname='y'):
        return self.getsol(self.yvar, isint=True)

    def getFullSolution(self):
        return [v.x for v in self.minlp.getVars()]

    def setYsolution(self, ysol: dict[Link, int]):
        for (a, y) in self.yvar.items():
            if ysol[a] == 1:
                y.lb = 1
                y.ub = 1
            else:
                y.lb = 0
                y.ub = 0

    def ksheuristic(self):
        budget = 0
        arcs = list(self.net.links2)
        arcs.sort(key=lambda arc: arc.cost)
        ysol = {a: 0 for a in self.net.links2}
        for a in arcs:
            if budget + a.cost > self.net.B:
                break
            ysol[a] = 1
            budget += a.cost
        return ysol

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
            tstt, musol, xsol = self.compute_val_OAcut(ysol)
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
            self.minlp.addConstr(self.cvar >= tstt - self.bigm * self.yvar.prod(musol))

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
            cback = DNDPvalueCallback(self)
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

    def simulate(self, ysol: dict, yname: str):
        """Optimize SO-TAP for a given design y: min_{x} sum_a c_a: TAP(x), y=0 => x=0, c_a= x_a.t(x_a)."""
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
        tstt = tstttapas
        print(f"solution cost= {tstt}  runtime={runtime:.2f}")
        return tstt

    def checkNodePrices(self, ysol: dict, yname: str):
        """ collect the node prices associated to a solution of TAP(y). """
        eps = 1e-6
        print(f"-- simulate solution {yname}: {ysol}")
        stime = time.time()
        tstttapas = self.net.tapas('SO', ysol)
        runtime = time.time() - stime
        nodeprices = {}
        for r in self.net.origins:
            self.net.dijkstras(r, 'SO')
            nodeprices[r] = {n: n.cost for n in self.net.nodes}
            for a in self.net.links:
                tt = a.gettt(a.x, 'SO')
                rc = a.end.cost - a.start.cost
                diff = tt - rc
                if -eps < diff < eps:
                    print(f"{a}: y={a.y} x={a.x}f'={tt} r={r} rc={rc} diff={diff} rci={a.start.cost} rcj={a.end.cost}")

    def getbigMmultipliers(self, ysol: dict):
        """ compute the optimal multipliers of the constraint x<= My. """
        tstt = self.net.tapas('SO', ysol)
        mu = {a: 0 for a in self.net.links2}
        for r in self.net.origins:
            self.net.dijkstras(r, 'SO')
            for a in mu.keys():
                if ysol[a] == 0:
                    slack = a.gettt(a.x, 'SO') - (a.end.cost - a.start.cost)
                    mu[a] = max(mu[a], -slack)
        return tstt, mu

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


class DNDPvalueCallback:
    """ a callback to generate an OA cut u >= v(y*) - M.mu*.(y - y*) at each integer node y*. """

    def __init__(self, gbm: GBValModel):
        self.gbm = gbm
        self.ub = GRB.INFINITY
        self.ycvars = [v for a, v in self.gbm.yvar.items()] + [self.gbm.cvar]
        self.ycvals = None

    # noinspection PyBroadException
    def __call__(self, m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                ysol, tstt = self.add_val_oacuts_at_mipsol(m)
                costmip = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                currentub = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if currentub >= GRB.INFINITY:
                    currentub = 0
                currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
                currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
                currentphase = int(m.cbGet(GRB.Callback.MIPSOL_PHASE))
                print(f"MIPSOL #{currentnode} P{currentphase}: "
                      f"obj={costmip} ub={currentub:.2f} lb={currentlb:.2f} tap={tstt:.2f}")

            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                m.terminate()

        elif where == GRB.Callback.MIPNODE:
            self.set_mipsol(m)

    def add_val_oacuts_at_mipsol(self, m):
        ysol = {a: round(ya) for (a, ya) in m.cbGetSolution(self.gbm.yvar).items()}
        tstt, musol, xsol = self.gbm.compute_val_OAcut(ysol)
        m.cbLazy(self.gbm.cvar >= tstt - self.gbm.bigm * self.gbm.yvar.prod(musol))
        if not self.ycvals or self.ycvals[-1] > tstt + 1e-12:
            self.ycvals = [ysol[a] for a, v in self.gbm.yvar.items()] + [(tstt+1e-12)]
        if self.ub > tstt:
            self.ub = tstt
            print(f"new ub={tstt} y={ysol}")
        return ysol, tstt

    def set_mipsol(self, m):
        if self.ycvals:
            m.cbSetSolution(self.ycvars, self.ycvals)
            usesolution = m.cbUseSolution()
            print(f"solution {usesolution}: {self.ycvals}")
            self.ycvals = None


class DNDPvalueOracle(Oracle):
    """ Concrete oracle for the value function of the SO-TAP problem:
    v(y):= min_{sum_a f_a(x_a): TAP(x), x <= M.y} and f_a(x_a) = x_a.t_a(x_a)
    v is convex and a subgradient at y* is -M.mu*
    with mu*_a = max(0, max_w -f'_a(x*_a) + p*_{wj} - p*_{wi}) for a=(i,j) if y*_a=0, and mu*_a = 0 if y*_a = 1
    where p* are the "node prices" of SO-TAP(y*):
    p*_{wj} is the min distance from origin(w) to j in DAG(V,A(y*),f_a(x_a*))
    """

    def __init__(self, id_: str, gbm: GBValModel):
        Oracle.__init__(self, id_, positive_quadrant=True, integer_points=True)
        self.gbm = gbm

    def oracle(self, y: list):
        """ get the zero and first information of v at point y

         Args:
             y (list[int]): point where to evaluate the function v

         Returns:
               v(y): (float) = min_{sum_a f_a(x_a): TAP(x), x <= M.y} optimal value of SO-TAP
               g: (list) a subgradient of v at y: -M.mu
               x: (list) flow solution x of SO-TAP(y)
        """
        ysol = {a: round(y[i]) for i, a in enumerate(self.gbm.net.links2)}
        logging.debug(f"candidate {ysol}")
        tstt, musol, xsol = self.gbm.compute_val_OAcut(ysol)
        g = [-self.gbm.bigm * musol[a] for a in musol.keys()]
        x = list(xsol.values())
        return tstt, g, x

    def has_lb(self):
        return False

    def eval_lb(self):
        return False

    def add_primal_constraint(self):
        return [a.cost for a in self.gbm.net.links2] + [self.gbm.net.B]


def solve(ntk, otype, noacuts=0):
    zestr = otypes[otype]
    if noacuts:
        zestr += " {noacuts} ctrs/arc"
    print(f"\n\n-- solve {zestr} --")
    model = GBValModel(ntk)
    if otype == 'pb':
        starttime = time.perf_counter()
        solver = ProximalBundle(DNDPvalueOracle(ins, model))
        yinit = [0 for _ in ntk.links2]
        cost, ysol = solver.solve(yinit)
        cpu = time.perf_counter() - starttime
        logging.info(f"best value {cost}")
        logging.info(f"best integer solution {ysol}")
        solver.show_iters()
    else:
        cost, cpu = model.solve(otype)
        ysol = model.getSolution()
    return {'c': cost, 't': cpu, 'y': ysol}


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
              "oat": "lpnlp algorithm",
              "pb": "integer proximal bundle algorithm"}

    netwk.params.min_gap = 1e-4
    netwk.params.warmstart = True

    result = {}
    result['oa'] = solve(netwk, 'oa')
    # result['oat'] = solve(netwk, 'oat')  # @todo cbLazy cutoffs the integer node
    # result['pb'] = solve(netwk, 'pb')  # @todo integer bundle is not ready yet

    printresults(result)
