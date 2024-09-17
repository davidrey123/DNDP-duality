#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:56:06 2024

Penalization and block coordination for SO-DNDP:
min_{x,y} f(x)= sum_a x_a.t_a(x_a): g.y <= B, y=0 => x=0, E.z^s = d^s, sum_s z^s <= x
by dualizing the flow coupling constraints sum_s z^s <= x

@author: Sophie Demassey
"""

import logging
import gurobipy as gp
from gurobipy import GRB

from src.cvxsolver.cvxsolver import CvxSolver, Oracle
from src.cvxsolver.subgradient import SubGradient
from src.cvxsolver.proximalbundle import ProximalBundle
from src.tapas import Network, Zone, Link

INDIR = "../../data/"
OUTDIR = "../../output/"

logging.basicConfig(
    handlers=[
        logging.StreamHandler()
    ],
    level=logging.WARN
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DualFlowOracle(Oracle):
    """Concrete oracle for min_{u>=0} -l(u) the opposite lagrangian dual function obtained by dualizing sum z_s <= x
    in SO-DNDP min_{x,y} f(x)= sum_a x_a.t_a(x_a): g.y <= B, y=0 => x=0, E.z^s = d^s, sum_s z^s <= x
    l(u) = lx(u) + sum_s ls(u) with lx(u) = min_{x: f(x) - u.x: g.y <= B, x <= My} and ls(u)= min_{u.z_s: E.z^s = d^s}
    -l is convex and a subgradient at u is (x-sum_s z^s) with x solves lx(u) and z^s solves ls(u) (lin min cost flow)
    """

    def __init__(self, id_: str, network: Network):
        Oracle.__init__(self, id_, positive_quadrant=True)
        self.net = network
        self.allclosed = {a: 0 for a in network.links2}
        self.lasty = {a: 0 for a in network.links2}
        self.modx, self.yvar, self.xvar = DualFlowOracle.build_model_x(network)
        self.mods = {}
        self.xsvar = {}
        for (i, s) in enumerate(network.zones):
            ms, vs = DualFlowOracle.build_model_s(network, s, i)
            self.mods[s] = ms
            self.xsvar[s] = vs

    @staticmethod
    def get_demand(network: Network, node, zone):
        return - sum(r.getDemand(node) for r in network.zones) if node.id == zone.id else \
            node.getDemand(zone) if isinstance(node, type(zone)) else 0

    @staticmethod
    def get_SO_poly_reverse(link: Link):
        """ c(x) = x.f(x) = t.x + c.x^5 """
        t = link.t_ff
        e = link.beta
        c = t * link.alpha / pow(link.C, e)
        assert e == 4, f"travel time exponent for link {link}: {e} != 4"
        return [c, 0, 0, 0, t, 0]

    @staticmethod
    def build_model_x(network: Network):
        minlp = gp.Model('xymodel')
        yvar = minlp.addVars(network.links2, vtype=GRB.BINARY, name="y")
        minlp.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")
        xvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=network.TD, obj=0.0, name="x")
        mctrs = minlp.addConstrs((xvar[a] <= yvar[a] * network.TD for a in network.links2), name="M")
        cvar = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, obj=1.0, name="c")
        for a in network.links:
            minlp.addGenConstrPoly(xvar[a], cvar[a], DualFlowOracle.get_SO_poly_reverse(a))
        minlp.update()
        # minlp.write('xymodel.lp')  # default sense is minimization
        minlp.setParam(GRB.Param.OutputFlag, True)
        minlp.setParam(GRB.Param.FuncNonlinear, 0)
        minlp.setParam(GRB.Param.FuncPieceRatio, 0)
        minlp.setParam(GRB.Param.FuncPieces, -1)
        minlp.setParam(GRB.Param.FuncPieceError, 1e-2)

        return minlp, yvar, xvar

    @staticmethod
    def build_model_s(network: Network, s: Zone, i: int) -> tuple[gp.Model, dict]:
        lps = gp.Model(f"smodel{i}-{s}")
        xsub = -DualFlowOracle.get_demand(network, s, s)
        xsvar = lps.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, ub=xsub, obj=0.0, name="x{i}")
        lps.addConstrs((sum(xsvar[a] for a in i.outgoing) - sum(xsvar[a] for a in i.incoming)
                        == DualFlowOracle.get_demand(network, i, s) for i in network.nodes), name="D")
        lps.update()
        # lps.write(f'smodel{i}.lp')
        lps.setParam(GRB.Param.OutputFlag, False)
        return lps, xsvar

    def set_multipliers(self, u: dict):
        for a, ua in u.items():
            self.xvar[a].obj = -ua
            for s, v in self.xsvar.items():
                v[a].obj = ua

    @staticmethod
    def solvemodel(model: gp.Model):
        """Solve the model."""
        model.setParam(GRB.Param.OutputFlag, False)
        model.optimize()
        if model.status == GRB.INFEASIBLE:
            logger.warning(f'{model.modelname}: no solution found')
        assert model.status == GRB.OPTIMAL, f"Optimization was stopped with status {model.status}"
        cost = model.objval
        runtime = model.runtime
        # logger.debug(f"{model.modelname}: solution cost={cost} runtime={runtime:.2f}")
        return cost, runtime

    def oracle(self, u: list):
        """ get the zero and first information of -l at point u

         Args:
             u: point where to evaluate the function -l

         Returns:
               -l(u): (float) = -(lx(u) + sum_s ls(u))
               g: (list) a subgradient of -l at u: x - sum_s z^s
               y: (list) integer solution y of lx(u)
        """
        udict = {a: u[i] for (i, a) in enumerate(self.net.links)}
        logger.debug(f"candidate {udict}")

        self.set_multipliers(udict)

        lx, rx = self.solvemodel(self.modx)
        x = {a: v.x for (a, v) in self.xvar.items()}
        y = {a: int(v.x) for (a, v) in self.yvar.items()}
        self.lasty = y
        fu = lx
        g = list(x.values())

        for (s, ms) in self.mods.items():
            ls, rs = self.solvemodel(ms)
            xs = {a: v.x for (a, v) in self.xsvar[s].items()}
            fu += ls
            for i, a in enumerate(x.keys()):
                g[i] -= xs[a]

        return -fu, g, y

    def has_lb(self):
        return True

    def eval_lb(self):
        y = self.lasty
        tstt = self.net.msa('SO', y, self.allclosed)
        x = {a: a.x for a in self.net.links2}
        logger.debug(f"TAP: SO={tstt} x= {x}")
        logger.info(f"heuristic={tstt} integer solution {y}")
        return -tstt, y


class Lagrangian:

    DATADIR = "../../data/"

    def __init__(self, instancename: str, mode: str):
        ntk = Lagrangian.buildinstance(instancename)
        self.solver = Lagrangian.buildsolver(instancename, ntk, mode)
        self.uinit = Lagrangian.init_dual_solution(ntk, mode)

    @staticmethod
    def buildinstance(instancename: str) -> Network:
        net = ''
        datadir = Lagrangian.DATADIR
        if instancename.startswith('SF'):
            net = 'SiouxFalls'
            datadir = Lagrangian.DATADIR + net + "/"
        logger.info(f"{net} {instancename}")
        assert net, f"no instance {instancename}"
        return Network.Network(datadir, instancename, 0.5, 1e-0, 1e-3)

    @staticmethod
    # initial dual point
    def init_dual_solution(ntk, mode: str):
        uinit = [0.0 for _ in ntk.links]
        logger.info(f"dual solution {uinit}")
        return uinit

    @staticmethod
    def buildsolver(ins: str, ntk: Network, mode='B') -> CvxSolver:
        solver = None
        if mode == 'B':
            solver = ProximalBundle(DualFlowOracle(ins, ntk))
        if mode == 'S':
            solver = SubGradient(DualFlowOracle(ins, ntk), lb_init=-8000)
        return solver

    def solve(self, plot=True):
        """Solve the convex problem [min_{u>=0} -L(u)] starting from uinit. """
        fu, u = self.solver.solve(self.uinit)
        logger.info(f"best dual cost {-fu}")
        logger.info(f"best dual solution {u}")
        if plot:
            self.solver.show_iters()
        lb, lbsol = self.solver.get_relaxed_solution()
        if lbsol:
            logger.info(f"best primal cost {-lb}")
            logger.info(f"best primal solution {lbsol}")


if __name__ == "__main__":

    modes = {'S': "lag subgradient", 'B': "lag bundle"}

    instance = 'SF_DNDP_10_1'
    m = 'B'

    logger.info(f"Solver = {modes[m]}")
    lagsolver = Lagrangian(instance, m)
    lagsolver.solve()

# todo setters for paramaters in Solvers (#iterations, LINE SEARCH)
# todo replace the approx PWL subproblem with exact method
