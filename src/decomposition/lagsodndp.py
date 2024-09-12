#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:28:06 2024

@author: Sophie Demassey
"""

import logging
import random
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

from src.cvxsolver.blocksolver import BlockOracle
from src.cvxsolver.cvxsolver import CvxSolver, Oracle
from src.cvxsolver.subgradient import SubGradient
from src.cvxsolver.proximalbundle import ProximalBundle
from src.cvxsolver.admm import Admm
from src.decomposition.gbmodel import GBModel
from src.tapas import Network

INDIR = "../../data/"
OUTDIR = "../../output/"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DNDP:
    """ Utility class for implementing oracles for lagrangian relaxations of DNDP """

    def __init__(self, id_: str, network: Network):
        self.id = id_
        self.network = network
        # self.tap_model = DNDPOracle.build_tap_model(network)
        self.knap_model = DNDP.build_knap_model(network)
        self.knap_vars = self.knap_model.getVars()
        self.allopen = {a: 1 for a in self.network.links2}
        self.allclosed = {a: 0 for a in self.network.links2}
        self.link_idx = {a: i for (i, a) in enumerate(self.network.links2)}
        self.last_knap_sol = {a: 0 for a in self.network.links2}
        self.last_knap_val = None
        self.changed_knap_sol = True
        self.allopen_tap = None

    @staticmethod
    def build_knap_model(network: Network):
        ksm = gp.Model('knap')
        ksm.Params.OutputFlag = 0
        ksm.ModelSense = GRB.MAXIMIZE
        yvar = ksm.addVars(network.links2, obj=1, vtype=GRB.BINARY, name="y")
        ksm.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")
        ksm.update()
        return ksm

    def udict(self, u: list):
        return {a: u[i] for (i, a) in enumerate(self.network.links2)}

    def dim(self):
        return len(self.network.links2)

    def solveKS(self, u: list):
        """
        solve the knapsack 0-1 subproblem with cost u over the budget constraint: max u.y: g.y <= B,
        :param u: the objective coefficient
        :return: the optimal cost and network configuration (also recorded in self.last_knap_sol)
        """
        for (i, a) in enumerate(self.network.links2):
            self.knap_vars[i].obj = u[i]
        self.knap_model.optimize()
        assert self.knap_model.Status == GRB.OPTIMAL, f"knapsack status={self.knap_model.Status}"
        ly = self.knap_model.objVal
        self.changed_knap_sol = self.update_last_knap_sol()
        if self.changed_knap_sol:
            self.last_knap_val = None
        return ly, self.last_knap_sol

    def update_last_knap_sol(self):
        change = 0
        for (i, a) in enumerate(self.network.links2):
            if self.last_knap_sol[a] != int(self.knap_vars[i].x):
                change += 1
                self.last_knap_sol[a] = int(self.knap_vars[i].x)
        if change:
            logger.debug(f"new knap sol {change}: {self.last_knap_sol}")
        return change > 0

    def solveTAP(self, costtype: str, y: dict, u: dict):
        """
        computes the optimal TAP flow for configuration y and cost perturbed with u
        f(y) = min_{x} x.(t(x) + u[0] + u[1]*x): TAP(x), y=0 => x=0

        Returns:
            tstt (float): the 'SO' flow cost sum x.t(x)
            f(y) (float): the 'costtype' flow cost
            sx (dict): the optimal flow solution
        """
        tstt = self.network.msa(costtype, y, u)
        lx = self.network.getCost(costtype)
        x = {a: a.x for a in self.network.links2}
        logger.debug(f"TAP: SO={tstt}, {costtype}={lx}")
        return tstt, lx, x

    def allopen_TAP(self):
        if not self.allopen_tap:
            tstt, lx, x = self.solveTAP("SO", self.allopen, self.allclosed)
            assert abs(tstt - lx) < 1e-5
            logger.debug(f"allopensol = {tstt} flow = {x}")
            self.allopen_tap = (tstt, x)
        return self.allopen_tap

    def eval_last_knap_sol(self):
        """
        computes the optimal TAP flow for the feasible configuration y=last_knap_sol in {0,1}^|A2|
        f(y) = min_{x} x.t(x): TAP(x), y=0 => x=0
        y being primal feasible (g.y <= B), then -f(y) <= -f* <= -l* <= -l(u) for all u dual feasible (u >= 0)

        Returns:
            -f(y) (float): a valid lower bound of the lagrangian dual -l*
            y (list): the feasible configuration
        """
        if self.last_knap_val:
            return self.last_knap_val, self.last_knap_sol

        tstt, lx, sx = self.solveTAP('SO', self.last_knap_sol, self.allclosed)
        self.last_knap_val = -tstt
        assert abs(tstt - lx) < 1e-5
        logger.info(f"heuristic={lx} integer solution {self.last_knap_sol}")
        return -tstt, self.last_knap_sol

    def randomy(self) -> dict:
        return {a: random.randint(0, 1) for a in self.network.links2}


class DNDPOracle(Oracle):
    """Concrete oracle for min_{u>=0} -l(u) the opposite lagrangian dual function obtained by dualizing x <= M.y in
    SO-DNDP min_{x,y} f(x)= sum_a x_a.t_a(x_a): TAP(x), g.y <= B, x <= M.y
    l(u) = lx(u) - ly(u) with lx(u)=min_{x: TAP(x)} sum_a x_a(t_a(x_a)+u_a) and ly(u)= max_{y: g.y <= B} M.u.y
    -l is convex and a subgradient at u is My-x with x solves lx(u) (perturbed TAP) and y solves ly(u) (0-1 knapsack)
    """

    def __init__(self, id_: str, network: Network):
        Oracle.__init__(self, id_, positive_quadrant=True)
        self.dndp = DNDP(id_, network)
        self.bigM = network.TD

    def oracle(self, u: list):
        """ get the zero and first information of -l at point u

         Args:
             u: point where to evaluate the function -l

         Returns:
               -l(u): (float) = ly(u) - lx(u) with lx(u)=min_{x: TAP(x)} x.(t(x)+mu) and ly(u)= max_{y: g.y<=B} M.u.y
               g: (list) a subgradient of -l at u: My-x with x solves lx(u) and y solves ly(u)
               y: (list) integer solution y of ly(u)
        """
        udict = self.dndp.udict(u)
        logger.debug(f"candidate {udict}")

        tstt, lx, sx = self.dndp.solveTAP("UEL", self.dndp.allopen, udict)
        ly, sy = self.dndp.solveKS(u)

        # get the bundle information
        fu = self.bigM * ly - lx
        g = [self.bigM * sy[a] - sx[a] for a in sx]

        return fu, g, sy

    def has_lb(self):
        return True

    def eval_lb(self):
        return self.dndp.eval_last_knap_sol()


class DNDPGBOracle(Oracle):
    """Concrete approximate oracle for min_{u,r>=0} -l(u,r) the opposite augmented lagrangian dual function
    l(y,x,u,r)=min_{y,x: TAP(x), g.y <= B} x.t'(x,y) with t'_a(x_a,y_a) = t_a(x_a) + (1-y_a)*(u_a + r/2.(1-y_a)x_a)
    an approximate solution is computed in 2 steps:
    1/ (y,x0) is obtained from the PWL approximation of gurobi
    2/ x is obtained from TAP(y)
    -l is convex and a subgradient at (u,r) is (x(1-y), |x(1-y)|^2/2) where (x,y) solves l(u,r)
    """

    def __init__(self, id_: str, network: Network):
        Oracle.__init__(self, id_, positive_quadrant=True)
        self.dndp = DNDP(id_, network)
        gbm = GBModel(network)
        gbm.minlp.remove(gbm.mctrs)
        gbm.minlp.setParam(GRB.Param.OutputFlag, False)
        gbm.minlp.setParam(GRB.Param.FuncNonlinear, 0)
        self.gbmodel = gbm
        self.clvar = DNDPGBOracle.addObjective(network, gbm.minlp, gbm.xvar, gbm.yvar, gbm.cvar)
        self.clctrs = None

    @staticmethod
    def addObjective(network, minlp, xvar, yvar, cvar):
        """ c_a = x_a.t_a(x_a) + (1-y_a) * cl_a
        gurobi does not allow yet to add the convex polynomial constraint c >= x.t(x)...
        thus we add the nonconvex constraint c == x.t(x) instead """
        c0var = minlp.addVars(network.links, vtype=GRB.CONTINUOUS, lb=0.0, name="c0")
        for a in network.links:
            minlp.addGenConstrPoly(xvar[a], c0var[a], GBModel.get_SO_poly_reverse(a), name=f"C0[{a}]")
            minlp.addConstr(cvar[a] >= c0var[a], name=f"CC[{a}]")

        clvar = minlp.addVars(network.links2, vtype=GRB.CONTINUOUS, lb=0.0, name="cl")
        for a in network.links2:
            minlp.addGenConstrIndicator(yvar[a], 0.0, cvar[a] >= c0var[a] + clvar[a])
        return clvar

    def updateObjective(self, u: list, r: float):
        """ cl_a = u_a.x_a + r/2.x_a^2 """
        self.gbmodel.minlp.update()
        if self.clctrs:
            self.gbmodel.minlp.remove(self.clctrs)
        self.clctrs = self.gbmodel.minlp.addConstrs(self.clvar[a] >= self.gbmodel.xvar[a] * u[i] +
                                                    self.gbmodel.xvar[a] * self.gbmodel.xvar[a] * r/2
                                                    for i, a in enumerate(self.clvar))

    def solveAugLagModel(self, u: list, r: float):
        self.updateObjective(u, r)
        gbm = self.gbmodel.minlp
        gbm.optimize()

        if gbm.status == GRB.INFEASIBLE:
            iisfilename = "oracle.iis"
            logger.warning(f'no solution found write IIS file {iisfilename}')
            gbm.computeIIS()
            gbm.write(iisfilename)

        assert gbm.status == GRB.OPTIMAL, f"Optimization was stopped with status {gbm.status}"

        cost = gbm.objval
        bctr = gbm.getConstrByName("B")
        runtime = gbm.runtime
        logger.debug(f"oracle: cost={cost} slack={bctr.Slack} ({bctr.RHS}) runtime={runtime:.2f}")

        return self.gbmodel.getSolution()

    def oracle(self, v: list):
        """ get the zero and first information of -l at point v=(u,r)
        """
        logger.info(f"candidate {v}")
        u = v[:-1]
        r = v[-1:][0]
        y = self.solveAugLagModel(u, r)
        self.update_last_knap_sol(y)

        logger.debug(f"-- get auglag-TAP solution for {y}")
        lbd = {a: (u[i] * (1 - y[a]), r * (1 - y[a]) / 2) for (i, a) in enumerate(y)}
        tstt, lx, x = self.dndp.solveTAP('AUEL', self.dndp.allopen, lbd)
        assert abs(tstt - self.dndp.network.getCost('SO')) < 1e-5
        logger.debug(f"lx={lx} x={x}")

        sg = [0 if y[a] == 1 else -x[a] for a in y]
        sgr = sum(v * v for v in sg) / 2
        sg.append(-sgr)

        return -lx, sg, y

    def has_lb(self):
        return True

    def eval_lb(self):
        return self.dndp.eval_last_knap_sol()

    def update_last_knap_sol(self, y: dict):
        change = 0
        for (a, ya) in y.items():
            if self.dndp.last_knap_sol[a] != y[a]:
                change += 1
            self.dndp.last_knap_sol[a] = y[a]
        if change:
            logger.debug(f"new knap sol {change}: {self.dndp.last_knap_sol}")
        self.dndp.changed_knap_sol = (change > 0)


class BlockDNDPOracle(BlockOracle):
    """Concrete oracle for min_{u,r>=0} -l(u,r) the opposite augmented lagrangian dual function
    obtained by dualizing the complementary equality x.(1-y) = 0 in SO-DNDP
    min_{x,y} f(x)= sum_a x_a.t_a(x_a): TAP(x), g.y <= B, y=0 => x=0
    we enforce the separation of the lagrangian function in two blocks:
    lx(u,y,r)=min_{x: TAP(x)} x.t'(x,y) with t'_a(x_a,y_a) = t_a(x_a) + (1-y_a)*(u_a + r/2.(1-y_a)x_a)
    ly(u,x,r)= min_{y: g.y <= B} sum_a (1-y_a).x_a.(u_a + r/2.x_a)
    -l is convex and a subgradient at (u,r) is (x(1-y), |x(1-y)|^2/2) where (x,y) solves l(u,r)
    although we do not solve l(u,r) exactly but approximately by iterating over y=lx(y) and x=ly(x)
    """
    ADM_IT_MAX = 100

    def __init__(self, id_: str, network: Network, partial_solution: dict):
        BlockOracle.__init__(self, id_, partial_solution, positive_quadrant=True)
        self.dndp = DNDP(id_, network)
        self.last_flowcost = 0

    def oracle_block_1(self, u: list, y: dict, r: float):
        """ solve the partial augmented lagrangian (flow solution) for the complementary formulation x(1-y)=0:
        solve the perturbed TAP: min_{x: TAP(x)} L(x,y,u,r) for fixed multiplier u, config y, penalty r
        with L(x,y,u,r) = x.t(x) + u.x.(1-y) + r/2|x.(1-y)|^2

         Args:
             u: (list) multipliers
             y: (dict) configuration
             r: (float) penalty

         Returns:
               -L(x*,y,u,r): (float) the optimal pertubed TAP value with objective x(t(x)+(1-y)(l+x.r/2))
               x*: (list) the optimal perturbed TAP solution
        """
        # SODNDP:  augmented lagrangian x(t(x) + (1-y).l + x.(1-y).r/2))
        logger.debug(f"B1 y={y} u={u} ,r={r}")

        if y == self.dndp.allopen:
            tstt, sx = self.dndp.allopen_TAP()
            self.last_flowcost = tstt
            return -tstt, sx

        lbd = {a: (u[i] * (1 - y[a]), r * (1 - y[a]) / 2) for (i, a) in enumerate(y)}
        tstt, lx, sx = self.dndp.solveTAP('AUEL', self.dndp.allopen, lbd)
        assert abs(tstt - self.dndp.network.getCost('SO')) < 1e-5
        self.last_flowcost = tstt
        logger.debug(f"B1 lx={lx} x={sx}")
        # self.z1_init = sx
        return -lx, sx

    def oracle_block_2(self, u: list, x: dict, r: float):
        """ solve the partial augmented lagrangian (config solution) for the complementary formulation x(1-y)=0:
        solve the knapsack problem: min_{y: g.y <= B} L(x,y,u,r) for fixed multiplier u, flow x, penalty r
        with L(x,y,u,r) = x.t(x) + u.x.(1-y) + r/2|x.(1-y)|^2

         Args:
             u: (list) multipliers
             x: (list) TAP flow solution
             r: (float) penalty

         Returns:
               -L(x,y*,u,r): (float) the optimal knapsack value with objective (1-y).x.(u+x.r/2)
               y*: (list) the optimal knapsack solution
        """
        cost = [x[a] * (u[i] + x[a] * r/2) for (i, a) in enumerate(x)]
        logger.debug(f"B2: knapcost={cost} x={x} u={u} r={r}")
        total_cost = sum(cost)

        ksopt, kssol = self.dndp.solveKS(cost)
        ly = self.last_flowcost + total_cost - ksopt
        logger.debug(f"B2: ly={ly} pen={total_cost - ksopt} y={kssol}")
        if self.has_changed() and self.has_lb():
            self.dndp.eval_last_knap_sol()
        # self.z2_init = kssol
        # self.z2_init = self.dndp.allclosed
        return -ly, kssol

    def has_changed(self):
        return self.dndp.changed_knap_sol

    def init_partial_sols(self):
        self.dndp.last_knap_sol = self.z2_init.copy()
        self.dndp.changed_knap_sol = True
        return None, self.z2_init

    def subgradient(self, x: dict, y: dict):
        """ returns the vector x.(y-1)"""
        h = [0 if y[a] == 1 else -x[a] for a in y]
        self.z2_init = {a: 0 if ya == 1 else 1 for a, ya in y.items()}
        dev = abs(min(h))
        return h, dev

    def has_lb(self):
        return True

    def eval_lb(self):
        return self.dndp.eval_last_knap_sol()


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
    # initial network configuration for BlockOracle
    def init_partial_primal_solution(ntk):
        # yinit_list = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        # yinit_list = [1,1,1,1,0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,0, 1,1,0,0]
        yinit_list = [0 for _ in ntk.links2]
        logger.info(f"primal partial solution {yinit_list}")
        return {a: yinit_list[i] for (i, a) in enumerate(ntk.links2)}

    @staticmethod
    # initial dual point
    def init_dual_solution(ntk, mode: str):
        uinit = [0.0 for _ in ntk.links2]
        if mode.startswith('AB'):
            uinit.append(1)
        logger.info(f"dual solution {uinit}")
        return uinit

    @staticmethod
    def buildsolver(ins: str, ntk: Network, mode='B') -> CvxSolver:
        solver = None
        if mode == 'B':
            solver = ProximalBundle(DNDPOracle(ins, ntk))
        if mode == 'S':
            solver = SubGradient(DNDPOracle(ins, ntk), lb_init=-8000)
        elif mode == 'A':
            yinit = Lagrangian.init_partial_primal_solution(ntk)
            solver = Admm(BlockDNDPOracle(ins, ntk, yinit), penalty=1)
        elif mode == 'AB':
            yinit = Lagrangian.init_partial_primal_solution(ntk)
            solver = ProximalBundle(BlockDNDPOracle(ins, ntk, yinit))
        elif mode == 'ABG':
            solver = ProximalBundle(DNDPGBOracle(ins, ntk))
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

    modes = {'S': "lag subgradient", 'B': "lag bundle",
             'A': "auglag admm (inexact dual)", 'AB': "auglag GS1 bdle (inexact dual)",
             'ABG': "auglag GBPWL bdle (inexact dual)"}

    instance = 'SF_DNDP_20_1'
    m = 'ABG'

    logger.info(f"Solver = {modes[m]}")
    lagsolver = Lagrangian(instance, m)
    lagsolver.solve()

    # 'AB': blockoracle gives a very bad estimate as the block iterations stop after the first KS computed solution
    # 'ABG': oracle based on solving the PWL approximate model - the dual bound is still inexact but works well
    # @todo oracle based on solving the OA-epsilon relaxed model: if OA feas-tol = OA opt-tol then = lag dual opt-tol
