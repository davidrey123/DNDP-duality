#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:28:06 2024

@author: Sophie Demassey
"""

import logging
from pathlib import Path
import gurobipy as gp

from src.cvxsolver.blocksolver import BlockOracle
from src.cvxsolver.cvxsolver import CvxSolver, Oracle
from src.cvxsolver.subgradient import SubGradient
from src.cvxsolver.proximalbundle import ProximalBundle
from src.cvxsolver.admm import Admm
from gurobipy import GRB
from src.tapas import Network

INDIR = "../../data/"
OUTDIR = "../../output/"

logging.basicConfig(
    handlers=[
        logging.FileHandler(Path(OUTDIR, "lag.log")),
        logging.StreamHandler()
    ],
    # format="%(levelname)s: %(message)s")
    # format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
    )


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

    @staticmethod
    def build_knap_model(network: Network):
        m = gp.Model('knap')
        m.Params.OutputFlag = 0
        m.ModelSense = GRB.MAXIMIZE
        yvar = m.addVars(network.links2, obj=1, vtype=GRB.BINARY, name="y")
        m.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")
        m.update()
        return m

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
        self.last_knap_sol = {a: int(self.knap_vars[i].x) for (i, a) in enumerate(self.network.links2)}
        return ly, self.last_knap_sol

    def solveTAP(self, costtype: str, y: dict, u: dict):
        """
        computes the optimal TAP flow for configuration y and cost perturbed with u
        f(y) = min_{x} x.(t(x) + u[0] + u[1]*x): TAP(x), y=0 => x=0

        Returns:
            tstt (float): the 'UE' flow cost
            f(y) (float): the 'costtype' flow cost
            sx (dict): the optimal flow solution
        """

        tstt = self.network.msa(costtype, y, u)
        lx = self.network.getTSTT(costtype)
        x = {a: a.x for a in self.network.links2}
        logging.debug(f"TAP: UE={tstt}, {costtype}={lx}")
        return tstt, lx, x

    def eval_last_knap_sol(self):
        """
        computes the optimal TAP flow for the feasible configuration y=last_knap_sol in {0,1}^|A2|
        f(y) = min_{x} x.t(x): TAP(x), y=0 => x=0
        y being primal feasible (g.y <= B), then -f(y) <= -f* <= -l* <= -l(u) for all u dual feasible (u >= 0)

        Returns:
            -f(y) (float): a valid lower bound of the lagrangian dual -l*
            y (list): the feasible configuration
        """

        tstt, lx, sx = self.solveTAP('UE', self.last_knap_sol, self.allclosed)
        assert tstt == lx
        logging.debug(f"heuristic={lx} integer solution {self.last_knap_sol}")
        return -tstt, self.last_knap_sol


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
        logging.debug(f"candidate {udict}")

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


class BlockDNDPOracle(BlockOracle):
    """Concrete oracle for min_{u,r>=0} -l(u,r) the opposite augmented lagrangian dual function
    obtained by dualizing the complementary equality x.(1-y) = 0 in SO-DNDP
    min_{x,y} f(x)= sum_a x_a.t_a(x_a): TAP(x), g.y <= B, y=0 => x=0
    we enforce the separation of the lagrangian function in two blocks:
    lx(u,y,r)=min_{x: TAP(x)} x.t(x,y) with t'_a(x_a,y_a) = t_a(x_a) + (1-y_a)*(u_a + r/2.(1-y_a)x_a
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
        with L(x,y,u,r) = x.t(x) + u.x.(1-y) + r/2|x.(1-y)|^2 : TAP(x), g.y <= B

         Args:
             u: (list) multipliers
             y: (list) configuration
             r: (float) penalty

         Returns:
               -L(x*,y,u,r): (float) the optimal pertubed TAP value with objective x(t(x)+(1-y)(l+x.r/2))
               x*: (list) the optimal perturbed TAP solution
        """
        logging.debug(f"y={y}")
        # SODNDP:  augmented lagrangian x(t(x) + (1-y).l + x.(1-y).r/2))
        lbd = {a: (u[i] * (1 - y[a]), r * (1 - y[a]) / 2) for (i, a) in enumerate(y)}
        tstt, lx, sx = self.dndp.solveTAP('AUEL', self.dndp.allopen, lbd)
        self.last_flowcost = self.dndp.network.getTSTT("UE")
        return -lx, sx

    def oracle_block_2(self, u: list, x: dict, r: float):
        """ solve the partial augmented lagrangian (config solution) for the complementary formulation x(1-y)=0:
        solve the knapsack problem: min_{y: g.y <= B} L(x,y,u,r) for fixed multiplier u, flow x, penalty r
        with L(x,y,u,r) = x.t(x) + u.x.(1-y) + r/2|x.(1-y)|^2 : TAP(x), g.y <= B

         Args:
             u: (list) multipliers
             x: (list) TAP flow solution
             r: (float) penalty

         Returns:
               -L(x,y*,u,r): (float) the optimal knapsack value with objective (1-y).x.(u+x.r/2)
               y*: (list) the optimal knapsack solution
        """
        cost = [x[a] * (u[i] + x[a] * r/2) for (i, a) in enumerate(x)]
        total_cost = sum(cost)

        logging.debug(f"x={x}")
        ksopt, kssol = self.dndp.solveKS(cost)
        ly = self.last_flowcost + total_cost - ksopt
        logging.debug(f"ly={ly}")
        return -ly, kssol

    def update_admm(self, u: list, x: dict, y: dict, r: float):
        """ update multipliers according to the ADMM policy when dualizing he complementary formulation x(1-y)=0:
        u += r.x*.(1-y*) with x* and y* the partial solutions of L(x,y,u,r) for fixed multiplier u, penalty r
        and L(x,y,u,r) = x.t(x) + u.x.(1-y) + r/2|x.(1-y)|^2 : TAP(x), g.y <= B
        Note that there is a priori no proof of convergence as the dualized constraint is not linear.

         Args:
             u: (list) multipliers
             x: (list) TAP flow partial solution
             y: (list) config partial solution
             r: (float) penalty

         Returns:
               v: (list) updated multipliers
               d: (float) maximal deviation max_a x_a(1-y_a)
        """
        v = [u[i] if y[a] == 1 else u[i] + r * x[a] for (i, a) in enumerate(y)]
        norminf = max(x[a]*(1-y[a]) for a in y)
        logging.info(f"deviation Linf = {norminf}")
        return v, norminf

    def violation(self, x: dict, y: dict):
        """ returns the vector x.(1-y)"""
        return [0 if y[a] == 1 else x[a] for a in y]

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
        logging.info(f"{net} {instancename}")
        assert net, f"no instance {instancename}"
        return Network.Network(datadir, instancename, 0.5, 1e-0, 1e-3)

    @staticmethod
    # initial network configuration for BlockOracle
    def init_partial_primal_solution(ntk):
        # yinit_list = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        yinit_list = [0 for _ in ntk.links2]
        logging.info(f"primal partial solution {yinit_list}")
        return {a: yinit_list[i] for (i, a) in enumerate(ntk.links2)}

    @staticmethod
    # initial dual point
    def init_dual_solution(ntk, mode: str):
        uinit = [0 for _ in ntk.links2]
        if mode == 'AB':
            uinit.append(0)
        logging.info(f"dual solution {uinit}")
        return uinit

    @staticmethod
    def buildsolver(ins: str, ntk: Network, mode='B') -> CvxSolver:
        solver = ProximalBundle(DNDPOracle(ins, ntk))
        if mode == 'S':
            solver = SubGradient(DNDPOracle(ins, ntk), lb_init=-8000)
        elif mode == 'A':
            yinit = Lagrangian.init_partial_primal_solution(ntk)
            solver = Admm(BlockDNDPOracle(ins, ntk, yinit), penalty=1000)
        elif mode == 'AB':
            yinit = Lagrangian.init_partial_primal_solution(ntk)
            solver = ProximalBundle(BlockDNDPOracle(ins, ntk, yinit))
        return solver

    def solve(self, plot=True):
        """Solve the convex problem [min_{u>=0} -L(u)] starting from uinit. """
        fu, u = self.solver.solve(self.uinit)
        logging.info(f"best dual cost {-fu}")
        logging.info(f"best dual solution {u}")
        if plot:
            self.solver.show_iters()
        lb, lbsol = self.solver.get_relaxed_solution()
        if lbsol:
            logging.info(f"best primal cost {-lb}")
            logging.info(f"best primal solution {lbsol}")


if __name__ == "__main__":

    instance = 'SF_DNDP_20_1'

    modes = {'S': "lag subgradient", 'B': "lag bundle", 'A': "auglag admm", 'AB': "auglag bdle"}
    lagsolver = Lagrangian(instance, mode='B')
    lagsolver.solve()
