#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:28:06 2024

Abstract classes for numerical algorithms for convex optimization unconstrained or on the positive quadrant:
- oracle: to implement the convex function to minimize and returns first-order information
- cvxsolver: to implement the numerical algorithm

@author: Sophie Demassey
"""
import sys

import matplotlib.pylab as plt
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Oracle:
    """Abstract oracle: get the zero and first order information for a convex function f defined over R^n
        oracle: Oracle objectmapping x -> (f(x), g, s) with f convex, g a subgradient of f at x, s
        positive_quadrant: is f defined on x >= 0 or not ?
    """

    def __init__(self, id_: str, positive_quadrant=False):
        self.id: str = id_
        self.positive_quadrant = positive_quadrant

    def oracle(self, x):
        """ get the zero and first information at point x as a tuple

         Args:
             x: point where to evaluate the function

         Returns:
               f(x): (float) value of the function at x
               g: (list) a subgradient of the function f at x
               y: (list) primal solution when f(x)=max_y g(x,y)
        """
        pass

    def has_lb(self):
        """
        Can we evaluate lower bounds lb <= f ?

        Returns:
            True of False
        """
        pass

    def eval_lb(self):
        """
        Computes and returns a valid lower bound lb <= f.
        For ex, if f is a lagrangian function f(x)=max_y c(y) + x.g(y) on x>=0,
        then for all y: g(y)>= 0, c(y) is valid lower bound associated to the primal feasible solution y

        Returns:
            lb (float): a valid lower bound
            y (list): the associated primal solution if any
        """
        pass


class CvxSolver:
    """Abstract solver: minimizes a convex function f on R^n or R^n_+ given first order information.

        oracle: Oracle object defining function f and, possibly, nonnegative constraints

        MAX_ITER: maximum iteration number
        TOL: optimality tolerance value
        xc: the stability center at the current iteration
        fxc: the function value at xc at the current iteration
    """
    MAX_ITER = 200
    TOL = 1e-7

    def __init__(self, oracle: Oracle, name: str):
        self.oracle_obj = oracle
        self.name = name
        self.xc = None
        self.fxc = 1e12
        self.final_solution = None
        self.iters = None
        self.iters_label = None
        self.axes = None
        self.nbsteps = None
        self.starttime = 0
        self.lb = -sys.maxsize
        self.lbsol = None

    def init_solve(self):
        self.xc = None
        self.fxc = 0
        self.final_solution = {}
        self.iters = {}
        self.starttime = time.perf_counter()

    def oracle(self, x):
        return self.oracle_obj.oracle(x)

    def solve(self, x0):
        pass

    def time(self):
        return time.perf_counter() - self.starttime

    def set_final_solution(self):
        self.final_solution = (self.fxc, self.xc)

    def set_iters_label(self, vals_label: tuple):
        self.iters_label = ('serious', 'fx', 'fxc', '|sg|', 'time') + vals_label
        self.axes = [-1, 0, 0, 1, -1]
        nexta = 2
        for v in vals_label:
            if v == 'lb' or v == 'heurlb':
                self.axes.append(0)
            else:
                self.axes.append(nexta)
                nexta += 1

    def store_iteration(self, serious: bool, it: int, fx: float, normsg: float, vals: list):
        self.iters[it] = [self.nbsteps['serious'], fx, self.fxc, normsg, self.time()] + vals
        valstr = f"It {it}({self.nbsteps['serious']}): "
        for i in range(1, len(self.iters[it])):
            valstr += f"{self.iters_label[i]}={self.iters[it][i]:.5f} "
        logger.debug(valstr)
        # if serious:
        logger.info(valstr)

    def show_iters(self):
        its, bounds = zip(*(sorted(self.iters.items())))
        vals = list(zip(*bounds))
        dim = max(self.axes)
        fig, axes = plt.subplots(nrows=1, ncols=dim+1, figsize=(20, 3))
        date = time.strftime("%y-%m-%d-%H:%M", time.gmtime())
        fig.suptitle(f"{self.oracle_obj.id} {self.name} {date} - cpu={self.time():.1f} it={max(its)+1}", fontsize=10)
        #cmap = plt.get_cmap('gnuplot')
        #colors = cmap(range(len(self.axes)))
        colors = "rgbcmyrgbcmyrgbcmy"
        for (i, a) in enumerate(self.axes):
            if a >= 0:
                axes[a].plot(its, vals[i], color=colors[i], label=self.iters_label[i])
        fig.tight_layout()
        fig.legend()
        plt.savefig('iterations.png')

    def update_lb(self):
        if not self.oracle_obj.has_lb():
            return None
        relaxed_val, relaxed_sol = self.oracle_obj.eval_lb()
        if relaxed_sol:
            logger.debug(f"relaxation: {relaxed_val}")
            if self.lb < relaxed_val:
                self.lb = relaxed_val
                self.lbsol = relaxed_sol
        return relaxed_val

    def get_relaxed_solution(self):
        return self.lb, self.lbsol

