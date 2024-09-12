from src import Params


class Link:

    # construct this Link with the given parameters
    def __init__(self, id, start, end, t_ff, C, alpha, beta, cost):
        self.id = id
        self.start = start
        self.end = end
        self.t_ff = t_ff
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.x = 0
        self.y = 1
        self.cost = cost # for DNDP
        
        self.visit_order = -1
        
        if start is not None:
            start.addOutgoingLink(self)
            
        if end is not None:
            end.addIncomingLink(self)
            
        self.xstar = 0
        self.lbdcost = 0
        self.lbdcost2 = 0

    def setlbdCost(self, lbdcost):
        self.lbdcost = lbdcost    

    def setlbdCost2(self, lbdcost2):
        self.lbdcost2 = lbdcost2

    def setFlow(self, x):
        self.x = x
    
    def __repr__(self):
        return str(self)

    def getTravelTime(self, x, type):
        """ return f(x) such that the link cost is c(x) = int_0^x f(v)dv for a given type. """
        if (self.y == 0 and type !='L'):
            return Params.INFTY
            
        # UE: f(x) = t(x) = tff(1+a.(x/C)^b)
        if type == 'UE':
            output = self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta))

        # SO: c(x) = xt(x) => f(x) = t(x)+xt'(x) = tff(1 + a.(x/C)^b.(b+1))
        elif type == 'SO':
            output = self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta))
            output += x * self.t_ff * self.alpha * self.beta * pow(x / self.C, self.beta-1) / self.C

        elif type == 'PRIM':
            output = self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta) / (self.beta+1))

        # Lagrangian SO: c(x) = x(t(x)+l) => f(x) = tff(1 + a.(x/C)^b.(b+1)) + l
        elif type == 'L':
            output = self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta))
            output += x * self.t_ff * self.alpha * self.beta * pow(x / self.C, self.beta-1) / self.C
            output += self.lbdcost

        # Lagrangian SO (same as 'L' above): c(x) = x(t(x)+l) => f(x) = tff(1 + a.(x/C)^b.(b+1)) + l
        elif type == 'UEL':
            output = self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta) * (self.beta+1))
            output += self.lbdcost

        # Augmented Lagrangian SOSODNDP:  c(x) = x(t(x) + l + m.x) => f(x) = tff(1 + a.(x/C)^b.(b+1)) + l + 2.m.x
        elif type == 'AUEL':
            output = self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta) * (self.beta+1))
            output += self.lbdcost + 2 * self.lbdcost2 * x

        else:
            raise Exception("wrong type "+str(type))

        return output

    def getCost(self, x, type):
        """ return link cost c(x) = int_0^x f(v)dv  with f travel time for a given type. """

        # UE: f(x) = t(x) = tff(1 + a.(x/C)^b), c(x)= tff.x.(1 + a.(x/C)^b/(b+1))
        if type == 'UE':
            output = x * self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta) / (self.beta + 1))

        # SO: c(x) = x.t(x)
        elif type == 'SO':
            output = x * self.t_ff * (1 + self.alpha * pow(x / self.C, self.beta))

        # Lagrangian SO: c(x) = x.t(x) + l.x
        elif type == 'UEL':
            output = self.getCost(x, 'SO') + self.lbdcost * x

        # Augmented Lagrangian SOSODNDP:  c(x) = x.t(x) + l.x + m.x^2
        elif type == 'AUEL':
            output = self.getCost(x, 'UEL') + self.lbdcost2 * x * x

        else:
            raise Exception("wrong type " + str(type))

        return output

    def getDerivativeTravelTime(self, x):
        
        if self.y == 0:
            return Params.INFTY
            
        return self.t_ff * self.alpha * self.beta * pow(x / self.C, self.beta-1) / self.C   

    def getCapacity(self):
        return self.C
    
    def getFlow(self):
        return self.x
        
    def __str__(self):
        return "(" + str(self.start.getId()) + "," + str(self.end.getId()) + ")"
        
    def addXstar(self, flow):
        self.xstar += flow   
    
    def calculateNewX(self, stepsize):        
        self.x = (1 - stepsize) * self.x + stepsize * self.xstar
        self.xstar = 0
        
    def hasHighReducedCost(self, type, percent):
        reducedCost = self.end.cost - self.start.cost
        tt = self.getTravelTime(self.x, type)        
        return tt - reducedCost > tt*percent
 
    def getReducedCost(self, type):
        reducedCost = self.end.cost - self.start.cost
        tt = self.getTravelTime(self.x, type)
        return tt - reducedCost
