# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:15:09 2024

@author: mcrav
"""

import numpy as np
import numdifftools as nd
import operator
from math import cos, sin, acos, pi
import matplotlib.pyplot as plt

#colors
dark_blue = [51/255, 34/255, 136/255]
dark_green = [17/255, 119/255, 51/255]
teal = [68/255, 170/255, 153/255]
light_blue = [136/255, 204/255, 238/255]
yellow = [221/255, 204/255, 119/255]
salmon = [204/255, 102/255, 119/255]
light_purple = [170/255, 68/255,153/255]
dark_purple = [136/255, 34/255, 85/255]

#groups code and helpers

class GM_Group:
    
    def __init__(self, operation, inverse_operation, identity):
        
        self.operation = operation
        self.inverse = inverse_operation
        self.identity = identity
        
    def element(self, value):
        
        return(GM_Element(self,value))
    
    def identity_element(self):
        return self.identity
    
class GM_Element:
    
    def __init__(self, group, value):
        self.group = group
        self.value = value

    def apply_operation(self, element):
        
        return(self.group.operation(self, element))
    
    def apply_inverse(self, element):
        
        return(self.group.operation(self, element))
    
    def left_action(self, other):
        
        return self.apply_operation(other)
    
    def right_action(self, other):
        
        return other.apply_operation(self)
    
    def left_inverse(self, other):
        
        return self.apply_inverse(other)
    
    def right_inverse(self, other):
        
        return other.apply_inverse(self)
    
    def left_lifted_action(self, vector):
        
        J = nd.Jacobian(self.left_action)
        new = vector.matmul(J)
        return new
    
    def right_lifted_action(self, vector):
        
        J = nd.Jacobian(self.right_action)
        new = vector.matmul(J)
        return new
    
    def left_lifted_inverse(self, vector):
        
        J = nd.Jacobian(self.left_inverse)
        new = vector.matmul(J)
        return new
    
    def right_lifted_inverse(self, vector):
        
        J = nd.Jacobian(self.right_inverse)
        new = vector.matmul(J)
        return new
    
    def Ad(self, vector):
        
        return self.right_lifted_inverse(self.left_lifted_action, vector)
        
    def __mul__(self, other):
        
        return self.left_action(other)
    
    def __rmul__(self, other):
        
        return self.right_action(other)
    
    
class GM_RepGroup(GM_Group):
    
    def __init__(self,rep_fcn, derep_fcn, identity):
        
        super().__init__(self.rep_op, self.rep_inv, identity)
        self.RepresentationFunction = rep_fcn
        self.DerepresentationFunction = derep_fcn
        
    def element(self, value, **kwags):
        
        return(GM_RepElement(self,value, **kwags))
    
    def rep_op(self, element1, element2):
        
        val = np.matmul(element1.value, element2.value)
        return(GM_RepElement(self, val))
    
    def rep_inv(self, element):
        
        val = np.linalg.inv(element.value)
        
        return(GM_RepElement(self, val))
        
    
class GM_RepElement(GM_Element):
    
    def __init__(self, group, value, from_params = False, from_list = False):
        
        if from_params:
            value = group.RepresentationFunction(value)
        elif from_list:
            sz = np.sqrt(len(value))
            value = np.reshape(value, [sz,sz])

        super().__init__(group, value)
        
        self.InverseFunction = np.linalg.inv
        
    def AD(self, other):
        
        return (self * other) * self.group.inverse(self)
    
def do_on_lists(l1, l2, operation):
    
    new = []
    for vector_index in range(len(l1)):
        new.append(operation(l1[vector_index], l2[vector_index]))
        
    return new

# 2.1
class GM_TangentVector():
    
    
    def __init__(self, config: list, value: list):
        
        self.config = config
        self.value = value
        
    def __add__(self, other):
                
        if self.config != other.config:
            raise Exception("Sorry, these tangent vectors belong to different spaces.") 
        
        new = do_on_lists(self.value, other.value, operator.add)
            
        return GM_TangentVector(self.config, new)
    
    def __mul__(self, other):
                
        new = do_on_lists(self.value, other.value, operator.mul)
            
        return GM_TangentVector(self.config, new)
    
    def __rmul__(self, val):
                
        if type(val) == float:
            return self.scalmul(val)
        else:
            return self.matmul(val)
    
    def matmul(self, matrix):
        
        new = np.matmul(np.array(matrix), self.value).tolist()
            
        return GM_TangentVector(self.config, new)
    
    def scalmul(self, scalar):
        
        np_new = scalar * np.array(self.value)
        new = np_new.tolist()
        
        return GM_TangentVector(self.config, new)

# 2.2
class GM_TangentBases():
    
    def __init__(self, bases, config=None):
        
        is_tangents =  [type(x) == GM_TangentVector for x in bases]
        
        if all(is_tangents):
            assert(self.check_configs_same(bases))
            self.value = bases
            self.config = bases[0].config
        elif type(bases) == np.ndarray:
            self.value = []
            self.config = config
            for column in range(bases.shape[1]):
                self.value.append(GM_TangentVector(config, bases[:,column].tolist()))
        else:
            np_bases = np.array(bases)
            self.value = []
            self.config = config
            for column in range(np_bases.shape[1]):
                self.value.append(GM_TangentVector(config, np_bases[:,column].tolist()))
                
            
    def check_configs_same(self, vectors):
        return all(v.config == vectors[0].config for v in vectors)
    
    def flatten(self):
        basis_matrix = np.zeros([len(self.value[0].value),len(self.value)])
        for i in range(len(self.value)):
            basis_matrix[:,i] = self.value[i].value
            
        return basis_matrix
    
    def __mul__(self, val):
        
        if type(val) == np.ndarray:
            rows = val.size[0]
        elif type(val) == list:
            rows = len(val)
            val = np.array(val)
        else:
            raise Exception("invalid input type for multiplication")
        
        if rows != len(self.value):
            raise Exception("invalid input size for multiplication. expected {} rows but got {}".format(len(self.value),rows))
        else:
            mat = np.multiply(self.flatten(), val)
            new_val = np.sum(mat, axis=1)
            return GM_TangentVector(self.config, new_val)
        
    def  __add__(self, other):
        
        if len(self.value) != len(other.value) or len(self.value[0].value) != len(other.value[0].value):
            raise Exception("size mismatch during addition")
        elif self.config != other.config:
            raise Exception("Sorry, these tangent bases belong to different spaces.") 
        else:
            new_bases = self.flatten() + other.flatten()
            return GM_TangentBases(new_bases, config=self.config)
            
    def invert(self):
        
        og_bases = self.flatten()
        inverted_bases = np.linalg.inv(og_bases)
        
        return GM_TangentBases(inverted_bases, config=self.config)
        
        

# 2.5 - untested
def get_mapping_jacobian(func, arr, mapping_inputs=None):
    
    if mapping_inputs != None:
        to_pass = lambda an_array : func(an_array, mapping_inputs)
    else:
        to_pass = func
        
    return nd.Jacobian(to_pass)

# 2.6 

def directional_derivative(diff_function, eval_point):
    
    to_pass = lambda scal : diff_function(eval_point, scal)
    val = nd.Derivative(to_pass)

    return GM_TangentVector(eval_point, val(0.0)) # Putting 0 here is sketch

# 2.7
    
class GM_DirectionDerivativeBasis():
    
    def __init__(self, funcs):
        
        self.funcs = funcs
        self.basis = None
        
    def __call__(self, func_inputs):
        
        vects = []
        
        for func in self.funcs:
            d_ddir = directional_derivative(func, func_inputs)
            vects.append(d_ddir)
            #print(d_ddir.value)
        self.basis = GM_TangentBases(vects)
        
        return self.basis
    
#functions for first deliverable
def f_rho(cartesian_coords, rho):
    
    x = cartesian_coords[0]
    y = cartesian_coords[1]
    
    x_new = (1 + rho/np.linalg.norm([x,y])) * x
    y_new = (1 + rho/np.linalg.norm([x,y])) * y
    
    return np.array([x_new, y_new])
        
def f_phi(cartesian_coords, phi):
    
    x = cartesian_coords[0]
    y = cartesian_coords[1]
    
    x_new = cos(phi) * x - sin(phi) * y
    y_new = sin(phi) * x + cos(phi) * y
    
    return np.array([x_new, y_new])

def f_rho_phi(cartesian_coords, params):
    
    x = cartesian_coords[0]
    y = cartesian_coords[1]
    
    rho = params[0]
    phi = params[1]
    
    x_new = (1 + rho/np.linalg.norm([x,y])) * (cos(phi) * x - sin(phi) * y)
    y_new = (1 + rho/np.linalg.norm([x,y])) * (sin(phi) * x + cos(phi) * y)
    
    return np.array([x_new, y_new])
    
#plotting stuff in service of first deliverable

def draw_transform_2d(ax, T):
        
    px = T[0,2]
    py = T[1,2]
    
    x = T[:,0]
    y = T[:,1]
    
    ax.quiver(px,py,x[0],x[1], color = dark_blue, linewidths = 4, scale_units='xy', scale=5)
    ax.quiver(px,py,y[0],y[1], color = light_blue, linewidths = 4,scale_units='xy', scale=5)
    
def plot_bases(bases, ax):
    directions = bases.flatten()
    transform = np.identity(3)
    transform[0:2,0:2] = directions
    transform[0:2,2] = bases.config
    draw_transform_2d(ax, transform)


def plot_velocities(bases, speeds, ax):
    directions = bases.flatten()
    coord_v = np.array(speeds).transpose()
    v =  np.matmul(directions,coord_v)
    ax.quiver(bases.config[0],bases.config[1],v[0],v[1], color = dark_blue, linewidths = 4, scale_units='xy', scale=5)
    
# deliver!

cart_expressed_polar = GM_DirectionDerivativeBasis([f_rho, f_phi])

f, [ax1, ax2] = plt.subplots(1,2)
for x in [-2,-1,0,1,2]:
    for y in [-2,-1,0,1,2]:
        bases = cart_expressed_polar([x,y])
        plot_bases(bases, ax1)
        plot_velocities(bases, [1.,1.], ax2)

ax1.axis("equal")
ax1.set_xlim([-3,3])
ax1.set_ylim([-3,3])
ax1.set_title("2.7 a re-creation")

ax2.axis("equal")
ax2.set_xlim([-3,3])
ax2.set_ylim([-3,3])
ax2.set_title("2.7 b re-creation")
plt.show()


class GM_GroupTangentVector(GM_TangentVector):
    
    def __init__(self, element: GM_Element, value):
        
        super().__init__(element.value, value)
        
        self.group = element.group
        
    def groupwise_basis(self, other):
        
        to_pass = lambda g : self.group.operation(self.value, g)
        TgLh = nd.Jacobian(to_pass)
        
        out = TgLh(other.value)
        
        return out
        
class ProductScaleShift(GM_Group):
    
    def __init__(self):
        
        super().__init__(self.compose, self.de_compose, [1,0])
        
    def compose(self, element1, element2):
        
        x = element1.value[0] * element2.value[0]
        y = element1.value[1] + element2.value[1]
        
        new_value = [x,y]
        
        new_element = GM_Element(self, new_value)
        
        return new_element
    
    def de_compose(self, element1, element2):
        
        x = element1.value[0] / element2.value[0]
        y = element1.value[1] - element2.value[1]
        
        new_value = [x,y]
        
        new_element = GM_Element(self, new_value)
        
        return new_element
    


#working examples

# vec1 = GM_TangentVector([1,1], [0,1])
# vec2 =GM_TangentVector([1,1], [1,2])
# scaled = 2.0*vec1
# unchanged = np.identity(2).tolist()*vec1
# added = vec1+vec2
# element_wise = vec1*vec2
# bases1 = GM_TangentBases([vec1, vec2])
# bases2 = GM_TangentBases([[1,2],[1,2]],config=[1,1])
# bases3 = GM_TangentBases(np.array([[1,2],[3,4]]),config=[1,1])
# multiplied = bases3 * [[0,1],[2,3]] 
# added = bases3 + bases2
# inverted = bases3.invert()