# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:20:32 2024

@author: mcrav
"""
import numpy as np
from matplotlib import pyplot as plt

#colors
dark_blue = [51/255, 34/255, 136/255]
dark_green = [17/255, 119/255, 51/255]
teal = [68/255, 170/255, 153/255]
light_blue = [136/255, 204/255, 238/255]
yellow = [221/255, 204/255, 119/255]
salmon = [204/255, 102/255, 119/255]
light_purple = [170/255, 68/255,153/255]
dark_purple = [136/255, 34/255, 85/255]


# %% Part 1

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
    
    def left_action(self, other):
        
        return self.apply_operation(other)
    
    def right_action(self, other):
        
        return other.apply_operation(self)
    
    def __mul__(self, other):
        
        return self.left_action(other)
    
    def __rmul__(self, other):
        
        return self.right_action(other)
        
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

def transform_from_coordinates(coords):
    
    x = coords[0]
    y = coords[1]
    theta = coords[2]
    
    H = np.array([[np.cos(theta), -1*np.sin(theta), x],[np.sin(theta), np.cos(theta), y],[0.0,0.0,1.0]])
    return H

class SE2(GM_Group):
    
    def __init__(self):
        
        super().__init__(self.compose, self.de_compose, [0,0,0])
        
    def compose(self, element1, element2):
        #I genuinely don't know how to do SE(2) actions without a matrix
        
        H1 = transform_from_coordinates(element1.value)
        H2 = transform_from_coordinates(element2.value)
        
        new_matrix = np.matmul(H1, H2)
        
        x = new_matrix[0,2]
        y = new_matrix[1,2]
        theta = np.arctan2(new_matrix[1,0], new_matrix[1,1])
        
        new_value = [x,y,theta]
        
        new_element = GM_Element(self, new_value)
        
        return new_element
    
    def de_compose(self, element1, element2):
        
        H1 = transform_from_coordinates(element1.value)
        H2 = transform_from_coordinates(element2.value)
        
        R2 = H2[0:2,0:2]
        p2 = H2[0:2,2]
        
        H2_inv = np.identity(3)
        H2_inv[0:2,0:2] = R2.transpose()
        H2_inv[0:2,2] = np.matmul(-1 * R2.transpose(), p2)
        
        #this product needs to be swapped depending on the original operation
        #group is non-commutative :'(
        new_matrix = np.matmul(H2_inv, H1)
        
        x = new_matrix[0,2]
        y = new_matrix[1,2]
        theta = np.arctan2(new_matrix[1,0], new_matrix[1,1])
       
        new_value = [x,y,theta]
        
        new_element = GM_Element(self, new_value)
        
        return new_element


def draw_transform_2d(ax, T):
        
    px = T[0,2]
    py = T[1,2]
    
    x = T[:,0]
    y = T[:,1]
    
    ax.quiver(px,py,x[0],x[1], color = dark_blue, linewidths = 4)
    ax.quiver(px,py,y[0],y[1], color = light_blue, linewidths = 4)
    
    return

def test_group(group, elmt1, elmt2):
    elmt3 = elmt1 * elmt2
    print(elmt3.value)
    elmt1_restored = group.de_compose(elmt3,elmt2)
    print(elmt1_restored.value)
    elmt2_restored = group.de_compose(elmt3,elmt1)
    print(elmt2_restored.value)
    
    
#example 1.3.16
pss = ProductScaleShift()
g1 = pss.element([3, -1])
g2 = pss.element([0.5, 1.5])
test_group(pss, g1, g2)


#SE2 tests
se2 = SE2()
h1 = se2.element([0,1,np.pi/4])
h2 = se2.element([1,2,np.pi/2])
test_group(se2, h1, h2)
test_group(se2, h2, h1)

#SE2 
fig = plt.figure()
ax1 = fig.add_subplot(121)
draw_transform_2d(ax1, transform_from_coordinates([0,0,0]))
draw_transform_2d(ax1, transform_from_coordinates(h1.value))
draw_transform_2d(ax1, transform_from_coordinates(h2.value))
#ax.quiver([0,0], [0,0], [h1.value[0],h2.value[0]], [h1.value[1],h2.value[1]], color = salmon, scale=1)
#ax.arrow(0, 0, h2.value[0], h2.value[1], length_includes_head = True, head_width = 0.1, color = salmon)
ax2 = fig.add_subplot(122)
draw_transform_2d(ax2, transform_from_coordinates([0,0,0]))
gh = h1 * h2
hg = h2 * h1
draw_transform_2d(ax2, transform_from_coordinates(gh.value))
draw_transform_2d(ax2, transform_from_coordinates(hg.value))

# %% Part 2

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
        
    #def apply_operation(self, element):
        
    #    return(self.group.operation(self.value, element.value))
    
        
class RepProductScaleShift(GM_RepGroup):
    
    def __init__(self):
        
        super().__init__(self.rep, self.de_rep, np.identity(3))
        
    def rep(self, value):
        
        mat = np.identity(3)
        mat[0,0] = value[0]
        mat[1,2] = value[1]
        
        return mat
    
    def de_rep(self, mat):
        
        val = [1,0]
        val[0] = mat[0,0]
        val[1] = mat[1,2]
        
        return val
    
    
        
    
def test_repgroup(group, elmt1, elmt2):
    elmt3 = elmt1 * elmt2
    print(elmt3.value)
    print(group.inverse(elmt3).value)
    elmt1_restored =  group.inverse(elmt2) * elmt3
    print(elmt1_restored.value)
    elmt2_restored = group.inverse(elmt1)*elmt3
    print(elmt2_restored.value)
    elmt3_adj = elmt1.AD(elmt2)*elmt1
    print(elmt3_adj.value)
    
pss_rep = RepProductScaleShift()
g1 = pss_rep.element([3, -1], from_params=True)
g2 = pss_rep.element([0.5, 1.5], from_params=True)
test_repgroup(pss_rep, g1, g2)
