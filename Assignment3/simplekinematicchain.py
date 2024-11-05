import sys
sys.path.append('../')
from geomotion import rigidbody as rb
import numpy as np
from matplotlib import pyplot as plt

# Set the group as SE2 from rigidbody
G = rb.SE2

class KinematicChain:
    """Simple implementation of a kinematic chain"""

    def __init__(self,
                 links,  # List of SE2 group elements
                 joint_axes):  # List of SE2 Lie algebra elements

        """Initialize a kinematic chain with a set of links and joint_axes"""
        
        # I added this for convenience :3
        dof = len(joint_axes)

        # Save the links and joint_axes as class attributes named links and joint axes
        self.links = links
        self.joint_axes = joint_axes

        # Create an attribute self.joint_angles to store the joint angles. This should be an ndarray of zeros with as
        # many entries as there are joint axes
        self.joint_angles = np.zeros(dof)

        # Create an attribute self.joint_transforms containing a list of joint transforms that is the same size as
        # the number of joint_axes. These transforms should start out as group identity elements
        self.joint_transforms = [G.identity_element()]*dof

        # Create an attribute self.link_positions containing a list of locations of the end points of the links
        # These transforms should be initialized as group identity elements
        self.link_positions = [G.identity_element()]*dof #not sure which group tbh

        return

    def set_configuration(self,
                          joint_angles):

        """Multiply the joint angles by the corresponding joint axes and exponentiate them, storing the resulting
        group elements as a list called 'joint_transforms' that is both saved as an attribute and returned by
        the function"""

        # Save the provided joint angles to self.joint_angles
        self.joint_angles = joint_angles

        # Set up a loop over the elements of self.joint_axes
        for i, alpha in enumerate(joint_angles):

            # Multiply the ith joint angle by the ith joint axis
            joint_motion = alpha * self.joint_axes[i]  #might need a .value an an np.multiply


            # Exponentiate the scaled axis (it's in the Lie algebra, so either exp_L or exp_R will work) and save it to
            # the ith element of self.joint_transforms
            
            joint_motion_exp = joint_motion.exp_L
            self.joint_transforms[i] = joint_motion_exp

        ########
        # Take the cumulative product of the joint transforms and link elements, saving the end point of each link to
        # self.link_positions

        # Save the product of the first joint transform and link to the first element of self.link_positions  
        self.link_positions[0] = self.joint_transforms[0] * self.links[0]

        # Set up a loop over the elements of self.link_positions, starting with the second position
        for i in range(1, len(self.link_positions)):

            # Set the ith element of self.link_positions to be the product of the (i-1)th element, the ith
            # joint transform, and the ith link transform
            self.link_positions[i] = self.link_positions[i-1] * self.joint_transforms[i] * self.links[i] #might need an np.matmul and also unsure of operation

        return self.link_positions

    def draw(self,
              ax):
        """ Draw the arm at its current set of joint angles, with its basepoint at the origin"""

        #######
        # Extract the x and y points for the ends of the links

        # Create lists for the x and y values of the link endpoints, initially one-element lists with zero values
        # (for the basepoint)
        x = [0]
        y = [0]


        # Loop over self.link_positions
        for l in self.link_positions:

            # Extract the value of the ith link_position
            pos = l.value

            # append the first and second elements of this value to the x and y lists
            x.append(pos[0])
            y.append(pos[1])
            

        # Plot the x and y values as a line
        
        ax.plot(x,y)

        # Set the plot aspect ratio to 'equal'
        ax.set_aspect('equal')

        return


if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc = KinematicChain(links, joint_axes)

    # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    kc.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])

    # Create a plotting axis
    ax = plt.subplot(1, 1, 1)

    # Draw the chain
    kc.draw(ax)

    # Tell pyplot to draw
    plt.show()
