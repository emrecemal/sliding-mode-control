import numpy as np


class SpringMassDamperModel:
    def __init__(self, mass, damping_coefficient, spring_constant):
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.spring_constant = spring_constant

    def state_space(self):
        A = np.array([[0, 1],
                      [-self.spring_constant / self.mass, -self.damping_coefficient / self.mass]])
        B = np.array([[0],
                      [1 / self.mass]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        return A, B, C, D
    
    
    