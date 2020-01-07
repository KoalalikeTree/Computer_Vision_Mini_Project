import math
import numpy as np
theta1 = 0
theta3 = 0

while theta1 <= 2*math.pi:
    while theta3 <= 2*math.pi:
        s1 = math.sin(theta1)
        s3 = math.sin(theta3)
        c1 = math.cos(theta1)
        c3 = math.cos(theta3)
        jacobian = np.array([[-9*s1-5*c3*s1, 0, -5*c1*s3],
                             [9*c1+5*c1*c3, 0, -5*s1*s3],
                             [0, 1, 5*c3]])
        det = np.linalg.det(jacobian)
        if abs(det) <= 1e-5:
            print("Singularity Config: theta1 = ", theta1, "theta3 = ", theta3)
        theta3 += math.pi / 2
    theta1 += math.pi/2
    theta3 = 0

