import numpy as np

M, R, G = 5.064, 0.081, 9.81            # measured values, not 5.1 / 0.20 / 0.16
L_R, H = 0.1892, 0.1605
RHO = float(np.hypot(L_R, H - R))
ALPHA = float(np.arctan2(H - R, L_R))
I_A = (1.0/12.0) * M * (0.53**2 + 0.30**2) + M * RHO**2
C_OMEGA = 0.2
KV, CV = 1.494, 0.0119

def nominal_accel(state, tau):
    x, v, theta, omega = state
    beta = ALPHA - theta
    omega_dot = (-(tau / R) * (R + RHO * np.sin(beta))
                 + M * G * RHO * np.cos(beta)) / I_A - C_OMEGA * omega
    if theta >= 0.0 and omega_dot > 0.0:
        omega_dot = 0.0
    return KV * tau - CV * v, omega_dot


# def nominal_accel(state, tau):
#     x, v, theta, omega = state

#     m = 5.1
#     r = 0.081
#     g = 9.81
#     l = 0.2
#     L_car = 0.53
#     H_body = 0.30

#     I_body = (1.0 / 12.0) * m * (L_car**2 + H_body**2)
#     I_eff = I_body + m * l**2

#     omega_dot = ((-tau + m * g * l * np.cos(theta)) / I_eff) 

#     if theta >= 0.0 and omega_dot > 0.0:
#         omega_dot = 0.0

#     x_dot = v
#     v_dot = (tau / (m * r))
#     theta_dot = omega

#     return v_dot, omega_dot