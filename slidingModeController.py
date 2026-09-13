import numpy as np


class SlidingModeController:
    """Boundary-Layer Sliding Mode Controller for a 2nd-order system.
    
    Dynamics assumed by controller: m_hat * x_ddot + c_hat * x_dot + k_hat * x = u
    """
    def __init__(self, m_hat, c_hat, k_hat, lam, K, phi, output_limits=None):
        # Nominal parameter guesses
        self.m_hat = m_hat
        self.c_hat = c_hat
        self.k_hat = k_hat

        # SMC Tuning parameters
        self.lam = lam        # Sliding surface slope (bandwidth: e_dot + lam * e = 0)
        self.K = K            # Robust switching gain to overpower parameter bounds
        self.phi = phi        # Boundary layer thickness to prevent chattering
        self.output_limits = output_limits

    def compute_control(self, x, x_dot, x_d, x_d_dot=0.0, x_d_ddot=0.0):
        # Step 1: Define Tracking Error and Sliding Surface (s)
        # s = e_dot + lambda * e
        e = x - x_d
        e_dot = x_dot - x_d_dot
        s = e_dot + self.lam * e

        # Step 2: Nominal Equivalent Control (u_eq)
        # Evaluated at nominal parameters to enforce s_dot = 0 under ideal conditions
        u_eq = self.c_hat * x_dot + self.k_hat * x + self.m_hat * (x_d_ddot - self.lam * e_dot)

        # Step 3: Robust Switching Control (u_sw) with Boundary Layer Saturation
        # Replaces discontinuous sgn(s) with continuous sat(s / phi) to kill chattering
        s_norm = s / self.phi
        if s_norm > 1.0:
            sat_val = 1.0
        elif s_norm < -1.0:
            sat_val = -1.0
        else:
            sat_val = s_norm

        u_sw = -self.m_hat * self.K * sat_val

        # Total control effort
        control_signal = u_eq + u_sw

        # Apply actuator saturation if defined
        if self.output_limits is not None:
            lower, upper = self.output_limits
            control_signal = np.clip(control_signal, lower, upper)

        return control_signal