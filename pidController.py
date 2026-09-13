import numpy as np


class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.integral_error = 0
        self.previous_error = 0

    def compute_control(self, error, dt):
        derivative_error = (error - self.previous_error) / dt if dt > 0 else 0
        proportional_derivative = self.Kp * error + self.Kd * derivative_error
        proposed_integral = self.integral_error + error * dt
        control_signal = proportional_derivative + self.Ki * proposed_integral

        if self.output_limits is not None:
            lower, upper = self.output_limits
            if control_signal > upper:
                control_signal = upper
                if error > 0:
                    proposed_integral = self.integral_error
            elif control_signal < lower:
                control_signal = lower
                if error < 0:
                    proposed_integral = self.integral_error

        self.integral_error = proposed_integral
        self.previous_error = error
        return control_signal