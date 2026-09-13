import numpy as np

from springMassDamperModel import SpringMassDamperModel
from pidController import PIDController


def simulate_discrete(system, controller, initial_state, setpoint, t_span):
    A, B, _, _ = system.state_space()
    n_steps = len(t_span)
    states = np.zeros((n_steps, 2))
    control_signals = np.zeros(n_steps)

    state = np.array(initial_state, dtype=float)
    states[0] = state

    for i in range(n_steps - 1):
        dt = t_span[i + 1] - t_span[i]
        error = setpoint - state[0]

        u = controller.compute_control(error, dt) if controller else 0.0
        control_signals[i] = u

        # Explicit Euler or RK4 step for the plant
        x_dot = A @ state.reshape(-1, 1) + B * u
        state = state + x_dot.flatten() * dt
        states[i + 1] = state

    control_signals[-1] = control_signals[-2]
    return np.column_stack((states, control_signals))
    
def plot_results(t_span, results, title):
    import matplotlib.pyplot as plt
    
    f, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t_span, results[:, 0], label='Position (m)')
    axs[0].plot(t_span, np.ones_like(t_span), 'r--', label='Setpoint (m)')
    axs[0].set_ylabel('Position Response')
    axs[0].legend()
    axs[0].grid()
    
    axs[1].plot(t_span, results[:, 1], label='Velocity (m/s)')
    axs[1].plot(t_span, np.zeros_like(t_span), 'r--', label='Setpoint (m)')
    axs[1].set_ylabel('Velocity Response')
    axs[1].legend()
    axs[1].grid()
    
    axs[2].plot(t_span, results[:, 2], label='Control Signal')
    axs[2].set_ylabel('Control Signal')
    axs[2].legend()
    axs[2].grid()
    
    f.suptitle(title)
    
    plt.show()
    

def main():
    mass = 1.0
    damping_coefficient = 0.5
    spring_constant = 2.0
    
    x0 = 0.5  # Initial position
    v0 = 0.0  # Initial velocity
    initial_state = [x0, v0]
    
    setpoint = 1.0  # Desired position
    
    t_span = np.linspace(0, 10, 10000)  # Time span for simulation
    system = SpringMassDamperModel(mass, damping_coefficient, spring_constant)
    
    # Simulate without any controller
    controller = None
    result_no_control = simulate_discrete(system, controller, initial_state, setpoint, t_span)
    plot_results(t_span, result_no_control, "No Control")
    
    # Simulate with hand-tuned PID controller
    controller = PIDController(10.0, 5.0, 2.0)  # Kp, Ki, Kd
    result_with_control = simulate_discrete(system, controller, initial_state, setpoint, t_span)
    plot_results(t_span, result_with_control, "Hand-tuned PID, Kp=10, Ki=5, Kd=2")
    
    # Simulate with system-tuned PID controller
    settling_time_desired = 5.0  # Desired settling time in seconds
    damping_ratio_desired = 0.7  # Desired damping ratio
    frequency_desired = 4 / damping_ratio_desired / settling_time_desired
    
    # Pole placement
    p3 = 3.0 * damping_ratio_desired * frequency_desired
    Kd = mass * (p3 + 2 * damping_ratio_desired * frequency_desired) - damping_coefficient
    Kp = mass * (frequency_desired ** 2 + 2 * damping_ratio_desired * frequency_desired * p3) - spring_constant
    Ki = mass * (p3 * (frequency_desired ** 2))  # Integral gain for zero steady-state error
    controller = PIDController(Kp, Ki, Kd)  # Kp, Ki, Kd
    result_with_control = simulate_discrete(system, controller, initial_state, setpoint, t_span)
    plot_results(t_span, result_with_control, f"System-tuned PID, T={settling_time_desired:.2f}, ζ={damping_ratio_desired:.2f}, ω={frequency_desired:.2f}, Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
        
    
if __name__ == "__main__":
    main()