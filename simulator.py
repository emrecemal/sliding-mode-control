import numpy as np
import matplotlib.pyplot as plt
from springMassDamperModel import SpringMassDamperModel
from pidController import PIDController
from slidingModeController import SlidingModeController


def simulate_discrete(system, controller, initial_state, setpoint, t_span):
    A, B, _, _ = system.state_space()
    n_steps = len(t_span)
    states = np.zeros((n_steps, 2))
    control_signals = np.zeros(n_steps)

    state = np.array(initial_state, dtype=float)
    states[0] = state

    for i in range(n_steps - 1):
        dt = t_span[i + 1] - t_span[i]
        x, v = state[0], state[1]
        error = setpoint - x

        if controller is None:
            u = 0.0
        elif isinstance(controller, PIDController):
            u = controller.compute_control(error, dt)
        elif isinstance(controller, SlidingModeController):
            u = controller.compute_control(x=x, x_dot=v, x_d=setpoint, x_d_dot=0.0, x_d_ddot=0.0)
        else:
            u = 0.0

        control_signals[i] = u

        # Plant state update (Explicit Euler)
        x_dot = A @ state.reshape(-1, 1) + B * u
        state = state + x_dot.flatten() * dt
        states[i + 1] = state

    control_signals[-1] = control_signals[-2]
    return np.column_stack((states, control_signals))


def compare_saturation_plot(t_span, res_pid, res_smc, setpoint, limits, title):
    f, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    lower_limit, upper_limit = limits

    # Position
    axs[0].plot(t_span, res_pid[:, 0], 'r--', linewidth=1.5, label='PID with Anti-Windup')
    axs[0].plot(t_span, res_smc[:, 0], 'b-', linewidth=1.5, label='SMC (Boundary Layer)')
    axs[0].plot(t_span, np.full_like(t_span, setpoint), 'k:', label='Setpoint')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_ylim([0, 1.5])
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    # Velocity
    axs[1].plot(t_span, res_pid[:, 1], 'r--', linewidth=1.5, label='PID Velocity')
    axs[1].plot(t_span, res_smc[:, 1], 'b-', linewidth=1.5, label='SMC Velocity')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_ylim([-0.5, 2.0])
    axs[1].legend(loc='lower right')
    axs[1].grid(True)

    # Saturated Control Effort
    axs[2].plot(t_span, res_pid[:, 2], 'r--', linewidth=1.5, label='PID Control Effort u(t)')
    axs[2].plot(t_span, res_smc[:, 2], 'b-', linewidth=1.5, label='SMC Control Effort u(t)')
    axs[2].axhline(upper_limit, color='k', linestyle='--', alpha=0.7, label='Actuator Upper Bound')
    axs[2].axhline(lower_limit, color='k', linestyle='--', alpha=0.7, label='Actuator Lower Bound')
    axs[2].set_ylabel('Force u(t) [N]')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_xlim([0, 10])
    axs[2].set_ylim([lower_limit * 1.3, upper_limit * 1.3])
    axs[2].legend(loc='lower right')
    axs[2].grid(True)

    f.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()


def main():
    # True Physical System Parameters
    true_mass = 2.0
    true_damping = 1.0
    true_spring = 8.0
    real_system = SpringMassDamperModel(true_mass, true_damping, true_spring)

    x0, v0 = 0.0, 0.0
    initial_state = [x0, v0]
    setpoint = 1.0
    t_span = np.linspace(0, 10, 10000)

    # Realistic Physical Limits: Max ±15 N of force authority
    # Steady-state required force at x=1.0 is: k * x = 8.0 * 1.0 = 8.0 N.
    # 15 N leaves only 7 N of dynamic margin during transient motion.
    actuator_limits = [-15.0, 15.0]

    # Desired pole placement metrics for PID tuning baseline
    settling_time_desired = 2.0
    zeta = 0.707
    wn = 4.0 / (zeta * settling_time_desired)
    p3 = 3.0 * zeta * wn

    # =========================================================================
    # EXPERIMENT 1: Saturated Actuators with EXACT Nominal Parameters
    # =========================================================================
    print("Running Experiment 1: Actuator Saturation with Exact Parameters...")
    Kd_exact = true_mass * (p3 + 2.0 * zeta * wn) - true_damping
    Kp_exact = true_mass * (wn**2 + 2.0 * zeta * wn * p3) - true_spring
    Ki_exact = true_mass * (p3 * (wn**2))

    pid_exact = PIDController(
        Kp=Kp_exact, Ki=Ki_exact, Kd=Kd_exact, 
        output_limits=actuator_limits
    )
    smc_exact = SlidingModeController(
        m_hat=true_mass, c_hat=true_damping, k_hat=true_spring,
        lam=4.0, K=10.0, phi=0.05, 
        output_limits=actuator_limits
    )

    res_pid_exact = simulate_discrete(real_system, pid_exact, initial_state, setpoint, t_span)
    res_smc_exact = simulate_discrete(real_system, smc_exact, initial_state, setpoint, t_span)
    compare_saturation_plot(
        t_span, res_pid_exact, res_smc_exact, setpoint, actuator_limits,
        "Experiment 1: Saturated Actuator (±15N) with Exact Parameters"
    )
    plt.savefig("docs/experiment1_exact.png", dpi=300)

    # =========================================================================
    # EXPERIMENT 2: Saturated Actuators with SEVERE PARAMETER MISMATCH (~60% Error)
    # =========================================================================
    print("Running Experiment 2: Actuator Saturation with Parameter Mismatch...")
    m_guess = 0.8   # Real is 2.0 kg
    c_guess = 0.3   # Real is 1.0 Ns/m
    k_guess = 3.0   # Real is 8.0 N/m

    Kd_mismatch = m_guess * (p3 + 2.0 * zeta * wn) - c_guess
    Kp_mismatch = m_guess * (wn**2 + 2.0 * zeta * wn * p3) - k_guess
    Ki_mismatch = m_guess * (p3 * (wn**2))

    pid_mismatched = PIDController(
        Kp=Kp_mismatch, Ki=Ki_mismatch, Kd=Kd_mismatch, 
        output_limits=actuator_limits
    )
    smc_mismatched = SlidingModeController(
        m_hat=m_guess, c_hat=c_guess, k_hat=k_guess,
        lam=4.0, K=15.0, phi=0.05, 
        output_limits=actuator_limits
    )

    res_pid_mismatch = simulate_discrete(real_system, pid_mismatched, initial_state, setpoint, t_span)
    res_smc_mismatch = simulate_discrete(real_system, smc_mismatched, initial_state, setpoint, t_span)
    compare_saturation_plot(
        t_span, res_pid_mismatch, res_smc_mismatch, setpoint, actuator_limits,
        "Experiment 2: Saturated Actuator (±15N) with Severe Parameter Mismatch"
    )
    plt.savefig("docs/experiment2_mismatch.png", dpi=300)
    
    plt.show()


if __name__ == "__main__":
    main()