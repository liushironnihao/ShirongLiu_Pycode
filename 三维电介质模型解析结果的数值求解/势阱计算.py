import numpy as np
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from functools import lru_cache
import time

# ====================
# Parameter Settings
# ====================
eps0 = 8.854e-12        # Vacuum permittivity F/m
H = 1.45e-6                # Dielectric layer thickness m
rho = 1e13              # Dielectric resistivity Ω·m
E_ext = -2e5            # External electric field V/m (Note: negative sign indicates downward direction)

# Physical parameters
n = 0.3e7               # Particle number density m⁻²
s = 3.5                 # Dielectric constant
e = 1.602e-19           # Elementary charge C
D = 2.2e-5              # Gas ion diffusion coefficient m²/s (serving as diffusion coefficient for surface charge species)
k = 1.380649e-23        # Boltzmann constant J/K
T = 298.0               # Temperature K


# Time parameters
sigma_initial = 1e-8    # Initial charge density C/m² (provide a small value to prevent initial numerical errors)
t_max = 60 * 60         # Final time s


# ====================
# Auxiliary Functions
# ====================
@lru_cache(maxsize=2000)
def solve_t_for_K(K_val):
    """Solve equation: (t²-1)/√(t⁴+t²+1) = K"""
    K_val = float(K_val)
    
    if K_val <= 1e-15:
        return 1.0 + np.sqrt(K_val/3)  # Small K approximation
    
    if K_val < 0.001:
        return 1.0 + np.sqrt(K_val/3) - K_val/6.0
    
    if K_val < 0.1:
        t_guess = 1.0 + np.sqrt(K_val/3)
    elif K_val < 1.0:
        t_guess = 1.2
    elif K_val < 10.0:
        t_guess = 1.5
    elif K_val < 100.0:
        t_guess = 2.0
    else:
        t_guess = 3.0
    
    def f(t):
        if t <= 1.0:
            return 1e6
        num = t**2 - 1
        den = np.sqrt(t**4 + t**2 + 1)
        return num/den - K_val
    
    try:
        # Try Brent's method
        t_sol = root_scalar(f, bracket=[1.001, 10.0], 
                          method='brentq', xtol=1e-12, rtol=1e-12)
        if t_sol.converged and t_sol.root > 1.0:
            return t_sol.root
    except:
        pass
    
    # Fallback: use fsolve
    try:
        t_sol = fsolve(f, t_guess, xtol=1e-10, maxfev=200)[0]
        if t_sol > 1.0:
            return t_sol
    except:
        pass
    
    # Last resort: use analytical approximation
    if K_val < 0.1:
        return 1.0 + np.sqrt(K_val/3)
    elif K_val < 1.0:
        return 1.0 + 0.5 * K_val**0.5
    else:
        return 1.0 + np.log(1 + K_val)

def compute_geometric_params(sigma):
    """Calculate geometric parameters based on formula: z_m(σ) and R_m(σ)"""
   
    sigma_eff = sigma / s  # σ/3   # Electric field correction for dielectric constant, replace sigma with sigma/s
    
    if sigma_eff < 1e-20:
        return 1e-12, 1e-12, 1.001, 0.0
    
    
    K = -2 * eps0 * E_ext / sigma_eff  
    
    # Handle extreme cases of K
    if K > 1000000:  # σ_eff very small
        return 1e-12, 1e-12, 1.001, K
    if K < 1e-15:  # σ_eff very large
        # Use asymptotic expansion
        delta = np.sqrt(K/3)
        t_val = 1.0 + delta
        # Use series expansion to avoid division by zero
        zm = 2 * H / (3 * delta + 3 * delta**2 + delta**3)
        Rm = 2 * H * (1 + delta) * np.sqrt((1 + delta)**2 + 1) / (3 * delta + 3 * delta**2 + delta**3)
        return zm, Rm, t_val, K
    
    t_val = solve_t_for_K(round(K, 12))  # Higher precision
    
    # Ensure t > 1
    if t_val <= 1.001:
        t_val = 1.001
    
    # Use formula from the figure
    denom = t_val**3 - 1
    
    # Use series expansion when denom is very small
    if abs(denom) < 1e-8:
        delta = t_val - 1.0
        # Series expansion: t^3 - 1 = 3δ + 3δ² + δ³
        zm = 2 * H / (3 * delta + 3 * delta**2 + delta**3)
        Rm = 2 * H * t_val * np.sqrt(t_val**2 + 1) / (3 * delta + 3 * delta**2 + delta**3)
    else:
        zm = 2 * H / denom
        Rm = 2 * H * t_val * np.sqrt(t_val**2 + 1) / denom
    
    # Physical constraints
    zm = min(max(zm, 1e-12), 1e-3)  # 1pm to 1mm
    Rm = min(max(Rm, 1e-12), 1e-2)  # 1pm to 1cm
    
    return zm, Rm, t_val, K

def compute_V_total(z, sigma, zm, Rm):
    """Calculate total potential based on formula: V_total(z; σ)"""

    sigma_eff = sigma / s  # σ/3
    
    if sigma_eff < 1e-20 or Rm < 1e-20:
        return -E_ext * z
    
    if abs(z) < 1e-20:  # z=0 case
        
        V_total = (sigma_eff/(2*eps0)) * (Rm - np.sqrt((2*H)**2 + Rm**2))
    else:
        sqrt1 = np.sqrt(max(z**2 + Rm**2, 1e-30))
        sqrt2 = np.sqrt(max((z + 2*H)**2 + Rm**2, 1e-30))
        # Replace sigma with sigma_eff in V_total formula
        V_total = (sigma_eff/(2*eps0)) * (sqrt1 - sqrt2) - E_ext * z
    
    return V_total

def compute_V0(sigma):
    """Calculate barrier height based on formula: V₀(σ) = V_total(0;σ) - V_total(z_m;σ)"""
    if sigma < 1e-20:
        return 0.0, 1e-9, 1e-9
    
    zm, Rm, _, _ = compute_geometric_params(sigma)
    
    V_at_0 = compute_V_total(0, sigma, zm, Rm)
    V_at_zm = compute_V_total(zm, sigma, zm, Rm)
    
    V0 = V_at_0 - V_at_zm
    V0 = max(V0, 0.0)  # Ensure non-negative
    
    return V0, zm, Rm

def compute_current_density(sigma):
    """Calculate current density J based on formula: J = (E_ext + σ/ε₀) / ρ"""
    # Note: E_ext is negative, σ is positive, so electric fields add
    # Total electric field = E_ext + σ/ε₀
    E_total = abs(E_ext) + sigma / eps0 / s  # Use absolute value to ensure total field is positive
    
    # Current density J = E_total / ρ
    J = E_total / rho
    
    return J


def dsigmadt_vectorized(t, y):
    sigma = float(y[0])
    
    if sigma < 1e-20:
        return np.array([0.0])
    
    V0, zm, _ = compute_V0(sigma)
    
    # Prevent zm from being too small
    if zm < 1e-12:
        zm = 1e-12
    
    # Charging term (corrected formula)
    particle_flux = 2 * D * n / (zm**2)  # Particle flux m⁻²/s
    exponent = -e * V0 / (k * T)
    
    # Limit exponent range
    if exponent > 50:
        exp_term = np.exp(50)
    elif exponent < -50:
        exp_term = np.exp(-50)
    else:
        exp_term = np.exp(exponent)
    
    # Convert particle flux to current density: J = e × particle flux × exp(-eV0/kT)
    charging_current = e * particle_flux * exp_term  # A/m² = C/(m²·s)
    
    # Loss term (current density)
    loss_current = compute_current_density(sigma)  # A/m²
    
    # Net current density change
    net_current = charging_current - loss_current
    
    # Convert current density to charge density rate: dσ/dt = J
    result = net_current
    
    # Ensure reasonable result
    if sigma < 1e-20 and result < 0:
        result = 0.0
    
    return np.array([result])

# ====================
# Robust Integration Function
# ====================
def robust_integration():
    """Robust integration function to handle numerical instability"""
    print("Starting robust integration...")
    print(f"Parameters: H={H:.1e}m, E_ext={E_ext:.1e}V/m")
    print(f"n={n:.1e}m⁻², D={D:.1e}m²/s, T={T}K")
    print(f"Resistivity ρ={rho:.1e}Ω·m")
    print(f"Initial σ₀={sigma_initial:.1e}C/m², integrate to t={t_max:.1f}s")
    print(f"Using effective charge density: σ_eff = σ/3")
    print("="*60)
    
    all_times = []
    all_sigmas = []
    
    # Fine time segmentation
    time_segments = [
        (0.0, 1e-12, 1e-14, 50),
        (1e-12, 1e-10, 1e-12, 50),
        (1e-10, 1e-8, 1e-10, 50),
        (1e-8, 1e-6, 1e-9, 100),
        (1e-6, 1e-4, 1e-7, 200),
        (1e-4, 1e-2, 1e-5, 300),
        (1e-2, 0.1, 1e-4, 400),
        (0.1, 1.0, 1e-3, 500),
        (1.0, 10.0, 0.01, 600),
        (10.0, 100.0, 0.1, 800),
        (100.0, 1000.0, 1.0, 1000),
        (1000.0, 10000.0, 10.0, 800),
        (10000.0, t_max, 100.0, 600),
    ]
    
    # Only integrate up to t_max
    time_segments = [(start, min(end, t_max), step, n_pts) 
                     for start, end, step, n_pts in time_segments 
                     if start < t_max]
    
    current_sigma = sigma_initial
    current_time = 0.0
    
    for seg_idx, (seg_start, seg_end, max_step, n_points) in enumerate(time_segments):
        if current_time >= t_max or seg_start >= t_max:
            break
        
        seg_start = max(seg_start, current_time)
        seg_end = min(seg_end, t_max)
        
        if seg_end <= seg_start:
            continue
        
        print(f"\nIntegration segment {seg_idx+1}: [{seg_start:.1e}s, {seg_end:.1e}s]")
        print(f"  Duration: {seg_end-seg_start:.1e}s, max step: {max_step:.1e}s")
        
        t_eval = np.linspace(seg_start, seg_end, n_points)
        
        try:
            sol = solve_ivp(
                dsigmadt_vectorized,
                (seg_start, seg_end),
                [current_sigma],
                t_eval=t_eval,
                method='Radau',  # Better for stiff problems
                rtol=1e-8,
                atol=1e-12,
                max_step=max_step
            )
            
            if sol.success:
                # Store results
                if len(all_times) == 0:
                    all_times = sol.t.copy()
                    all_sigmas = sol.y[0].copy()
                else:
                    all_times = np.concatenate([all_times, sol.t[1:]])
                    all_sigmas = np.concatenate([all_sigmas, sol.y[0, 1:]])
                
                current_sigma = sol.y[0, -1]
                current_time = seg_end
                
                sigma_start = sol.y[0, 0]
                sigma_end = sol.y[0, -1]
                
                print(f"  Success: σ from {sigma_start:.2e} to {sigma_end:.2e} C/m²")
                print(f"  Effective charge density σ_eff: {sigma_start/s:.2e} to {sigma_end/s:.2e} C/m²")
                print(f"  Relative change: {sigma_end/sigma_start:.2f} times")
                
                # Check for saturation
                if len(sol.y[0]) > 5:
                    dsigma_dt_last = dsigmadt_vectorized(sol.t[-1], [sol.y[0, -1]])[0]
                    if abs(dsigma_dt_last) < 1e-20:
                        print(f"  Net charging rate very small ({dsigma_dt_last:.1e}), approaching equilibrium")
                        break
                        
            else:
                print(f"  Segment integration failed: {sol.message}")
                # Simple Euler method as backup
                dt = max_step / 10
                t_vals = np.arange(seg_start, seg_end, dt)
                if len(t_vals) == 0:
                    t_vals = np.array([seg_start, seg_end])
                
                sigma_vals = [current_sigma]
                for t in t_vals[1:]:
                    dsigma = dsigmadt_vectorized(t, [sigma_vals[-1]])[0] * dt
                    sigma_new = sigma_vals[-1] + dsigma
                    if sigma_new > 1e-2:  # Reasonable upper limit
                        sigma_new = 1e-2
                    sigma_vals.append(sigma_new)
                
                all_times = np.concatenate([all_times, t_vals[1:]]) if len(all_times) > 0 else t_vals
                all_sigmas = np.concatenate([all_sigmas, sigma_vals[1:]]) if len(all_sigmas) > 0 else sigma_vals
                current_sigma = sigma_vals[-1]
                current_time = t_vals[-1]
                
                print(f"  Euler method: σ from {sigma_vals[0]:.2e} to {sigma_vals[-1]:.2e} C/m²")
                print(f"  Effective charge density σ_eff: {sigma_vals[0]/s:.2e} to {sigma_vals[-1]/s:.2e} C/m²")
                
        except Exception as e:
            print(f"  Integration error: {e}")
            break
    
    return np.array(all_times), np.array(all_sigmas)

# ====================
# Main Program
# ====================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Run integration
    times, sigmas = robust_integration()
    
    if len(times) == 0 or len(sigmas) == 0:
        print("\nIntegration failed!")
    else:
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Integration completed! Total time: {total_time:.2f} seconds")
        print(f"Computed up to time: t = {times[-1]:.6f} s")
        print(f"Final σ = {sigmas[-1]:.6e} C/m²")
        print(f"Final effective charge density σ_eff = {sigmas[-1]/s:.6e} C/m²")
        print(f"Growth factor: {sigmas[-1]/sigma_initial:.2f}")
        print("="*60)
        
        # 2. Compute related physical quantities
        print("\nComputing related physical quantities...")
        
        # Sample calculation, avoid computing all points
        sample_indices = np.linspace(0, len(sigmas)-1, min(1000, len(sigmas)), dtype=int)
        
        zm_vals = []
        Rm_vals = []
        V0_vals = []
        dsigma_dt_vals = []
        K_vals = []
        charging_rates = []
        loss_rates = []
        current_densities = []
        
        for i in sample_indices:
            sigma_val = sigmas[i]
            V0, zm, Rm = compute_V0(sigma_val)
            zm_vals.append(zm)
            Rm_vals.append(Rm)
            V0_vals.append(V0)
            
            dsigma_dt = dsigmadt_vectorized(times[i], [sigma_val])[0]
            dsigma_dt_vals.append(dsigma_dt)
            
            # Corrected charging rate calculation (current density form)
            exponent = -e * V0 / (k * T)
            exp_term = np.exp(exponent) 
            particle_flux = 2 * D * n / (zm**2)
            charging_rate = e * particle_flux * exp_term  # A/m²
            charging_rates.append(charging_rate)
            
            # Loss rate (current density)
            loss_rate = compute_current_density(sigma_val)
            loss_rates.append(loss_rate)
            current_densities.append(loss_rate)
            
            # Calculate K value (using effective charge density)
            sigma_eff = sigma_val / s  # σ/3
            K = -2 * eps0 * E_ext / sigma_eff if sigma_val > 1e-20 else 0
            K_vals.append(K)
        
        zm_vals = np.array(zm_vals)
        Rm_vals = np.array(Rm_vals)
        V0_vals = np.array(V0_vals)
        dsigma_dt_vals = np.array(dsigma_dt_vals)
        K_vals = np.array(K_vals)
        charging_rates = np.array(charging_rates)
        loss_rates = np.array(loss_rates)
        current_densities = np.array(current_densities)
        sample_times = times[sample_indices]
        
        # 3. Plot results
        print("Plotting results...")
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        
        # Subplot 1: σ(t) log scale
        ax1 = axes[0, 0]
        ax1.plot(times, sigmas, 'b-', linewidth=1, alpha=0.7, label='σ')
        ax1.plot(times, sigmas/s, 'b--', linewidth=1, alpha=0.5, label='σ_eff = σ/3')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Charge Density (C/m²)')
        ax1.set_title('Surface Charge Density Evolution')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: σ(t) linear scale
        ax2 = axes[0, 1]
        ax2.plot(times, sigmas, 'r-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('σ (C/m²)')
        ax2.set_title('σ(t) (Linear Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Charging and loss rates (corrected)
        ax3 = axes[0, 2]
        
        ax3.plot(sample_times, loss_rates, 'r-', linewidth=1, label='Loss Rate')
        ax3.plot(sample_times, dsigma_dt_vals, 'b-', linewidth=1.5, label='Net Rate (dσ/dt)')
        ax3.plot(sample_times, charging_rates, 'g-', linewidth=1, label='Charging Rate')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Current Density (A/m²)')
        ax3.set_title('Charging and Loss Rates (Corrected Units)')
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: zm(t)
        ax4 = axes[1, 0]
        ax4.plot(sample_times, zm_vals, 'm-', linewidth=1)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('$z_m$ (m)')
        ax4.set_title('Optimal Height $z_m$')
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Rm(t)
        ax5 = axes[1, 1]
        ax5.plot(sample_times, Rm_vals, 'c-', linewidth=1)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('$R_m$ (m)')
        ax5.set_title('Optimal Radius $R_m$')
        ax5.set_yscale('log')
        ax5.set_xscale('log')
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: zm/Rm
        ax6 = axes[1, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            zm_ratio = zm_vals / Rm_vals
            zm_ratio[np.isinf(zm_ratio)] = np.nan
            zm_ratio = np.nan_to_num(zm_ratio, nan=0.0)
        ax6.plot(sample_times, zm_ratio, 'orange', linewidth=1)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('$z_m/R_m$')
        ax6.set_title('Aspect Ratio $z_m/R_m$')
        ax6.set_xscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Subplot 7: V0(t)
        ax7 = axes[2, 0]
        ax7.plot(sample_times, V0_vals, 'brown', linewidth=1)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('$V_0$ (V)')
        ax7.set_title('Potential Barrier $V_0$')
        ax7.set_xscale('log')
        ax7.grid(True, alpha=0.3)
        
        # Subplot 8: eV0/kT
        ax8 = axes[2, 1]
        eV0_kT = e * V0_vals / (k * T)
        ax8.plot(sample_times, eV0_kT, 'purple', linewidth=1)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('$eV_0/(kT)$')
        ax8.set_title('Reduced Barrier Height')
        ax8.set_xscale('log')
        ax8.grid(True, alpha=0.3)
        
        # Subplot 9: Current density J
        ax9 = axes[2, 2]
        ax9.plot(sample_times, current_densities, 'gray', linewidth=1)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Current Density (A/m²)')
        ax9.set_title('Current Density J = (E_ext + σ/ε₀)/ρ')
        ax9.set_yscale('log')
        ax9.set_xscale('log')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 4. Detailed analysis
        print("\n" + "="*60)
        print("Charging kinetics detailed analysis:")
        print("="*60)
        
        # Check 10-1000 second interval
        print("\nChecking 10-1000 second interval:")
        mask_10_1000 = (times >= 10.0) & (times <= 1000.0)
        if np.any(mask_10_1000):
            times_10_1000 = times[mask_10_1000]
            sigmas_10_1000 = sigmas[mask_10_1000]
            
            print(f"Number of data points in interval: {len(times_10_1000)}")
            print(f"σ range: {sigmas_10_1000[0]:.2e} to {sigmas_10_1000[-1]:.2e} C/m²")
            print(f"σ_eff range: {sigmas_10_1000[0]/s:.2e} to {sigmas_10_1000[-1]/s:.2e} C/m²")
            
            if len(times_10_1000) > 1:
                avg_rate = (sigmas_10_1000[-1] - sigmas_10_1000[0]) / (times_10_1000[-1] - times_10_1000[0])
                print(f"Average net charging rate: {avg_rate:.2e} C/m²/s")
                
                # Check for anomalies
                diff_sigmas = np.diff(sigmas_10_1000)
                if np.any(diff_sigmas < 0):
                    print("Warning: Detected regions where σ decreases!")
                if np.max(np.abs(diff_sigmas)) > 1e-6:
                    print("Warning: Detected large jumps!")
        
        # Find equilibrium point
        print("\nFinding equilibrium point...")
        if len(sigmas) > 10:
            # Check if last few points have reached equilibrium
            last_n = min(20, len(sigmas))
            recent_rates = []
            for i in range(-last_n, 0):
                if i + 1 < 0:
                    rate = (sigmas[i] - sigmas[i-1]) / (times[i] - times[i-1])
                    recent_rates.append(abs(rate))
            
            if recent_rates:
                avg_recent_rate = np.mean(recent_rates)
                print(f"Average change rate for recent {last_n} time points: {avg_recent_rate:.2e} C/m²/s")
                
                if avg_recent_rate < 1e-15:
                    print("Steady-state equilibrium reached!")
                    
                    # Compute charging and loss rates at equilibrium
                    final_sigma = sigmas[-1]
                    final_J = compute_current_density(final_sigma)
                    
                    # Compute charging rate
                    V0_final, zm_final, _ = compute_V0(final_sigma)
                    exponent = -e * V0_final / (k * T)
                    exp_term = np.exp(exponent) if exponent > -50 else np.exp(-50)
                    particle_flux = 2 * D * n / (zm_final**2)
                    charging_final = e * particle_flux * exp_term
                    
                    print(f"Charging rate at equilibrium: {charging_final:.2e} A/m²")
                    print(f"Loss rate (J) at equilibrium: {final_J:.2e} A/m²")
                    print(f"Difference between them: {abs(charging_final - final_J):.2e} A/m²")
        
        # 5. Final state
        print("\n" + "="*60)
        print("Final state (t = {:.2f} s):".format(times[-1]))
        print("="*60)
        
        final_sigma = sigmas[-1]
        final_sigma_eff = final_sigma / s  # σ/3
        V0_final, zm_final, Rm_final = compute_V0(final_sigma)
        dsigma_final = dsigmadt_vectorized(times[-1], [final_sigma])[0]
        
        # Compute charging and loss rates
        exponent = -e * V0_final / (k * T)
        exp_term = np.exp(exponent) if exponent > -50 else np.exp(-50)
        particle_flux = 2 * D * n / (zm_final**2)
        charging_final = e * particle_flux * exp_term
        loss_final = compute_current_density(final_sigma)
        
        print(f"Charge density: σ = {final_sigma:.6e} C/m²")
        print(f"Effective charge density: σ_eff = {final_sigma_eff:.6e} C/m²")
        print(f"Net charging rate: dσ/dt = {dsigma_final:.6e} C/m²/s")
        print(f"Charging rate: {charging_final:.6e} A/m²")
        print(f"Loss rate (J): {loss_final:.6e} A/m²")
        print(f"Equilibrium difference: {charging_final - loss_final:.6e} A/m²")
        
        print(f"\nElectric field related:")
        E_from_sigma = final_sigma / eps0 / s
        E_total = abs(E_ext) + E_from_sigma
        print(f"  Electric field from σ: σ/ε₀ = {E_from_sigma:.2e} V/m")
        print(f"  Total electric field: E_total = E_ext + σ/ε₀ = {E_total:.2e} V/m")
        print(f"  Current density: J = E_total/ρ = {loss_final:.2e} A/m²")
        
        if dsigma_final > 0:
            time_to_double = np.log(2) * final_sigma / dsigma_final
            print(f"Current doubling time: {time_to_double:.2e} s")
        
        print(f"\nGeometric parameters:")
        print(f"  zₘ = {zm_final:.6e} m")
        print(f"  Rₘ = {Rm_final:.6e} m")
        print(f"  zₘ/H = {zm_final/H:.4f}")
        print(f"  Rₘ/H = {Rm_final/H:.4f}")
        
        print(f"\nBarrier parameters:")
        print(f"  V₀ = {V0_final:.6f} V")
        print(f"  e·V₀/(kT) = {e*V0_final/(k*T):.4f}")
        print(f"  exp(-e·V₀/(kT)) = {np.exp(-e*V0_final/(k*T)):.2e}")
        
        print(f"\nRelated parameters:")
        # Calculate K using effective charge density
        K_final = -2 * eps0 * E_ext / final_sigma_eff
        print(f"  K = -2ε₀E_ext/(σ/3) = {K_final:.6f}")
        print(f"  t(K) = {solve_t_for_K(K_final):.6f}")
        
        print("="*60)