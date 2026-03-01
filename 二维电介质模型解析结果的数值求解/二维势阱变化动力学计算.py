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
h = 1.5e-6             # Electrode separation distance m (replaces H)
rho_s = 1e13            # Surface resistivity Ω (2D equivalent, replaces rho)
E_ext = -1e5            # External electric field V/m (Note: negative sign indicates downward direction)

# Physical parameters
c = 2.2e9*1e6                 # Particle number density in air m⁻3
s = 3.5                 # Dielectric constant
e = 1.602e-19           # Elementary charge C
D = 2.2e-5              # Gas ion diffusion coefficient m²/s (serving as diffusion coefficient for surface charge species)
k = 1.380649e-23        # Boltzmann constant J/K
T = 298.0               # Temperature K
Δh = 20e-9               # Small vertical displacement for potential difference calculation         
d_p = 0.3e-9              # Particle diameter

# Time parameters
sigma_initial = 1e-16    # Initial charge density C/m² (provide a small value to prevent initial numerical errors)
t_max = 0.4e-3         # Final time s

# ====================
# Auxiliary Functions (2D version)
# ====================
def compute_L(sigma):
    """Calculate L(σ) = y_min = -h + sqrt(h² - σh/(πε₀E_ext)) from the image"""
    if abs(sigma) < 1e-20 or abs(E_ext) < 1e-20:
        return 1e-12
    
    # Ensure the expression under sqrt is positive
    discriminant = h**2 - sigma * h / (np.pi * eps0 * E_ext)
    if discriminant < 0:
        # If discriminant negative, use small value
        return 1e-12
    
    L = -h + np.sqrt(discriminant)
    return max(L, 1e-12)  # Ensure positive value

def compute_V_y(y, sigma):
    """Calculate V_y = σ/(2πε₀) * ln(1+2h/y) - E_ext * y from the image"""
    if y <= 0:
        y = 1e-12
    
    term1 = sigma / (2 * np.pi * eps0) * np.log(1 + 2*h/y)
    term2 = E_ext * y
    return term1 - term2

def compute_U(sigma):
    """Calculate U(σ) = V(y=Δh) - V(y_min) from the image"""
    y_min = compute_L(sigma)
    
    V_Δh = compute_V_y(Δh, sigma)
    V_ymin = compute_V_y(y_min, sigma)
    
    U = V_Δh - V_ymin
    return U

def compute_J(sigma):
    """Calculate J(σ) = -E_y=-h/ρ_s = -σ/(πε₀hρ_s) - E_ext/ρ_s from the image"""
    J=0
    #J = -sigma / (np.pi * eps0 * h * rho_s) + E_ext / rho_s
    return J

def dsigmadt_vectorized(t, y):
    """Calculate dσ/dt according to the 2D equation from the image"""
    sigma = float(y[0])
    
    if sigma < 1e-20:
        return np.array([0.0])
    
    # Extract parameters from image equation
    L_sigma = compute_L(sigma)
    U_sigma = compute_U(sigma)
    J_sigma = compute_J(sigma)
    
    # Charging term (first term in equation)
    charging_term = 2 * D * c * d_p * e / (L_sigma**2) * np.exp(-e * U_sigma / (k * T))
    
    # Full equation: dσ/dt = charging_term - J(σ)
    result = charging_term + J_sigma
    
    # Ensure reasonable bounds
    if sigma < 1e-20 and result < 0:
        result = 0.0
    
    return np.array([result])

# ====================
# Robust Integration Function
# ====================
def robust_integration():
    """Robust integration function to handle numerical instability"""
    print("Starting robust integration (2D version)...")
    print(f"Parameters: h={h:.1e}m, E_ext={E_ext:.1e}V/m")
    print(f"c={c:.1e}m⁻², D={D:.1e}m²/s, T={T}K")
    print(f"Surface resistivity ρ_s={rho_s:.1e}Ω")
    print(f"Initial σ₀={sigma_initial:.1e}C/m², integrate to t={t_max:.1f}s")
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
        (10000.0, t_max, 100.0, 600)
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
                print(f"  Relative change: {sigma_end/sigma_start:.2f} times")
                
                # Check for saturation
                if len(sol.y[0]) > 5:
                    dsigma_dt_last = dsigmadt_vectorized(sol.t[-1], [sol.y[0, -1]])[0]
                    if abs(dsigma_dt_last) < 1e-25:
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
        print(f"Growth factor: {sigmas[-1]/sigma_initial:.2f}")
        print("="*60)
        
        # 2. Compute related physical quantities
        print("\nComputing related physical quantities (2D version)...")
        
        # Sample calculation, avoid computing all points
        sample_indices = np.linspace(0, len(sigmas)-1, min(1000, len(sigmas)), dtype=int)
        
        L_vals = []
        U_vals = []
        J_vals = []
        dsigma_dt_vals = []
        charging_rates = []
        V_Δh_vals = []
        V_ymin_vals = []
        
        for i in sample_indices:
            sigma_val = sigmas[i]
            
            # Compute 2D geometric parameters
            L = compute_L(sigma_val)
            U = compute_U(sigma_val)
            J = compute_J(sigma_val)
            
            L_vals.append(L)
            U_vals.append(U)
            J_vals.append(J)
            
            # Potential at y=Δh and y=y_min
            V_Δh = compute_V_y(Δh, sigma_val)
            V_ymin = compute_V_y(L, sigma_val)
            V_Δh_vals.append(V_Δh)
            V_ymin_vals.append(V_ymin)
            
            # Rate of change
            dsigma_dt = dsigmadt_vectorized(times[i], [sigma_val])[0]
            dsigma_dt_vals.append(dsigma_dt)
            
            # Charging rate (first term in equation)
            charging_rate = 2 * D * c * d_p * e / (L**2) * np.exp(-e * U / (k * T))
            charging_rates.append(charging_rate)
        
        L_vals = np.array(L_vals)
        U_vals = np.array(U_vals)
        J_vals = np.array(J_vals)
        dsigma_dt_vals = np.array(dsigma_dt_vals)
        charging_rates = np.array(charging_rates)
        V_Δh_vals = np.array(V_Δh_vals)
        V_ymin_vals = np.array(V_ymin_vals)
        sample_times = times[sample_indices]
        
        # 3. Plot results
        print("Plotting results...")
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        
        # Subplot 1: σ(t) log scale
        ax1 = axes[0, 0]
        ax1.plot(times, sigmas, 'b-', linewidth=1, alpha=0.7, label='σ')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Surface Charge Density (C/m)')
        ax1.set_title('Surface Charge Density Evolution (2D)')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: σ(t) linear scale
        ax2 = axes[0, 1]
        ax2.plot(times, sigmas, 'r-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('σ (C/m)')
        ax2.set_xscale('log')
        ax2.set_title('σ(t) (Linear Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Charging and loss rates
        ax3 = axes[0, 2]
        ax3.plot(sample_times, -J_vals, 'r-', linewidth=1, label='Loss Rate -J(σ)')
        ax3.plot(sample_times, dsigma_dt_vals, 'b-', linewidth=1.5, label='Net Rate (dσ/dt)')
        ax3.plot(sample_times, charging_rates, 'g-', linewidth=1, label='Charging Rate')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Rate (C/m/s)')
        ax3.set_title('Charging and Loss Rates (2D)')
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: L(σ) = y_min
        ax4 = axes[1, 0]
        ax4.plot(sample_times, L_vals, 'm-', linewidth=1)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('L(σ) = y_min (m)')
        ax4.set_title('Optimal Position y_min (2D)')
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: U(σ) = V(Δh) - V(y_min)
        ax5 = axes[1, 1]
        ax5.plot(sample_times, U_vals, 'c-', linewidth=1)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('U(σ) (V)')
        ax5.set_title('electric field potential well depth U(σ)')
        ax5.set_xscale('log')
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: y_min/h ratio
        ax6 = axes[1, 2]
        ax6.plot(sample_times, L_vals*1e6, 'orange', linewidth=1)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('L(σ) (μm)')
        ax6.set_title(' electric field potential well width L(σ)')
        ax6.set_xscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Subplot 7: V(Δh) and V(y_min)
        ax7 = axes[2, 0]
        ax7.plot(sample_times, V_Δh_vals, 'brown', linewidth=1, label='V(Δh)')
        ax7.plot(sample_times, V_ymin_vals, 'gray', linewidth=1, label='V(y_min)')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Potential (V)')
        ax7.set_title('Potentials at y=Δh and y=y_min')
        ax7.set_xscale('log')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Subplot 8: eU(σ)/kT
        ax8 = axes[2, 1]
        eU_kT = e * U_vals / (k * T)
        ax8.plot(sample_times, eU_kT, 'purple', linewidth=1)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('$eU/(kT)$')
        ax8.set_title('Reduced Potential Barrier')
        ax8.set_xscale('log')
        ax8.grid(True, alpha=0.3)
        
        # Subplot 9: Exponential factor
        ax9 = axes[2, 2]
        exp_factor = np.exp(-e * U_vals / (k * T))
        ax9.plot(sample_times, exp_factor, 'gray', linewidth=1)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('exp(-eU/kT)')
        ax9.set_title('Boltzmann Factor')
        ax9.set_yscale('log')
        ax9.set_xscale('log')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 4. Detailed analysis
        print("\n" + "="*60)
        print("2D Charging kinetics detailed analysis:")
        print("="*60)
        
        # Check 10-1000 second interval
        print("\nChecking 10-1000 second interval:")
        mask_10_1000 = (times >= 10.0) & (times <= 1000.0)
        if np.any(mask_10_1000):
            times_10_1000 = times[mask_10_1000]
            sigmas_10_1000 = sigmas[mask_10_1000]
            
            print(f"Number of data points in interval: {len(times_10_1000)}")
            print(f"σ range: {sigmas_10_1000[0]:.2e} to {sigmas_10_1000[-1]:.2e} C/m²")
            
            if len(times_10_1000) > 1:
                avg_rate = (sigmas_10_1000[-1] - sigmas_10_1000[0]) / (times_10_1000[-1] - times_10_1000[0])
                print(f"Average net charging rate: {avg_rate:.2e} C/m²/s")
                
                # Check for anomalies
                diff_sigmas = np.diff(sigmas_10_1000)
                if np.any(diff_sigmas < 0):
                    print("Warning: Detected regions where σ decreases!")
        
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
                    final_J = compute_J(final_sigma)
                    
                    # Compute charging rate
                    L_final = compute_L(final_sigma)
                    U_final = compute_U(final_sigma)
                    charging_final = 2 * D * c * d_p * e / (L_final**2) * np.exp(-e * U_final / (k * T))
                    
                    print(f"Charging rate at equilibrium: {charging_final:.2e} C/m²/s")
                    print(f"Loss rate (J) at equilibrium: {final_J:.2e} C/m²/s")
                    print(f"Difference: {abs(charging_final - final_J):.2e} C/m²/s")
        
        # 5. Final state
        print("\n" + "="*60)
        print(f"Final state (t = {times[-1]:.2f} s):")
        print("="*60)
        
        final_sigma = sigmas[-1]
        dsigma_final = dsigmadt_vectorized(times[-1], [final_sigma])[0]
        
        # Compute final parameters
        L_final = compute_L(final_sigma)
        U_final = compute_U(final_sigma)
        J_final = compute_J(final_sigma)
        V_Δh_final = compute_V_y(Δh, final_sigma)
        V_ymin_final = compute_V_y(L_final, final_sigma)
        
        charging_final = 2 * D * c * d_p * e / (L_final**2) * np.exp(-e * U_final / (k * T))
        
        print(f"Surface charge density: σ = {final_sigma:.6e} C/m²")
        print(f"Net charging rate: dσ/dt = {dsigma_final:.6e} C/m²/s")
        print(f"Charging rate: {charging_final:.6e} C/m²/s")
        print(f"Loss rate (J): {J_final:.6e} C/m²/s")
        print(f"Equilibrium difference: {charging_final - J_final:.6e} C/m²/s")
        
        print(f"\nGeometric parameters:")
        print(f"  L(σ) = y_min = {L_final:.6e} m")
        print(f"  y_min/h = {L_final/h:.4f}")
        
        print(f"\nPotential parameters:")
        print(f"  U(σ) = V(Δh) - V(y_min) = {U_final:.6f} V")
        print(f"  V(Δh) = {V_Δh_final:.6f} V")
        print(f"  V(y_min) = {V_ymin_final:.6f} V")
        
        print(f"\nEnergy parameters:")
        print(f"  e·U/(kT) = {e*U_final/(k*T):.4f}")
        print(f"  exp(-e·U/(kT)) = {np.exp(-e*U_final/(k*T)):.2e}")
        
        print(f"\nDiscriminant check:")
        discriminant = h**2 - final_sigma * h / (np.pi * eps0 * E_ext)
        print(f"  h² - σh/(πε₀E_ext) = {discriminant:.6e}")
        print(f"  Condition: {discriminant > 0}")
        
        print("="*60)