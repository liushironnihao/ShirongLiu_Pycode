import numpy as np
import matplotlib.pyplot as plt


# Physical parameters
sigma0= 3.419e-5  # C/m^2, surface charge density taken from potential well calculation
s= 3.5            # Dielectric constant, consistent with value in potential well calculation
sigma = sigma0/s  # C/m^2  dielectric correction
eps0 = 8.854e-12  # F/m
r0 = 4e-6  # m, maximum influence radius, Rm taken from potential well calculation
H = 1.45e-6  # m dielectric thickness, consistent with value in potential well calculation
E_ext = -2e5  # V/m, external field strength, negative sign indicates downward direction, consistent with value in potential well calculation

# Add coincidence threshold definition
coincidence_threshold = 0.01e-6  # Coincidence threshold, unit: m


coeff = sigma / (2 * eps0)  # Electric field coefficient V/m

# Take points upward from the center of dielectric plane
z_start = -H
z_end = 5e-6  
z_m = np.linspace(z_start, z_end, 1000)   # z coordinate (m)

# Define electric field calculation function
def calculate_electric_field(z, r_val):
    """Calculate electric field and potential at position z for given r value"""
    R_dynamic = r0 + r_val
    
    def V_annulus(z_point, z0, r_inner=r_val, R_outer=R_dynamic):
        d = z_point - z0
        term = np.sqrt(d**2 + R_outer**2) - np.sqrt(d**2 + r_inner**2)
        return coeff * term
    
    # Total potential
    V_total = (V_annulus(z, 0) - V_annulus(z, -2*H) + -E_ext * z)
    
    # Calculate total electric field using numerical differentiation
    dz = z[1] - z[0]
    E_total_numerical = -np.gradient(V_total, dz)
    
    return E_total_numerical, V_total, R_dynamic

# Define function to find distance between two zeros
def find_zero_separation(r_val):
    """Calculate minimum distance between two zeros and zero positions for given r value"""
    E_field, _, _ = calculate_electric_field(z_m, r_val)
    
    # Find all zero crossings
    zero_crossings = np.where(np.diff(np.sign(E_field)))[0]
    
    if len(zero_crossings) >= 2:
        # Calculate all zero positions
        zero_positions = []
        for crossing in zero_crossings:
            idx1, idx2 = crossing, crossing + 1
            z1, z2 = z_m[idx1], z_m[idx2]
            E1, E2 = E_field[idx1], E_field[idx2]
            zero_exact = z1 - E1 * (z2 - z1) / (E2 - E1)
            zero_positions.append(zero_exact)
        
        # Only consider zeros with z >= -2H
        valid_zeros = [z for z in zero_positions if z >= -2*H]
        
        if len(valid_zeros) >= 2:
            # Return minimum distance and zero positions
            sorted_zeros = sorted(valid_zeros)
            min_sep = np.min(np.diff(sorted_zeros))
            return min_sep, sorted_zeros
        else:
            return np.inf, []  # Not enough valid zeros
    else:
        return np.inf, []  # No two zeros

# Find r value where two zeros coincide
print(f"Searching for the r value where two zeros coincide (threshold: {coincidence_threshold*1e6:.1f} μm)...")

# Search within reasonable range (0.1μm to 2.5μm)
r_search = np.linspace(0.005e-6, 5e-6, 10000)
separations = []
zero_positions_list = []

# Check candidate r values during search
candidate_r_values = []
for r_val in r_search:
    sep, zeros = find_zero_separation(r_val)
    separations.append(sep)
    zero_positions_list.append(zeros)
    
    # Check if coincidence condition is satisfied
    if sep < coincidence_threshold:
        candidate_r_values.append((r_val, sep))

# Find r value corresponding to minimum separation
min_sep_idx = np.argmin(separations)
optimal_r = r_search[min_sep_idx]
min_separation = separations[min_sep_idx]
optimal_zeros = zero_positions_list[min_sep_idx]

# Add coincidence judgment
if min_separation < coincidence_threshold:
    print(f"\n*** Found qualifying r value: {optimal_r*1e6:.3f} μm ***")
    print(f"*** Minimum zero distance: {min_separation*1e6:.3f} μm < threshold {coincidence_threshold*1e6:.1f} μm ***")
    print(f"*** Zero positions: {[z*1e6 for z in optimal_zeros]} μm ***")
else:
    print(f"\n*** No r value satisfying coincidence condition found ***")
    print(f"*** Minimum zero distance: {min_separation*1e6:.3f} μm > threshold {coincidence_threshold*1e6:.1f} μm ***")
    print(f"*** Closest r value: {optimal_r*1e6:.3f} μm ***")

# Print candidate r values
if candidate_r_values:
    print(f"\nCandidate r values (distance<{coincidence_threshold*1e6:.1f}μm):")
    for r_val, sep in candidate_r_values[:5]:  # Only show first 5
        print(f"  r = {r_val*1e6:.3f} μm, distance = {sep*1e6:.3f} μm")
else:
    print(f"\nNo candidate r values (zero distance for all r values > {coincidence_threshold*1e6:.1f} μm)")

# Calculate detailed electric field and potential distribution for optimal r value
z_detailed = np.linspace(-3e-6, 5e-6, 2000)  # More dense sampling
E_optimal, V_optimal, R_optimal = calculate_electric_field(z_detailed, optimal_r)

# Create main figure - only show distribution for optimal r value
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot electric field distribution
ax1.plot(z_detailed*1e6, E_optimal/1e3, 'b-', linewidth=2, label='Electric Field')
ax1.set_xlabel('z (μm)')
ax1.set_ylabel('E (kV/m)')
ax1.set_title(f'Electric Field Distribution for Optimal r = {optimal_r*1e6:.3f} μm')
ax1.grid(True, alpha=0.3)

# Plot potential distribution
ax2.plot(z_detailed*1e6, V_optimal, 'r-', linewidth=2, label='Electric Potential')
ax2.set_xlabel('z (μm)')
ax2.set_ylabel('V (V)')
ax2.set_title(f'Electric Potential Distribution for r = {optimal_r*1e6:.3f} μm')
ax2.grid(True, alpha=0.3)

# Mark zeros and extreme points
if len(optimal_zeros) >= 2:
    for i, zero_pos in enumerate(optimal_zeros):
        # Mark zero points on electric field plot
        ax1.axvline(x=zero_pos*1e6, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Zero {i+1}: {zero_pos*1e6:.3f} μm' if i == 0 else "")
        ax1.plot(zero_pos*1e6, 0, 'ro', markersize=6)
        
        # Mark corresponding points on potential plot
        idx_zero = np.argmin(np.abs(z_detailed - zero_pos))
        V_at_zero = V_optimal[idx_zero]
        ax2.plot(zero_pos*1e6, V_at_zero, 'ro', markersize=6, 
                label=f'Zero {i+1}' if i == 0 else "")
        
        # Add zero annotation
        ax1.annotate(f'Zero {i+1}\n{zero_pos*1e6:.3f} μm', 
                    xy=(zero_pos*1e6, 0), 
                    xytext=(zero_pos*1e6 + 0.5*i, -300 + i*150), 
                    arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax2.annotate(f'Zero {i+1}', 
                    xy=(zero_pos*1e6, V_at_zero), 
                    xytext=(zero_pos*1e6 + 0.5*i, V_at_zero -0.5+ i*0.2), 
                    arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    fontsize=8)

# Add reference lines
for ax in [ax1, ax2]:
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='z = 0 (Top plate)')
    ax.axvline(x=-2*H*1e6, color='orange', linestyle='--', alpha=0.7, label=f'z = {-2*H*1e6:.1f} μm (Bottom plate)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()

# Add parameter text box
status_color = 'lightgreen' if min_separation < coincidence_threshold else 'lightyellow'
param_text = f'Parameters:\n• r = {optimal_r*1e6:.3f} μm\n• R = {R_optimal*1e6:.3f} μm\n• Zero sep = {min_separation*1e6:.3f} μm\n• Threshold = {coincidence_threshold*1e6:.1f} μm\n• Status: {"PASS" if min_separation < coincidence_threshold else "FAIL"}'
props = dict(boxstyle='round', facecolor=status_color, alpha=0.8)
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()

# Plot separation versus r
fig2, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(r_search*1e6, np.array(separations)*1e6, 'b-', linewidth=2)
ax3.set_xlabel('Hole Radius r (μm)')
ax3.set_ylabel('Minimum Distance Between Two Zeros (μm)')
ax3.set_title(f'Zero Separation vs. Hole Radius')
ax3.grid(True)
ax3.axhline(y=coincidence_threshold*1e6, color='red', linestyle='--', alpha=0.7, 
           label=f'Coincidence threshold ({coincidence_threshold*1e6:.3f} μm)')
ax3.axvline(x=optimal_r*1e6, color='purple', linestyle='--', alpha=0.7, 
           label=f'Optimal r = {optimal_r*1e6:.2f} μm')
ax3.legend()
plt.tight_layout()

plt.show()

# Print detailed results
print("="*50)
print("Optimal r Value Analysis Results")
print("="*50)
print(f"Coincidence threshold: {coincidence_threshold*1e6:.1f} μm")
print(f"Optimal r = {optimal_r*1e6:.3f} μm")
print(f"Corresponding R = {R_optimal*1e6:.3f} μm")
print(f"Minimum zero separation = {min_separation*1e6:.3f} μm")
print(f"Number of zeros found = {len(optimal_zeros)}")
print(f"Zero positions = {[z*1e6 for z in optimal_zeros]} μm")



# Print electric field and potential values at key points
print("\nKey Point Values:")
print("z(μm)\tE(kV/m)\tV(V)")
for z in [-2, -1, 0, 1, 2, 5]:
    idx = np.argmin(np.abs(z_detailed - z*1e-6))
    E_val = E_optimal[idx]/1e3
    V_val = V_optimal[idx]
    print(f"{z}\t{E_val:.3f}\t{V_val:.3f}")