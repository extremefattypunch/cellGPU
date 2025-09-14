import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

total_tsteps=131072
total_sims=22*32
def find_steady_state(times, series, window_size=total_tsteps, tol=1e-5):
    if len(series) < window_size:
        return len(series) // 2
    smoothed = savgol_filter(series, min(total_tsteps, len(series) - (len(series) % 2 + 1)), 2)
    last_trans = 0
    for i in range(len(smoothed) - window_size):
        win_t = times[i:i+window_size]
        win_s = smoothed[i:i+window_size]
        slope = np.polyfit(win_t, win_s, 1)[0]
        if abs(slope) > tol:
            last_trans = i
    ss_start = last_trans + window_size // 2
    if len(series) - ss_start < total_tsteps:
        return len(series) // 2
    return ss_start

cache_file = 'phase_post_process.parquet'
data_dir = '../raw_outputs/analysis/'

if not os.path.exists(cache_file):
    files = [f for f in os.listdir(data_dir) if f.startswith('aligningVoronoi_modified_') and f.endswith('.h5')]

    J_list = []
    p0_list = []
    phi_list = []
    chi_list = []
    q_list = []
    count=0
    t0_phi=0
    t0_q=0
    for filename in files:
        count+=1
        parts = filename.split('_')
        J = float(parts[2])
        p0_str = parts[3].split('.h5')[0]
        p0 = float(p0_str)
        
        full_path = os.path.join(data_dir, filename)
        with h5py.File(full_path, 'r') as f:
            times = f['time'][:].flatten()
            vicsek = f['vicsekOrderParam'][:].flatten()
            shape_index = f['meanShapeIndex'][:].flatten()
        
        if len(times) == 0:
            continue
        
        if count==1:
            t0_phi = find_steady_state(times, vicsek)
            t0_q = find_steady_state(times, shape_index)
        print(f"progress:{count}/{total_sims},found t0_phi:{t0_phi},t0_q:{t0_q} for J:{J},p0:{p0_str}")
        t0 = max(t0_phi, t0_q)
        
        phi_avg = np.mean(vicsek[t0:])
        chi_phi = np.var(vicsek[t0:])
        q_avg = np.mean(shape_index[t0:])
        
        J_list.append(J)
        p0_list.append(p0)
        phi_list.append(phi_avg)
        chi_list.append(chi_phi)
        q_list.append(q_avg)

    df = pd.DataFrame({
        'J': J_list,
        'p0': p0_list,
        'phi': phi_list,
        'chi': chi_list,
        'q': q_list
    })

    df.to_parquet(cache_file)
else:
    df = pd.read_parquet(cache_file)

# Prepare data for plotting
unique_p0 = np.sort(df['p0'].unique())
unique_J = np.sort(df['J'].unique())

P0, J_grid = np.meshgrid(unique_p0, unique_J)

phi_2d = np.full((len(unique_J), len(unique_p0)), np.nan)
q_2d = np.full((len(unique_J), len(unique_p0)), np.nan)
chi_2d = np.full((len(unique_J), len(unique_p0)), np.nan)

for i, j_val in enumerate(unique_J):
    for k, p0_val in enumerate(unique_p0):
        row = df[(df['J'] == j_val) & (df['p0'] == p0_val)]
        if not row.empty:
            phi_2d[i, k] = row['phi'].values[0]
            q_2d[i, k] = row['q'].values[0]
            chi_2d[i, k] = row['chi'].values[0]

# Plot the phase diagram
fig, ax = plt.subplots(figsize=(8, 6))

# Color map for <phi>
pcm = ax.pcolormesh(P0, J_grid, phi_2d, cmap='RdYlBu_r', shading='auto')
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label(r'$\langle \varphi \rangle$')

# Red line: solid-liquid transition where <q> ≈ 3.813
trans_p0_red = []
for j_val in unique_J:
    sub = df[df['J'] == j_val]
    if not sub.empty:
        idx = np.argmin(np.abs(sub['q'] - 3.813))
        trans_p0_red.append(sub['p0'].iloc[idx])
ax.plot(trans_p0_red, unique_J, 'ro-', label='Solid/Liquid transition')

# Green line: flocking transition, peak in chi_phi
trans_J_green = []
for p0_val in unique_p0:
    sub = df[df['p0'] == p0_val]
    if not sub.empty:
        idx = np.argmax(sub['chi'])
        trans_J_green.append(sub['J'].iloc[idx])
ax.plot(unique_p0, trans_J_green, 'go-', label='Flocking transition')

# Black dashed: gas transition
# ax.axvline(4.05, ls='-',linewidth=1, color='k', label='Transition to gas-like state')

# Blue dashed: theoretical for J=0
# ax.axvline(3.813, ls='--', color='b', label=r'Theoretical $J=0$')

# Labels for regions
ax.text(3.2, 4, 'Solid flock', fontsize=12, ha='center')
ax.text(3.7, 4, 'Liquid flock', fontsize=12, ha='center')
ax.text(4.05, 4, 'Gas', fontsize=12, ha='center')

# Axes labels
ax.set_xlabel(r'Shape index $p_0$')
ax.set_ylabel(r'Alignment interaction $J$')
ax.set_xlim(2.9, 4.1)
ax.set_ylim(0, 8)

# Legend
ax.legend(loc='upper right')

plt.title('Phase diagram')
plt.savefig('phase_diagram.png')
plt.show()
