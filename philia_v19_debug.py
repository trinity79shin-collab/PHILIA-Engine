import numpy as np
import pandas as pd

BASE_PATH = r"C:\Users\trini\OneDrive\Desktop"

# 데이터 로드
dielectron = pd.read_csv(f"{BASE_PATH}\\dielectron.csv")["M"].values
higgs_raw = pd.read_csv(f"{BASE_PATH}\\higgs.csv")["DER_mass_MMC"].values
higgs = higgs_raw[higgs_raw != -999.0]

data = np.concatenate([dielectron, higgs])
data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)

print("=== 데이터 진단 ===")
print(f"dielectron shape: {dielectron.shape}")
print(f"higgs shape (필터 후): {higgs.shape}")
print(f"data_norm min: {data_norm.min():.4f}")
print(f"data_norm max: {data_norm.max():.4f}")
print(f"data_norm NaN 있음: {np.any(np.isnan(data_norm))}")

# 초기값 진단
N = 50
np.random.seed(42)
S = np.random.rand(N)
zeta_imag = np.array([
    14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
    37.5862, 40.9187, 43.3271, 48.0052, 49.7738,
    52.9703, 56.4462, 59.3470, 60.8318, 65.1125,
    67.0798, 69.5464, 72.0672, 75.7047, 77.1448,
    79.3374, 82.9104, 84.7355, 87.4253, 88.8091,
    92.4919, 94.6513, 95.8706, 98.8312, 101.3178,
    103.7255, 105.4466, 107.1686, 111.0295, 111.8747,
    114.3202, 116.2267, 118.7908, 121.3701, 122.9468,
    124.2568, 127.5167, 129.5787, 131.0877, 133.4977,
    134.7565, 138.1160, 139.7362, 141.1237, 143.1118
])
gamma_scale = 2.0
gamma_i = gamma_scale * (zeta_imag / np.max(zeta_imag))
gamma_i = np.clip(gamma_i, 0.05, None)

print(f"\n=== gamma_i 진단 ===")
print(f"gamma_i min: {gamma_i.min():.4f}")
print(f"gamma_i max: {gamma_i.max():.4f}")

# t=0 첫 스텝 진단
SR = np.zeros(N)
phase_i = np.random.uniform(0, 2*np.pi, N)
omega_i = zeta_imag

Phi = np.tanh(np.sin(omega_i * 0 * 0.001 + phase_i))
Phi_eff = Phi * (1 - SR)
Reward = data_norm[0]

print(f"\n=== t=0 첫 스텝 진단 ===")
print(f"Phi_eff min: {Phi_eff.min():.4f}, max: {Phi_eff.max():.4f}")
print(f"Reward: {Reward:.4f}")
denom = 1 + gamma_i * (S + Phi_eff)
print(f"denom min: {denom.min():.4f}, max: {denom.max():.4f}")
print(f"denom 음수 있음: {np.any(denom <= 0)}")

S_new = (S + Phi_eff + Reward) / np.maximum(denom, 1e-8)
print(f"S_new NaN 있음: {np.any(np.isnan(S_new))}")
print(f"S_new min: {np.nanmin(S_new):.4f}, max: {np.nanmax(S_new):.4f}")
