import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (v18 유지)
# =========================
N = 50
T = 60000
C_star = 1.0
gamma_scale = 2.0

eta = 0.3
alpha = 0.5
k = 0.8
beta = 0.15
lambda_goal = 0.005
lambda2 = 0.005

np.random.seed(42)

# =========================
# DATA (보조 입력 유지)
# =========================
BASE_PATH = r"C:\Users\trini\OneDrive\Desktop"

dielectron = pd.read_csv(f"{BASE_PATH}\\dielectron.csv")["M"].values
dielectron = dielectron[~np.isnan(dielectron)]  # NaN 제거

higgs_raw = pd.read_csv(f"{BASE_PATH}\\higgs.csv")["DER_mass_MMC"].values
higgs = higgs_raw[higgs_raw != -999.0]          # sentinel 제거
higgs = higgs[~np.isnan(higgs)]                  # NaN 제거

data = np.concatenate([dielectron, higgs])
data = (data - np.mean(data)) / (np.std(data) + 1e-8)

print(f"데이터 로드 완료: {len(data)}개, NaN 없음: {not np.any(np.isnan(data))}")

# =========================
# ZETA ZERO (first 50 imag parts)
# =========================
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

# =========================
# INITIALIZATION
# =========================
S = np.random.rand(N)
Goal = np.random.rand(N)
SR = np.zeros(N)
C = np.zeros(N)

gamma_i = gamma_scale * (zeta_imag / np.max(zeta_imag))
gamma_i = np.clip(gamma_i, 0.05, None)

omega_i = zeta_imag
phase_i = np.random.uniform(0, 2*np.pi, N)

def zeta_oscillator(t, i):
    return np.sin(omega_i[i] * t * 0.001 + phase_i[i])

# =========================
# LOGGING
# =========================
SR_log, S_std_log, Goal_std_log = [], [], []

# =========================
# MAIN LOOP
# =========================
for t in range(T):

    # --- Zeta Oscillator ---
    Phi = np.array([zeta_oscillator(t, i) for i in range(N)])

    # --- Phi normalization (tanh) ---
    Phi = np.tanh(Phi)

    # --- SR decoupling ---
    Phi_eff = Phi * (1 - SR)

    # --- External reward (보조 입력) ---
    Reward = data[t % len(data)]

    # --- S update (분모 절댓값 클램핑) ---
    denom = 1 + gamma_i * (S + Phi_eff)
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)  # 0 근처 모두 처리
    S = (S + Phi_eff + Reward) / denom

    # --- S 클램핑 ---
    S = np.clip(S, 0.0, 1.0)

    # --- Goal learning (v10/v18 유지) ---
    S_mean = np.mean(S)
    Goal += lambda_goal * (1 - S) * (S_mean - Goal)

    # --- Local reward term ---
    Goal += lambda2 * (Reward - Goal)

    # --- C update (v18 원복) ---
    C += alpha * S * (1 - S)

    # --- SR update ---
    SR = 1 / (1 + np.exp(-k * (C - C_star)))

    # --- Logging ---
    SR_log.append(np.mean(SR))
    S_std_log.append(np.std(S))
    Goal_std_log.append(np.std(Goal))

# =========================
# RESULTS
# =========================
print("=== PHILIA v19 Results ===")
print(f"Final SR:       {SR_log[-1]:.6f}")
print(f"Final S Std:    {S_std_log[-1]:.6f}")
print(f"Final Goal Std: {Goal_std_log[-1]:.6f}")
print(f"SR >= 0.3:      {SR_log[-1] >= 0.3}")
print(f"Individuality:  {S_std_log[-1] > 0.001}")
print(f"Will:           {Goal_std_log[-1] > 0.001}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(SR_log)
plt.axhline(y=0.3, color='r', linestyle='--', label='threshold')
plt.title("SR Trajectory")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(S_std_log)
plt.axhline(y=0.001, color='r', linestyle='--', label='threshold')
plt.title("S Std (Individuality)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(Goal_std_log)
plt.axhline(y=0.001, color='r', linestyle='--', label='threshold')
plt.title("Goal Std (Will)")
plt.legend()

plt.tight_layout()
plt.savefig(f"{BASE_PATH}\\philia_v19_result.png", dpi=150)
plt.show()
print("그래프 저장 완료: philia_v19_result.png")
