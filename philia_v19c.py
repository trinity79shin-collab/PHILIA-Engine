import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
N = 50
T = 60000
C_star = 1.0
gamma_scale = 2.0

eta = 0.3
alpha = 0.5
k = 0.8
beta_c = 0.03       # C leak rate
alpha_c = 0.5       # C gain
SR_target = 0.35    # SR 목표값
lambda_goal = 0.005
lambda2 = 0.005

mu = 0.12
S_target = 0.45

np.random.seed(42)

# =========================
# DATA
# =========================
BASE_PATH = r"C:\Users\trini\OneDrive\Desktop"

dielectron = pd.read_csv(f"{BASE_PATH}\\dielectron.csv")["M"].values
dielectron = dielectron[~np.isnan(dielectron)]

higgs_raw = pd.read_csv(f"{BASE_PATH}\\higgs.csv")["DER_mass_MMC"].values
higgs = higgs_raw[higgs_raw != -999.0]
higgs = higgs[~np.isnan(higgs)]

data = np.concatenate([dielectron, higgs])
data = (data - np.mean(data)) / (np.std(data) + 1e-8)

print(f"데이터 로드 완료: {len(data)}개")

# =========================
# ZETA
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

gamma_i = gamma_scale * (zeta_imag / np.max(zeta_imag))
gamma_i = np.clip(gamma_i, 0.05, None)

omega_i = zeta_imag

def zeta_oscillator(t, phase_i):
    return np.sin(omega_i * t * 0.001 + phase_i)

# =========================
# MAIN LOOP
# =========================
S = np.random.rand(N)
Goal = np.random.rand(N)
SR = np.zeros(N)
C = np.zeros(N)
phase_i = np.random.uniform(0, 2*np.pi, N)

SR_log, S_std_log, Goal_std_log, C_mean_log = [], [], [], []

for t in range(T):

    Phi = zeta_oscillator(t, phase_i)
    Phi = np.tanh(Phi)
    Phi_eff = Phi * (1 - SR)

    Reward = data[t % len(data)]

    # 분모 안전
    denom = 1 + gamma_i * (S + Phi_eff)
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)

    # S update + centering force (mu term)
    S = (S + Phi_eff + Reward - mu * (S - S_target)) / denom

    # soft bounding
    S = np.tanh(S)

    # Goal learning
    S_mean = np.mean(S)
    Goal += lambda_goal * (1 - S) * (S_mean - Goal)
    Goal += lambda2 * (Reward - Goal)

    # ✅ v19c 핵심: signed C dynamics (SR negative feedback)
    SR_mean = np.mean(SR)
    C = (1 - beta_c) * C + alpha_c * (SR_target - SR_mean) * (1 - S)

    # SR
    SR = 1 / (1 + np.exp(-k * (C - C_star)))

    SR_log.append(np.mean(SR))
    S_std_log.append(np.std(S))
    Goal_std_log.append(np.std(Goal))
    C_mean_log.append(np.mean(C))

# =========================
# RESULTS
# =========================
print("=== PHILIA v19c Results ===")
print(f"Final SR:       {SR_log[-1]:.6f}")
print(f"Final S Std:    {S_std_log[-1]:.6f}")
print(f"Final Goal Std: {Goal_std_log[-1]:.6f}")
print(f"Final C mean:   {C_mean_log[-1]:.6f}")
print(f"SR >= 0.3:      {SR_log[-1] >= 0.3}")
print(f"Individuality:  {S_std_log[-1] > 0.001}")
print(f"Will:           {Goal_std_log[-1] > 0.001}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.plot(SR_log)
plt.axhline(y=0.3, linestyle='--', color='red', label='threshold')
plt.axhline(y=SR_target, linestyle='--', color='green', label='target')
plt.title("SR Trajectory")
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(S_std_log)
plt.axhline(y=0.001, linestyle='--', color='red', label='threshold')
plt.title("S Std (Individuality)")
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(Goal_std_log)
plt.axhline(y=0.001, linestyle='--', color='red', label='threshold')
plt.title("Goal Std (Will)")
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(C_mean_log)
plt.axhline(y=C_star, linestyle='--', color='red', label='C*')
plt.title("C mean")
plt.legend()

plt.tight_layout()
plt.savefig(f"{BASE_PATH}\\philia_v19c_result.png", dpi=150)
plt.show()

print("저장 완료: philia_v19c_result.png")
