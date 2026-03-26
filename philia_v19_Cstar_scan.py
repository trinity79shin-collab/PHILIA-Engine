import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
N = 50
T = 60000
gamma_scale = 2.0

eta = 0.3
alpha = 0.5
k = 0.8
beta = 0.15
lambda_goal = 0.005
lambda2 = 0.005

C_star_list = [1.0, 2.0, 3.0, 4.0, 6.0]

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

# =========================
# OSCILLATOR
# =========================
def zeta_oscillator(t, phase_i):
    return np.sin(omega_i * t * 0.001 + phase_i)

# =========================
# RUN FUNCTION
# =========================
def run_simulation(C_star):

    # 독립 초기화
    S = np.random.rand(N)
    Goal = np.random.rand(N)
    SR = np.zeros(N)
    C = np.zeros(N)
    phase_i = np.random.uniform(0, 2*np.pi, N)

    SR_log, S_std_log, Goal_std_log = [], [], []

    for t in range(T):

        Phi = zeta_oscillator(t, phase_i)
        Phi = np.tanh(Phi)
        Phi_eff = Phi * (1 - SR)

        Reward = data[t % len(data)]

        denom = 1 + gamma_i * (S + Phi_eff)
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)

        S = (S + Phi_eff + Reward) / denom
        S = np.clip(S, 0.0, 1.0)

        S_mean = np.mean(S)
        Goal += lambda_goal * (1 - S) * (S_mean - Goal)
        Goal += lambda2 * (Reward - Goal)

        # v18 C 유지
        C += alpha * S * (1 - S)

        SR = 1 / (1 + np.exp(-k * (C - C_star)))

        SR_log.append(np.mean(SR))
        S_std_log.append(np.std(S))
        Goal_std_log.append(np.std(Goal))

    return SR_log, S_std_log, Goal_std_log

# =========================
# RUN ALL
# =========================
results = {}

for C_star in C_star_list:
    print(f"Running C* = {C_star} ...")
    SR_log, S_std_log, Goal_std_log = run_simulation(C_star)

    results[C_star] = {
        "SR": SR_log,
        "S_std": S_std_log,
        "Goal_std": Goal_std_log
    }

    print(f"  Final SR: {SR_log[-1]:.6f} | S Std: {S_std_log[-1]:.6f} | Goal Std: {Goal_std_log[-1]:.6f}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
for C_star in C_star_list:
    plt.plot(results[C_star]["SR"], label=f"C*={C_star}")
plt.axhline(y=0.3, linestyle='--', color='red', label='threshold')
plt.title("SR Trajectory")
plt.legend()

plt.subplot(1, 3, 2)
for C_star in C_star_list:
    plt.plot(results[C_star]["S_std"], label=f"C*={C_star}")
plt.axhline(y=0.001, linestyle='--', color='red', label='threshold')
plt.title("S Std (Individuality)")
plt.legend()

plt.subplot(1, 3, 3)
for C_star in C_star_list:
    plt.plot(results[C_star]["Goal_std"], label=f"C*={C_star}")
plt.axhline(y=0.001, linestyle='--', color='red', label='threshold')
plt.title("Goal Std (Will)")
plt.legend()

plt.tight_layout()
plt.savefig(f"{BASE_PATH}\\philia_v19_Cstar_scan.png", dpi=150)
plt.show()

print("C* scan 완료")
