import math
import pandas as pd

# ============================================================
# 全局工程假设（你可以后面系统性调这些）
# ============================================================

PHASES = 3

# SVPWM 下，相电压 RMS ≈ 0.58~0.61 Vdc
V_UTIL = 0.60

# -------- 磁链经验模型（关键修正点） --------
# 在“参考极数”下的结构磁链（经验量级）
PSI_REF = 0.015      # Wb，典型中等尺寸 PMSM
P_REF   = 10         # 参考极数（不是极对数）

# -------- 损耗经验模型（概念设计级） --------
COPPER_LOSS_COEFF = 0.015   # ∝ I^2
IRON_LOSS_COEFF   = 0.0008  # ∝ f_e

# -------- 可枚举的常见槽/极组合 --------
SLOT_CANDIDATES = [12, 18, 24, 27, 36, 48]
POLE_CANDIDATES = [8, 10, 14, 16, 20, 22, 24, 28]


# ============================================================
# 主扫描函数
# ============================================================

def motor_feasibility_scan(
    inner_rotor: bool,
    rotor_outer_diameter_mm: float,
    rated_torque_nm: float,
    rated_speed_rpm: float,
    dc_voltage_v: float
):
    results = []

    # -------- 基本量 --------
    omega_m = 2 * math.pi * rated_speed_rpm / 60.0
    v_phase_max = dc_voltage_v * V_UTIL

    for Z in SLOT_CANDIDATES:
        for P in POLE_CANDIDATES:
            if P % 2 != 0:
                continue

            p = P // 2

            # 每极每相槽数
            q = Z / (P * PHASES)

            # 工程经验：过滤明显不可用拓扑
            if q < 0.25 or q > 2.5:
                continue

            # -------- 电频率 --------
            f_e = p * rated_speed_rpm / 60.0
            omega_e = 2 * math.pi * f_e

            # ====================================================
            # 关键修正：磁链不是直接等于电压极限
            # ====================================================

            # ① 结构可实现磁链（极数越多，单极磁通越小）
            psi_struct = PSI_REF * (P_REF / P)

            # ② 电压限制磁链（反电动势上限）
            psi_voltage_limit = v_phase_max / omega_e

            # ③ 实际可用磁链
            psi = min(psi_struct, psi_voltage_limit)

            # -------- 转矩 → 电流 --------
            iq = rated_torque_nm / (1.5 * p * psi)

            # -------- 简化损耗模型 --------
            copper_loss = COPPER_LOSS_COEFF * iq**2
            iron_loss   = IRON_LOSS_COEFF * f_e
            total_loss  = copper_loss + iron_loss

            mech_power = rated_torque_nm * omega_m
            efficiency = mech_power / (mech_power + total_loss)

            results.append({
                "Slots": Z,
                "Poles": P,
                "q": round(q, 3),
                "ElecFreq_Hz": round(f_e, 1),
                "Psi_Wb": round(psi, 5),
                "Iq_est_A": round(iq, 1),
                "Eff_est": round(efficiency, 3)
            })

    df = pd.DataFrame(results)

    # 更符合工程直觉的排序方式
    return df.sort_values(
        by=["Eff_est", "Iq_est_A", "ElecFreq_Hz"],
        ascending=[False, True, True]
    )


# ============================================================
# 测试输入（你的案例）
# ============================================================

df = motor_feasibility_scan(
    inner_rotor=True,
    rotor_outer_diameter_mm=114,
    rated_torque_nm=5.23,
    rated_speed_rpm=2400,
    dc_voltage_v=48
)

print(df.head(20))