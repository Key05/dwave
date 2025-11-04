import math
from collections import defaultdict

import numpy as np
from openjij import SQASampler

import csv
import datetime as dt

from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite

np.set_printoptions(threshold=np.inf, linewidth=180)

token = 'EN2g-3e1980b9a7f3674b5ab09f45cd809a11006fd00e'
endpoint = 'https://cloud.dwavesys.com/sapi/'


dw_sampler = DWaveSampler(solver='  ', token=token, endpoint=endpoint)
sampler = EmbeddingComposite(dw_sampler)


# 設定
a = 11
b = 7
v_num = 5

# ビット重み
w = np.array([2.0, 1.0, 0.5, 0.25, 0.125])

# 例: 対称 [-3.75, 3.875] を実現
x0 = -4
y0 = -4
beta = 2.0  # = (上限 - 下限) / sum(w)

# 罰則強度（補助変数拘束用）
lam = 100


# 変数のインデックス割当
sv_num = math.comb(v_num, 2)

# xビット: 0 .. v_num-1
# x補助 z_ij: v_num .. v_num+sv_num-1
# yビット: v_num+sv_num .. v_num+sv_num+v_num-1
# y補助 t_ij: その後ろ
def pair_index(i, j, n):
    #i<j のペアを 0..C(n,2)-1 に詰めるインデックス
    # 辞書順 i<j
    idx = 0
    for p in range(n):
        for q in range(p+1, n):
            if p == i and q == j:
                return idx
            idx += 1
    raise RuntimeError("pair_index error")

x_bits_idx = list(range(v_num))
x_aux_idx0 = v_num
y_bits_idx0 = v_num + sv_num
y_aux_idx0 = y_bits_idx0 + v_num

N = v_num + sv_num + v_num + sv_num

# QUBO 辞書構築のユーティリティ
def add(Q, i, j, val):
    if abs(val) < 1e-14:
        return
    if i > j:
        i, j = j, i
    Q[(i, j)] += float(val)

def add_lin(Q, i, val):
    add(Q, i, i, val)

def add_quad(Q, i, j, val):
    add(Q, i, j, val)

# ------------------------------
# 式の係数を準備
# X = x0 + beta * sum_i w_i b_i
# Y = y0 + beta * sum_i w_i c_i
# X^2 = x0^2 + 2 x0 beta * sum_i w_i b_i + beta^2 * ( sum_i w_i^2 b_i + 2 sum_{i<j} w_i w_j z_ij )
# 同様に Y^2
# A = X^2 + Y - a
# B = X + Y^2 - b
# A^2 + B^2 を QUBO に落とす（b_i, z_ij, c_i, t_ij の2次まで）
# ------------------------------
def build_qubo(v_num, w, x0, y0, beta, a, b, lam):
    Q = defaultdict(float)

    def z_index(i, j):
        return x_aux_idx0 + pair_index(i, j, v_num)

    def t_index(i, j):
        return y_aux_idx0 + pair_index(i, j, v_num)

    # --- 線形項 (X, Y), 二次化表現の各係数 ---
    # X^2 の成分を「定数 + b_i (lin) + z_ij (lin)」に分解
    X2_const = x0 * x0
    X2_lin_b = 2 * x0 * beta * w + (beta**2) * (w**2)
    # z_ij の係数は 2 * beta^2 * w_i w_j
    def X2_lin_z(i, j):
        return 2 * (beta**2) * w[i] * w[j]

    # Y^2 も同様
    Y2_const = y0 * y0
    Y2_lin_c = 2 * y0 * beta * w + (beta**2) * (w**2)
    def Y2_lin_t(i, j):
        return 2 * (beta**2) * w[i] * w[j]

    # Y = y0 + beta * sum w_j c_j
    # X = x0 + beta * sum w_i b_i

    # --- A = X^2 + Y - a ---
    A_const = X2_const + y0 - a
    A_lin_b = X2_lin_b.copy()                               # b_i
    A_lin_z = {}                                            # z_ij
    for i in range(v_num):
        for j in range(i+1, v_num):
            A_lin_z[(i, j)] = X2_lin_z(i, j)
    A_lin_c = beta * w                                      # c_j

    # --- B = X + Y^2 - b ---
    B_const = x0 + Y2_const - b
    B_lin_b = beta * w                                      # b_i
    B_lin_c = Y2_lin_c.copy()                               # c_j
    B_lin_t = {}
    for i in range(v_num):
        for j in range(i+1, v_num):
            B_lin_t[(i, j)] = Y2_lin_t(i, j)

    # ========== A^2 を展開 ==========
    # (c + sum α_i b_i + sum ζ_k z_k + sum γ_j c_j)^2
    # = c^2 + 2c(...) + Σ α_i^2 b_i + 2 Σ_{i<j} α_i α_j b_i b_j + Σ ζ_k^2 z_k + 2 Σ ζ_k ζ_l z_k z_l
    #   + 2 Σ α_i ζ_k b_i z_k + 2 Σ α_i γ_j b_i c_j + 2 Σ ζ_k γ_j z_k c_j + Σ γ_j^2 c_j + 2 Σ_{j<l} γ_j γ_l c_j c_l
    # b_i^2=b_i, z_k^2=z_k, c_j^2=c_j
    # ---- 定数は無視してOK（エネルギーの基準シフト） ----
    # 2 c (...)
    for i in range(v_num):
        add_lin(Q, x_bits_idx[i], 2 * A_const * A_lin_b[i])
    for (i, j), val in A_lin_z.items():
        add_lin(Q, z_index(i, j), 2 * A_const * val)
    for j in range(v_num):
        add_lin(Q, y_bits_idx0 + j, 2 * A_const * A_lin_c[j])

    # α_i^2 b_i
    for i in range(v_num):
        add_lin(Q, x_bits_idx[i], A_lin_b[i]**2)

    # 2 α_i α_j b_i b_j
    for i in range(v_num):
        for j in range(i+1, v_num):
            add_quad(Q, x_bits_idx[i], x_bits_idx[j], 2 * A_lin_b[i] * A_lin_b[j])

    # ζ_k^2 z_k
    for (i, j), val in A_lin_z.items():
        add_lin(Q, z_index(i, j), val**2)

    # 2 ζ_k ζ_l z_k z_l
    A_z_keys = list(A_lin_z.keys())
    for u in range(len(A_z_keys)):
        (i1, j1) = A_z_keys[u]
        for v in range(u+1, len(A_z_keys)):
            (i2, j2) = A_z_keys[v]
            add_quad(Q, z_index(i1, j1), z_index(i2, j2), 2 * A_lin_z[(i1, j1)] * A_lin_z[(i2, j2)])

    # 2 α_i ζ_k b_i z_k
    for i in range(v_num):
        for (p, q), val in A_lin_z.items():
            add_quad(Q, x_bits_idx[i], z_index(p, q), 2 * A_lin_b[i] * val)

    # 2 α_i γ_j b_i c_j
    for i in range(v_num):
        for j in range(v_num):
            add_quad(Q, x_bits_idx[i], y_bits_idx0 + j, 2 * A_lin_b[i] * A_lin_c[j])

    # 2 ζ_k γ_j z_k c_j
    for (p, q), val in A_lin_z.items():
        for j in range(v_num):
            add_quad(Q, z_index(p, q), y_bits_idx0 + j, 2 * val * A_lin_c[j])

    # γ_j^2 c_j
    for j in range(v_num):
        add_lin(Q, y_bits_idx0 + j, A_lin_c[j]**2)

    # 2 γ_j γ_l c_j c_l
    for j in range(v_num):
        for l in range(j+1, v_num):
            add_quad(Q, y_bits_idx0 + j, y_bits_idx0 + l, 2 * A_lin_c[j] * A_lin_c[l])

    # ========== B^2 を展開（同様） ==========
    # (c' + Σ β_i b_i + Σ δ_j c_j + Σ τ_k t_k)^2
    # 2 c'(...)
    for i in range(v_num):
        add_lin(Q, x_bits_idx[i], 2 * B_const * B_lin_b[i])
    for j in range(v_num):
        add_lin(Q, y_bits_idx0 + j, 2 * B_const * B_lin_c[j])
    for (i, j), val in B_lin_t.items():
        add_lin(Q, t_index(i, j), 2 * B_const * val)

    # β_i^2 b_i
    for i in range(v_num):
        add_lin(Q, x_bits_idx[i], B_lin_b[i]**2)
    # 2 β_i β_k b_i b_k
    for i in range(v_num):
        for k in range(i+1, v_num):
            add_quad(Q, x_bits_idx[i], x_bits_idx[k], 2 * B_lin_b[i] * B_lin_b[k])

    # δ_j^2 c_j
    for j in range(v_num):
        add_lin(Q, y_bits_idx0 + j, B_lin_c[j]**2)
    # 2 δ_j δ_l c_j c_l
    for j in range(v_num):
        for l in range(j+1, v_num):
            add_quad(Q, y_bits_idx0 + j, y_bits_idx0 + l, 2 * B_lin_c[j] * B_lin_c[l])

    # τ_k^2 t_k
    for (i, j), val in B_lin_t.items():
        add_lin(Q, t_index(i, j), val**2)
    # 2 τ_k τ_m t_k t_m
    B_t_keys = list(B_lin_t.keys())
    for u in range(len(B_t_keys)):
        (i1, j1) = B_t_keys[u]
        for v in range(u+1, len(B_t_keys)):
            (i2, j2) = B_t_keys[v]
            add_quad(Q, t_index(i1, j1), t_index(i2, j2), 2 * B_lin_t[(i1, j1)] * B_lin_t[(i2, j2)])

    # 2 β_i δ_j b_i c_j
    for i in range(v_num):
        for j in range(v_num):
            add_quad(Q, x_bits_idx[i], y_bits_idx0 + j, 2 * B_lin_b[i] * B_lin_c[j])

    # 2 β_i τ_k b_i t_k
    for i in range(v_num):
        for (p, q), val in B_lin_t.items():
            add_quad(Q, x_bits_idx[i], t_index(p, q), 2 * B_lin_b[i] * val)

    # 2 δ_j τ_k c_j t_k
    for j in range(v_num):
        for (p, q), val in B_lin_t.items():
            add_quad(Q, y_bits_idx0 + j, t_index(p, q), 2 * B_lin_c[j] * val)

    # ========== 補助変数の拘束ペナルティ ==========
    #  λ*(b_i b_j - 2 b_i z_ij - 2 b_j z_ij + 3 z_ij)  を全ペアに付与
    for i in range(v_num):
        for j in range(i+1, v_num):
            bij = (x_bits_idx[i], x_bits_idx[j])
            zij = z_index(i, j)
            add_quad(Q, *bij, lam)             # + λ b_i b_j
            add_quad(Q, x_bits_idx[i], zij, -2*lam)  # -2λ b_i z
            add_quad(Q, x_bits_idx[j], zij, -2*lam)  # -2λ b_j z
            add_lin(Q, zij, 3*lam)             # +3λ z

    for i in range(v_num):
        for j in range(i+1, v_num):
            cij = (y_bits_idx0 + i, y_bits_idx0 + j)
            tij = t_index(i, j)
            add_quad(Q, *cij, lam)
            add_quad(Q, y_bits_idx0 + i, tij, -2*lam)
            add_quad(Q, y_bits_idx0 + j, tij, -2*lam)
            add_lin(Q, tij, 3*lam)

    return Q, N

# ------------------------------
# 実行
# ------------------------------
Q, N = build_qubo(v_num, w, x0, y0, beta, a, b, lam)

sampleset = sampler.sample_qubo(Q,num_reads=1000)
print(sampleset.record)
