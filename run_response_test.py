"""
run_response_test.py  (방식 B: STDP 모델 유지 + D=0 + reward=0 으로 학습 억제)

저장된 E->E (tril-recover), E->I, I->I, I->E 를 모두 수동 로드해 네트워크를
구성하고, dopamine D 를 0 으로 초기화 + reward 를 전혀 주지 않아 STDP 에 의한
weight 변화를 억제한 상태에서, stimuli 만 주고 spike 반응을 기록한다.

  - E->E : izhikevich_stdp_model 로 로드 (원본과 동일), g=저장값, c=0
  - E->I : izhikevich_stdp_model 로 로드, g=저장값, c=0
  - I->I : 저장된 구조 로드, g=inh_weight 상수, StaticPulse (GLOBALG)
  - I->E : 저장된 구조 로드, g=inh_weight 상수, StaticPulse (GLOBALG)
  - 뉴런 E : izhikevich_dopamine_model (원본과 동일), D=0 초기화
  - reward bitmask : 전부 0  -> injectDopamine 항상 false -> D 계속 0
                     -> STDP 의 g 변화항 (c * D_pre * scale) 이 항상 0 -> g 고정

네 시냅스 그룹을 전부 set_sparse_connections 로 지정 -> RNG 불사용,
원본 run 의 연결 구조를 그대로 재현.

[검증] 시뮬레이션 후 EE/EI 의 g 가 로드값과 동일한지 확인한다 (학습 억제 확인).
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from common_w_delay_long import (
    izhikevich_dopamine_model,
    izhikevich_stdp_model,
    get_params,
    convert_spikes,
)

# ============================================================================
# 사용자가 조정할 부분
# ============================================================================
EE_RECORD_DIR = "/media/inhoi/12tbstorage/eprop_triplet_sdf_classifier/data/Triplet_Raster/260806_PRE_analysis/260806_data_5_5/data_5_5(4)/record"
EI_RECORD_DIR = EE_RECORD_DIR
INH_RECORD_DIR = EE_RECORD_DIR   # II / IE 구조 파일 위치

# 파일명 규칙:
#   EE: "{DT1},{DT2}ms interval, weight in {TIME}s_(data|row|col).npy"
#   EI: "{DT1},{DT2}ms interval, EI weight in {TIME}s_(data|row|col).npy"
#   II: "{DT1},{DT2}ms interval, II_(row|col).npy"
#   IE: "{DT1},{DT2}ms interval, IE_(row|col).npy"
DT1 = 5
DT2 = 5
WEIGHT_TIME_S = 1200

# II / IE 를 GLOBALG 로 로드 (안 되면 True 로 바꿔 INDIVIDUALG 사용)
USE_INDIVIDUALG_FOR_INH = False

# 반응성 측정용 실험 파라미터
TEST_DURATION_MS = 1.5*6000.0 * 1000.0
RECORD_CHUNK_MS = 10.0 * 1000.0
INTER_STIMULI_INTERVAL_MS = 100*1000.0
LEARNING_TIME_MS = 0.0                # 0 이면 처음부터 probe 패턴을 준다
OUT_DIR = "/media/inhoi/12tbstorage/eprop_triplet_sdf_classifier/data/Triplet_Raster/260806_PRE_analysis/response_test_5_5_time1200(4)_original1200"
# ============================================================================


def load_triplet(record_dir, base, with_g=True):
    row = np.load(os.path.join(record_dir, base + "_row.npy")).astype(np.uint32)
    col = np.load(os.path.join(record_dir, base + "_col.npy")).astype(np.uint32)
    assert len(row) == len(col), f"{base}: row/col 길이 불일치"
    if with_g:
        g = np.load(os.path.join(record_dir, base + "_data.npy")).astype(np.float64)
        assert len(g) == len(row), f"{base}: data/row/col 길이 불일치"
        return g, row, col
    return row, col


# ----------------------------------------------------------------------------
# Current source (원래 메인 스크립트와 동일)
# ----------------------------------------------------------------------------
stim_noise_model = genn_model.create_custom_current_source_class(
    "stim_noise",
    param_names=["n", "stimMagnitude"],
    var_name_types=[
        ("startStim", "unsigned int"),
        ("endStim", "unsigned int", VarAccess_READ_ONLY),
    ],
    extra_global_params=[("stimTimes", "scalar*")],
    injection_code="""
        scalar current = ($(gennrand_uniform) * $(n) * 2.0) - $(n);
        if($(startStim) != $(endStim) && $(t) >= $(stimTimes)[$(startStim)]) {
           current += $(stimMagnitude);
           $(startStim)++;
        }
        $(injectCurrent, current);
        """,
)


def get_start_end_stim(stim_counts):
    end_stimuli = np.cumsum(stim_counts)
    start_stimuli = np.empty_like(end_stimuli)
    start_stimuli[0] = 0
    start_stimuli[1:] = end_stimuli[0:-1]
    return start_stimuli, end_stimuli


# ----------------------------------------------------------------------------
# Frozen network 빌드 (STDP 유지, D=0 + reward=0 으로 학습 억제)
# ----------------------------------------------------------------------------
def build_frozen_model(name, params, reward_timesteps,
                       g_ee, row_ee, col_ee,
                       g_ei, row_ei, col_ei,
                       row_ii, col_ii,
                       row_ie, col_ie):
    model = genn_model.GeNNModel("double", name)
    model.dT = params["timestep_ms"]
    model._model.set_merge_postsynaptic_models(True)
    model._model.set_default_narrow_sparse_ind_enabled(True)
    if "seed" in params:
        model._model.set_seed(params["seed"])
    model.timing_enabled = params["measure_timing"]

    # --- 뉴런 (원본과 동일: dopamine 모델, 단 D=0 초기화) ---
    exc_params = {
        "a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0,
        "tauD": params["tau_d"], "dStrength": params["dopamine_strength"],
    }
    exc_init = {"V": -65.0, "U": -13.0, "D": 0.0}   # D=0
    inh_params = {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0}
    inh_init = {"V": -65.0, "U": -13.0}

    e_pop = model.add_neuron_population(
        "E", params["num_excitatory"], izhikevich_dopamine_model, exc_params, exc_init
    )
    i_pop = model.add_neuron_population(
        "I", params["num_inhibitory"], "Izhikevich", inh_params, inh_init
    )

    # reward bitmask 전부 0 -> injectDopamine 항상 false -> D 계속 0
    e_pop.set_extra_global_param("rewardTimesteps", reward_timesteps)

    e_pop.spike_recording_enabled = params["use_genn_recording"]
    i_pop.spike_recording_enabled = params["use_genn_recording"]

    delay_steps = 5  # 원본과 동일한 5ms delay

    # STDP 파라미터/초기값 (원본 build_model 과 동일)
    stdp_params = {
        "tauPlus": 20.0, "tauMinus": 20.0, "tauC": 1000.0,
        "tauD": params["tau_d"], "aPlus": 0.1, "aMinus": 0.15,
        "wMin": 0.0, "wMax": params["max_exc_weight"],
    }
    ### inh weight 제거
    #params["inh_weight"] = -1
    # --- E->E : STDP 모델, g=저장값 배열, c=0 ---
    e_e_pop = model.add_synapse_population(
        "EE", "SPARSE_INDIVIDUALG", delay_steps,
        "E", "E",
        izhikevich_stdp_model, stdp_params, {"g": g_ee, "c": 0.0},
        {}, {},
        "DeltaCurr", {}, {},
    )
    e_e_pop.set_sparse_connections(row_ee, col_ee)

    # --- E->I : STDP 모델, g=저장값 배열, c=0 ---
    e_i_pop = model.add_synapse_population(
        "EI", "SPARSE_INDIVIDUALG", delay_steps,
        "E", "I",
        izhikevich_stdp_model, stdp_params, {"g": g_ei, "c": 0.0},
        {}, {},
        "DeltaCurr", {}, {},
    )
    e_i_pop.set_sparse_connections(row_ei, col_ei)

    # --- I->I, I->E : static, 저장된 구조 로드, g=inh_weight ---
    if USE_INDIVIDUALG_FOR_INH:
        g_ii = np.full(len(row_ii), params["inh_weight"], dtype=np.float64)
        g_ie = np.full(len(row_ie), params["inh_weight"], dtype=np.float64)
        i_i_pop = model.add_synapse_population(
            "II", "SPARSE_INDIVIDUALG", delay_steps,
            "I", "I", "StaticPulse", {}, {"g": g_ii}, {}, {},
            "DeltaCurr", {}, {},
        )
        i_i_pop.set_sparse_connections(row_ii, col_ii)
        i_e_pop = model.add_synapse_population(
            "IE", "SPARSE_INDIVIDUALG", delay_steps,
            "I", "E", "StaticPulse", {}, {"g": g_ie}, {}, {},
            "DeltaCurr", {}, {},
        )
        i_e_pop.set_sparse_connections(row_ie, col_ie)
    else:
        i_i_pop = model.add_synapse_population(
            "II", "SPARSE_GLOBALG", delay_steps,
            "I", "I", "StaticPulse", {}, {"g": params["inh_weight"]}, {}, {},
            "DeltaCurr", {}, {},
        )
        i_i_pop.set_sparse_connections(row_ii, col_ii)
        i_e_pop = model.add_synapse_population(
            "IE", "SPARSE_GLOBALG", delay_steps,
            "I", "E", "StaticPulse", {}, {"g": params["inh_weight"]}, {}, {},
            "DeltaCurr", {}, {},
        )
        i_e_pop.set_sparse_connections(row_ie, col_ie)

    return model, e_pop, i_pop, e_e_pop, e_i_pop, i_i_pop, i_e_pop


# ============================================================================
# 메인
# ============================================================================
params = get_params(build_model=True, measure_timing=False, use_genn_recording=True)

params["duration_ms"] = TEST_DURATION_MS
params["duration_timestep"] = int(round(TEST_DURATION_MS / params["timestep_ms"]))
params["record_time_ms"] = RECORD_CHUNK_MS
params["record_time_timestep"] = int(round(RECORD_CHUNK_MS / params["timestep_ms"]))
params["min_inter_stimuli_interval_ms"] = INTER_STIMULI_INTERVAL_MS
params["max_inter_stimuli_interval_ms"] = INTER_STIMULI_INTERVAL_MS+10
params["min_inter_stimuli_interval_timestep"] = int(round(INTER_STIMULI_INTERVAL_MS/ params["timestep_ms"]))
params["max_inter_stimuli_interval_timestep"] = int(round(INTER_STIMULI_INTERVAL_MS+10/ params["timestep_ms"]))
params["learning_time_ms"] = LEARNING_TIME_MS
params["learning_time_timestep"] = int(round(LEARNING_TIME_MS / params["timestep_ms"]))
params["dt1"] = DT1
params["dt2"] = DT2
params["use_weight_record"] = False

assert (params["duration_timestep"] % params["record_time_timestep"]) == 0, \
    "duration 이 record chunk 의 배수가 아닙니다"

os.makedirs(OUT_DIR, exist_ok=True)

num_cells = params["num_excitatory"] + params["num_inhibitory"]

# --- 저장된 EE / EI / II / IE 로드 ---
ee_base = f"{DT1},{DT2}ms interval, weight in {WEIGHT_TIME_S}s"
ei_base = f"{DT1},{DT2}ms interval, EI weight in {WEIGHT_TIME_S}s"
ii_base = f"{DT1},{DT2}ms interval, II"
ie_base = f"{DT1},{DT2}ms interval, IE"

g_ee, row_ee, col_ee = load_triplet(EE_RECORD_DIR, ee_base, with_g=True)
g_ei, row_ei, col_ei = load_triplet(EI_RECORD_DIR, ei_base, with_g=True)
row_ii, col_ii = load_triplet(INH_RECORD_DIR, ii_base, with_g=False)
row_ie, col_ie = load_triplet(INH_RECORD_DIR, ie_base, with_g=False)

print(f"Loaded E->E: {len(g_ee)} synapses, g [{g_ee.min():.3f}, {g_ee.max():.3f}]")
print(f"Loaded E->I: {len(g_ei)} synapses, g [{g_ei.min():.3f}, {g_ei.max():.3f}]")
print(f"Loaded I->I: {len(row_ii)} synapses (g={params['inh_weight']})")
print(f"Loaded I->E: {len(row_ie)} synapses (g={params['inh_weight']})")

# 인덱스 범위 확인
assert row_ee.max() < params["num_excitatory"], "EE row(pre) out of E range"
assert col_ee.max() < params["num_excitatory"], "EE col(post) out of E range"
assert row_ei.max() < params["num_excitatory"], "EI row(pre) out of E range"
assert col_ei.max() < params["num_inhibitory"], "EI col(post) out of I range"
assert row_ii.max() < params["num_inhibitory"], "II row(pre) out of I range"
assert col_ii.max() < params["num_inhibitory"], "II col(post) out of I range"
assert row_ie.max() < params["num_inhibitory"], "IE row(pre) out of I range"
assert col_ie.max() < params["num_excitatory"], "IE col(post) out of E range"

# ----------------------------------------------------------------------------
# Stimuli 생성 (reward 관련 전부 제거)
# ----------------------------------------------------------------------------
input_sets = np.load(os.path.join(EE_RECORD_DIR,'5,5ms interval, input_sets.npy'))
print(input_sets)
print(input_sets.shape)
input_sets = list(input_sets)
input_sets.append(np.array([]))
stimuli_triplet_set_timing = []

stimuli_triplet_set_timing.append(np.array([0, 1, 2, params["dt1"], params["dt2"]]))
stimuli_triplet_set_timing.append(np.array([0, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([1, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([2, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([3, 4 ,5, params["dt1"], params["dt2"]]))
    

total_stimulus_times = []
neuron_stimuli_times = [[] for _ in range(num_cells)]

next_stimuli_timestep = np.random.randint(
    params["min_inter_stimuli_interval_timestep"],
    params["max_inter_stimuli_interval_timestep"],
)

while next_stimuli_timestep < params["duration_timestep"]:

    stimuli_set = stimuli_triplet_set_timing[
        np.random.randint(0, len(stimuli_triplet_set_timing))
        ]

    for n in input_sets[stimuli_set[0]]:
        neuron_stimuli_times[n].append(next_stimuli_timestep * params["timestep_ms"])

    if stimuli_set[1] is not None:
        for n in input_sets[stimuli_set[1]]:
            neuron_stimuli_times[n].append(
                (next_stimuli_timestep + (stimuli_set[3] or 0)) * params["timestep_ms"])

    if stimuli_set[2] is not None:
        delay1 = stimuli_set[3] or 0
        delay2 = stimuli_set[4] or 0
        for n in input_sets[stimuli_set[2]]:
            neuron_stimuli_times[n].append(
                (next_stimuli_timestep + delay1 + delay2) * params["timestep_ms"])

    total_stimulus_times.append(
        (next_stimuli_timestep * params["timestep_ms"],
         stimuli_set[0], stimuli_set[1], stimuli_set[2],
         stimuli_set[3], stimuli_set[4]))

    next_stimuli_timestep += np.random.randint(
        params["min_inter_stimuli_interval_timestep"],
        params["max_inter_stimuli_interval_timestep"],
    )

neuron_stimuli_counts = [len(n) for n in neuron_stimuli_times]

# ----------------------------------------------------------------------------
# 네트워크 빌드
# ----------------------------------------------------------------------------
reward_timesteps = np.zeros((params["duration_timestep"] + 31) // 32, dtype=np.uint32)

model, e_pop, i_pop, e_e_pop, e_i_pop, i_i_pop, i_e_pop = build_frozen_model(
    "izhikevich_response_test", params, reward_timesteps,
    g_ee, row_ee, col_ee, g_ei, row_ei, col_ei,
    row_ii, col_ii, row_ie, col_ie
)

curr_source_params = {"n": 6.5, "stimMagnitude": params["stimuli_current"]}
start_exc_stimuli, end_exc_stimuli = get_start_end_stim(
    neuron_stimuli_counts[: params["num_excitatory"]])
start_inh_stimuli, end_inh_stimuli = get_start_end_stim(
    neuron_stimuli_counts[params["num_excitatory"]:])

e_curr_pop = model.add_current_source(
    "ECurr", stim_noise_model, "E", curr_source_params,
    {"startStim": start_exc_stimuli, "endStim": end_exc_stimuli})
i_curr_pop = model.add_current_source(
    "ICurr", stim_noise_model, "I", curr_source_params,
    {"startStim": start_inh_stimuli, "endStim": end_inh_stimuli})

e_curr_pop.set_extra_global_param(
    "stimTimes", np.hstack(neuron_stimuli_times[: params["num_excitatory"]]))
i_curr_pop.set_extra_global_param(
    "stimTimes", np.hstack(neuron_stimuli_times[params["num_excitatory"]:]))

print("Building model")
model.build()
print("Loading model")
model.load(num_recording_timesteps=params["record_time_timestep"])

# ----------------------------------------------------------------------------
# [검증 1] 로드된 EE / EI / II / IE 가 저장값과 일치하는지
# ----------------------------------------------------------------------------
def roundtrip_check(pop, row_ref, col_ref, label, g_ref=None):
    pop.pull_connectivity_from_device()
    row_chk = np.array(pop.get_sparse_pre_inds())
    col_chk = np.array(pop.get_sparse_post_inds())
    ok = (len(row_chk) == len(row_ref)
          and np.array_equal(row_chk, row_ref)
          and np.array_equal(col_chk, col_ref))
    if g_ref is not None:
        pop.pull_var_from_device("g")
        g_chk = pop.get_var_values("g")
        ok = ok and len(g_chk) == len(g_ref) and np.allclose(g_chk, g_ref)
    print(f"[{label} round-trip check] {'OK' if ok else 'MISMATCH -- 확인 필요'}")
    return ok

roundtrip_check(e_e_pop, row_ee, col_ee, "E->E", g_ref=g_ee)
roundtrip_check(e_i_pop, row_ei, col_ei, "E->I", g_ref=g_ei)
roundtrip_check(i_i_pop, row_ii, col_ii, "I->I")
roundtrip_check(i_e_pop, row_ie, col_ie, "I->E")

# ----------------------------------------------------------------------------
# 시뮬레이션 (weight recording 제거, spike recording 만)
# ----------------------------------------------------------------------------
print("Simulating")
sim_start = perf_counter()
total_exc_spikes = []
total_inh_spikes = []

while model.timestep < params["duration_timestep"]:
    model.step_time()
    if params["use_genn_recording"]:
        if model.timestep % params["record_time_timestep"] == 0:
            model.pull_recording_buffers_from_device()
            e_times, e_ids = e_pop.spike_recording_data
            i_times, i_ids = i_pop.spike_recording_data
            total_exc_spikes.append((e_times, e_ids))
            total_inh_spikes.append((i_times, i_ids))

print("Simulation time: %fms" % ((perf_counter() - sim_start) * 1000.0))

# ----------------------------------------------------------------------------
# [검증 2] 시뮬레이션 후 EE/EI g 가 로드값과 동일한지 (학습 억제 확인)
# ----------------------------------------------------------------------------
e_e_pop.pull_var_from_device("g")
g_ee_after = e_e_pop.get_var_values("g")
e_i_pop.pull_var_from_device("g")
g_ei_after = e_i_pop.get_var_values("g")
print(f"[EE g unchanged] {np.allclose(g_ee_after, g_ee)} "
      f"(max |dg| = {np.abs(g_ee_after - g_ee).max():.3e})")
print(f"[EI g unchanged] {np.allclose(g_ei_after, g_ei)} "
      f"(max |dg| = {np.abs(g_ei_after - g_ei).max():.3e})")

# ----------------------------------------------------------------------------
# 저장
# ----------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "5_5_izhikevich_e_spikes.csv"), "w", newline="\n") as f:
    w = csv.writer(f, delimiter=",")
    w.writerow(["Time [ms]", " Neuron ID"])
    for e in total_exc_spikes:
        for t, i in zip(e[0], e[1]):
            w.writerow([t, i])

with open(os.path.join(OUT_DIR, "5_5_izhikevich_i_spikes.csv"), "w", newline="\n") as f:
    w = csv.writer(f, delimiter=",")
    w.writerow(["Time [ms]", " Neuron ID"])
    for e in total_inh_spikes:
        for t, i in zip(e[0], e[1]):
            w.writerow([t, i])

with open(os.path.join(OUT_DIR, "5_5_izhikevich_stimulus_times.csv"), "w", newline="\n") as f:
    w = csv.writer(f, delimiter=",")
    for s in total_stimulus_times:
        w.writerow([s[0], s[1], s[2], s[3], s[4], s[5]])

# 간단 raster (전체 구간)
all_e = [np.hstack([e[0] for e in total_exc_spikes]),
         np.hstack([e[1] for e in total_exc_spikes])]
all_i = [np.hstack([e[0] for e in total_inh_spikes]),
         np.hstack([e[1] for e in total_inh_spikes])]
fig, ax = plt.subplots(figsize=(20, 8))
ax.scatter(all_e[0], all_e[1], s=1, color="red", edgecolors="none")
ax.scatter(all_i[0], all_i[1] + params["num_excitatory"], s=1, color="blue",
           edgecolors="none")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Neuron ID")
ax.set_ylim(0, num_cells)
plt.savefig(os.path.join(OUT_DIR, "raster.png"), dpi=150)
print(f"Done. Outputs in {OUT_DIR}")
