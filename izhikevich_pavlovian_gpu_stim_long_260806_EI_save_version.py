import sys

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from time import perf_counter

#from common import (
# replace "common" by "common_w_delay" (02/06/2025 이경진) and 추가로 recording 시간을 늘림 (06/16/2025) 
from common_w_delay_long import (    
    izhikevich_dopamine_model,
    izhikevich_stdp_model,
    build_model,
    get_params,
    plot,
    convert_spikes,
)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def get_start_end_stim(stim_counts):
    end_stimuli = np.cumsum(stim_counts)
    start_stimuli = np.empty_like(end_stimuli)
    start_stimuli[0] = 0
    start_stimuli[1:] = end_stimuli[0:-1]

    return start_stimuli, end_stimuli


# ----------------------------------------------------------------------------
# Custom models
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

# ----------------------------------------------------------------------------
# Stimuli generation
# ----------------------------------------------------------------------------
# Get standard model parameters
params = get_params(build_model=True, measure_timing=False, use_genn_recording=True, size_scale_factor = 2) #original, no size scale factor
if "seed" in params:
    np.random.seed(params["seed"])
# Generate stimuli sets of neuron IDs
num_cells = params["num_excitatory"] + params["num_inhibitory"]
stim_gen_start_time = perf_counter()
input_sets = [
    np.random.choice(num_cells, params["stimuli_set_size"], replace=False)
    for _ in range(params["num_stimuli_sets"])
]
input_sets.append(np.array([]))

params["dt1"] = int(sys.argv[1])
params["dt2"] = int(sys.argv[2])
#params["cwd"] = "./data_"+str(params["dt1"])+"_"+str(params["dt2"])
params["cwd"] = f"260806_data_5_5/data_{params['dt1']}_{params['dt2']}_size_scale_factor2_no_weight_scale"

_base_cwd = params["cwd"]
_run_idx = 1
while os.path.exists(f"{_base_cwd}({_run_idx})"):
    _run_idx += 1
params["cwd"] = f"{_base_cwd}({_run_idx})"

os.makedirs(params["cwd"], exist_ok=True) 
os.makedirs(os.path.join(params["cwd"], params["record_path"]), exist_ok=True)
os.makedirs(os.path.join(params["cwd"], params["csv_path"]), exist_ok=True)
# ---- input_sets 저장 (마지막 빈 배열 제외한 100개, 각 길이 50) ----
input_sets_arr = np.array(input_sets[:params["num_stimuli_sets"]])  # (100, 50)
np.save(
    "%s/%s/%d,%dms interval, input_sets"
    % (params["cwd"], params["record_path"], params["dt1"], params["dt2"]),
    input_sets_arr,
)
#os.mkdir(params["cwd"], exist_ok=True)
target_triplet_set = np.array([0, 1, 2, params["dt1"], params["dt2"]])
stimuli_triplet_set_timing = []
for d1 in range(3, 17, 2):
    for d2 in range(3, 17, 2):
        stimuli_triplet_set_timing.append(np.array([0, 1, 2, d1, d2]))
for d1 in range(3, 17, 2):
    stimuli_triplet_set_timing.append(
        np.array([0, 1, params["num_stimuli_sets"], d1, 0])
    )
for d2 in range(3, 17, 2):
    stimuli_triplet_set_timing.append(
        np.array([0, params["num_stimuli_sets"], 2, 0, d2])
    )
stimuli_triplet_set_timing.append(
    np.array(
        [
            0,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            params["dt1"],
            params["dt2"],
        ]
    )
)
stimuli_triplet_set_timing.append(
    np.array(
        [
            1,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            params["dt1"],
            params["dt2"],
        ]
    )
)
stimuli_triplet_set_timing.append(
    np.array(
        [
            2,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            params["dt1"],
            params["dt2"],
        ]
    )
)
#stimuli_triplet_set_timing.append(np.array([4, 5, 6, params["dt1"], params["dt2"]]))
#stimuli_triplet_set_timing.append(np.array([40, 50, 60, params["dt1"], params["dt2"]]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 3, 70]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 30]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 50]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 70]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 90]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 110]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 130]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 150]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 300]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 5, 600]))
#stimuli_triplet_set_timing.append(np.array([0, 1, 2, 30, 30]))
###inhoi  (0, None, None, None, None), (1, None, None, None, None), (2, None, None, None, None) 추가
stimuli_triplet_set_timing.append(np.array([0, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([1, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([2, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([3, None, None, None, None]))
stimuli_triplet_set_timing.append(np.array([5, None, None, None, None]))
# Lists of stimulus and reward_times for use when plotting
total_stimulus_times = []
total_reward_times = []


# Create list for each neuron
neuron_stimuli_times = [[] for _ in range(num_cells)]
total_num_exc_stimuli = 0
total_num_inh_stimuli = 0

# Create zeroes numpy array to hold reward timestep bitmask
reward_timesteps = np.zeros((params["duration_timestep"] + 31) // 32, dtype=np.uint32)

# Loop while stimuli are within simulation duration

next_stimuli_timestep = np.random.randint(
    params["min_inter_stimuli_interval_timestep"],
    params["max_inter_stimuli_interval_timestep"],
)
cycle_num = 0
while next_stimuli_timestep < params["duration_timestep"]:
    # Pick a stimuli set to present at this timestep
    # 100번에 한번씩은 target 자극가함!
    if cycle_num != 0 and cycle_num % 100 == 0:    
        stimuli_set = target_triplet_set
    # 그리고 일반적으로는 (예: Sp, Sq, Sr, dt1=11, dt2=11) 자극 전달 
    else:
        stimuli_set = np.append(
            np.random.choice(params["num_stimuli_sets"], 3, replace=False),
            [params["dt1"], params["dt2"]],        
        )

    # 후반부에서는 앞서 위에서 미리 만들어둔 "stimuli_triplet_set_timing"에서 패턴을 무작위로 뽑는다.
    # 위에서 range를 설정하기 나름 ....
    ###inhoi params["learning_time_ms"] 까지만 학습하고, 나머지는 다 test 하도록 설정한다.
    if next_stimuli_timestep > (                
        params["learning_time_timestep"]
    ):
        stimuli_set = stimuli_triplet_set_timing[
            np.random.randint(0, len(stimuli_triplet_set_timing))
        ]

    # Loop through neurons in stimuli set and add time to list (즉, 매 iteration시점에 자극 받을 뉴런들 설정)
    for n in input_sets[stimuli_set[0]]:
        neuron_stimuli_times[n].append(next_stimuli_timestep * params["timestep_ms"])

    # 나머지는 None 체크
    if stimuli_set[1] is not None:
        for n in input_sets[stimuli_set[1]]:
            neuron_stimuli_times[n].append(
            (next_stimuli_timestep + (stimuli_set[3] or 0)) * params["timestep_ms"]
        )

    if stimuli_set[2] is not None:
        delay1 = stimuli_set[3] or 0
        delay2 = stimuli_set[4] or 0
        for n in input_sets[stimuli_set[2]]:
            neuron_stimuli_times[n].append(
            (next_stimuli_timestep + delay1 + delay2) * params["timestep_ms"]
        )

    # Count the number of excitatory neurons in input set and add to total
    num_exc_in_input_set = np.sum(input_sets[stimuli_set[0]] < params["num_excitatory"])
    total_num_exc_stimuli += num_exc_in_input_set
    total_num_inh_stimuli += num_cells - num_exc_in_input_set
        
    
    ###inhoi start_stimulus_times와 end_stimulus_times를 나누지 말고, total_stimulus_times를 저장한다.
    total_stimulus_times.append(
            (
                next_stimuli_timestep * params["timestep_ms"],
                stimuli_set[0],
                stimuli_set[1],
                stimuli_set[2],
                stimuli_set[3],
                stimuli_set[4],
            )
        )
    
    
    # **** If this is the rewarded stimuli (그리고 전달 시점 < duration_ms - record_time_ms (eg. 3,600 - 1,200 = 2,400인 경우 reward 주는 시점들을 확인해서 reward_timestep에 넣어줌) ****
    
    
    if (stimuli_set == target_triplet_set).all() and next_stimuli_timestep <= (
        params["duration_timestep"]
    ):
        # Determine time of next reward
        reward_timestep = next_stimuli_timestep + np.random.randint(
            params["max_reward_delay_timestep"]
        )

        # If this is within simulation
        ###inhoi learning time ms까지 reward를 준다.
        if reward_timestep < params["learning_time_timestep"]:
            # Set bit in reward timesteps bitmask
            reward_timesteps[reward_timestep // 32] |= 1 << (reward_timestep % 32)

            # If we should be recording at this point, add reward to list
            
            total_reward_times.append(reward_timestep * params["timestep_ms"])
            
                
    # Advance to next stimuli
    next_stimuli_timestep += np.random.randint(
        params["min_inter_stimuli_interval_timestep"],
        params["max_inter_stimuli_interval_timestep"],
    )
    cycle_num += 1

# Count stimuli each neuron should emit
neuron_stimuli_counts = [len(n) for n in neuron_stimuli_times]

stim_gen_end_time = perf_counter()
print(
    "Stimulus generation time: %fms"
    % ((stim_gen_end_time - stim_gen_start_time) * 1000.0)
)

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
# Assert that duration is a multiple of record time
assert (params["duration_timestep"] % params["record_time_timestep"]) == 0

# Build base model  *** Kyoung aeede i_i_pop and i_e_pop in the next line
model, e_pop, i_pop, e_e_pop, e_i_pop, i_i_pop, i_e_pop = build_model(
    "izhikevich_pavlovian_gpu_stim", params, reward_timesteps
)

# Current source parameters
curr_source_params = {"n": 6.5, "stimMagnitude": params["stimuli_current"]}

# Calculate start and end indices of stimuli to be injected by each current source
start_exc_stimuli, end_exc_stimuli = get_start_end_stim(
    neuron_stimuli_counts[: params["num_excitatory"]]
)
start_inh_stimuli, end_inh_stimuli = get_start_end_stim(
    neuron_stimuli_counts[params["num_excitatory"] :]
)

# Current source initial state
exc_curr_source_init = {"startStim": start_exc_stimuli, "endStim": end_exc_stimuli}
inh_curr_source_init = {"startStim": start_inh_stimuli, "endStim": end_inh_stimuli}

# Add background current sources
e_curr_pop = model.add_current_source(
    "ECurr", stim_noise_model, "E", curr_source_params, exc_curr_source_init
)
i_curr_pop = model.add_current_source(
    "ICurr", stim_noise_model, "I", curr_source_params, inh_curr_source_init
)

# Set stimuli times
e_curr_pop.set_extra_global_param(
    "stimTimes", np.hstack(neuron_stimuli_times[: params["num_excitatory"]])
)
i_curr_pop.set_extra_global_param(
    "stimTimes", np.hstack(neuron_stimuli_times[params["num_excitatory"] :])
)

if params["build_model"]:
    print("Building model")
    model.build()

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
# Load model, allocating enough memory for recording
print("Loading model")
model.load(num_recording_timesteps=params["record_time_timestep"])


print("Simulating")
# Loop through timesteps
sim_start_time = perf_counter()

###inhoi start와 end를 나누지 않고 spike를 저장한다.
total_exc_spikes = []
total_inh_spikes = []

i_i_pop.pull_connectivity_from_device()
row_i_i = np.array(i_i_pop.get_sparse_pre_inds())
col_i_i = np.array(i_i_pop.get_sparse_post_inds())
np.save(
        "%s/%s/%d,%dms interval, II_row"
        % (params["cwd"], params["record_path"],
        params["dt1"], params["dt2"]),
        row_i_i,
        )
np.save(
        "%s/%s/%d,%dms interval, II_col"
        % (params["cwd"], params["record_path"],
        params["dt1"], params["dt2"]),
        col_i_i,
        )

i_e_pop.pull_connectivity_from_device()
row_i_e = np.array(i_e_pop.get_sparse_pre_inds())
col_i_e = np.array(i_e_pop.get_sparse_post_inds())
np.save(
        "%s/%s/%d,%dms interval, IE_row"
        % (params["cwd"], params["record_path"],
        params["dt1"], params["dt2"]),
        row_i_e,
        )
np.save(
        "%s/%s/%d,%dms interval, IE_col"
         % (params["cwd"], params["record_path"],
         params["dt1"], params["dt2"]),
         col_i_e,
         )
while model.timestep <= params["duration_timestep"]:
    if params["use_weight_record"] == True:
        if model.t % (10 * 1000) == 0:
            e_e_pop.pull_var_from_device("g")
            e_e_pop.pull_connectivity_from_device()
            
            g_e_e = e_e_pop.get_var_values("g")
            row_e_e = np.array(e_e_pop.get_sparse_pre_inds())
            col_e_e = np.array(e_e_pop.get_sparse_post_inds())

            np.save(
                "%s/%s/%d,%dms interval, weight in %ds_data"
                % (
                    params["cwd"],
                    params["record_path"],
                    params["dt1"],
                    params["dt2"],
                    ((model.t // (1 * 1000)) * 1),
                ),
                g_e_e,
            )
            np.save(
                "%s/%s/%d,%dms interval, weight in %ds_row"
                % (
                    params["cwd"],
                    params["record_path"],
                    params["dt1"],
                    params["dt2"],
                    ((model.t // (1 * 1000)) * 1),
                ),
                row_e_e,
            )
            np.save(
                "%s/%s/%d,%dms interval, weight in %ds_col"
                % (
                    params["cwd"],
                    params["record_path"],
                    params["dt1"],
                    params["dt2"],
                    ((model.t // (1 * 1000)) * 1),
                ),
                col_e_e,
            )
            # ---- E->I weight 저장 (E->E 와 동일 방식, 태그만 'EI weight') ----
            e_i_pop.pull_var_from_device("g")
            e_i_pop.pull_connectivity_from_device()

            g_e_i = e_i_pop.get_var_values("g")
            row_e_i = np.array(e_i_pop.get_sparse_pre_inds())
            col_e_i = np.array(e_i_pop.get_sparse_post_inds())

            np.save(
                "%s/%s/%d,%dms interval, EI weight in %ds_data"
                % (params["cwd"], params["record_path"],
                   params["dt1"], params["dt2"], ((model.t // 1000) * 1)),
                g_e_i,
            )
            np.save(
                "%s/%s/%d,%dms interval, EI weight in %ds_row"
                % (params["cwd"], params["record_path"],
                   params["dt1"], params["dt2"], ((model.t // 1000) * 1)),
                row_e_i,
            )
            np.save(
                "%s/%s/%d,%dms interval, EI weight in %ds_col"
                % (params["cwd"], params["record_path"],
                   params["dt1"], params["dt2"], ((model.t // 1000) * 1)),
                col_e_i,
            )



    # Simulation
    model.step_time()
    
    
    if params["use_genn_recording"]:
        # If we've just finished simulating the initial recording interval
        if model.timestep % params["record_time_timestep"] ==0:
            # Download recording data
            model.pull_recording_buffers_from_device()
            e_times, e_ids = e_pop.spike_recording_data
            i_times, i_ids = i_pop.spike_recording_data
            

            total_exc_spikes.append((e_times, e_ids))
            total_inh_spikes.append((i_times, i_ids))
            
            

            

sim_end_time = perf_counter()
print("Simulation time: %fms" % ((sim_end_time - sim_start_time) * 1000.0))

if params["measure_timing"]:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tPresynaptic update:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tPostsynaptic update:%f" % (1000.0 * model.postsynaptic_update_time))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
"""
plot(
    total_exc_spikes,
    total_inh_spikes,
    total_stimulus_times,
    total_reward_times,
    10000.0,
    params,
)
"""
# Save plot
#plt.savefig(f"{params['cwd']}/{params['fig_path']}/{params['dt1']}_{params['dt2']}_plot.jpg")

# Read rewards
with open(
    f"{params['cwd']}/{params['csv_path']}/{params['dt1']}_{params['dt2']}_izhikevich_stimulus_times.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    for stimuls in total_stimulus_times:
        spamwriter.writerow(
            [stimuls[0], stimuls[1], stimuls[2], stimuls[3], stimuls[4], stimuls[5]]
        )

with open(
    f"{params['cwd']}/{params['csv_path']}/{params['dt1']}_{params['dt2']}_izhikevich_reward_times.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    for reward in total_reward_times:
        spamwriter.writerow([reward])


with open(
    f"{params['cwd']}/{params['csv_path']}/{params['dt1']}_{params['dt2']}_izhikevich_e_spikes.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    spamwriter.writerow(["Time [ms]", " Neuron ID"])
    for exc_spikes in total_exc_spikes:
        for time, ids in zip(exc_spikes[0], exc_spikes[1]):
            spamwriter.writerow([time, ids])

with open(
    f"{params['cwd']}/{params['csv_path']}/{params['dt1']}_{params['dt2']}_izhikevich_i_spikes.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    spamwriter.writerow(["Time [ms]", " Neuron ID"])
    for inh_spikes in total_inh_spikes:
        for time, ids in zip(inh_spikes[0], inh_spikes[1]):
            spamwriter.writerow([time, ids])
