import os, sys, shutil
import collections
import time
from itertools import repeat
import numpy as np
import matplotlib
matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from shared.io import print_stdout_forced, print_job_output, print_cg_itp_file, update_cg_itp_obj, print_ff_file, print_mdp_file
from shared.helpers import get_rounded_parameters_set
import config
from shared.scoring import get_particle_score
from shared.analysis import get_score_and_plot_graphs_for_single_job
from shared.hpc_jobs import HPCJobs
from datetime import datetime
from statistics import mean
import multiprocessing
from shared.io import backup_swarm_iter_logs_and_checkpoint
from shared.simulations import run_parallel


def create_files_and_dirs_for_swarm_iter(ns, parameters_sets, nb_eval_particles_range_over_complete_opti):
    # NOTE: we have some redundancy in the setup simulation files that are written, but this probably does NOT
    #       need refactorizing atm because we are going towards more flexibility anyways, so all this will change soon

    parameters_sets_rounded, updated_cg_itps = [], []

    for ii, nb_eval_particle in enumerate(nb_eval_particles_range_over_complete_opti):
        current_eval_dir = f"{ns.user_config['exec_folder']}/{config.iteration_sim_files_dirname}{nb_eval_particle}"

        # 0. round the parameters given by FST-PSO to the required precision, given in config file
        parameters_set = parameters_sets[ii]
        parameters_set_rounded = get_rounded_parameters_set(ns, parameters_set)
        parameters_sets_rounded.append(parameters_set_rounded)

        # 0. update the temporary CG ITPs container with parameters according to current evaluation type + write for sims
        #    all molecules (= lipids atm) are then updated with the bonded parameters
        current_cg_itps, param_cursor = update_cg_itp_obj(ns, parameters_set=parameters_set_rounded)
        updated_cg_itps.append(current_cg_itps)

        for lipid_code in ns.user_config['lipids_codes']:
            for temp in ns.user_config['lipids_codes'][lipid_code]:

                # 1. create all subdirs for each lipid and temperature
                os.makedirs(f'{current_eval_dir}/{lipid_code}_{temp}', exist_ok=True)

                # 2. copy starting frames (different for each lipid/temperature) + TOP file (the same for each temperature) + NDX
                shutil.copy(f"{config.cg_setups_data_dir}/{lipid_code}_{ns.user_config['mapping_type']}_{ns.user_config['solv']}/start_frame_{temp}.gro",
                            f'{current_eval_dir}/{lipid_code}_{temp}/start_frame.gro')
                shutil.copy(f"{config.cg_setups_data_dir}/{lipid_code}_{ns.user_config['mapping_type']}_{ns.user_config['solv']}/system_{temp}.top",
                            f'{current_eval_dir}/{lipid_code}_{temp}/system.top')
                shutil.copy(f"{config.cg_setups_data_dir}/{lipid_code}_{ns.user_config['mapping_type']}_{ns.user_config['solv']}/index_{temp}.ndx",
                            f'{current_eval_dir}/{lipid_code}_{temp}/index.ndx')

                # 3. copy MDP files + adapt temperature etc on the fly
                print_mdp_file(ns, lipid_code, temp=None, mdp_filename_in=f'{config.cg_setups_data_dir}/{ns.cg_mini_mdp}',
                               mdp_filename_out=f'{current_eval_dir}/{lipid_code}_{temp}/mini.mdp', sim_type='mini')
                print_mdp_file(ns, lipid_code, temp, mdp_filename_in=f'{config.cg_setups_data_dir}/{ns.cg_equi_mdp}',
                               mdp_filename_out=f'{current_eval_dir}/{lipid_code}_{temp}/equi.mdp', sim_type='equi')
                print_mdp_file(ns, lipid_code, temp, mdp_filename_in=f'{config.cg_setups_data_dir}/{ns.cg_prod_mdp}',
                               mdp_filename_out=f'{current_eval_dir}/{lipid_code}_{temp}/prod.mdp', sim_type='prod')

                # 4. create the new ITPs for each lipid (= the bonded parameters for the simulations happening within a swarm particle)
                print_cg_itp_file(current_cg_itps[lipid_code], out_path_itp=f'{current_eval_dir}/{lipid_code}_{temp}/{lipid_code}.itp')

                # 5. create force field file with new sig/eps tries + plot the LJ EPS matrix between CG beads (= the non-bonded)
                print_ff_file(ns, parameters_set_rounded, param_cursor, out_dir=f'{current_eval_dir}/{lipid_code}_{temp}/')

    return parameters_sets_rounded, updated_cg_itps


# TODO: disabled atm because we don't care too much in fact, so I have not re-integrated this
# def plot_ff_error(ns, nb_eval_particle, lipid_code, temp, sim_status):
#
#     # TODO/NOTE: in code blocks below the averaging gives more weight to a lipid if several temperatures are used for this given lipid -- this is more or less desired and at worst considered harmless atm
#
#     nb_beads_types = len(ns.all_beads_types)
#
#     # EPS -- data
#     err_eps = collections.Counter()
#     err_mat_eps = np.zeros((nb_beads_types, nb_beads_types), dtype=np.float)
#     pair_counter = collections.Counter()
#
#     # EPS -- aggregation
#     for lipid_code in ns.lipids_codes:
#         for temp in ns.lipids_codes[lipid_code]:
#
#             for i in range(nb_beads_types):  # error matrix of eps
#                 for j in range(nb_beads_types):
#                     if j >= i:
#                         bead_type_1, bead_type_2 = ns.all_beads_types[i], ns.all_beads_types[j]
#                         pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
#                         try:
#                             err_eps[pair_type] += all_error_data[lipid_code][temp]['rdf_pair'][pair_type]
#                             err_mat_eps[i, j] += all_error_data[lipid_code][temp]['rdf_pair'][pair_type]
#                             pair_counter[pair_type] += 1
#                         except KeyError:  # those beads pairs that can be missing in a given lipid
#                             pass
#                     else:
#                         err_mat_eps[i, j] = None
#
#     # EPS -- averaging
#     for i in range(nb_beads_types):
#         for j in range(nb_beads_types):
#             if j >= i:
#                 bead_type_1, bead_type_2 = ns.all_beads_types[i], ns.all_beads_types[j]
#                 pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
#                 # print(err_mat_eps[i, j], '--', pair_counter[pair_type])
#                 if pair_counter[pair_type] != 0:  # to avoid a warning
#                     err_eps[pair_type] /= pair_counter[pair_type]
#                     err_mat_eps[i, j] /= pair_counter[pair_type]
#             # print('Bead type 1', bead_type_1, '-- Bead type 2', bead_type_2, '-- Counter', pair_counter[pair_type])
#
#     # EPS -- plot error mat
#     fig, ax = plt.subplots()
#     cmap = CM.get_cmap('OrRd')
#     cmap.set_bad('lavender')
#     im = ax.imshow(err_mat_eps, cmap=cmap, vmin=0, aspect='equal')
#     ax.set_xticks(np.arange(nb_beads_types))
#     ax.set_yticks(np.arange(nb_beads_types))
#     ax.set_xticklabels(ns.all_beads_types)
#     ax.set_yticklabels(ns.all_beads_types)
#     ax.xaxis.tick_top()
#     cbar = ax.figure.colorbar(im, ax=ax)
#     cbar.ax.set_ylabel('Error', rotation=-90, va="bottom")
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.suptitle('Overall -- Error quantif. for EPS')
#     # plt.show()
#     plt.savefig(ns.exec_folder + '/' + current_eval_dir + '/EPS_ERROR_all.png')
#     plt.close(fig)


# evaluation function to be optimized using FST-PSO
def eval_function_parallel_swarm(parameters_sets, args):

    ns = args

    ts_start_swarm_iter = datetime.now().timestamp()
    ts_start_swarm_iter_str = time.strftime('%H:%M:%S on %d-%m-%Y')

    print('\n######################################################################')
    print('          SWARM ITERATION', ns.n_swarm_iter, 'started at', ts_start_swarm_iter_str)
    print('######################################################################\n')

    backup_swarm_iter_logs_and_checkpoint(ns)

    # NOTE: if using ns.variables within the parallelized functions (i.e. anything starting HERE)
    #       they SHALL only read from the namespace, BUT NEVER WRITE !!
    # NOTE: variable prefix 'p_' means 'pool parallel', NOT particle

    # 1. data structure for storing results from THE WHOLE SWARM ITERATION
    # NOTE: currently the key is: #iteration (over complete execution) + lipid_code + temp
    #       but in the future we will want to have as key the simulation, for using mixed molecules types in simulations
    swarm_res = {}
    jobs = {}  # jobs variables
    p_nb_eval_particle, p_lipid_code, p_temp = [], [], []  # for multiprocessing pool in block 4.
    p_job_name, p_job_exec_dir = [], []  # for running jobs in LOCAL in block 3.

    nb_eval_particles_range_over_complete_opti = range((ns.n_swarm_iter - 1) * ns.nb_particles + 1,
                                                       ns.n_swarm_iter * ns.nb_particles + 1)
    for nb_eval_particle in nb_eval_particles_range_over_complete_opti:
        swarm_res[nb_eval_particle] = {}
        for lipid_code in ns.user_config['lipids_codes']:
            swarm_res[nb_eval_particle][lipid_code] = {}
            for temp in ns.user_config['lipids_codes'][lipid_code]:
                swarm_res[nb_eval_particle][lipid_code][temp] = {}

                # TODO: when doing something else than lipids+temp we need to adapt these jobs names
                # NOTE: the master job name must stay included in variable 'job_name' below, so there is not mix-up
                #       if 2 masters are running on the same HPC or SUPSI machine
                job_name = f"{ns.user_config['master_job_name']}_{nb_eval_particle}_{lipid_code}_{temp}"
                job_exec_dir = f"{ns.user_config['exec_folder']}/{config.iteration_sim_files_dirname}{nb_eval_particle}/{lipid_code}_{temp}"

                jobs[job_name] = {'job_exec_dir': job_exec_dir}

                p_nb_eval_particle.append(nb_eval_particle)  # for multiprocessing pool in block 4.
                p_lipid_code.append(lipid_code)
                p_temp.append(temp)

                p_job_name.append(job_name)  # for running jobs in LOCAL in block 3.
                p_job_exec_dir.append(job_exec_dir)

    # 2. write all the files that will be needed for the WHOLE SWARM ITERATION
    #    with the parameters chosen by FST-PSO for each particle of the swarm
    parameters_sets_rounded, updated_cg_itps = create_files_and_dirs_for_swarm_iter(ns, parameters_sets, nb_eval_particles_range_over_complete_opti)

    p_updated_cg_itps = []
    for i in range(len(nb_eval_particles_range_over_complete_opti)):
        for lipid_code in ns.user_config['lipids_codes']:
            for _ in ns.user_config['lipids_codes'][lipid_code]:
                p_updated_cg_itps.append(updated_cg_itps[i])

    # 3. run the simulations in queue for HPC or LOCAL (SUPSI machines)
    ts_start_jobs_execution = datetime.now().timestamp()

    if ns.user_config['nb_hpc_slots'] != 0:  # CASE: running on HPC with SLURM

        # NOTE: all of this is good only for SLURM -- for HPC that do NOT use SLURM we will need to make another job runner
        # NOTE: JOB TIME -- For lipids I tested DLPC and DOPC (having the min/max of particles among all lipids simulations)
        #                   and they run mini/equi/prod (15 ns / 200 ns) in 29m04s and 31m50s, so I set job time to 40 min
        hpc_username = 'cempereu'
        slurm_single_job_args = {
            'sbatch': {
                'account': 's1125',  # this HAS to be provided (CSCS account for billing the node hours)
                'time': '00:25:00',  # this HAS to be format: '00:30:00' for a 30 min job
                'nodes': 1,  # number of nodes requested for a given job
                'ntasks-per-node': 12,  # number of MPI ranks
                'cpus-per-task': 1,
                'partition': 'normal',  # normal / low / debug (debug still bills the account, it just has a higher priority)
                'constraint': 'gpu',  # potential specific request about architecture
                'hint': 'nomultithread'
            },
            'env_modules': ['daint-gpu', 'GROMACS'],  # environment modules will be loaded in the order they are provided here
            'bash_exports': ['OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK', 'CRAY_CUDA_MPS=1']  # bash exports will be made in the order they are provided here
        }

        hpc_jobs = HPCJobs(hpc_username, jobs, ns.user_config['nb_hpc_slots'], slurm_single_job_args, gmx_path=ns.user_config['gmx_path'])

        # check how much exec time is remaining on the Master execution and verify we can run a full SWARM iteration
        ns.master_job_id, ts_master_elapsed, ts_master_total = hpc_jobs.get_master_time(master_job_name=ns.user_config['master_job_name'])
        delta_ts_master_remaining = ts_master_total - ts_master_elapsed
        # 50% margin because we do NOT know at which speed future jobs will enter the SLURM queue
        # 0% margin if we run on the long queue because we hardly will have the last iter being worst than the worst of 7 days
        time_margin = 0.00
        ts_master_elapsed_h = round(ts_master_elapsed / (60 * 60), 3)
        ts_master_total_h = round(ts_master_total / (60 * 60), 3)
        ts_master_total_time_perc = round(ts_master_elapsed / ts_master_total * 100, 2)
        delta_ts_master_remaining_h = round(delta_ts_master_remaining / (60 * 60), 3)
        print(f'Starting new SWARM iteration with MASTER elapsed time: {ts_master_elapsed_h} h of {ts_master_total_h} h ({ts_master_total_time_perc} %)\n')
        if delta_ts_master_remaining < ns.max_delta_ts_swarm_iter * (1 + time_margin):
            time_estimate_swarm_iter_h = round(ns.max_delta_ts_swarm_iter * (1 + time_margin) / (60 * 60), 3)
            print(
                f'Master remaining SLURM time: {delta_ts_master_remaining_h} h'
                f'\nInsufficient to perform a complete SWARM iteration estimated to maximum: {time_estimate_swarm_iter_h} h ({int(time_margin * 100)}% margin applied)'
                f'\n\n## FINISHING NOW + KILLING MASTER ##\n\n'
            )
            hpc_jobs.kill_slurm_job(job_id=ns.master_job_id)
            sys.exit()

        while not hpc_jobs.finished:
            hpc_jobs.run()

    else:  # CASE: running on LOCAL (SUPSI machines)

        # here we are NOT really using a MASTER SLURM process, but we still have a master process and record exec time
        ts_master_elapsed = datetime.now().timestamp() - ns.opti_start_ts
        ts_master_elapsed_h = round(ts_master_elapsed / (60 * 60), 3)
        delta_ts_master_remaining_h = 'INFINITE'
        print(f'Starting new SWARM iteration running in LOCAL (= without SLURM) using {ns.nb_slots} slots')

        # NOTE: this is NOT ENOUGH to only have lock=True here and array access has to be also managed with get_lock() or something
        # slots_states = multiprocessing.Array('i', ns.nb_slots, lock=True)
        # for i in range(ns.nb_slots):  # multiprocessing.Array does NOT like list comprehension
        #     slots_states[i] = 1  # mark slot as available

        # with multiprocessing.Pool(processes=ns.nb_slots, initializer=init_process, initargs=(slots_states,)) as pool:
        with multiprocessing.Pool(processes=ns.nb_slots) as pool:
            p_args = zip(repeat(ns), p_job_exec_dir, p_nb_eval_particle, p_lipid_code, p_temp)
            p_res = pool.starmap(run_parallel, p_args)
            p_time_start_str, p_time_end_str, p_time_elapsed_str = list(map(list, zip(*p_res)))

    delta_ts_start_jobs_execution = datetime.now().timestamp() - ts_start_jobs_execution
    delta_ts_start_jobs_execution_min = round(delta_ts_start_jobs_execution / 60, 2)
    delta_ts_start_jobs_execution_h = round(delta_ts_start_jobs_execution / (60 * 60), 3)
    print(f'\nWall time for all jobs of the SWARM iteration: {delta_ts_start_jobs_execution_min} min ({delta_ts_start_jobs_execution_h} hours)')

    # 4. get times for start, end, elapsed on the LOCAL / HPC to be written in our logs
    #    also check that simulations finished correctly, by verifying that the final .gro file exists
    ts_start_jobs_statuses = datetime.now().timestamp()

    if ns.user_config['nb_hpc_slots'] != 0:  # CASE: running on HPC with SLURM
        jobs_stats = hpc_jobs.get_stats()
    else:  # CASE: running on LOCAL (SUPSI machines)
        jobs_stats = {}
        for i, job_name in enumerate(p_job_name):
            jobs_stats[job_name] = {
                'time_start_str': p_time_start_str[i],
                'time_end_str': p_time_end_str[i],
                'time_elapsed_str': p_time_elapsed_str[i],
            }

    for job_name in jobs_stats:
        sp_job_name = job_name.split('_')
        nb_eval_particle = int(sp_job_name[-3])
        lipid_code = sp_job_name[-2]
        temp = sp_job_name[-1]
        swarm_res[nb_eval_particle][lipid_code][temp]['time_start_str'] = jobs_stats[job_name]['time_start_str']
        swarm_res[nb_eval_particle][lipid_code][temp]['time_end_str'] = jobs_stats[job_name]['time_end_str']
        swarm_res[nb_eval_particle][lipid_code][temp]['time_elapsed_str'] = jobs_stats[job_name]['time_elapsed_str']

        checked_file_path = f"{ns.user_config['exec_folder']}/{config.iteration_sim_files_dirname}{nb_eval_particle}/{lipid_code}_{temp}/prod.gro"
        swarm_res[nb_eval_particle][lipid_code][temp]['status'] = 'failure'
        if os.path.isfile(checked_file_path):
            swarm_res[nb_eval_particle][lipid_code][temp]['status'] = 'success'

    p_sim_status = []  # for multiprocessing pool in block 4.
    for nb_eval_particle, lipid_code, temp in zip(p_nb_eval_particle, p_lipid_code, p_temp):
        p_sim_status.append(swarm_res[nb_eval_particle][lipid_code][temp]['status'])

    delta_ts_start_jobs_statuses = datetime.now().timestamp() - ts_start_jobs_statuses
    delta_ts_start_jobs_statuses_min = round(delta_ts_start_jobs_statuses, 1)
    print(f'Wall time for getting all jobs statuses from SLURM accounting: {delta_ts_start_jobs_statuses_min} sec')

    # 5. perform the analysis of EACH JOB in OpenMP parallel in the master node
    #    here we are analyzing separately EACH (AND ALL) JOB THAT BELONG TO EACH (AND ALL) PARTICLES OF THE SWARM
    ts_start_jobs_analysis = datetime.now().timestamp()

    # multiprocessing pool for OpenMP calculation of all the geoms/rdfs at the end of all the GROMACS calculations
    # that happen within a swarm iteration (i.e. for all particles)
    # NOTE: if using a HPC, the number of processes should be set to the number of booked cores (taking hyperthreading into
    # account, if enabled), and if using a SUPSI machine, the number of processes should amount to the sum of all cores
    # available in all the 'GPU slots' (as we call them) that have been selected for this optimization run (i.e. if you book
    # 3 GPU slots with each -nt 9 and each a different GPU unit, then here you can select number of processes = 3*9 = 27)
    with multiprocessing.Pool(processes=ns.user_config['nb_cores_analysis']) as pool:

        p_args = zip(repeat(ns), p_nb_eval_particle, p_lipid_code, p_temp, p_sim_status, p_updated_cg_itps)
        p_res = pool.starmap(get_score_and_plot_graphs_for_single_job, p_args)
        p_score_part, p_geoms_time, p_rdfs_time, p_error_data = list(map(list, zip(*p_res)))

    for nb_eval_particle, lipid_code, temp, score_part, geoms_time, rdfs_time, error_data \
            in zip(p_nb_eval_particle, p_lipid_code, p_temp, p_score_part, p_geoms_time, p_rdfs_time, p_error_data):
        swarm_res[nb_eval_particle][lipid_code][temp]['score_part'] = score_part
        swarm_res[nb_eval_particle][lipid_code][temp]['geoms_time'] = geoms_time
        swarm_res[nb_eval_particle][lipid_code][temp]['rdfs_time'] = rdfs_time
        swarm_res[nb_eval_particle][lipid_code][temp]['error_data'] = error_data

    delta_ts_start_jobs_analysis = datetime.now().timestamp() - ts_start_jobs_analysis
    delta_ts_start_jobs_analysis_min = round(delta_ts_start_jobs_analysis / 60, 2)
    delta_ts_start_jobs_analysis_h = round(delta_ts_start_jobs_analysis / (60 * 60), 3)
    print(f'Wall time for performing all jobs analysis on the master node: {delta_ts_start_jobs_analysis_min} min ({delta_ts_start_jobs_analysis_h} hours)\n')

    # 6. get score for EACH PARTICLE OF THE SWARM
    eval_scores = []  # to be returned to FST-PSO at the end of each swarm iteration for deciding next moves
    eval_scores_details = {}  # for writting our logs
    for i, nb_eval_particle in enumerate(nb_eval_particles_range_over_complete_opti):
        # print('swarm_res[nb_eval] for nb_eval_particle:', nb_eval_particle, '\n\n', swarm_res[nb_eval_particle], '\n')
        eval_score = get_particle_score(ns, nb_eval_particle, swarm_res)
        eval_scores.append(eval_score)
        eval_scores_details[nb_eval_particle] = {
            'eval_score': eval_score,
            'parameters_set_rounded': parameters_sets_rounded[i]
        }

    # 7. write the log std output
    for nb_eval_particle in nb_eval_particles_range_over_complete_opti:

        # note best score -- these are actually never used because we have a single opti cycle atm
        if eval_scores_details[nb_eval_particle]['eval_score'] < ns.best_fitness[0]:
            # err_dict = [err_bonds, err_angles, err_sig, err_eps] # keys for each dict: geom_grp (B), geom_grp (A), bead_type, pair_type
            err_dict = None
            ns.best_fitness = eval_scores_details[nb_eval_particle]['eval_score'], nb_eval_particle, err_dict

        print('*******************************************************************')
        print_stdout_forced()
        print_stdout_forced(f"SWARM ITER {ns.n_swarm_iter} -- PARTICLE {nb_eval_particle}     (equi runs: {ns.user_config['cg_time_equi']} ns / prod runs: {ns.prod_sim_time} ns)")
        print_stdout_forced('\nRounded parameters:', ['{:.3f}'.format(param) for param in eval_scores_details[nb_eval_particle]['parameters_set_rounded']])
        print()

        for lipid_code in sorted(ns.user_config['lipids_codes']):
            for temp in ns.user_config['lipids_codes'][lipid_code]:

                # write stdout + logs for each job (job = a simulation that has been performed as part of a particle)
                print_job_output(ns, swarm_res, nb_eval_particle, lipid_code, temp)

        print(f"SWARM ITER {ns.n_swarm_iter} -- PARTICLE {nb_eval_particle} -- SCORE: {round(eval_scores_details[nb_eval_particle]['eval_score'], 4)}")
        # TODO: bring back the monitoring of total opti time, but this can wait
        # ns.total_eval_time += current_eval_time
        # TODO: try to bring this back in a meaningful way (now the geoms/rdfs are done via multiprocessing on the Master), but it's not urgent
        # print_stdout_forced('Particle total time:', current_eval_time, 'min')
        print()

        # results log -- particle score + parameters
        eval_score = eval_scores_details[nb_eval_particle]['eval_score']
        parameters_set_rounded = ['{:.3f}'.format(param) for param in eval_scores_details[nb_eval_particle]['parameters_set_rounded']]
        with open(f"{ns.user_config['exec_folder']}/{ns.opti_moves_file}", 'a') as fp:
            fp.write(f'{ns.n_cycle} {ns.n_swarm_iter} {nb_eval_particle} {round(eval_score, 4)} ' + ' '.join(parameters_set_rounded) + '\n')

    delta_ts_swarm_iter = datetime.now().timestamp() - ts_start_swarm_iter
    delta_ts_swarm_iter_min = round(delta_ts_swarm_iter / 60, 2)
    delta_ts_swarm_iter_h = round(delta_ts_swarm_iter / (60 * 60), 3)
    ns.all_delta_ts_swarm_iter.append(delta_ts_swarm_iter)
    min_delta_ts_swarm_iter_h = round(min(ns.all_delta_ts_swarm_iter) / (60 * 60), 3)
    max_delta_ts_swarm_iter_h = round(max(ns.all_delta_ts_swarm_iter) / (60 * 60), 3)
    avg_delta_ts_swarm_iter_h = round(mean(ns.all_delta_ts_swarm_iter) / (60 * 60), 3)
    print('-------------------------------------------------------------------\n')
    print_stdout_forced(f'SWARM ITERATION WALL CLOCK TIME: {delta_ts_swarm_iter_min} min ({delta_ts_swarm_iter_h} h)')
    print(f'Average (updated) swarm iteration wall clock time: {avg_delta_ts_swarm_iter_h} h (range: {min_delta_ts_swarm_iter_h} h to {max_delta_ts_swarm_iter_h} h)\n')

    # write timings in log
    with open(f"{ns.user_config['exec_folder']}/{ns.opti_moves_times_file}", 'a') as fp:
        # master elapsed and remaining time are calculated at the very beginning of the swarm iteration, NOT at the end
        fp.write(f'{ns.n_cycle} {ns.n_swarm_iter} {round(delta_ts_swarm_iter)} {delta_ts_swarm_iter_h} {delta_ts_start_jobs_execution_h} {delta_ts_start_jobs_analysis_h} {ts_master_elapsed_h} {delta_ts_master_remaining_h}\n')

    # update the max SWARM iteration time for stopping the MASTER accordingly
    # (do NOT start a new SWARM iteration if the MASTER won't have time to finish it)
    if delta_ts_swarm_iter > ns.max_delta_ts_swarm_iter:
        ns.max_delta_ts_swarm_iter = delta_ts_swarm_iter
    ns.n_swarm_iter += 1

    return eval_scores


