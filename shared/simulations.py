import os
import sys
import signal
import subprocess
import time
from datetime import datetime
from shared.context_managers import working_dir
import config


# build gromacs command with arguments
def gmx_args(gmx_cmd, nb_threads, gpu_id, gmx_additional_args_str, gmx_gpu_cancel_str):
    if int(nb_threads) > 0:
        gmx_cmd += f" -nt {nb_threads}"
    if gpu_id == 'X':
        gmx_cmd += f" {gmx_gpu_cancel_str}"
    else:
        gmx_cmd += f" -gpu_id {gpu_id}"
    gmx_cmd += f" {gmx_additional_args_str}"
    return gmx_cmd


def run_sims(ns, slot_nt, slot_gpu_id):

    # start from frames provided in directory START_CG_SETUPS, and we will do mini/equi/prod
    gmx_start = datetime.now().timestamp()
    starting_frame = 'start_frame.gro'

    # grompp -- MINI
    gmx_cmd = f"{ns.user_config['gmx_path']} grompp -c {starting_frame} -p system.top -f mini.mdp -o mini -maxwarn 2"
    gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gmx_out = gmx_process.communicate()[1].decode()

    if os.path.isfile('mini.tpr'):
        # mdrun -- MINI
        gmx_cmd = gmx_args(f"{ns.user_config['gmx_path']} mdrun -deffnm mini", slot_nt, slot_gpu_id, ns.user_config['gmx_mini_additional_args_str'], ns.user_config['gmx_gpu_cancel_str'])
        gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)  # create a process group for the MINI run

        # check if MINI run is stuck because of instabilities
        cycles_check = 0
        last_log_file_size = 0
        while gmx_process.poll() is None:  # while process is alive
            time.sleep(ns.process_alive_time_sleep)
            cycles_check += 1

            if cycles_check % ns.process_alive_nb_cycles_dead == 0:  # every minute or so, kill process if we determine it is stuck because the .log file's bytes size has not changed
                if os.path.isfile('mini.log'):
                    log_file_size = os.path.getsize(
                        'mini.log')  # get size of .log file in bytes, as a mean of detecting the MINI run is stuck
                else:
                    log_file_size = last_log_file_size  # MINI is stuck if the process was not able to create log file at start
                if log_file_size == last_log_file_size:  # MINI is stuck if the process is not writing to log file anymore
                    os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
                    # sim_status = 'Minimization run failed (stuck simulation was killed)'
                else:
                    last_log_file_size = log_file_size
    else:
        # pass
        # sim_status = 'Minimization run failed (simulation crashed)'
        print('\n\n'+config.header_gmx_error+gmx_out+'\n'+config.header_error+'Gmx grompp failed at the MINIMIZATION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')
        sys.exit('\n\n'+config.header_gmx_error+gmx_out+'\n'+config.header_error+'Gmx grompp failed at the MINIMIZATION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    # if MINI finished properly, we just check for the .gro file printed in the end
    if os.path.isfile('mini.gro'):

        # grompp -- EQUI
        gmx_cmd = f"{ns.user_config['gmx_path']} grompp -c mini.gro -p system.top -f equi.mdp -n index.ndx -o equi -maxwarn 2"
        gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gmx_out = gmx_process.communicate()[1].decode()

        if os.path.isfile('equi.tpr'):
            # mdrun -- EQUI
            gmx_cmd = gmx_args(f"{ns.user_config['gmx_path']} mdrun -deffnm equi", slot_nt, slot_gpu_id, ns.user_config['gmx_equi_additional_args_str'], ns.user_config['gmx_gpu_cancel_str'])
            gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)  # create a process group for the EQUI run

            # check if EQUI run is stuck because of instabilities
            cycles_check = 0
            last_log_file_size = 0
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(ns.process_alive_time_sleep)
                cycles_check += 1

                if cycles_check % ns.process_alive_nb_cycles_dead == 0:  # every minute or so, kill process if we determine it is stuck because the .log file's bytes size has not changed
                    if os.path.isfile('equi.log'):
                        log_file_size = os.path.getsize(
                            'equi.log')  # get size of .log file in bytes, as a mean of detecting the EQUI run is stuck
                    else:
                        log_file_size = last_log_file_size  # EQUI is stuck if the process was not able to create log file at start
                    if log_file_size == last_log_file_size:  # EQUI is stuck if the process is not writing to log file anymore
                        os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
                        # sim_status = 'Equilibration run failed (stuck simulation was killed)'
                    else:
                        last_log_file_size = log_file_size
        else:
            pass
            # sim_status = 'Equilibration run failed (simulation crashed)'
            # sys.exit('\n\n'+config.header_gmx_error+gmx_out+'\n'+config.header_error+'Gmx grompp failed at the EQUILIBRATION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    # if EQUI finished properly, we just check for the .gro file printed in the end
    if os.path.isfile('equi.gro'):

        # grompp -- PROD
        gmx_cmd = f"{ns.user_config['gmx_path']} grompp -c equi.gro -p system.top -f prod.mdp -n index.ndx -o prod -maxwarn 2"
        gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gmx_out = gmx_process.communicate()[1].decode()

        if os.path.isfile('prod.tpr'):
            # mdrun -- PROD
            gmx_cmd = gmx_args(f"{ns.user_config['gmx_path']} mdrun -deffnm prod", slot_nt, slot_gpu_id, ns.user_config['gmx_prod_additional_args_str'], ns.user_config['gmx_gpu_cancel_str'])
            gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           preexec_fn=os.setsid)  # create a process group for the MD run

            # check if PROD run is stuck because of instabilities
            cycles_check = 0
            last_log_file_size = 0
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(ns.process_alive_time_sleep)
                cycles_check += 1

                if cycles_check % ns.process_alive_nb_cycles_dead == 0:  # every minute or so, kill process if we determine it is stuck because the .log file's bytes size has not changed
                    if os.path.isfile('prod.log'):
                        log_file_size = os.path.getsize(
                            'prod.log')  # get size of .log file in bytes, as a mean of detecting the MD run is stuck
                    else:
                        log_file_size = last_log_file_size  # MD run is stuck if the process was not able to create log file at start
                    if log_file_size == last_log_file_size:  # MD run is stuck if the process is not writing to log file anymore
                        os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
                        # sim_status = 'MD run failed (stuck simulation was killed)'
                    else:
                        last_log_file_size = log_file_size
        else:
            pass
            # sim_status = 'MD run failed (simulation crashed)'
            # sys.exit('\n\n'+config.header_gmx_error+gmx_out+'\n'+config.header_error+'Gmx grompp failed at the PRODUCTION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    gmx_time = datetime.now().timestamp() - gmx_start

    return gmx_time


# for making a shared multiprocessing.Array() to handle slots states when running simulations in LOCAL (= NOT HPC)
def init_process(arg):
    global g_slots_states
    g_slots_states = arg


def run_parallel(ns, job_exec_dir, nb_eval_particle, lipid_code, temp):
    while True:
        time.sleep(1)
        for i in range(len(g_slots_states)):
            if g_slots_states[i] == 1:  # if slot is available

                g_slots_states[i] = 0  # mark slot as busy
                print(f'  Starting simulation for particle {nb_eval_particle} {lipid_code} {temp} on slot {i + 1}')
                slot_nt = ns.slots_nts[i]
                slot_gpu_id = ns.slots_gpu_ids[i]
                # print(f'  Slot uses -nt {slot_nt} and -gpu_id {slot_gpu_id} and in directory {job_exec_dir}')
                with working_dir(job_exec_dir):
                    gmx_time = run_sims(ns, slot_nt, slot_gpu_id)
                g_slots_states[i] = 1  # mark slot as available
                # print(f'Finished simulation for particle {nb_eval_particle} with {lipid_code} {temp} on slot {i + 1}')

                time_start_str, time_end_str = '', ''  # NOTE: this is NOT displayed anywhere atm & we don't care much
                time_elapsed_str = time.strftime('%H:%M:%S', time.gmtime(round(gmx_time)))

                return time_start_str, time_end_str, time_elapsed_str




