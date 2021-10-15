import os, sys, time
import subprocess
from shared.context_managers import working_dir
from datetime import datetime

# TODO: check the average computation time during the opti process and submit longer jobs first, this should allow
#       some gain overall (because NOT all simulations have the same runtime, in particular because of domain decomposition)

class HPCJobs:
    """
    A job is indexed by job_name and has attributes:
        - job_id (int, attributed when the job is submitted to SLURM)
        - status: master_pending, slurm_pending, slurm_running, finished (finished does NOT mean it successfully finished)
        - exec_dir: directory in which the .sh file to be sbatch-submitted to SLURM will be written and executed from
        - last_log_write_time: timestamp (int)
    """

    # TODO: 28-04-2021 I notice that because of a bug in SLURM (most probably), it is possible that a job is left hanging
    #       with status "COMPLETING" and will actually never die --> HANDLE THIS SHITTY CASE

    # NOTE: do NOT reduce argument 'kill_delay'

    def __init__(self, hpc_username, jobs, hpc_nb_slots, slurm_single_job_config, kill_delay=600, gmx_path='gmx_mpi'):
        self.hpc_username = hpc_username
        self.jobs = jobs
        for job_name in jobs:
            self.jobs[job_name]['status'] = 'master_pending'
            self.jobs[job_name]['logs'] = {
                'ts_last_write': None,  # for the stuck jobs killing after delay
                'mini_byte_size': 0,
                'equi_byte_size': 0,
                'prod_byte_size': 0
            }
        self.hpc_nb_slots = int(hpc_nb_slots)
        self.slurm_single_job_config = slurm_single_job_config  # TODO: check that the provided SLURM arguments are all valid
        self.kill_delay = kill_delay  # number of seconds after which we kill a job that has NOT been writting in its log files
        self.gmx_path = gmx_path

        self.nb_jobs_master_pending = len(jobs)
        self.nb_jobs_slurm_all = 0
        self.nb_jobs_slurm_pending = 0
        self.nb_jobs_slurm_running = 0
        self.finished = False
        self.job_priority = ['SDPC', 'PDPC', 'DOPC', 'POPC', 'DPPC', 'DMPC', 'DLPC']

    def check_slurm_queue(self):
        self.nb_jobs_slurm_pending = 0
        self.nb_jobs_slurm_running = 0
        self.nb_jobs_slurm_all = 0

        # TODO: IMPORTANT !! -- it seems that the command squeue might not always correctly respond, and this triggers
        #       finishing the swarm iteration while only some of the jobs have processed, while many others actually
        #       did NOT and will be considered FAILURES -- SO THIS FUCKS UP EVERYTHING !!!!

        # first check the queue of the HPC to find our waiting/running jobs
        squeue_stdout = subprocess.run(f'squeue -u {self.hpc_username} -o "%.30j %.30M"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()
        lines = squeue_stdout.split('\n')[1:]  # split lines and remove the header from the returned stdout

        for line in lines:
            job_name = line[:30].strip()
            job_time = line[31:].strip()

            # check only jobs that are children of the Master (NOT the Master and NOT other unrelated jobs)
            # this is safe because the job name contains the #SBATCH --job-name given to the master
            if job_name in self.jobs:
                if job_time == '0:00':  # if time is 0:00 then it has been submitted and it's pending
                    self.jobs[job_name]['status'] = 'slurm_pending'
                    self.nb_jobs_slurm_pending += 1
                else:  # if time is not 0:00 then it's running
                    self.jobs[job_name]['status'] = 'running'
                    self.nb_jobs_slurm_running += 1

        # check which jobs finished already + if some are bugged/stuck
        for job_name in self.jobs:
            if self.jobs[job_name]['status'] == 'running':
                with working_dir(self.jobs[job_name]['job_exec_dir']):

                    # check if job finished
                    if os.path.isfile('prod.gro'):
                        self.jobs[job_name]['status'] = 'finished'
                        # send a SIGKILL also at job completion, because it seems jobs sometimes
                        # continue up to the time limit and waste resources, for reasons I ignore atm
                        self.kill_slurm_job(job_name=job_name)

                    # check if we need to kill
                    # TODO: this part about checking for stuck jobs and killing them is NOT well tested
                    #       actually if too many jobs are allowed to start (when Daint is lightly used
                    #       and we could actually run like crazy) + the kill_delay is small (60 sec)
                    #       then it behaves badly and will kill jobs that are running OK !!
                    else:
                        if self.jobs[job_name]['logs']['ts_last_write'] is None:  # job just started
                            self.jobs[job_name]['logs']['ts_last_write'] = datetime.now().timestamp()
                        else:
                            for sim_step in ['mini', 'equi', 'prod']:
                                if os.path.isfile(f'{sim_step}.log'):
                                    byte_size = os.path.getsize(f'{sim_step}.log')
                                    if byte_size > self.jobs[job_name]['logs'][f'{sim_step}_byte_size']:
                                        self.jobs[job_name]['logs'][f'{sim_step}_byte_size'] = byte_size
                                        self.jobs[job_name]['logs']['ts_last_write'] = datetime.now().timestamp()
                        if datetime.now().timestamp() > self.jobs[job_name]['logs']['ts_last_write'] + self.kill_delay:
                            self.kill_slurm_job(job_name=job_name)
                            print(f'  --> Killing {job_name} because it seems stuck (simulation may have crashed by itself already)')
                            self.jobs[job_name]['status'] = 'finished'
                            # NOTE: here we do NOT update self.nb_jobs_slurm_running because the job may
                            #       take 10-20 sec to die, so we respect the max amount of SLURM slots
                            #       specified by NOT updating this value and letting the next check figure it out

        self.nb_jobs_slurm_all = self.nb_jobs_slurm_pending + self.nb_jobs_slurm_running
        print(f"  Jobs status on {time.strftime('%d-%m-%Y at %H:%M:%S')} -- Master Pending: {self.nb_jobs_master_pending} -- Slurm Pending: {self.nb_jobs_slurm_pending} -- Slurm Running: {self.nb_jobs_slurm_running}")

    def submit_slurm_jobs(self):
        for job_prio in self.job_priority:
            for job_name in self.jobs:
                if job_prio in job_name:
                    if self.jobs[job_name]['status'] == 'master_pending':
                        if self.nb_jobs_slurm_all < self.hpc_nb_slots:  # limit number of jobs either in the HPC SLURM queue or running

                            with working_dir(self.jobs[job_name]['job_exec_dir']):
                                self._write_slurm_bash_file(job_name)
                                sbatch_stdout = subprocess.run(f'sbatch run_{job_name}.sh', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()

                            if sbatch_stdout.startswith('Submitted batch job'):
                                job_id = int(sbatch_stdout.strip().split()[3])
                                self.jobs[job_name]['job_id'] = job_id
                                self.jobs[job_name]['status'] = 'slurm_pending'
                                self.nb_jobs_slurm_pending += 1
                                self.nb_jobs_slurm_all += 1
                                self.nb_jobs_master_pending -= 1
                                # print('Submitted JOB NAME', job_name, 'with JOB ID', job_id)
                            else:

                                # TODO: if a job submission failed, MAYBE stop the complete optimization process
                                #       this should never happen in principle, so it would mean that a SLURM argument was
                                #       inadequate (it can also be the account of the user that is incorrectly provided)
                                # TODO (UPDATE): it seems that jobs actually correctly get queued even when cmd stdout
                                #                reported failure such as:
                                #   - ERROR: invalid partition "whatever partition" requested
                                #   - ERROR: sbatch: error: Batch job submission failed: Socket timed out on send/recv operation

                                print('WARNING: Attempt to SBATCH submit JOB NAME', job_name, 'seems to have failed, with error:\n ', sbatch_stdout)

                                # TODO: try to understand if the following is enough for robustness (atm I don't want to try
                                #       sbatch'ing multiple times)
                                n_tries = 10
                                job_ok = False
                                while n_tries > 0 and job_ok is False:
                                    time.sleep(10)  # wait a bit because we don't know wtf is happening, in fact
                                    sacct_stdout = subprocess.run(f'sacct -n -X --format jobid --name {job_name}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()
                                    try:
                                        job_id = int(sacct_stdout.strip())
                                        self.jobs[job_name]['job_id'] = job_id
                                        self.jobs[job_name]['status'] = 'slurm_pending'
                                        self.nb_jobs_slurm_pending += 1
                                        self.nb_jobs_slurm_all += 1
                                        self.nb_jobs_master_pending -= 1
                                        job_ok = True
                                    except ValueError:  # if something else than an integer job id is returned
                                        pass
                                    n_tries -= 1

                                if job_ok is False:  # we will kill all jobs and exit
                                    print('  --> FAILURE TO START JOB, KILLING EVERYTHING\n')
                                    for job_name in self.jobs:
                                        subprocess.run(f'scancel --name {job_name}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                                    sys.exit('  --> KILLING MASTER NOW\n')
                                else:
                                    print('  --> The job submission finally went through, continuing\n')

    def get_stats(self):
        jobs_stats = {}
        for job_name in self.jobs:
            job_id = self.jobs[job_name]['job_id']
            sacct_stdout = subprocess.run(f'sacct -j {job_id} -o start,end,elapsed -X', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()
            line = sacct_stdout.split('\n')[2]  # get the only line we are interested in
            time_start_str = line[:20].strip()
            time_end_str = line[20:39].strip()
            time_elapsed_str = line[40:51].strip()
            jobs_stats[job_name] = {
                'time_start_str': time_start_str,
                'time_end_str': time_end_str,
                'time_elapsed_str': time_elapsed_str,
            }
        return jobs_stats

    def check_completion(self):
        if self.nb_jobs_master_pending == 0 and self.nb_jobs_slurm_all == 0:
            self.finished = True

    def get_master_time(self, master_job_name):
        sacct_stdout = subprocess.run(f'sacct --name {master_job_name} --format="JobIDRaw,Elapsed,Timelimit,state" -X', shell=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT).stdout.decode()
        # get the only line we are interested in
        lines = sacct_stdout.split('\n')[2:]  # all non-header lines, as there might be several ones
        # print(master_job_name)
        # print(lines)
        master_job_id, ts_master_elapsed, ts_master_total = 0, None, None

        for line in lines:  # make sure to select the master that is running, and not one that would be pending with dependency
            if line != '':
                job_id = int(line[:12].strip())
                ts_elapsed = self._get_sec_from_slurm_time_str(line[13:23])
                ts_total = self._get_sec_from_slurm_time_str(line[24:34])
                job_state = line[35:45].strip()
                if job_state.startswith('RUNNING'):  # other masters queued with dependencies are 'PENDING'
                    master_job_id = job_id
                    ts_master_elapsed = ts_elapsed
                    ts_master_total = ts_total

        return master_job_id, ts_master_elapsed, ts_master_total

    def _get_sec_from_slurm_time_str(self, time_str):
        hours, minutes, seconds = time_str.strip().split(':')
        days = 0
        if '-' in hours:  # if the master time is over 24h, there might be a prefix: nb of days and a dash
            days, hours = hours.split('-')
        return int(days) * (24 * 60 * 60) + int(hours) * (60 * 60) + int(minutes) * 60 + int(seconds)

    def _write_slurm_bash_file(self, job_name):
        # TODO: make all of this include all useful arguments, just like the MDP writter does
        #       also check if an argument is provided before writting the given line
        #       also load a list of given modules
        # TODO: make sure that the --job_name arg is not provided as part of slurm_single_job_config
        #       this would be an easy mistake, notably for us
        file_name = f'run_{job_name}.sh'
        with open(file_name, 'w') as fp:
            fp.write(
                    f"#!/bin/bash -l\n"
                    f"\n#SBATCH --job-name={job_name}"
            )
            for arg_sbatch in self.slurm_single_job_config['sbatch']:
                fp.write(f"\n#SBATCH --{arg_sbatch}={self.slurm_single_job_config['sbatch'][arg_sbatch]}")
            fp.write(f"\n")
            for env_module in self.slurm_single_job_config['env_modules']:
                fp.write(f"\nmodule load {env_module}")
            fp.write(f"\n")
            for bash_export in self.slurm_single_job_config['bash_exports']:
                fp.write(f"\nexport {bash_export}")
            fp.write(f"\n")

            # prepare files and run mini/equi/prod
            # NOTE: here we have 2 types of calls: 'gmx' and 'srun gmx_mpi' that are used for grompp and mdrun
            # TODO: for WET -rdd should be 1.5 adapted automatically (actually, put +0.2 with respect to cut-off used in MDPs)
            # TODO: make something for options -bonded and -nb (non-bonded)
            fp.write(
                f'''\ngmx grompp -c start_frame.gro -p system.top -f mini.mdp -n index.ndx -o mini -maxwarn 1'''  # SETUP MINI
                f'''\nif test -f "mini.tpr"; then'''  # IF MINI SETUP IS OK
                f'''\n  srun {self.gmx_path} mdrun -deffnm mini -ntomp $OMP_NUM_THREADS -pin on'''  # RUN MINI
                f'''\n  if test -f "mini.gro"; then'''  # IF MINI FINISHED CORRECTLY
                f'''\n    gmx grompp -c mini.gro -p system.top -f equi.mdp -n index.ndx -o equi'''  # SETUP EQUI
                f'''\n    if test -f "equi.tpr"; then'''  # IF EQUI SETUP IS OK
                f'''\n      sleep $[ ( $RANDOM % 11 )  + 5 ]s'''  # sleep a little in case all jobs would start at the same time
                # f'''\n      srun {self.gmx_path} mdrun -deffnm equi -ntomp $OMP_NUM_THREADS -pin on -dlb no -dd 2 2 2 -rdd 1.8 -bonded cpu -nb cpu'''  # RUN EQUI
                # f'''\n      srun {self.gmx_path} mdrun -deffnm equi -ntomp $OMP_NUM_THREADS -pin on -dlb no -dd 2 2 2 -rdd 1.8 -bonded cpu -nb cpu'''  # RUN EQUI
                f'''\n      srun {self.gmx_path} mdrun -deffnm equi -ntomp $OMP_NUM_THREADS -pin on -dlb no -dd 1 1 4  -rdd 1.8 -bonded cpu -nb cpu'''  # 8BEADS
                f'''\n      if test -f "equi.gro"; then'''  # IF EQUI FINISHED CORRECTLY
                f'''\n        gmx grompp -c equi.gro -p system.top -f prod.mdp -n index.ndx -o prod'''  # SETUP PROD
                f'''\n        if test -f "prod.tpr"; then'''  # IF PROD SETUP IS OK
                f'''\n          sleep $[ ( $RANDOM % 11 )  + 5 ]s'''  # sleep a little in case all jobs would start at the same time
                # f'''\n          srun {self.gmx_path} mdrun -deffnm prod -ntomp $OMP_NUM_THREADS -pin on -dlb no -dd 2 2 2 -rdd 1.8 -bonded cpu -nb cpu'''  # RUN PROD
                # f'''\n          srun {self.gmx_path} mdrun -deffnm prod -ntomp $OMP_NUM_THREADS -pin on -dlb no -dd 2 2 2 -rdd 1.8 -bonded cpu -nb cpu'''  # RUN PROD
                f'''\n          srun {self.gmx_path} mdrun -deffnm prod -ntomp $OMP_NUM_THREADS -pin on -dlb no -dd 1 1 4  -rdd 1.8 -bonded cpu -nb cpu'''  # 8BEADS
                f'''\n        fi'''
                f'''\n      fi'''
                f'''\n    fi'''
                f'''\n  fi'''
                f'''\nfi'''
                f'''\n\nexit'''
            )

    def kill_slurm_job(self, job_name=None, job_id=None):
        # no need to check for one or the other, we can execute these commands without nasty effects
        if job_name is not None:
            subprocess.run(f'scancel -s 9 -f -H --jobname {job_name}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if job_id is not None:
            subprocess.run(f'scancel -s 9 -f -H {job_id}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self, squeue_check_freq=30):
        self.check_slurm_queue()
        self.submit_slurm_jobs()
        self.check_completion()
        time.sleep(squeue_check_freq)  # seconds
