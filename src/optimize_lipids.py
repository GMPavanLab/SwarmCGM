#!/usr/bin/env python3

import os, sys, re, shutil, subprocess, time, copy, contextlib, pickle, itertools, warnings

warnings.filterwarnings("ignore")
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote
from fstpso import FuzzyPSO
import numpy as np
from datetime import datetime
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from shared.io import check_log_files, backup_swarm_iter_logs_and_checkpoint


import config
from opti_CG import par_wrap, load_aa_data, read_cg_itp_file_grp_comments, \
    get_search_space_boundaries, get_initial_guess_list, create_bins_and_dist_matrices, read_ndx_atoms2beads, \
    make_aa_traj_whole_for_selected_mols, get_AA_bonds_distrib, get_AA_angles_distrib, \
    get_atoms_weights_in_beads, get_beads_MDA_atomgroups, set_MDA_backend, initialize_cg_traj, \
    map_aa2cg_traj, get_cycle_restart_guess_list
from shared.eval_func_parallel import eval_function_parallel_swarm  # new implementation of parallelization
from shared.io import update_cg_itp_obj

# import matplotlib
# matplotlib.use('TkAgg') # for interactive plotting (not to file)
import matplotlib.pyplot as plt

# filter MDAnalysis (I think, maybe it actually comes from my code) + numpy deprecation stuff that is annoying
from numpy import VisibleDeprecationWarning

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

#####################################
# ARGUMENTS HANDLING / HELP DISPLAY #
#####################################

# print(header_package('                    Module: CG model optimization\n'))

args_parser = ArgumentParser(description='''\
Script used for the optimization of lipids according to AA reference membranes simulations.''',
                             formatter_class=lambda prog: RawTextHelpFormatter(prog, width=135,
                                                                               max_help_position=52),
                             add_help=False, usage=SUPPRESS)

req_args_header = config.sep_close + '\n|                                     REQUIRED ARGUMENTS                                      |\n' + config.sep_close
opt_args_header = config.sep_close + '\n|                                     OPTIONAL ARGUMENTS                                      |\n' + config.sep_close

# bullet = '❭'
# bullet = '★'
# bullet = '|'
bullet = ' '

required_args = args_parser.add_argument_group(bullet + 'REFERENCE DATA')
# required_args.add_argument('-lipids', dest='str_lipids', help='String of codes of lipids to use, for example: \'POPC DOPC\'', type=str, required=True, metavar='')
required_args.add_argument('-mapping', dest='mapping_type',
                           help='Mapping type, either: MARTINI2, MARTINI3, SIRAH, OPTION5, SUPERCG, etc.', type=str,
                           required=True, metavar='')
required_args.add_argument('-pos', dest='map_center', help='Mapped CG particles positioning, either: COM or COG',
                           type=str, required=True, metavar='')
required_args.add_argument('-solv', dest='solv', help='Presence of explicit solvent, either: WET or DRY', type=str,
                           required=True, metavar='')
required_args.add_argument('-reset', dest='reset',
                           help='Do NOT use stored ref. AA data and re-generate them from the AA trajs (takes no argument)',
                           action='store_true', default=False)
required_args.add_argument('-next_cycle', dest='next_cycle',
                           help='If checkpoints are present, restart from checkpoint in the next planned optimization cycle (takes no argument)',
                           action='store_true', default=False)
required_args.add_argument('-aa_ff', dest='aa_ff', help='Reference AA force field: slipids, charmm36u, etc.',
                           type=str, default='slipids', metavar='   (slipids)')

optional_args1 = args_parser.add_argument_group(bullet + 'GROMACS SETTINGS')
optional_args1.add_argument('-gmx', dest='gmx_path', help=config.help_gmx_path, type=str, default=config.gmx_path,
                            metavar='                  ' + par_wrap(config.gmx_path))
optional_args1.add_argument('-nb_cores_analysis', dest='nb_cores_analysis', help='Number of cores to use on the Master node for analysis of the simulations\nRequired for either HPC or SUPSI machines', type=int,
                            metavar='', required=True)
optional_args1.add_argument('-master_job_name', dest='master_job_name',
                            help='Job name of the Master process in the HPC SLURM queue', type=str, default='MASTER', metavar='MASTER')
optional_args1.add_argument('-nb_hpc_slots', dest='nb_hpc_slots',
                            help='Max nb of slots in the SLURM HPC queue, used for full swarm parallelization (triggers IGNORING -nt and -gpu_id)\nThis can be equal to the number of nodes to use, or not.\nse the SLURM config accordingly.',
                            type=int, default='0', metavar="                     '0'")
optional_args1.add_argument('-nt', dest='nb_threads',
                            help='String (use quotes) space-separated list of number of threads to use for each slot (forwarded to gmx mdrun -nt)',
                            type=str, default='0', metavar="                     '0'")
optional_args1.add_argument('-gpu_id', dest='gpu_id',
                            help='String (use quotes) space-separated list of GPU device IDs for each slot',
                            type=str, default='', metavar='')
optional_args1.add_argument('-sim_kill_delay', dest='sim_kill_delay',
                            help='Time (s) after which to kill a simulation that has not been\nwriting into its log file, in case a simulation gets stuck',
                            type=int, default=30, metavar='        (30)')

optional_args5 = args_parser.add_argument_group(bullet + 'CG MODEL SCORING')
optional_args5.add_argument('-b2a_score_fact', dest='bonds2angles_scoring_factor',
                            help='C constant of the eval score', type=float,
                            default=config.bonds2angles_scoring_factor,
                            metavar='       ' + par_wrap(config.bonds2angles_scoring_factor))
# optional_args5.add_argument('-g2a_score_fact', dest='geoms2apl_scoring_factor', help='G constant of the eval score', type=float, default=config.geoms2apl_scoring_factor, metavar='     '+par_wrap(config.geoms2apl_scoring_factor))
# optional_args5.add_argument('-t2a_score_fact', dest='thickness2apl_scoring_factor', help='K constant of the eval score', type=float, default=config.thickness2apl_scoring_factor, metavar='       '+par_wrap(config.thickness2apl_scoring_factor))

# optional_args5.add_argument('-r2a_score_fact', dest='rdfs2apl_scoring_factor', help='RDFs constant of the eval score', type=float, default=config.rdfs2apl_scoring_factor, metavar='       '+par_wrap(config.rdfs2apl_scoring_factor))
optional_args5.add_argument('-bw_constraints', dest='bw_constraints', help=config.help_bw_constraints, type=float,
                            default=config.bw_constraints, metavar='    ' + par_wrap(config.bw_constraints))
optional_args5.add_argument('-bw_bonds', dest='bw_bonds', help=config.help_bw_bonds, type=float,
                            default=config.bw_bonds, metavar='           ' + par_wrap(config.bw_bonds))
optional_args5.add_argument('-bw_angles', dest='bw_angles', help=config.help_bw_angles, type=float,
                            default=config.bw_angles, metavar='              ' + par_wrap(config.bw_angles))
optional_args5.add_argument('-bw_dihedrals', dest='bw_dihedrals', help=config.help_bw_dihedrals, type=float,
                            default=config.bw_dihedrals, metavar='           ' + par_wrap(config.bw_dihedrals))
optional_args5.add_argument('-bonds_max_range', dest='bonded_max_range', help=config.help_bonds_max_range,
                            type=float, default=config.bonds_max_range,
                            metavar='        ' + par_wrap(config.bonds_max_range))
optional_args5.add_argument('-bw_rdfs', dest='bw_rdfs', help='Bandwidth for radial distribution functions (nm)',
                            type=float, default=config.bw_rdfs, metavar='         ' + par_wrap(config.bw_rdfs))
optional_args5.add_argument('-cut_rdfs', dest='cut_rdfs_short',
                            help='Cutoff for radial distribution functions (nm)', type=float, default='2.5',
                            metavar='           (2.5)')  # WARNING: LET THIS AT 2.5 nm or more, OTHERWISE THE RDF COUNT NORM WILL BE SCREWED UP FOR COUNT NORMALIZATION !!!

optional_args3 = args_parser.add_argument_group(bullet + 'OTHERS')
optional_args3.add_argument('-o', dest='output_folder', help='Directory to create for storing all outputs of Swarm-CG',
                            type=str, default='', metavar='')
optional_args3.add_argument('-disable_x_scaling', dest='row_x_scaling',
                            help='Disable auto-scaling of X axis across each row on geoms distribs plots',
                            default=True, action='store_false')
optional_args3.add_argument('-disable_y_scaling', dest='row_y_scaling',
                            help='Disable auto-scaling of Y axis across each row on geoms distribs plots',
                            default=True, action='store_false')
# optional_args3.add_argument('-keep_all_sims', dest='keep_all_sims', help='Store all gmx files for all simulations, may use disk space', action='store_true', default=False)
optional_args3.add_argument('-h', '--help', help='Show this help message and exit', action='help')
optional_args3.add_argument('-v', '--verbose', dest='verbose', help=config.help_verbose, action='store_true',
                            default=False)

# arguments handling, display command line if help or no arguments provided
ns = args_parser.parse_args()
input_cmdline = ' '.join(map(cmd_quote, sys.argv))
ns.exec_folder = time.strftime(
    'LIPIDS_OPTI_' + ns.mapping_type + '_' + ns.map_center + '_' + ns.solv + '_STARTED_%d-%m-%Y_%Hh%Mm%Ss')  # default folder name for all files of this optimization run, in case none is provided
if ns.output_folder != '':
    ns.exec_folder = ns.output_folder
print('Working directory:', os.getcwd())
print('Command line:', input_cmdline)
print('Results directory:', ns.exec_folder)

# more parameters for scoring: accepting error and ensuring APL/Dhh lead the score
# at first I had put there the experimental error, but in fact the average experimental measures are just fine
# so I chose 2/3 of the experimental error, which allows the optimizer to have some margin for looking into solutions,
# while also providing very good results in the end (close to perfection, as long as the parameters that are
# NOT optimized are also reasonable/good)
# ns.apl_exp_error = 0.01 # nm^2
# ns.dhh_exp_error = 0.075 # nm
ns.apl_exp_error = 0.008  # nm^2
ns.dhh_exp_error = 0.065  # nm

if ns.mapping_type == '8BEADS':  # 8 BEADS
    # WARNING: At this resolution, check that the VDW cut-off and rlist BOTH = 1.5 !!
    ns.tune_geoms = True
    ns.tune_bonds_equi_val = True
    ns.specific_bonds_groups_for_tuning_equi_vals = []  # superseeds ns.tune_bonds_equi_val (= will tune eq val even if the other variable is False) -- Requires ns.tune_geoms == True
    ns.bonds_equi_val_from_config = False
    ns.tune_radii_in_groups = {  # if ns.tune_radii is True, the same radii can be specified here for several beads
        'small': ['MB', 'UB', 'MS', 'SB'],
        'big': ['K0', 'K1']
    }
    ns.tune_eps = True  # score and optimize all EPS independently between a given interval
    ns.eps_flat_range = 10  # float or None, limit FLAT range of EPS variations around (+/-) the EPS value given in the config file
    ns.apl_base_perc_error = 10
    ns.dhh_base_perc_error = 10
    ns.apl_cap_perc_error = 40
    ns.dhh_cap_perc_error = 40

else:
    sys.exit('\nUnknown mapping type')

# MDP files, will be searched for in directory: config.cg_setups_data_dir
ns.cg_mini_mdp = 'mini.mdp'
ns.cg_equi_mdp = 'equi.mdp'
ns.cg_prod_mdp = 'prod.mdp'

# namespace variables not directly linked to arguments for plotting or for global package interpretation
ns.mismatch_order = False
ns.row_x_scaling = True
ns.row_y_scaling = True
ns.ncols_max = 0  # 0 to display all
ns.process_alive_time_sleep = 5  # nb of seconds between process alive check cycles
ns.process_alive_nb_cycles_dead = int(
    ns.sim_kill_delay / ns.process_alive_time_sleep)  # nb of cycles without .log file bytes size changes to determine that the MD run is stuck
ns.rdf_frames = 250  # number of frames to take from trajectory for RDFs calculation (evenly spaced)
# ns.rdf_frames = 10  # for tests

####################
# ARGUMENTS CHECKS #
####################

print()
print(config.sep_close)
print('| PRE-PROCESSING AND CONTROLS                                                                 |')
print(config.sep_close)

# logs of all the results that interest us: parameters used in opti runs + their scores (also breakdown per molecule)
ns.opti_moves_file = 'opti_moves.log'
ns.opti_moves_details_lipid_temp_file = 'opti_moves_details_lipid_temp.log'  # score parts and all info per lipid+temp
ns.opti_moves_times_file = 'opti_moves_times.log'

# careful if continuing into an output directory of a previous optimization run
fstpso_checkpoint_in = None  # by default we assume there is no checkpoint
fstpso_checkpoint_in_nb = 0
n_swarm_iter = 1  # default start without checkpoints
n_cycle = 1
max_delta_ts_swarm_iter = 0  # maximum wall clock time (sec) for a complete SWARM iteration
# NOTE: the maximum time a SWARM iteration takes will be evaluated during this opti and does NOT use the history in logs atm

if os.path.isdir(ns.exec_folder):
    print(
        '\nProvided output folder already exists! We then assume this optimization is a restart from FST-PSO checkpoint.\n'
        'Swarm-CG will try to load the FST-PSO checkpoint and its own log files.\n'
        '\nNOTE: Swarm-CG will NOT verify that the parameters and objectives used in the previous part of the optimization\n'
        '      are identical to the ones used for this current execution.\n'
    )  # TODO: do something about this + maybe actually backup the logs or everything before continuing

    # look for FST-PSO checkpoints from which to restart the opti and use the last one (ALWAYS THE LAST ONE ATM)
    for filename in os.listdir(ns.exec_folder):
        if filename.startswith('fstpso_checkpoint_'):
            max_found_checkpoint_nb = int(filename.split('.')[0].split('_')[2])
            if max_found_checkpoint_nb > fstpso_checkpoint_in_nb:
                fstpso_checkpoint_in_nb = max_found_checkpoint_nb
    if fstpso_checkpoint_in_nb > 0 and not ns.next_cycle:
        fstpso_checkpoint_in = f'fstpso_checkpoint_{fstpso_checkpoint_in_nb}.obj'
        print('Loading FST-PSO checkpoint file:', fstpso_checkpoint_in, '\n')

    # find where the previous optimization stopped + small check of logs integrity
    # get number of previous iterations to correctly append the log files
    path_log_file_1 = f'{ns.exec_folder}/{ns.opti_moves_file}'
    path_log_file_2 = f'{ns.exec_folder}/{ns.opti_moves_details_lipid_temp_file}'
    path_log_file_3 = f'{ns.exec_folder}/{ns.opti_moves_times_file}'

    n_cycle_1, n_swarm_iter_1, n_particle_1 = check_log_files(path_log_file_1)
    n_cycle_2, n_swarm_iter_2, n_particle_2 = check_log_files(path_log_file_2)
    n_cycle_3, n_swarm_iter_3 = check_log_files(path_log_file_3, particle=False)

    # if logs are consistent
    if n_cycle_1 == n_cycle_2 == n_cycle_3 and n_swarm_iter_1 == n_swarm_iter_2 == n_swarm_iter_3 and n_particle_1 == n_particle_2:

        # set the point of restart for writting logs (n_particle is calculated via n_swarm_iter)
        # it is possible that there is no FST-PSO checkpoint but the exec directory still exists
        # this happens when the opti did not reach the end of SWARM iter 2)
        if n_cycle_1 is not None:
            if fstpso_checkpoint_in is not None or ns.next_cycle:
                try:
                    n_cycle = n_cycle_1  # continue within same opti cycle
                    n_swarm_iter = n_swarm_iter_1 + 1  # continue with the next swarm iteration
                    n_particle = n_particle_1

                    # if we want to do a calibrated re-initialization of the swarm from best results of the previous opti cycle
                    if ns.next_cycle:
                        n_cycle += 1
                        print('--> Going for a calibrated restart in a new optimization cycle')
                    else:
                        print('--> Going to continue an on-going optimization cycle')

                except ValueError:  # means we have just read the headers (not sure this below is relevant anymore)
                    n_swarm_iter = 1  # default start without checkpoints
                    n_cycle = 1
                    n_particle = 0

            # also get the average time per swarm iteration ?
            # TODO: fill ns.all_delta_ts_swarm_iter with the logged values

        else:
            n_swarm_iter = 1  # default start without checkpoints
            n_cycle = 1
            n_particle = 0

        print(
            f'Restarting at optimization cycle: {n_cycle}   (1-indexed)\n'
            f'Restarting at swarm iteration: {n_swarm_iter}   (1-indexed)\n'
            f'Restarting after particle: {n_particle}   (1-indexed)\n'
        )

        # remove directories that may have been created for running the simulations after the FST-PSO checkpoint
        # this should happen only if the routine for killing the MASTER (we kill the MASTER if it won't have time to run
        # a complete SWARM iteration) uses an improper time limit OR if the MASTER was killed not by himself, but by
        # power loss or something
        removed_dirs = False
        for n_part in range(n_particle + 1, n_particle + 200):  # this 200 limit just has to be more than nb of particles used
            irrelevant_sim_dir = f'{ns.exec_folder}/{config.iteration_sim_files_dirname}{n_part}'
            if os.path.isdir(irrelevant_sim_dir):
                shutil.rmtree(irrelevant_sim_dir)
                removed_dirs = True
                print(f'Removed irrelevant dir: {irrelevant_sim_dir}')
        if removed_dirs:
            print()
    else:
        sys.exit(
            'Swarm-CG log files did not write up to the same point in the previous optimization run.\n'
            'Please verify the integrity of log files, which should be also consistent with the FST-PSO checkpoint.'
        )

else:
    # directory to write all files for current execution of optimizations routines
    os.mkdir(ns.exec_folder)

    # just for safety, we will do a backup of everything useful and the end of each swarm iteration
    os.mkdir(f'{ns.exec_folder}/CHECKPOINTS_BACKUP')
    os.mkdir(f'{ns.exec_folder}/LOGS_SWARMCG_BACKUP')

# the new FST-PSO checkpoint we are going to write
fstpso_checkpoint_out_nb = fstpso_checkpoint_in_nb + 1
ns.fstpso_checkpoint_out = f'fstpso_checkpoint_{fstpso_checkpoint_out_nb}.obj'

# check that gromacs alias is correct
with open(os.devnull, 'w') as devnull:
    try:
        subprocess.call(ns.gmx_path, stdout=devnull, stderr=devnull)
    except OSError:
        sys.exit(
            config.header_error + 'Cannot find GROMACS using alias \'' + ns.gmx_path + '\', please provide the right GROMACS alias or path')

# check choice arguments
ns.map_center = ns.map_center.upper()
if ns.map_center != 'COM' and ns.map_center != 'COG':
    sys.exit(
        'Please provide as argument either COG or COM to indicate how beads should be centered with respect to mapped atoms')
ns.solv = ns.solv.upper()
if ns.solv != 'WET' and ns.solv != 'DRY':
    sys.exit('Please provide as argument either WET or DRY to indicate if solvent should be explicit or not')

# check parallelization arguments
if ns.nb_hpc_slots != 0 and (ns.nb_threads != '0' or ns.gpu_id != ''):
    sys.exit(
        'You have to choose between specifying -nb_hpc_slots OR (-nt AND -gpu_id).\n'
        'The former allows to run on a HPC, while the latter is for using SUPSI machines.\n'
        'NOTE: only HPC using SLURM are supported atm.'
    )

# check arguments for calculation slots
if ns.nb_hpc_slots == 0:  # if we are on local machines
    ns.slots_nts = ns.nb_threads.split(' ')
    if ns.gpu_id != '':  # here we assume that gpu slots HAVE TO BE provided either for EACH SLOT, or for NONE OF THE SLOTS
        ns.slots_gpu_ids = ns.gpu_id.split(' ')
        if len(ns.slots_nts) != len(ns.slots_gpu_ids):
            sys.exit(
                'Please provide the same number of arguments for -nt and -gpu_id in a space-separated list, \n'
                'as string (use quotes), as these define the slots available for calculation'
            )
    else:  # this actually will make GROMACS auto-detect available GPUs within each slot (= does NOT disable GPU usage)
        ns.slots_gpu_ids = [''] * len(ns.slots_nts)
    ns.nb_slots = len(ns.slots_nts)


##################
# INITIALIZATION #
##################

ns.opti_start_ts = datetime.now().timestamp()

# MDAnalysis backend
ns.mda_backend = 'serial'

print('\nSelected lipids and temperatures for this parallel optimization\n ', ns.lipids_codes)

# list which experimental data are available or not
print('\nAvailable experimental data from config file')
for lipid_code in ns.opti_config.lipids_codes:
    for temp in ns.opti_config.lipids_codes[lipid_code]:

        try:
            exp_apl = ns.opti_config.exp_data[lipid_code][temp]['apl']  # just check if the value is defined
        except KeyError:
            try:
                ns.opti_config.exp_data[lipid_code][temp]['apl'] = None
            except KeyError:
                try:
                    ns.opti_config.exp_data[lipid_code][temp] = {'apl': None, 'Dhh': None}
                except KeyError:
                    ns.opti_config.exp_data[lipid_code] = {temp: {'apl': None, 'Dhh': None}}
            exp_apl = None

        try:
            exp_dhh = ns.opti_config.exp_data[lipid_code][temp]['Dhh']  # just check if the value is defined
        except KeyError:
            ns.opti_config.exp_data[lipid_code][temp]['Dhh'] = None
            exp_dhh = None

        # just format the display
        if exp_apl is None:
            str_exp_apl = None
        else:
            str_exp_apl = str(exp_apl) + ' nm²'
        if exp_dhh is None:
            str_exp_dhh = None
        else:
            str_exp_dhh = str(exp_dhh) + ' nm²'

        print(' ', lipid_code, '--', temp, '-- APL:', str_exp_apl, '-- Dhh:', exp_dhh, 'nm -- Weight of the AA ref data:', ns.opti_config.exp_data_error[lipid_code][temp] * 100, '%')

# check AA files for given lipids -- THIS IS IF WE LIST THE TEMPERATURES BASED ON THE ARGUMENTS STRING
for lipid_code in ns.lipids_codes:
    for temp in ns.lipids_codes[lipid_code]:
        aa_lipid_and_temp_dir = f"{lipid_code}_unbiased_200ns_{temp}"
        if not (os.path.isfile(
                config.aa_data_dir + '/' + ns.aa_ff + '/' + aa_lipid_and_temp_dir + '/' + config.aa_tpr_filename) and os.path.isfile(
                config.aa_data_dir + '/' + ns.aa_ff + '/' + aa_lipid_and_temp_dir + '/' + config.aa_traj_filename)):
            sys.exit(
                f"{config.header_error}Cannot find TPR & XTC for {lipid_code} at temperature {temp} for AA force field {ns.aa_ff}")

# check mapping files for given lipids
ns.mapping_type = ns.mapping_type.upper()
for lipid_code in ns.lipids_codes:
    if os.path.isfile(
            config.cg_models_data_dir + '/CG_' + lipid_code + '_' + ns.mapping_type + '_' + ns.solv + '.itp'):
        if os.path.isfile(
                config.cg_models_data_dir + '/MAP_' + lipid_code + '_' + ns.mapping_type + '_DRY.ndx'):  # here DRY is hardcoded because in WET or DRY the mapping is the same, but we started naming files with DRY
            pass
        else:
            sys.exit(f"{config.header_error}Cannot find file of mapping {ns.mapping_type} for {lipid_code}")
    else:
        sys.exit(
            f"{config.header_error}Cannot find CG model ITP file for {lipid_code} for mapping {ns.mapping_type} with solvent {ns.solv}")

#############################################
# BIG DATA LOADING LOOP HERE                #
# CHECK IF ANYTHING IS MISSING FROM PICKLES #
#   IF YES RELOAD ALL FOR CURRENT LIPID     #
#   IF NO, USE EXISTING PICKLES             #
#############################################

# data structures for loading CG data + AA reference lipids simulations, from pickles or gromacs files
ns.cg_itps = {}
create_bins_and_dist_matrices(ns)  # bins for EMD calculations

# find masses of the CG beads
ns.beads_masses = config.beads_masses[ns.mapping_type][ns.solv]

# find radii of the CG beads
ns.user_config.init_beads_radii = config.user_config.init_beads_radii[ns.mapping_type][ns.solv]

ns.rdf_indep_weights = {}  # attempt to weight the independant RDF in the scoring that will come later
ns.all_delta_ts_swarm_iter = []  # keep track of average SWARM iteration time

# find which data have to be read or pickled
for lipid_code in ns.lipids_codes:

    print()
    print(config.sep)
    print()
    print('Processing', lipid_code, 'mapping', ns.mapping_type, '...')

    ns.cg_itp_filename = config.cg_models_data_dir + '/CG_' + lipid_code + '_' + ns.mapping_type + '_' + ns.solv + '.itp'
    ns.cg_map_filename = config.cg_models_data_dir + '/MAP_' + lipid_code + '_' + ns.mapping_type + '_DRY.ndx'  # here DRY is hardcoded because in WET or DRY the mapping is the same, but we started naming files with DRY
    pickle_file = lipid_code + '_' + ns.mapping_type + '_' + ns.map_center + '_' + ns.solv + '.pickle'

    ns.rdf_indep_weights[lipid_code] = {}

    if ns.reset or not os.path.isfile(config.data_aa_storage_dir + '/' + ns.aa_ff + '/' + pickle_file):

        for temp in ns.lipids_codes[lipid_code]:

            aa_lipid_and_temp_dir = lipid_code + '_unbiased_200ns_' + temp
            ns.rdf_indep_weights[lipid_code][temp] = {}

            # read atomistic trajectory
            print()
            print('  Reading AA trajectory for', lipid_code, temp, 'with mapping', ns.mapping_type, 'and solvent',
                  ns.solv, flush=True)
            aa_tpr_filepath = config.aa_data_dir + '/' + ns.aa_ff + '/' + aa_lipid_and_temp_dir + '/' + config.aa_tpr_filename
            aa_traj_filepath = config.aa_data_dir + '/' + ns.aa_ff + '/' + aa_lipid_and_temp_dir + '/' + config.aa_traj_filename
            aa_universe = mda.Universe(aa_tpr_filepath, aa_traj_filepath, in_memory=True, refresh_offsets=True,
                                          guess_bonds=False)  # setting guess_bonds=False disables angles, dihedrals and improper_dihedrals guessing, which is activated by default
            print('    Found', len(aa_universe.trajectory), 'frames in AA trajectory file', flush=True)

            # TODO: not useful atm, according to the AA trajs we use, but making the AA trajs whole
            #       should be re-enabled by default
            # load_aa_data(ns)
            # make_aa_traj_whole_for_selected_mols(ns)

            # this is based on lipid type and common for different temperatures, ASSUMING INDEXING IS IDENTICAL ACROSS TEMPERATURES
            # therefore execute only for first temperature encountered for lipid
            if temp == ns.lipids_codes[lipid_code][0]:

                ns.nb_mol_instances = len(aa_universe.residues)
                ns.nb_mol_atoms = len(aa_universe.residues[0].atoms)
                print('    Found', ns.nb_mol_instances, 'instances of', lipid_code)
                print('    Found', ns.nb_mol_atoms, 'atoms for', lipid_code)
                print()
                print('  Reading mapping file')
                read_ndx_atoms2beads(ns)  # read mapping, get atoms accurences in beads
                get_atoms_weights_in_beads(ns)  # get weights of atoms within beads
                print('    Found', ns.nb_beads_initial, 'CG particles')

                # read starting CG ITP file
                with open(ns.cg_itp_filename, 'r') as fp:
                    itp_lines = fp.read().split('\n')
                    itp_lines = [itp_line.strip() for itp_line in itp_lines]
                    print()
                    read_cg_itp_file_grp_comments(ns,
                                                  itp_lines)  # loads ITP object that contains our reference atomistic data -- won't ever be modified during execution

                # check that number of beads is appropriate between CG model ITP and mapping
                if ns.nb_beads_initial != ns.nb_beads_itp:  # obtained from the mapping file and itp
                    sys.exit(
                        '\nNumer of beads is inconsistent between CG ITP model and NDX mapping for lipid ' + lipid_code)

                # Storage at the PER LIPID LEVEL -- Create only once per lipid in this code block
                ns.cg_itp['rdf'] = {}  # for RDFs
                all_avg_Dhh_delta = []  # for Dhh AA-CG delta of phosphate positions

                # first index beads ids on beads types
                ns.cg_itp['beads_ids_per_beads_types_mult'] = {}

                for bead_id_mult in range(ns.nb_mol_instances * ns.nb_beads_itp):

                    bead_id_sing = bead_id_mult % ns.nb_beads_initial
                    bead_type = ns.cg_itp['atoms'][bead_id_sing]['bead_type']

                    if bead_type in ns.cg_itp['beads_ids_per_beads_types_mult']:
                        ns.cg_itp['beads_ids_per_beads_types_mult'][bead_type].append(bead_id_mult)
                    else:
                        ns.cg_itp['beads_ids_per_beads_types_mult'][bead_type] = [bead_id_mult]

                # next we prepare the vector of beads for InterRDF
                ns.cg_itp['rdf_pairs'] = {}  # key: beadtype1_beadtype2 sorted alphabetical
                for bead_type_1 in ns.cg_itp['beads_ids_per_beads_types_mult']:
                    for bead_type_2 in ns.cg_itp['beads_ids_per_beads_types_mult']:
                        sorted_beads_types = sorted([bead_type_1, bead_type_2])  # conserve the ordering
                        pair_type = '_'.join(sorted_beads_types)
                        ns.cg_itp['rdf_pairs'][pair_type] = [
                            ns.cg_itp['beads_ids_per_beads_types_mult'][sorted_beads_types[0]],
                            ns.cg_itp['beads_ids_per_beads_types_mult'][sorted_beads_types[1]]]

            # some more consistency checks for the other temperatures found after the very first one for given lipid
            else:
                if ns.nb_mol_instances != len(aa_universe.residues):
                    sys.exit(
                        '\nNumber of molecules are inconsistent in traj files across temperatures for lipid ' + lipid_code)
                elif ns.nb_mol_atoms != len(aa_universe.residues[0].atoms):
                    sys.exit(
                        '\nNumber of atoms of 1st molecule found in traj files are inconsistent across temperatures for lipid ' + lipid_code)
                else:
                    # print('  Number of molecules and atoms is consistent with first read trajectory')
                    pass

            # storage of AA data: APL and Dhh for each temperature
            ns.cg_itp['exp_data_' + temp] = {}

            # it seems MDA AtomGroups are copies, then it's necessary to re-create the atom groups for each new reference trajectory
            mda_beads_atom_grps, mda_weights_atom_grps = get_beads_MDA_atomgroups(ns, aa_universe)  # for each CG bead, create atom groups for mapped geoms calculation using COM or COG

            print('\n  Mapping the trajectory from AA to CG representation')
            aa2cg_universe = initialize_cg_traj(ns)
            aa2cg_universe = map_aa2cg_traj(ns, aa_universe, aa2cg_universe, mda_beads_atom_grps, mda_weights_atom_grps)

            # calculate Dhh delta for perfect thickness calculations
            # NOTE: it works because trajectories are WHOLE for each molecule
            print('\n  Calculating Dhh delta of phosphate position between AA-CG')
            head_type = lipid_code[2:]  # TODO: when we start incorporating lipids for which the code is not 4 letters this will NOT work
            phosphate_atom_id = ns.user_config.phosphate_pos[head_type]['AA']
            phosphate_bead_id = ns.user_config.phosphate_pos[head_type]['CG']
            phosphate_bead_id -= 1  # retrieve 0-indexing
            phosphate_atom_id -= 1
            all_Dhh_deltas = np.empty(len(aa_universe.trajectory) * ns.nb_mol_instances, dtype=np.float32)

            for ts in aa_universe.trajectory:
                for i in range(ns.nb_mol_instances):

                    bead_id = i * ns.nb_beads_itp + phosphate_bead_id
                    if ns.map_center == 'COM':
                        cg_z_pos = mda_beads_atom_grps[bead_id].center(mda_weights_atom_grps[bead_id])[2]
                    else:
                        cg_z_pos = mda_beads_atom_grps[bead_id].center_of_geometry()[2]
                    atom_id = i * ns.nb_mol_atoms + phosphate_atom_id
                    aa_z_pos = aa_universe.atoms[atom_id].position[2]

                    # find if delta should be positive or negative by checking the position of an atom of the arm, far away from the head (here atom_id +60 with respect to phosphate id)
                    ref_z_pos = aa_universe.atoms[atom_id + 60].position[2]
                    if abs(cg_z_pos - ref_z_pos) > abs(aa_z_pos - ref_z_pos):
                        all_Dhh_deltas[ts.frame * ns.nb_mol_instances + i] = -abs(
                            aa_z_pos - cg_z_pos) / 10  # retrieve nm
                    else:
                        all_Dhh_deltas[ts.frame * ns.nb_mol_instances + i] = abs(
                            aa_z_pos - cg_z_pos) / 10  # retrieve nm

            avg_Dhh_delta = np.median(all_Dhh_deltas)
            all_avg_Dhh_delta.append(avg_Dhh_delta)
            print('    Median delta:', round(avg_Dhh_delta, 3), 'nm')

            # calculate APL for AA data
            print('\n  Calculating APL for AA data')
            x_boxdims = []
            for ts in aa_universe.trajectory:
                x_boxdims.append(ts.dimensions[
                                     0])  # X-axis box size, Y is in principle identical and Z size is orthogonal to the bilayer
            x_boxdims = np.array(x_boxdims)

            apl_avg = round(np.mean(x_boxdims ** 2 / (ns.nb_mol_instances / 2)) / 100, 4)
            apl_std = round(np.std(x_boxdims ** 2 / (ns.nb_mol_instances / 2)) / 100, 4)
            print('    APL avg:', apl_avg, 'nm2')
            print('    APL std:', apl_std, 'nm2')
            ns.cg_itp['exp_data_' + temp]['apl_avg'] = apl_avg
            ns.cg_itp['exp_data_' + temp]['apl_std'] = apl_std

            # calculate Dhh for AA data
            print('\n  Calculating Dhh thickness for AA data')

            # Dhh thickness definition: here we will use the Phosphates positions to get average Z-axis positive and negative values, then use the distance between those points
            # NOTE: we decided not to care about flip flops to calculate Z-axis positions, as this is probably done the same way in experimental calculations
            #       (anyway the impact would be very small but really, who knows ??)

            # get the id of the bead that should be used as reference for Dhh calculation + the delta for Dhh calculation, if any
            head_type = lipid_code[2:]  # TODO: when we start incorporating lipids for which the code is not 4 letters this may NOT work
            phosphate_atom_id = ns.user_config.phosphate_pos[head_type]['AA']
            phosphate_atom_id -= 1

            # to ensure thickness calculations are not affected by bilayer being split on Z-axis PBC
            # for each frame, calculate 2 thicknesses: 1. without changing anything AND 2. by shifting the upper half of the box below the lower half THEN 3. take minimum thickness value
            phos_z_dists = []

            for ts in aa_universe.trajectory:

                z_all = np.empty(ns.nb_mol_instances)
                for i in range(ns.nb_mol_instances):
                    id_phos = i * ns.nb_mol_atoms + phosphate_atom_id
                    z_phos = aa_universe.atoms[id_phos].position[2]
                    z_all[i] = z_phos

                # 1. without correction
                z_avg = np.mean(
                    z_all)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
                z_pos, z_neg = [], []
                for i in range(ns.nb_mol_instances):
                    if z_all[i] > z_avg:
                        z_pos.append(z_all[i])
                    else:
                        z_neg.append(z_all[i])
                phos_z_dists_1 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

                # 2. with correction
                for i in range(ns.nb_mol_instances):
                    if z_all[i] > ts.dimensions[2] / 2:  # Z-axis box size
                        z_all[i] -= ts.dimensions[2]
                z_avg = np.mean(
                    z_all)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
                z_pos, z_neg = [], []
                for i in range(ns.nb_mol_instances):
                    if z_all[i] > z_avg:
                        z_pos.append(z_all[i])
                    else:
                        z_neg.append(z_all[i])
                phos_z_dists_2 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

                # 3. choose the appropriate thickness measurement
                if phos_z_dists_1 <= phos_z_dists_2:
                    phos_z_dists.append(phos_z_dists_1)
                else:
                    phos_z_dists.append(phos_z_dists_2)

            Dhh_avg = round(np.mean(phos_z_dists), 4)
            Dhh_std = round(np.std(phos_z_dists), 4)
            print('    Dhh avg:', Dhh_avg, 'nm')
            print('    Dhh std:', Dhh_std, 'nm')
            ns.cg_itp['exp_data_' + temp]['Dhh_avg'] = Dhh_avg
            ns.cg_itp['exp_data_' + temp]['Dhh_std'] = Dhh_std

            # calculate RDFs
            print('\n  Calculating reference AA-mapped RDFs')
            rdf_start = datetime.now().timestamp()
            ns.cg_itp['rdf_' + temp + '_short'], ns.cg_itp['rdf_' + temp + '_long'] = {}, {}

            for pair_type in sorted(ns.cg_itp['rdf_pairs']):
                bead_type_1, bead_type_2 = pair_type.split('_')

                ag1 = mda.AtomGroup(ns.cg_itp['rdf_pairs'][pair_type][0], aa2cg_universe)
                ag2 = mda.AtomGroup(ns.cg_itp['rdf_pairs'][pair_type][1], aa2cg_universe)

                # here we ignore all pairs of beads types that are involved in bonded interactions (i.e. from the same molecule),
                # as this would only add noise into the scoring function
                # note that the exclusion block is NOT the number of atoms per molecule,
                # but the numbers of ATOMS PROVIDED PER MOLECULE in the input 2 first arguments to InterRDF
                irdf_short = rdf.InterRDF(ag1, ag2, nbins=round(ns.cut_rdfs_short / ns.bw_rdfs),
                                          range=(0, ns.cut_rdfs_short * 10), exclusion_block=(
                    len(ns.cg_itp['beads_ids_per_beads_types_sing'][bead_type_1]),
                    len(ns.cg_itp['beads_ids_per_beads_types_sing'][bead_type_2])))

                irdf_short.run(step=round(len(aa2cg_universe.trajectory) / ns.rdf_frames))

                rdf_norm = irdf_short.count / ns.vol_shell
                rdf_count = irdf_short.count  # scale for percentage error

                # attempt to weight the independant RDF in the scoring that will come later
                last_id = np.where(ns.bins_vol_shell == 1.5)[0][
                    0]  # index of the bin for which the volume is 1.5 nm
                ns.rdf_indep_weights[lipid_code][temp][pair_type] = np.sum(rdf_count[:last_id + 1])

                ns.cg_itp['rdf_' + temp + '_short'][pair_type] = rdf_count, rdf_norm

            rdf_time = datetime.now().timestamp() - rdf_start
            print('    Time for RDFs calculation:', round(rdf_time / 60, 2),
                  'min (' + str(ns.rdf_frames) + ' frames)')

            print('\n  Calculating reference AA-mapped geoms distributions')
            geoms_start = datetime.now().timestamp()

            # create all ref atom histograms to be used for pairwise distributions comparisons + find average geoms values as first guesses (without BI at this point)
            # get ref atom hists + find very first distances guesses for constraints groups
            for grp_constraint in range(ns.nb_constraints):
                constraint_avg, constraint_hist, constraint_values = get_AA_bonds_distrib(ns, beads_ids=
                ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraint', uni=aa2cg_universe)
                # ns.cg_itp['constraint'][grp_constraint]['value'] = constraint_avg # taken from ITP
                ns.cg_itp['constraint'][grp_constraint]['avg_' + temp] = constraint_avg
                ns.cg_itp['constraint'][grp_constraint]['hist_' + temp] = constraint_hist
                dom_restricted = np.flatnonzero(constraint_hist > np.max(constraint_hist) * config.eq_val_density_thres_constraints)
                eq_val_min, eq_val_max = ns.bins_constraints[dom_restricted[0] + 1], ns.bins_constraints[dom_restricted[-1]]
                ns.cg_itp['constraint'][grp_constraint]['values_dom_' + temp] = [
                    round(eq_val_min, 3),
                    round(eq_val_max, 3)]  # boundaries of geom values

            # get ref atom hists + find very first distances and force constants guesses for bonds groups
            for grp_bond in range(ns.nb_bonds):
                bond_avg, bond_hist, bond_values = get_AA_bonds_distrib(ns, beads_ids=
                ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bond', uni=aa2cg_universe)
                # ns.cg_itp['bond'][grp_bond]['value'] = bond_avg # taken from ITP
                ns.cg_itp['bond'][grp_bond]['avg_' + temp] = bond_avg
                ns.cg_itp['bond'][grp_bond]['hist_' + temp] = bond_hist
                dom_restricted = np.flatnonzero(bond_hist > np.max(bond_hist) * config.eq_val_density_thres_bonds)
                eq_val_min, eq_val_max = ns.bins_bonds[dom_restricted[0] + 1], ns.bins_bonds[dom_restricted[-1]]
                ns.cg_itp['bond'][grp_bond]['values_dom_' + temp] = [round(eq_val_min, 3),
                                                                     round(eq_val_max, 3)]  # boundaries of geom values

            # get ref atom hists + find very first values and force constants guesses for angles groups
            for grp_angle in range(ns.nb_angles):
                angle_avg, angle_hist, angle_values_deg = get_AA_angles_distrib(ns, beads_ids=
                ns.cg_itp['angle'][grp_angle]['beads'], uni=aa2cg_universe)
                # ns.cg_itp['angle'][grp_angle]['value'] = angle_avg # taken from ITP
                ns.cg_itp['angle'][grp_angle]['avg_' + temp] = angle_avg
                ns.cg_itp['angle'][grp_angle]['hist_' + temp] = angle_hist
                dom_restricted = np.flatnonzero(angle_hist > np.max(angle_hist) * config.eq_val_density_thres_angles)
                eq_val_min, eq_val_max = ns.bins_angles[dom_restricted[0] + 1], ns.bins_angles[dom_restricted[-1]]
                ns.cg_itp['angle'][grp_angle]['values_dom_' + temp] = [round(eq_val_min, 2),
                                                                       round(eq_val_max, 2)]  # boundaries of geom values

            # # get ref atom hists + find very first values and force constants guesses for dihedrals groups
            # for grp_dihedral in range(ns.nb_dihedrals):

            #   dihedral_avg, dihedral_hist, dihedral_values_deg = get_AA_dihedrals_distrib_single(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'], uni=aa2cg_universe)
            #   # ns.cg_itp['dihedral'][grp_dihedral]['value'] = dihedral_avg # taken from ITP
            #   ns.cg_itp['dihedral'][grp_dihedral]['avg_'+temp] = dihedral_avg
            #   ns.cg_itp['dihedral'][grp_dihedral]['hist_'+temp] = dihedral_hist
            #   ns.cg_itp['dihedral'][grp_dihedral]['values_dom_'+temp] = [round(np.min(dihedral_values_deg), 2), round(np.max(dihedral_values_deg), 2)] # boundaries of geom values

            geoms_time = datetime.now().timestamp() - geoms_start
            print('    Time for geoms distributions calculation:', round(geoms_time / 60, 2),
                  'min (' + str(len(aa2cg_universe.trajectory)) + ' frames)')

        # store some metadata
        ns.cg_itp['meta'] = {'nb_mols': ns.nb_mol_instances, 'nb_beads': ns.nb_beads_itp,
                             'nb_constraints': ns.nb_constraints, 'nb_bonds': ns.nb_bonds,
                             'nb_angles': ns.nb_angles, 'nb_dihedrals': ns.nb_dihedrals,
                             'temps': ns.lipids_codes[lipid_code].copy(), 'delta_Dhh': np.mean(all_avg_Dhh_delta)}

        # now that we have read all temperatures, do the pickling
        with open(config.data_aa_storage_dir + '/' + ns.aa_ff + '/' + pickle_file, 'wb') as fp:
            pickle.dump(ns.cg_itp, fp)

    else:

        with open(config.data_aa_storage_dir + '/' + ns.aa_ff + '/' + pickle_file, 'rb') as fp:
            ns.cg_itp = pickle.load(fp)
        print('  Loaded from pickle file')

    # all data needed for optimization
    ns.cg_itps[lipid_code] = copy.deepcopy(ns.cg_itp)

#################################
# DEFINE PARAMS TO BE OPTIMIZED #
#################################

print()
print(config.sep_close)
print('| DEFINING PARAMETERS TO BE OPTIMIZED                                                         |')
print(config.sep_close)

# find all types of beads in presence in the input CG models
# also record beads per lipid, to find LJ used for each lipid and save opti results per lipid with minimal/required set of parameters
ns.all_beads_types, ns.lipid_beads_types = set(), {}
for lipid_code in ns.lipids_codes:
    ns.lipid_beads_types[lipid_code] = set()
    for bead_id in range(
            len(ns.cg_itps[lipid_code]['atoms'])):  # here 'atoms' are actually the [atoms] field of the CG ITPs
        ns.all_beads_types.add(ns.cg_itps[lipid_code]['atoms'][bead_id]['bead_type'])
        ns.lipid_beads_types[lipid_code].add(ns.cg_itps[lipid_code]['atoms'][bead_id]['bead_type'])
ns.all_beads_types = sorted(ns.all_beads_types)

# check if there are specific groups of radii to be tuned together
ns.reverse_radii_mapping = {}
if ns.user_config.tune_radii_in_groups is not None:
    for radii_grp in ns.user_config.tune_radii_in_groups:
        for bead_type in ns.user_config.tune_radii_in_groups[radii_grp]:
            ns.reverse_radii_mapping[bead_type] = radii_grp
else:  # we do a 1-to-1 mapping of the radii to beads
    for bead_type in ns.user_config.init_beads_radii:
        radii_grp = bead_type
        ns.reverse_radii_mapping[bead_type] = radii_grp
    for bead_type in ns.all_beads_types:  # here: bead_type == radii_grp
        ns.user_config.tune_radii_in_groups[bead_type] = bead_type

for lipid_code in ns.lipid_beads_types:
    ns.lipid_beads_types[lipid_code] = sorted(ns.lipid_beads_types[lipid_code])

    # dump it for external usage and comparisons of AA-mapped data and different AA FF
    pickle_file = lipid_code + '_' + ns.mapping_type + '_matrix_beads_types.pickle'
    with open(config.data_aa_storage_dir + '/' + pickle_file, 'wb') as fp:
        pickle.dump(ns.lipid_beads_types[lipid_code], fp)

print()
print('Found the following CG BEADS TYPES across all input CG models:\n ', ns.all_beads_types)

# create the matrix of CG LJ bead to bead interactions -- all
ns.all_beads_pairs = [sorted(pair) for pair in
                      itertools.combinations(ns.all_beads_types, 2)]  # pairwise combinations excluding identities
for bead_type in ns.all_beads_types:
    ns.all_beads_pairs.append(
        [bead_type, bead_type])  # here we add also identical ones (i.e. we can have ['C1', 'C1'])
ns.all_beads_pairs = sorted(ns.all_beads_pairs)
print()
print('All LJ pairs are defined as follows:\n ', ns.all_beads_pairs)

# create the matrix of CG LJ bead to bead interactions -- per lipid usage
ns.lipid_beads_pairs = {}
print()
for lipid_code in ns.lipid_beads_types:
    ns.lipid_beads_pairs[lipid_code] = [sorted(pair) for pair in
                                        itertools.combinations(ns.lipid_beads_types[lipid_code],
                                                               2)]  # pairwise combinations excluding identities
    for bead_type in ns.lipid_beads_types[lipid_code]:
        ns.lipid_beads_pairs[lipid_code].append([bead_type, bead_type])  # add identities
    ns.lipid_beads_pairs[lipid_code] = sorted(ns.lipid_beads_pairs[lipid_code])
    print(lipid_code, 'LJ pairs are defined as follows:\n ', ns.lipid_beads_pairs[lipid_code])

# remove beads pairs that cannot be tuned, because they are never found in the same simulation
print()
print('Checking for beads pairs that cannot be tuned for radius and LJ')
all_tunable_beads_pairs = []
for bead_pair in ns.all_beads_pairs:
    found_pair = False
    for lipid_code in ns.lipid_beads_pairs:
        if bead_pair in ns.lipid_beads_pairs[lipid_code]:
            found_pair = True
    if found_pair:
        all_tunable_beads_pairs.append(bead_pair)
    else:
        print('  Will NOT be able to tune bead pair:', bead_pair)
if len(all_tunable_beads_pairs) == len(ns.all_beads_pairs):
    print('  Will be able to tune all beads pairs that were detected')
ns.all_beads_pairs = sorted(all_tunable_beads_pairs)

# in next code blocks also record geoms per lipid to store optimization results using reduced set of parameters
# (lipid+temp as key, record score parts to be able to modify weights later in the scoring function = recycle data at max if necessary)
ns.lipid_params_opti = {}

# find all types of constraints in presence in the input CG models, then attribute relevant constraint_ids to their geom_grp
ns.all_constraints_types = {}
for lipid_code in ns.lipids_codes:

    ns.lipid_params_opti[lipid_code] = []

    for constraint_id in range(len(ns.cg_itps[lipid_code]['constraint'])):
        geom_grp = ns.cg_itps[lipid_code]['constraint'][constraint_id]['geom_grp']

        if geom_grp not in ns.all_constraints_types:
            ns.all_constraints_types[geom_grp] = {}

        if lipid_code in ns.all_constraints_types[geom_grp]:
            ns.all_constraints_types[geom_grp][lipid_code].append(constraint_id)
        else:
            ns.all_constraints_types[geom_grp][lipid_code] = [constraint_id]

print()
print('Found the following CG CONSTRAINTS TYPES across all input CG models:\n ', ns.all_constraints_types)

# find all types of bonds in presence in the input CG models, then attribute relevant bond_ids to their geom_grp
ns.all_bonds_types = {}
for lipid_code in ns.lipids_codes:

    for bond_id in range(len(ns.cg_itps[lipid_code]['bond'])):
        geom_grp = ns.cg_itps[lipid_code]['bond'][bond_id]['geom_grp']

        if geom_grp not in ns.all_bonds_types:
            ns.all_bonds_types[geom_grp] = {}

        if lipid_code in ns.all_bonds_types[geom_grp]:
            ns.all_bonds_types[geom_grp][lipid_code].append(bond_id)
        else:
            ns.all_bonds_types[geom_grp][lipid_code] = [bond_id]

        if geom_grp not in ns.lipid_params_opti[lipid_code]:
            ns.lipid_params_opti[lipid_code].append(geom_grp)
print()
print('Found the following CG BONDS TYPES across all input CG models:\n ', ns.all_bonds_types)

# find all types of angles in presence in the input CG models, then attribute relevant angle_ids to their geom_grp
ns.all_angles_types = {}
for lipid_code in ns.lipids_codes:

    for angle_id in range(len(ns.cg_itps[lipid_code]['angle'])):
        geom_grp = ns.cg_itps[lipid_code]['angle'][angle_id]['geom_grp']

        if geom_grp not in ns.all_angles_types:
            ns.all_angles_types[geom_grp] = {}

        if lipid_code in ns.all_angles_types[geom_grp]:
            ns.all_angles_types[geom_grp][lipid_code].append(angle_id)
        else:
            ns.all_angles_types[geom_grp][lipid_code] = [angle_id]

        if geom_grp not in ns.lipid_params_opti[lipid_code]:
            ns.lipid_params_opti[lipid_code].append(geom_grp)
print()
print('Found the following CG ANGLES TYPES across all input CG models:\n ', ns.all_angles_types)

# add radii used per lipid (i.e. beads present in a given lipid)
if ns.tune_radii:
    for bead_type in ns.all_beads_types:
        for lipid_code in ns.lipid_beads_pairs:
            for bead_id in range(len(ns.cg_itps[lipid_code]['atoms'])):  # here 'atoms' are actually the [atoms] field of the CG ITPs
                if bead_type == ns.cg_itps[lipid_code]['atoms'][bead_id]['bead_type'] and f'r_{ns.reverse_radii_mapping[bead_type]}' not in ns.lipid_params_opti[lipid_code]:
                    ns.lipid_params_opti[lipid_code].append(f'r_{ns.reverse_radii_mapping[bead_type]}')

# add LJ used per lipid
if ns.tune_eps:
    for i in range(len(ns.all_beads_pairs)):
        LJ_curr = '_'.join(ns.all_beads_pairs[i])
        for lipid_code in ns.lipid_beads_pairs:
            if ns.all_beads_pairs[i] in ns.lipid_beads_pairs[lipid_code] and LJ_curr not in ns.lipid_params_opti[lipid_code]:
                # ns.lipid_params_opti[lipid_code].append('LJ'+str(i+1))
                ns.lipid_params_opti[lipid_code].append('LJ_' + LJ_curr)

# gather values and force constants from config file, used to fill the input ITPs and for initialization of the swarm
for lipid_code in ns.lipids_codes:
    try:
        for bond_id in range(len(ns.cg_itps[lipid_code]['bond'])):
            geom_grp = ns.cg_itps[lipid_code]['bond'][bond_id]['geom_grp']
            # if ns.geoms_val_from_cfg:
            # 	ns.cg_itps[lipid_code]['bond'][bond_id]['value'] = ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['val']
            ns.cg_itps[lipid_code]['bond'][bond_id]['fct'] = ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['fct']
        # print('GEOM GRP:', geom_grp, '-- BOND ID:', bond_id, '-- FCT:', ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['fct'])
        for angle_id in range(len(ns.cg_itps[lipid_code]['angle'])):
            geom_grp = ns.cg_itps[lipid_code]['angle'][angle_id]['geom_grp']
            ns.cg_itps[lipid_code]['angle'][angle_id]['value'] = ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp][
                'val']
            ns.cg_itps[lipid_code]['angle'][angle_id]['fct'] = ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp][
                'fct']
        # print('GEOM GRP:', geom_grp, '-- ANGLE ID:', angle_id, '-- FCT:', ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['fct'])
    except:
        sys.exit(
            '\nMissing default value and fct for ' + ns.mapping_type + ' geom grp ' + geom_grp + ' in lipid ' + lipid_code + ' with solvent ' + ns.solv)  # +')\nIf argument -geoms_val_from_cfg is active ('+str(ns.geoms_val_from_cfg)+') then angles values that should be fixed at 90°, 120° or 180° must be defined, all the others can be set to None in config file')

# find dimensions of the search space for optimization -- the extensive definition that can be reduced later during each optimization cycle
# in particular, find if both value and fct should be optimized for angles
ns.all_params_opti = []
ns.params_val = {}

print()
print('Collecting bonds and angles values from AA mapped data')

for geom_grp in ns.all_constraints_types:

    ns.params_val[geom_grp] = {'range': [np.inf, -np.inf],
                               'avg': []}  # find min/max values of bonds in the geom group at all selected temperatures + average of their average value
    # ns.all_params_opti.append({geom_grp: X}) # for constraints we tune nothing, there is no force constant and the equi value comes either from AA or config file
    geom_grp_val = []

    for lipid_code in ns.all_constraints_types[geom_grp]:
        for constraint_id in ns.all_constraints_types[geom_grp][lipid_code]:
            for temp in ns.lipids_codes[lipid_code]:
                values_dom = ns.cg_itps[lipid_code]['constraint'][constraint_id]['values_dom_' + temp]
                ns.params_val[geom_grp]['range'][0] = min(ns.params_val[geom_grp]['range'][0], values_dom[0])
                ns.params_val[geom_grp]['range'][1] = max(ns.params_val[geom_grp]['range'][1], values_dom[1])
                # print('Geom', geom_grp, 'got his range defined/updated to:', ns.params_val[geom_grp]['range'])

                # if ns.geoms_val_from_cfg:
                # 	ns.params_val[geom_grp]['avg'].append(ns.cg_itps[lipid_code]['bond'][bond_id]['value']) # to start from config file bonds lengths
                # else:
                # ns.params_val[geom_grp]['avg'].append(ns.cg_itps[lipid_code]['constraint'][constraint_id]['avg_'+temp]) # to start from average constraints lengths from AA mapped data
                geom_grp_val.append(ns.cg_itps[lipid_code]['constraint'][constraint_id]['avg_' + temp])

    geom_grp_std = round(np.std(geom_grp_val), 3)
    geom_grp_avg = round(np.mean(geom_grp_val), 3)
    ns.params_val[geom_grp]['avg'] = geom_grp_avg

    if ns.bonds_equi_val_from_config:
        equi_val = ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['val']
    else:
        equi_val = geom_grp_avg

    for lipid_code in ns.all_constraints_types[geom_grp]:  # attribute average value
        for constraint_id in ns.all_constraints_types[geom_grp][lipid_code]:
            ns.cg_itps[lipid_code]['constraint'][constraint_id]['value'] = equi_val

    print(' ', geom_grp, '-- Avg:', geom_grp_avg, '-- Used:', equi_val, '-- Std:', geom_grp_std, '-- Range:',
          ns.params_val[geom_grp]['range'])

for geom_grp in ns.all_bonds_types:

    ns.params_val[geom_grp] = {'range': [np.inf, -np.inf],
                               'avg': []}  # find min/max values of bonds in the geom group at all selected temperatures + average of their average value
    if ns.tune_geoms:
        if ns.tune_bonds_equi_val or geom_grp in ns.specific_bonds_groups_for_tuning_equi_vals:
            ns.all_params_opti.append({geom_grp: 2})  # for bonds we tune both value and fct
        else:
            ns.all_params_opti.append(
                {geom_grp: 1})  # for bonds we tune fct only, the equi value comes either from AA or config file
    geom_grp_val = []

    for lipid_code in ns.all_bonds_types[geom_grp]:
        for bond_id in ns.all_bonds_types[geom_grp][lipid_code]:
            for temp in ns.lipids_codes[lipid_code]:

                values_dom = ns.cg_itps[lipid_code]['bond'][bond_id]['values_dom_' + temp]

                if ns.bonds_equi_val_from_config:
                    ns.params_val[geom_grp]['range'][0] = min(ns.params_val[geom_grp]['range'][0], values_dom[0],
                                                              ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp][
                                                                  'val'])
                    ns.params_val[geom_grp]['range'][1] = max(ns.params_val[geom_grp]['range'][1], values_dom[1],
                                                              ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp][
                                                                  'val'])
                else:
                    ns.params_val[geom_grp]['range'][0] = min(ns.params_val[geom_grp]['range'][0], values_dom[0])
                    ns.params_val[geom_grp]['range'][1] = max(ns.params_val[geom_grp]['range'][1], values_dom[1])
                # print('Geom', geom_grp, 'got his range defined/updated to:', ns.params_val[geom_grp]['range'])

                # if ns.geoms_val_from_cfg:
                # 	ns.params_val[geom_grp]['avg'].append(ns.cg_itps[lipid_code]['bond'][bond_id]['value']) # to start from config file bonds lengths
                # else:
                # ns.params_val[geom_grp]['avg'].append(ns.cg_itps[lipid_code]['bond'][bond_id]['avg_'+temp]) # to start from average bonds lengths from AA mapped data
                geom_grp_val.append(ns.cg_itps[lipid_code]['bond'][bond_id]['avg_' + temp])

    geom_grp_std = round(np.std(geom_grp_val), 3)
    geom_grp_avg = round(np.mean(geom_grp_val), 3)
    ns.params_val[geom_grp]['avg'] = geom_grp_avg

    if ns.bonds_equi_val_from_config:
        equi_val = ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['val']
    else:
        equi_val = geom_grp_avg

    for lipid_code in ns.all_bonds_types[geom_grp]:  # attribute average value
        for bond_id in ns.all_bonds_types[geom_grp][lipid_code]:
            ns.cg_itps[lipid_code]['bond'][bond_id]['value'] = equi_val

    print(' ', geom_grp, '-- Avg:', ns.params_val[geom_grp]['avg'], '-- Used:', equi_val, '-- Std:', geom_grp_std,
          '-- Range:', ns.params_val[geom_grp]['range'])

for geom_grp in ns.all_angles_types:

    ns.params_val[geom_grp] = {'range': [np.inf, -np.inf], 'avg': [],
                               'avg_real': []}  # find min/max values of angles in the geom group at all selected temperatures + average of their average value
    geom_grp_std = []

    if ns.tune_geoms:
        if ns.opti_config.init_bonded[ns.mapping_type][ns.solv][geom_grp]['val'] in ns.opti_config.fixed_angle_equi_vals:
            ns.all_params_opti.append({
                                          geom_grp: 1})  # for pre-defined angles the equilibrium value will be left untouched, but the force constant is tuned
        else:
            ns.all_params_opti.append(
                {geom_grp: 2})  # otherwise we tune both the equilibrium value and force constant

    for lipid_code in ns.all_angles_types[geom_grp]:
        for angle_id in ns.all_angles_types[geom_grp][lipid_code]:
            for temp in ns.lipids_codes[lipid_code]:
                values_dom = ns.cg_itps[lipid_code]['angle'][angle_id]['values_dom_' + temp]
                ns.params_val[geom_grp]['range'][0] = min(ns.params_val[geom_grp]['range'][0], values_dom[0])
                ns.params_val[geom_grp]['range'][1] = max(ns.params_val[geom_grp]['range'][1], values_dom[1])

                if ns.cg_itps[lipid_code]['angle'][angle_id]['value'] in ns.opti_config.fixed_angle_equi_vals:
                    ns.params_val[geom_grp]['avg'].append(ns.cg_itps[lipid_code]['angle'][angle_id][
                                                              'value'])  # to start from config file angle values
                else:
                    ns.params_val[geom_grp]['avg'].append(ns.cg_itps[lipid_code]['angle'][angle_id][
                                                              'avg_' + temp])  # to start from average angles values from AA mapped data
                ns.params_val[geom_grp]['avg_real'].append(ns.cg_itps[lipid_code]['angle'][angle_id]['avg_' + temp])
                geom_grp_std.append(ns.cg_itps[lipid_code]['angle'][angle_id]['avg_' + temp])

    geom_grp_std = round(np.std(geom_grp_std), 3)
    ns.params_val[geom_grp]['avg'] = round(np.mean(ns.params_val[geom_grp]['avg']), 1)
    ns.params_val[geom_grp]['avg_real'] = round(np.mean(ns.params_val[geom_grp]['avg_real']), 1)

    print(' ', geom_grp, '-- Avg:', ns.params_val[geom_grp]['avg_real'], '-- Used:', ns.params_val[geom_grp]['avg'],
          '-- Std:', geom_grp_std, '-- Range:', ns.params_val[geom_grp]['range'])

if ns.tune_radii:
    for radii_grp in sorted(ns.user_config.tune_radii_in_groups):
        # the radius of each bead type, that will be summed between pairs of beads to get the LJ sigmas
        ns.all_params_opti.append({f'r_{radii_grp}': 1})

if ns.tune_eps:
    # get the ranges within given percentage of variation allowed for EPS, when tuning within constrained range
    ns.eps_ranges = []  # will be an array (LJ nb) of arrays (min/max EPS), used only if ns.eps_perc_range != None

    for i in range(len(ns.all_beads_pairs)):
        LJ_curr = '_'.join(ns.all_beads_pairs[i])
        ns.all_params_opti.append({'LJ_' + LJ_curr: 1})
        bead_pair = ' '.join(ns.all_beads_pairs[i])

        if ns.eps_flat_range is not None:
            min_eps = max(ns.user_config.init_nonbonded[bead_pair]['eps'] - ns.eps_flat_range, ns.user_config.min_epsilon)
            max_eps = min(ns.user_config.init_nonbonded[bead_pair]['eps'] + ns.eps_flat_range, ns.user_config.max_epsilon)
            ns.eps_ranges.append([min_eps, max_eps])

print()
print('Number of parameters to be optimized per geom grp or LJ for all lipids:\n ', ns.all_params_opti)

# total number of free parameters to be optimized
nb_free_params = 0
for i in range(len(ns.all_params_opti)):
    for param in ns.all_params_opti[i]:
        nb_free_params += ns.all_params_opti[i][param]
print('\nNumber of free parameters to be optimized:', nb_free_params)

# find parameters associated to each lipid
print()
print('Parameters associated to each lipid')
for lipid_code in ns.lipid_params_opti:
    print(' ', lipid_code, 'parameters groups:\n   ', ns.lipid_params_opti[lipid_code])

###################################################################
# PLOT RDF FOR EACH BEADS PAIR FOR EACH LIPID AT ALL TEMPERATURES #
###################################################################

# RDFs -- Short cutoff
for lipid_code in ns.lipids_codes:

    nb_beads_types = len(ns.lipid_beads_types[lipid_code])
    fig = plt.figure(figsize=(nb_beads_types * 3, nb_beads_types * 3))
    ax = fig.subplots(nrows=nb_beads_types, ncols=nb_beads_types, squeeze=False)

    for i in range(nb_beads_types):  # matrix of LJ
        for j in range(nb_beads_types):

            if j >= i:
                for temp in ns.lipids_codes[lipid_code]:
                    bead_type_1, bead_type_2 = ns.lipid_beads_types[lipid_code][i], \
                                               ns.lipid_beads_types[lipid_code][j]
                    pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
                    _, rdf = ns.cg_itps[lipid_code]['rdf_' + temp + '_short'][pair_type]
                    rdf_norm = rdf / np.sum(rdf)  # norm to sum 1 for solo display
                    ax[i][j].plot(ns.bins_vol_shell - ns.bw_rdfs / 2, rdf_norm, label=temp)

                ax[i][j].set_title(bead_type_1 + ' ' + bead_type_2)
                ax[i][j].grid()
                ax[i][j].set_xlim(0, ns.cut_rdfs_short)
            # ax[i][j].legend()
            else:
                ax[i][j].set_visible(False)

    plt.suptitle(lipid_code + ' ' + ' '.join(ns.lipids_codes[lipid_code]))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        ns.exec_folder + '/RDF_ref_AA-mapped_' + lipid_code + '_cutoff_' + str(ns.cut_rdfs_short) + '_nm.png')
    plt.close(fig)


################################
# STORAGE OF RELATED OPTI DATA #
################################

params_all_str = ''  # comprehensive string of the parameters being optimized
nb_LJ = 0
for param_dict in ns.all_params_opti:  # list of dict having unique keys
    for param in param_dict:  # accessing each single key of each dict

        if param.startswith('B'):
            if param_dict[param] == 2:
                params_all_str += param + '_val '
            params_all_str += param + '_fct '

        elif param.startswith('A'):
            if param_dict[param] == 2:
                params_all_str += param + '_val '
            params_all_str += param + '_fct '

        elif param.startswith('r_'):
            params_all_str += param + ' '

        elif param.startswith('LJ'):
            params_all_str += 'LJ_' + '_'.join(ns.all_beads_pairs[nb_LJ]) + ' '
            nb_LJ += 1

params_all_str = params_all_str[:-1]  # remove the terminal whitespace

# next blocks are log files that store results at different levels of aggregation
# careful not to erase previous log files if we are doing a restart from FST-PSO checkpoint
if fstpso_checkpoint_in is None and not ns.next_cycle:
    with open(f'{ns.exec_folder}/{ns.opti_moves_file}', 'w') as fp:
        fp.write('n_cycle n_swarm_iter n_particle particle_score ' + params_all_str + '\n')
    with open(f'{ns.exec_folder}/{ns.opti_moves_details_lipid_temp_file}', 'w') as fp:
        fp.write('n_cycle n_swarm_iter n_particle lipid_code temp apl_avg apl_exp apl_delta thick_avg thick_exp thick_delta area_compress geoms_score apls_score_real apls_score_adapt thicks_score_real thicks_score_adapt rdfs_score\n')
    with open(f'{ns.exec_folder}/{ns.opti_moves_times_file}', 'w') as fp:
        fp.write('n_cycle n_swarm_iter swarm_iter_wc_time_sec swarm_iter_wc_time_h jobs_exec_wc_time_h jobs_analysis_wc_time_h master_elapsed_time_h master_remaining_time_h\n')

##################################
# ITERATIVE OPTIMIZATION PROCESS #
##################################

# somehow long cycles with calibrated restarts using ALL parameters
# (= no selection of bonds/angles/dihedrals/whatever like in Swarm-CG bonded version)
opti_cycles = {
    1: {'sim_time': ns.user_config.cg_time_prod, 'cg_sampling': 4 / 3 * ns.cg_time_prod, 'max_sw_iter': 100, 'max_sw_iter_no_new_best': 15}
}

# for tests without opti cycles
# opti_cycles = {1: {'sim_time': ns.cg_time_prod, 'cg_sampling': 4 / 3 * ns.cg_time_prod, 'max_sw_iter': 1, 'max_sw_iter_no_new_best': 1}}

# for tests with opti cycles
# opti_cycles = {1: {'sim_time': ns.cg_time_short, 'cg_sampling': 4/3*ns.cg_time_short, 'max_sw_iter': 1, 'max_sw_iter_no_new_best': 1, 'GEOMS': True, 'LJ': True},
# 			   2: {'sim_time': ns.cg_time_short, 'cg_sampling': 4/3*ns.cg_time_short, 'max_sw_iter': 1, 'max_sw_iter_no_new_best': 1, 'GEOMS': True, 'LJ': True},
# 			   3: {'sim_time': ns.cg_time_short, 'cg_sampling': 4/3*ns.cg_time_short, 'max_sw_iter': 1, 'max_sw_iter_no_new_best': 1, 'GEOMS': True, 'LJ': True}}

# NOTE: currently, due to an issue in FST-PSO, number of swarm iterations performed is +2 when compared to the numbers we feed

# state variables for the cycles of optimization
ns.best_fitness = [np.inf, None, None]  # fitness_score, eval_step_best_scored, err_dict_best_scored_eval
ns.lag_ref_params = [None,
                     None]  # best params yet for calculating the lag, best params found WITHIN the current swarm iteration (that will become the reference for the next swarm iter)

#############################
# START OPTIMIZATION CYCLES #
#############################

ns.start_opti_ts = datetime.now().timestamp()
ns.n_swarm_iter = n_swarm_iter  # also allows to count n_particles
ns.max_delta_ts_swarm_iter = max_delta_ts_swarm_iter  # maximum time (sec) for a complete SWARM iteration

# TODO: bring this back and find adequate criterion for convergence
# not used anymore because we monitor manually so the opti never really finishes / go to the end of script to print those
ns.total_eval_time, ns.total_gmx_time, ns.total_model_eval_time = 0, 0, 0

for i in range(n_cycle, len(opti_cycles) + 1):

    ns.n_cycle = i

    # parameters specific to current opti cycle
    ns.prod_sim_time = opti_cycles[i]['sim_time']
    ns.cg_sampling = opti_cycles[i]['cg_sampling']
    max_swarm_iter = opti_cycles[i]['max_sw_iter']
    max_swarm_iter_without_new_global_best = opti_cycles[i]['max_sw_iter_no_new_best']

    print()
    print(config.sep_close)
    print('| STARTING OPTIMIZATION CYCLE', ns.n_cycle, '                                                              |')
    print(config.sep_close)
    print()

    # build vector for search space boundaries + create variations around the BI initial guesses
    # no need to round the boundaries of the search space, even if parameters are rounded they are initially defined and stay within the intervals
    search_space_boundaries = get_search_space_boundaries(ns)
    # print(search_space_boundaries)

    ns.worst_fit_score = 100  # now that we consider everything is a percentage of error, the worst is 100% mismatch

    # formula used by FST-PSO to choose nb of particles, which initially comes from some statistical study
    # demonstrating this choice is optimal -- this also defines the number of initial guesses we need to provde
    ns.nb_particles = int(round(10 + 2 * np.sqrt(len(search_space_boundaries))))
    # ns.nb_particles = 2  # for tests

    # if we are starting the very first optimization cycle: initial guesses from config + calibrated noise
    if ns.n_swarm_iter == 1:
        initial_guess_list = get_initial_guess_list(ns)
    # if we restart in a new optimization cycle: initial guesses from previous selected points in previous opti cycle
    elif ns.next_cycle:
        initial_guess_list = get_cycle_restart_guess_list(ns)
    # if we are continuing within a currently on-going optimization cycle: NO guesses
    else:
        initial_guess_list = None

        # restart from a given checkpoint, if one was provided
        if fstpso_checkpoint_in is not None and not ns.next_cycle:
            print('Restarting from checkpoint:', fstpso_checkpoint_in, '\n')
            fstpso_checkpoint_in = f'{ns.exec_folder}/{fstpso_checkpoint_in}'

    # actual optimization
    FP = FuzzyPSO()
    FP.set_search_space(search_space_boundaries)
    FP.set_swarm_size(ns.nb_particles)
    FP.set_parallel_fitness(fitness=eval_function_parallel_swarm, arguments=ns, skip_test=True)
    result = FP.solve_with_fstpso(max_iter=max_swarm_iter, initial_guess_list=initial_guess_list,
                                  max_iter_without_new_global_best=max_swarm_iter_without_new_global_best,
                                  restart_from_checkpoint=fstpso_checkpoint_in,
                                  save_checkpoint=f'{ns.exec_folder}/{ns.fstpso_checkpoint_out}',
                                  verbose=False)
    backup_swarm_iter_logs_and_checkpoint(ns)

    # update data with parameters from the best scored models in the current opti cycle
    opti_cycle_best_params = result[0].X
    param_id, nb_LJ = 0, 0

    for param_dict in ns.all_params_opti:  # list of dict having unique keys
        for param in param_dict:  # accessing each single key of each dict

            # bonds, tune either: (1) both value and force constants or (2) fct only and use previously chosen bonds lengths (from average across geoms from all AA distribs, for example)
            if param.startswith('B') and ns.tune_geoms:
                if param_dict[param] == 2:
                    ns.params_val[param]['avg'] = opti_cycle_best_params[param_id]  # val
                    ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] = opti_cycle_best_params[
                        param_id + 1]  # fct
                else:
                    ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] = opti_cycle_best_params[param_id]  # fct
                param_id += param_dict[param]

            # angles, tune either: (1) both value and force constants or (2) fct only if angle equilibrium value was pre-defined
            if param.startswith('A') and ns.tune_geoms:
                if param_dict[param] == 2:
                    ns.params_val[param]['avg'] = opti_cycle_best_params[param_id]  # val
                    ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] = opti_cycle_best_params[
                        param_id + 1]  # fct
                else:
                    ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] = opti_cycle_best_params[param_id]  # fct
                param_id += param_dict[param]

            if param.startswith('r_'):
                radii_grp = param.split('_')[1]
                ns.user_config.init_beads_radii[radii_grp] = opti_cycle_best_params[param_id]
                param_id += param_dict[param]

            if param.startswith('LJ') and ns.tune_eps:
                ns.user_config.init_nonbonded[' '.join(ns.all_beads_pairs[nb_LJ])]['eps'] = opti_cycle_best_params[param_id]
                param_id += param_dict[param]
                nb_LJ += 1

print(f'\n\n## FINISHING NOW, AFTER {ns.n_swarm_iter - 1} SWARM ITERATIONS ##\n\n')

# NOTE: this is commented because we never go to the end of this script, and instead monitor manually the state of opti while running forever
# print some stats
# total_time_sec = datetime.now().timestamp() - ns.start_opti_ts
# total_time = round(total_time_sec / (60 * 60), 2)
# fitness_eval_time = round(ns.total_eval_time / (60 * 60), 2)
# init_time = round((total_time_sec - ns.total_eval_time) / (60 * 60), 2)
# ns.total_gmx_time = round(ns.total_gmx_time / (60 * 60), 2)
# ns.total_model_eval_time = round(ns.total_model_eval_time / (60 * 60), 2)
# print()
# print(config.sep_close)
# print('  FINISHED PROPERLY')
# print(config.sep_close)
# print()
# print('Total nb of evaluation steps:', ns.nb_eval_particle)
# print('Best model obtained at evaluation step number:', ns.best_fitness[1])
# print()
# print('Total execution time       :', total_time, 'h')
# print('Initialization time        :', init_time, 'h ('+str(round(init_time/total_time*100, 2))+' %)')
# print('Simulations/retrieval time :', ns.total_gmx_time, 'h ('+str(round(ns.total_gmx_time/total_time*100, 2))+' %)')
# print('Models scoring time        :', ns.total_model_eval_time, 'h ('+str(round(ns.total_model_eval_time/total_time*100, 2))+' %)')
# print()

