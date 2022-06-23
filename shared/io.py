import contextlib
import os, sys, shutil
import config
import numpy as np
import matplotlib

matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import copy


# print forced stdout enabled
def print_stdout_forced(*args, **kwargs):
    with contextlib.redirect_stdout(sys.__stdout__):
        print(*args, **kwargs, flush=True)

    return


# modify MDP file to adjust simulation temperature, duration or other parameters
def print_mdp_file(ns, lipid_code, temp, mdp_filename_in, mdp_filename_out, sim_type):
    # read input
    with open(mdp_filename_in, 'r') as fp:
        mdp_lines_in = fp.read().split('\n')
        mdp_lines = [mdp_line.split(';')[0].strip() for mdp_line in mdp_lines_in]  # split for comments

    # go through file and replace what was requested
    if sim_type != 'mini':  # for mini there is no temperature/dt/nsteps/whatever so it will be a simple copy
        for i in range(len(mdp_lines)):
            mdp_line = mdp_lines[i]

            # simulation time and output writing frequency (xtc)
            if mdp_line.startswith('dt'):
                dt = float(mdp_line.split()[2])
            if mdp_line.startswith('nsteps'):
                if sim_type == 'prod':
                    sim_steps = str(round(ns.prod_sim_time * 1000 / dt))
                elif sim_type == 'equi':
                    sim_steps = str(round(ns.user_config['cg_time_equi'][lipid_code][temp] * 1000 / dt))
                mdp_lines_in[i] = mdp_line.replace('PLACEHOLDER', sim_steps) + '    ; automatically modified'
            if mdp_line.startswith('nstxout-compressed'):
                xtc_write_freq = str(round(ns.cg_sampling / dt))
                mdp_lines_in[i] = mdp_line.replace('PLACEHOLDER', xtc_write_freq) + '    ; automatically modified'

            # temperature
            if mdp_line.startswith('ref-t') or mdp_line.startswith('ref_t'):
                if 'PLACEHOLDER PLACEHOLDER' in mdp_line:
                    mdp_lines_in[i] = mdp_line.replace('PLACEHOLDER PLACEHOLDER',
                                                       temp[:-1] + ' ' + temp[:-1]) + '    ; automatically modified'
                else:
                    mdp_lines_in[i] = mdp_line.replace('PLACEHOLDER', temp[:-1]) + '    ; automatically modified'
            elif mdp_line.startswith('gen-temp') or mdp_line.startswith('gen_temp'):
                mdp_lines_in[i] = mdp_line.replace('PLACEHOLDER', temp[:-1]) + '    ; automatically modified'

    # replace MDP file's content by the modified one
    with open(mdp_filename_out, 'w') as fp:
        for mdp_line in mdp_lines_in:
            fp.write(mdp_line + '\n')


# print the readable + log output for a single job that was executed as part the calculations within a single particle
# NOTE: SUPER careful the day we put f-strings everywhere because this output is basically all the results we later analyze
def print_job_output(ns, swarm_res, nb_eval_particle, lipid_code, temp):

    # enter simulation directory for lipid + temperature
    os.chdir(f"{ns.user_config['exec_folder']}/{config.iteration_sim_files_dirname}{nb_eval_particle}/{lipid_code}_{temp}")

    sim_status = swarm_res[nb_eval_particle][lipid_code][temp]['status']
    score_part = swarm_res[nb_eval_particle][lipid_code][temp]['score_part']

    print(lipid_code, temp, '---------------------------------------------------------')

    if sim_status == 'success':
        print()

        with open('../../' + ns.opti_moves_details_lipid_temp_file, 'a') as fp:

            if ns.user_config['exp_data'][lipid_code][temp]['apl'] is not None:
                str_delta_apl = str(round(score_part['delta_apl'], 4))
                str_perc_delta_apl_real = str(round(score_part['perc_delta_apl_real'], 4))
                str_perc_delta_apl_adapt = str(round(score_part['perc_delta_apl_adapt'], 4))
                print('  Δ APL:', round(score_part['delta_apl'], 4),
                      '(' + str(round(score_part['perc_delta_apl_real'], 2)) + '%)', '              (APL avg:',
                      round(score_part['cg_apl']['avg'], 3), '+/-',
                      str(round(score_part['cg_apl']['std'], 3)) + ')      Score adapted APL:',
                      str(round(score_part['perc_delta_apl_adapt'], 2)) + '%')
            else:
                str_delta_apl = 'None'
                str_perc_delta_apl_real = 'None'
                str_perc_delta_apl_adapt = 'None'
                print('  CG APL:', round(score_part['cg_apl']['avg'], 4), '(no exp val)', '              (APL avg:',
                      round(score_part['cg_apl']['avg'], 3),
                      '+/-', str(round(score_part['cg_apl']['std'], 3)) + ')')

            if ns.user_config['exp_data'][lipid_code][temp]['Dhh'] is not None:
                str_delta_thick = str(round(score_part['delta_thick'], 4))
                str_perc_delta_thick_real = str(round(score_part['perc_delta_thick_real'], 4))
                str_perc_delta_thick_adapt = str(round(score_part['perc_delta_thick_adapt'], 4))
                print('  Δ Thick Dhh:', round(score_part['delta_thick'], 4),
                      '(' + str(round(score_part['perc_delta_thick_real'], 2)) + '%)',
                      '       (Dhh avg:', round(score_part['cg_thick']['avg'], 3), '+/-',
                      str(round(score_part['cg_thick']['std'], 3)) + ')      Score adapted Dhh:',
                      str(round(score_part['perc_delta_thick_adapt'], 2)) + '%')
            else:
                str_delta_thick = 'None'
                str_perc_delta_thick_real = 'None'
                str_perc_delta_thick_adapt = 'None'

                print('  CG Thick Dhh:', round(score_part['cg_thick']['avg'], 4), '(no exp val)',
                      '       (Dhh avg:',
                      round(score_part['cg_thick']['avg'], 3),
                      '+/-', str(round(score_part['cg_thick']['std'], 3)) + ')')

            # if user has specified that we make use of the simulations available for this lipid for bottom-up scoring
            str_rdf_scoring = ""
            if not ns.user_config['score_rdfs']:
                str_rdf_scoring = "(absent from score aggregation)"
            if ns.user_config['reference_AA_weight'][lipid_code] > 0:
                print(f"  Δ Geoms: {round(score_part['perc_delta_geoms'], 4)} ({round(score_part['perc_delta_geoms'], 2)} %)")
                print(f"  Δ RDFs: {round(score_part['perc_delta_rdfs'], 4)} ({round(score_part['perc_delta_rdfs'], 2)} %)   {str_rdf_scoring}")
            else:
                print(f"  Δ Geoms: Inactive (no ref. AA traj)")
                print(f"  Δ RDFs: Inactive (no ref. AA traj)")
            print(f"\n  Area compressibility: {round(score_part['area_compress'], 1)} mN/m     (NOT in the score atm)\n")

            fp.write(str(ns.n_cycle) + ' ' + str(ns.n_swarm_iter) + ' ' + str(nb_eval_particle) + ' ' + lipid_code + ' ' + temp + ' ' + str(
                round(score_part['cg_apl']['avg'], 4)) + ' ' + str(
                ns.user_config['exp_data'][lipid_code][temp]['apl']) + ' ' + str_delta_apl + ' ' + str(
                round(score_part['cg_thick']['avg'], 4)) + ' ' + str(
                ns.user_config['exp_data'][lipid_code][temp]['Dhh']) + ' ' + str_delta_thick + ' ' + str(
                round(score_part['area_compress'], 1)) + ' ' + str(round(score_part['perc_delta_geoms'],
                                                                         4)) + ' ' + str_perc_delta_apl_real + ' ' + str_perc_delta_apl_adapt + ' ' + str_perc_delta_thick_real + ' ' + str_perc_delta_thick_adapt + ' ' + str(
                round(score_part['perc_delta_rdfs'], 4)) + '\n')

    else:
        print()
        print_stdout_forced(f'  {sim_status.upper()}\n')

        # record parameters
        with open(f'../../{ns.opti_moves_details_lipid_temp_file}', 'a') as fp:
            fp.write(f'{ns.n_cycle} {ns.n_swarm_iter} {nb_eval_particle} {lipid_code} {temp} None None None None None None None None None None None None None\n')

    nb_cg_frames = 'TODO'  # TODO: put the count of CG frames here
    geoms_time = round(swarm_res[nb_eval_particle][lipid_code][temp]['geoms_time'] / 60, 2)
    rdfs_time = round(swarm_res[nb_eval_particle][lipid_code][temp]['rdfs_time'] / 60, 2)
    print(f"  Time for simulation: {swarm_res[nb_eval_particle][lipid_code][temp]['time_elapsed_str']} (hh:mm:ss)")
    print(f"  Time for geoms distrib. calculation: {geoms_time} min  ({nb_cg_frames} frames)")
    print(f"  Time for RDFs calculation: {rdfs_time} min  ({ns.rdf_frames} frames)")
    print()

    os.chdir('../../..')  # exit lipid + temp simulation directory

    return


# update coarse-grain ITP
def update_cg_itp_obj(ns, parameters_set):
    current_cg_itps = copy.deepcopy(ns.cg_itps)

    param_cursor = 0
    for param in ns.all_params_opti:  # list of dict having unique keys
        param_short = param.split('_')[0]

        if param.startswith('B') and param.endswith('val'):
            for lipid_code in ns.all_bonds_types[param_short]:
                for bond_id in ns.all_bonds_types[param_short][lipid_code]:
                    current_cg_itps[lipid_code]['bond'][bond_id]['value'] = round(parameters_set[param_cursor], 3)
            param_cursor += 1

        elif param.startswith('B') and param.endswith('fct'):
            for lipid_code in ns.all_bonds_types[param_short]:
                for bond_id in ns.all_bonds_types[param_short][lipid_code]:
                    current_cg_itps[lipid_code]['bond'][bond_id]['fct'] = round(parameters_set[param_cursor], 2)
            param_cursor += 1

        elif param.startswith('A') and param.endswith('val'):
            for lipid_code in ns.all_angles_types[param_short]:
                for angle_id in ns.all_angles_types[param_short][lipid_code]:
                    current_cg_itps[lipid_code]['angle'][angle_id]['value'] = round(parameters_set[param_cursor], 3)
            param_cursor += 1

        elif param.startswith('A') and param.endswith('fct'):
            for lipid_code in ns.all_angles_types[param_short]:
                for angle_id in ns.all_angles_types[param_short][lipid_code]:
                    current_cg_itps[lipid_code]['angle'][angle_id]['fct'] = round(parameters_set[param_cursor], 2)
            param_cursor += 1

    return current_cg_itps, param_cursor


# print coarse-grain ITP
def print_cg_itp_file(itp_obj, out_path_itp, print_sections=['constraint', 'bond', 'angle', 'dihedral', 'exclusion']):
    with open(out_path_itp, 'w') as fp:

        fp.write('[ moleculetype ]\n')
        fp.write('; molname        nrexcl\n')
        fp.write('{0:<4} {1:>13}\n'.format(itp_obj['moleculetype']['molname'], itp_obj['moleculetype']['nrexcl']))

        fp.write('\n\n[ atoms ]\n')
        fp.write('; id type resnr residue   atom  cgnr    charge     mass\n\n')

        for i in range(len(itp_obj['atoms'])):

            if 'mass_and_eol' in itp_obj['atoms'][i]:
                fp.write('{0:<4} {1:>4}    {6:>2}  {2:>6} {3:>6}  {4:<4} {5:9.5f} {7}\n'.format(
                    itp_obj['atoms'][i]['bead_id'] + 1, itp_obj['atoms'][i]['bead_type'],
                    itp_obj['atoms'][i]['residue'], itp_obj['atoms'][i]['atom'], i + 1, itp_obj['atoms'][i]['charge'],
                    itp_obj['atoms'][i]['resnr'], itp_obj['atoms'][i]['mass_and_eol']))
            else:
                fp.write('{0:<4} {1:>4}    {6:>2}  {2:>6} {3:>6}  {4:<4} {5:9.5f}\n'.format(
                    itp_obj['atoms'][i]['bead_id'] + 1, itp_obj['atoms'][i]['bead_type'],
                    itp_obj['atoms'][i]['residue'], itp_obj['atoms'][i]['atom'], i + 1, itp_obj['atoms'][i]['charge'],
                    itp_obj['atoms'][i]['resnr']))

        if 'constraint' in print_sections and 'constraint' in itp_obj and len(itp_obj['constraint']) > 0:
            fp.write('\n\n[ constraints ]\n')
            fp.write(';   i     j   funct   length                     geom_grp\n')

            for j in range(len(itp_obj['constraint'])):
                grp_val = itp_obj['constraint'][j]['value']
                fp.write('{beads[0]:>5} {beads[1]:>5} {0:>7} {1:8.3f}                     ; {2}\n'.format(
                    itp_obj['constraint'][j]['funct'], grp_val, itp_obj['constraint'][j]['geom_grp'],
                    beads=[bead_id + 1 for bead_id in itp_obj['constraint'][j]['beads'][0]]))

        if 'bond' in print_sections and 'bond' in itp_obj and len(itp_obj['bond']) > 0:
            fp.write('\n\n[ bonds ]\n')
            fp.write(';   i     j   funct   length   force.c.          geom_grp\n')

            for j in range(len(itp_obj['bond'])):
                grp_val, grp_fct = itp_obj['bond'][j]['value'], itp_obj['bond'][j]['fct']
                fp.write('{beads[0]:>5} {beads[1]:>5} {0:>7} {1:8.3f}   {2:>7.2f}           ; {3}\n'.format(
                    itp_obj['bond'][j]['funct'], grp_val, grp_fct, itp_obj['bond'][j]['geom_grp'],
                    beads=[bead_id + 1 for bead_id in itp_obj['bond'][j]['beads'][0]]))

        if 'angle' in print_sections and 'angle' in itp_obj and len(itp_obj['angle']) > 0:
            fp.write('\n\n[ angles ]\n')
            fp.write(';   i     j     k   funct     angle   force.c.        geom_grp\n')

            for j in range(len(itp_obj['angle'])):
                grp_val, grp_fct = itp_obj['angle'][j]['value'], itp_obj['angle'][j]['fct']
                fp.write('{beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {0:>7} {1:9.2f} {2:>7.2f}           ; {3}\n'.format(
                    itp_obj['angle'][j]['funct'], grp_val, grp_fct, itp_obj['angle'][j]['geom_grp'],
                    beads=[bead_id + 1 for bead_id in itp_obj['angle'][j]['beads'][0]]))

        if 'dihedral' in print_sections and 'dihedral' in itp_obj and len(itp_obj['dihedral']) > 0:
            fp.write('\n\n[ dihedrals ]\n')
            fp.write(';   i     j     k     l   funct     dihedral   force.c.   mult.    geom_grp\n')

            for j in range(len(itp_obj['dihedral'])):
                grp_val, grp_fct = itp_obj['dihedral'][j]['value'], itp_obj['dihedral'][j]['fct']
                multiplicity = ''  # NOTE: this is not ready for dihedrals
                fp.write(
                    '{beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {beads[3]:>5} {0:>7}    {1:9.2f} {2:>7.2f}       {5}     ; {3}\n'.format(
                        itp_obj['dihedral'][j]['funct'], grp_val, grp_fct, itp_obj['dihedral'][j]['geom_grp'],
                        multiplicity, beads=[bead_id + 1 for bead_id in itp_obj['dihedral'][j]['beads'][0]]))
                # TODO: check this the day we use dihedrals again

        if 'exclusion' in print_sections and 'exclusion' in itp_obj and len(itp_obj['exclusion']) > 0:
            fp.write('\n\n[ exclusions ]\n')
            fp.write(';   i     j\n\n')

            for j in range(len(itp_obj['exclusion'])):
                fp.write(('{:>4} ' * len(itp_obj['exclusion'][j]) + '\n').format(
                    *[bead_id + 1 for bead_id in itp_obj['exclusion'][j]]))

        fp.write('\n\n')

    return


# create FF ITP file
def print_ff_file(ns, parameters_set, param_cursor, out_dir):

    # TODO: handle the printing of the solvent parameters better than just hard-coded here
    #       this means either printing all solvent-to-other beads LJ parameters
    #       or just the necessary ones
    #       ALSO importantly (!!) make sure that we are not optimizing the LJ interactions
    #       for beads that are the ones used to represent the solvent, in the case of MARTINI

    with open(f'{out_dir}/force_field.itp', 'w') as fp:
        fp.write('''[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
  1      2         no        1.0     1.0

[ atomtypes ]
; name      mass  chrg ptype sig eps\n''')

        for bead_type in ns.all_beads_types:
            fp.write('  {0:<8} {1:>3.1f}  0.0  A     0   0\n'.format(bead_type, ns.user_config['beads_masses'][bead_type]))

        # write WATER bead description if we are in WET
        if ns.user_config['solv'] == 'WET' and ns.user_config['mapping_type'] == 'MARTINI2':
            fp.write('  {0:<8} {1:>3.1f}  0.0  A     0   0\n'.format('P4', 72))

        elif ns.user_config['solv'] == 'WET' and ns.user_config['mapping_type'] == 'MARTINI3REMAP':
            fp.write('  {0:<8} {1:>3.1f}  0.0  A     0   0\n'.format('W', 72))

        elif ns.user_config['solv'] == 'WET' and ns.user_config['mapping_type'] == 'MARTINI3':
            fp.write('  {0:<8} {1:>3.1f}  0.0  A     0   0\n'.format('W', 72))

        fp.write('''\n[ nonbond_params ]
;  part part   func sig      eps\n''')

        # radii come before LJ EPS in the vector of parameters
        if ns.user_config['tune_radii']:
            new_radii = {}
            for radii_grp in sorted(ns.user_config['tune_radii_in_groups']):
                new_radii[radii_grp] = parameters_set[param_cursor]
                param_cursor += 1

        # plot a matrix of LJ EPS interactions strength
        nb_beads_types = len(ns.all_beads_types)
        mat_eps = np.zeros((nb_beads_types, nb_beads_types), dtype=np.float)
        mat_eps = mat_eps - 1  # for upper triangle display

        # write the FF file
        for nb_LJ in range(len(ns.all_beads_pairs)):
            bead_type_1, bead_type_2 = ns.all_beads_pairs[nb_LJ]
            bead_pair = ' '.join(ns.all_beads_pairs[nb_LJ])

            if ns.user_config['tune_radii']:
                sig = new_radii[ns.reverse_radii_mapping[bead_type_1]] + new_radii[ns.reverse_radii_mapping[bead_type_2]]  # here sig is the sum of radii of each bead type
            else:
                sig = ns.user_config['init_nonbonded'][bead_pair]['sig']  # here the sig is predefined

            if ns.user_config['tune_epsilons'] == 'all' or bead_pair in ns.user_config['tune_epsilons']:
                eps = parameters_set[param_cursor]  # get the LJ from the optimizer
                param_cursor += 1
            else:
                eps = ns.user_config['init_nonbonded'][' '.join(ns.all_beads_pairs[nb_LJ])]['eps']  # get initial LJ from existing FF data

            fp.write('{0:>7} {1:<7}1    {2:1.4f}   {3:1.4f}\n'.format(bead_type_1, bead_type_2, sig, eps))

            i, j = ns.all_beads_types.index(bead_type_1), ns.all_beads_types.index(bead_type_2)
            mat_eps[i, j] = eps

        mat_eps[mat_eps == -1] = None  # for upper triangle display

        # plot a matrix of LJ EPS interactions strength
        fig, ax = plt.subplots()
        cmap = CM.get_cmap('YlGnBu')
        cmap.set_bad('lavender')
        im = ax.imshow(mat_eps, cmap=cmap, vmin=0, aspect='equal')
        ax.set_xticks(np.arange(nb_beads_types))
        ax.set_yticks(np.arange(nb_beads_types))
        ax.set_xticklabels(ns.all_beads_types)
        ax.set_yticklabels(ns.all_beads_types)
        ax.xaxis.tick_top()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('LJ EPS', rotation=-90, va="bottom")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('FF all LJ EPS')
        # plt.show()
        plt.savefig(f'{out_dir}/EPS_force_field.png')
        plt.close(fig)

        if ns.user_config['solv'] == 'WET' and ns.user_config['mapping_type'] == 'MARTINI3REMAP':

            water2beads = {
                'Q1':   '     W Q1     1    0.465   5.220\n',
                'Q5':   '     W Q5     1    0.465   6.340\n',
                'N4a':  '     W N4a    1    0.465   3.500\n',
                'SN4a': '     W SN4a   1    0.425   3.000\n',
                'C1':   '     W C1     1    0.470   2.060\n',
                'C4h':  '     W C4h    1    0.465   2.420\n',
                'SC2':  '     W SC2    1    0.425   1.690\n',
                'SC4h': '     W SC4h   1    0.425   1.800\n'
            }

            fp.write('     W W     1    0.470   4.650\n')  # this one is always present
            for bead_type in ns.all_beads_types:  # the other are not forcefully mandatory
                fp.write(water2beads[bead_type])

            fp.write('''

;;;;;; WATER (representing 4 molecules)

[ moleculetype ]
; molname  	nrexcl
  W  	    	1

[ atoms ]
;id     type    resnr   residu  atom    cgnr    charge
 1      W      1        W       W      1       0

''')

        elif ns.user_config['solv'] == 'WET' and ns.user_config['mapping_type'] == 'MARTINI3':

            water2beads = {
                'Q1':   '     W  Q1     1    0.465   5.220\n',
                'Q5':   '     W  Q5     1    0.465   6.340\n',
                'N4a':  '     W  N4a    1    0.465   3.500\n',
                'C1':   '     W  C1     1    0.470   2.060\n',
                'C4h':  '     W  C4h    1    0.465   2.420\n',
                'SN4a': '     W  SN4a   1    0.425   3.000\n'
            }

            fp.write('     W  W      1    0.470   4.650\n')  # this one is always present
            for bead_type in ns.all_beads_types:  # the other are not forcefully mandatory
                fp.write(water2beads[bead_type])

            fp.write('''

    ;;;;;; WATER (representing 4 H2O molecules)

    [ moleculetype ]
    ; molname  	nrexcl
      W  	    	1

    [ atoms ]
    ;id 	type 	resnr 	residu 	atom 	cgnr 	charge
     1 	W  	1 	W  	W 	1 	0 

    ''')

    return


def backup_swarm_iter_logs_and_checkpoint(ns):
    # BACKUP THE FST-PSO CHECKPOINT THAT WAS JUST WRITTEN AT THE END OF PREVIOUS SWARM ITERATION
    # so the checkpoints swarm iteration index corresponds to the END of an iteration
    # anyways if we need these files on day, we will treat checkpoints and logs manually
    # this is just for safety in case something goes seriously wrong (= code crashes and we need to recover an opti)
    checkpoint_to_backup = f"{ns.user_config['exec_folder']}/{ns.fstpso_checkpoint_out}"
    # remove file extension and add details
    fstpso_checkpoint_out_copy = f'{ns.fstpso_checkpoint_out[:-4]}_end_swarm_iter_{ns.n_swarm_iter - 1}.obj'
    if os.path.isfile(checkpoint_to_backup):
        # just for security when we manually change directories for continuing opti cycles
        if not os.path.isdir(f"{ns.user_config['exec_folder']}/CHECKPOINTS_BACKUP"):
            os.mkdir(f"{ns.user_config['exec_folder']}/CHECKPOINTS_BACKUP")
        shutil.copy(checkpoint_to_backup, f"{ns.user_config['exec_folder']}/CHECKPOINTS_BACKUP/{fstpso_checkpoint_out_copy}")

    # also the output logs just in case
    shutil.copy(f"{ns.user_config['exec_folder']}/{ns.opti_moves_file}", f"{ns.user_config['exec_folder']}/{ns.opti_moves_file[:-4]}_end_swarm_iter_{ns.n_swarm_iter - 1}.log")
    shutil.copy(f"{ns.user_config['exec_folder']}/{ns.opti_moves_details_lipid_temp_file}", f"{ns.user_config['exec_folder']}/{ns.opti_moves_details_lipid_temp_file[:-4]}_end_swarm_iter_{ns.n_swarm_iter - 1}.log")
    shutil.copy(f"{ns.user_config['exec_folder']}/{ns.opti_moves_times_file}", f"{ns.user_config['exec_folder']}/{ns.opti_moves_times_file[:-4]}_end_swarm_iter_{ns.n_swarm_iter - 1}.log")

def check_log_files(path_log_file, particle=True):
    if os.path.isfile(path_log_file):
        with open(path_log_file, 'r') as fp:
            if particle:
                try:
                    n_cycle, n_swarm_iter, n_particle = map(int, fp.read().splitlines()[-1].split(' ')[:3])
                except ValueError:  # we have just read the header
                    n_cycle, n_swarm_iter, n_particle = None, None, None
                return n_cycle, n_swarm_iter, n_particle
            else:
                try:
                    n_cycle, n_swarm_iter = map(int, fp.read().splitlines()[-1].split(' ')[:2])
                except ValueError:  # we have just read the header
                    n_cycle, n_swarm_iter = None, None
                return n_cycle, n_swarm_iter
    else:
        sys.exit(
            f'\nExpected Swarm-CG log file at: {path_log_file}'
            f'\nIt seems that the optimization that you are trying to continue did not go far enough to restart,'
            f'\nor maybe files have been altered in some way.'
            f'\nTo be able to restart an optimization, this optimization must have gone past the 2 first swarm iterations.'
        )
