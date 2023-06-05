from shared.compare_models import compare_models_lipids
import config
import numpy as np
import matplotlib
matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import os


def get_score_and_plot_graphs_for_single_job(ns, nb_eval_particle, lipid_code, temp, sim_status, updated_cg_itps):

    # enter simulation directory for lipid + temperature
    os.chdir(f"{ns.user_config['exec_folder']}/{config.iteration_sim_files_dirname}{nb_eval_particle}/{lipid_code}_{temp}")

    if sim_status == 'success':

        delta_geoms, delta_geoms_per_grp, raw_delta_geoms, cg_apl, cg_thick, delta_rdfs, delta_rdfs_per_grp, error_data, area_compress, geoms_time, rdfs_time \
            = compare_models_lipids(ns, lipid_code, temp, updated_cg_itps)

        # scoring
        perc_delta_geoms = delta_geoms

        if ns.user_config['exp_data'][lipid_code][temp]['apl'] is not None:
            delta_apl = cg_apl['avg'] - ns.user_config['exp_data'][lipid_code][temp]['apl']
            delta_apl_abs = abs(delta_apl)
            perc_delta_apl_real = delta_apl_abs / ns.user_config['exp_data'][lipid_code][temp]['apl'] * 100
            perc_delta_apl_adapt = ns.user_config['apl_base_perc_error'] + max(0, delta_apl_abs - ns.apl_exp_error) / ns.user_config['exp_data'][lipid_code][temp]['apl'] * 100
            if perc_delta_apl_adapt > ns.user_config['apl_cap_perc_error']:
                perc_delta_apl_adapt = ns.user_config['apl_cap_perc_error']
        else:
            delta_apl, perc_delta_apl_real, perc_delta_apl_adapt = None, None, None

        if ns.user_config['exp_data'][lipid_code][temp]['Dhh'] is not None:
            delta_thick = cg_thick['avg'] - ns.user_config['exp_data'][lipid_code][temp]['Dhh']
            delta_thick_abs = abs(delta_thick)
            perc_delta_thick_real = delta_thick_abs / ns.user_config['exp_data'][lipid_code][temp]['Dhh'] * 100
            perc_delta_thick_adapt = ns.user_config['dhh_base_perc_error'] + max(0, delta_thick_abs - ns.dhh_exp_error) / ns.user_config['exp_data'][lipid_code][temp]['Dhh'] * 100
            if perc_delta_thick_adapt > ns.user_config['dhh_cap_perc_error']:
                perc_delta_thick_adapt = ns.user_config['dhh_cap_perc_error']
        else:
            delta_thick, perc_delta_thick_real, perc_delta_thick_adapt = None, None, None

        perc_delta_rdfs = delta_rdfs

        score_part = {'perc_delta_geoms': perc_delta_geoms, 'cg_apl': cg_apl, 'delta_apl': delta_apl, 'perc_delta_apl_real': perc_delta_apl_real,
                      'perc_delta_apl_adapt': perc_delta_apl_adapt, 'cg_thick': cg_thick, 'delta_thick': delta_thick,
                      'perc_delta_thick_real': perc_delta_thick_real, 'perc_delta_thick_adapt': perc_delta_thick_adapt,
                      'perc_delta_rdfs': perc_delta_rdfs, 'area_compress': area_compress, 'delta_geoms_per_grp': delta_geoms_per_grp,
                      'delta_rdfs_per_grp': delta_rdfs_per_grp}

        # try to locate error for later directed swarm initialization
        # if user has specified that we make use of the simulations available for this lipid for bottom-up scoring
        if ns.user_config['reference_AA_weight'][lipid_code][temp] > 0:
            nb_beads_types = len(ns.lipid_beads_types[lipid_code])
            err_mat_eps = np.zeros((nb_beads_types, nb_beads_types), dtype=np.float)

            for i in range(nb_beads_types):  # error matrix of eps
                for j in range(nb_beads_types):
                    if j >= i:
                        bead_type_1, bead_type_2 = ns.lipid_beads_types[lipid_code][i], ns.lipid_beads_types[lipid_code][j]
                        pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
                        err_mat_eps[i, j] = error_data['rdf_pair'][pair_type]
                    else:
                        err_mat_eps[i, j] = None

            # plot error mat for eps
            fig, ax = plt.subplots()
            cmap = CM.get_cmap('OrRd')
            cmap.set_bad('lavender')
            im = ax.imshow(err_mat_eps, cmap=cmap, vmin=0, vmax=25, aspect='equal')
            ax.set_xticks(np.arange(nb_beads_types))
            ax.set_yticks(np.arange(nb_beads_types))
            ax.set_xticklabels(ns.lipid_beads_types[lipid_code])
            ax.set_yticklabels(ns.lipid_beads_types[lipid_code])
            ax.xaxis.tick_top()
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Error', rotation=-90, va="bottom")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.suptitle(lipid_code + ' ' + temp + ' -- Error quantif. for EPS')
            # plt.show()
            plt.savefig('EPS_ERROR_' + lipid_code + '_' + temp + '.png')
            plt.close(fig)

    else:  # simulation did NOT finish properly, attribute the pre-defined worst score

        geoms_time, rdfs_time = 0, 0
        error_data = {}
        score_part = {'perc_delta_geoms': ns.worst_fit_score, 'cg_apl': ns.worst_fit_score, 'delta_apl': ns.worst_fit_score, 'perc_delta_apl_real': ns.worst_fit_score,
                      'perc_delta_apl_adapt': ns.worst_fit_score, 'cg_thick': ns.worst_fit_score, 'delta_thick': ns.worst_fit_score,
                      'perc_delta_thick_real': ns.worst_fit_score, 'perc_delta_thick_adapt': ns.worst_fit_score,
                      'perc_delta_rdfs': ns.worst_fit_score, 'area_compress': None, 'delta_geoms_per_grp': {},
                      'delta_rdfs_per_grp': {}}

    os.chdir('../../..')  # exit lipid + temp simulation directory

    return score_part, geoms_time, rdfs_time, error_data
