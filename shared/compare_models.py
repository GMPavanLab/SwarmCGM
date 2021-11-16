import matplotlib
matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import numpy as np
import config
from datetime import datetime
import os, sys
import collections
from pyemd import emd
from scipy import constants


# calculate bonds distribution from CG trajectory
def get_CG_bonds_distrib_lipids(ns, beads_ids, cg_iter_universe, bins, bw):
    bond_values = np.empty(len(cg_iter_universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

    for ts in cg_iter_universe.trajectory:  # no need for PBC handling, trajectories were made wholes for each molecule

        for i in range(len(beads_ids)):
            bead_id_1, bead_id_2 = beads_ids[i]
            bead_pos_1[i], bead_pos_2[i] = cg_iter_universe.atoms[bead_id_1].position, cg_iter_universe.atoms[
                bead_id_2].position

        mda.lib.distances.calc_bonds(bead_pos_1, bead_pos_2, backend=ns.mda_backend, box=None, result=frame_values)
        bond_values[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values / 10  # retrieved nm

    bond_avg = round(np.mean(bond_values), 3)
    bond_hist = np.histogram(bond_values, bins, density=True)[0] * bw  # retrieve 1-sum densities

    return bond_avg, bond_hist


# calculate angles using MDAnalysis
def get_CG_angles_distrib_lipids(ns, beads_ids, cg_iter_universe):
    angle_values_rad = np.empty(len(cg_iter_universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

    for ts in cg_iter_universe.trajectory:  # no need for PBC handling, trajectories were made wholes for each molecule

        for i in range(len(beads_ids)):
            bead_id_1, bead_id_2, bead_id_3 = beads_ids[i]
            bead_pos_1[i], bead_pos_2[i], bead_pos_3[i] = cg_iter_universe.atoms[bead_id_1].position, \
                                                          cg_iter_universe.atoms[bead_id_2].position, \
                                                          cg_iter_universe.atoms[bead_id_3].position

        mda.lib.distances.calc_angles(bead_pos_1, bead_pos_2, bead_pos_3, backend=ns.mda_backend, box=None,
                                      result=frame_values)
        angle_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    angle_values_deg = np.rad2deg(angle_values_rad)
    angle_avg = round(np.mean(angle_values_deg), 3)
    angle_hist = np.histogram(angle_values_deg, ns.bins_angles, density=True)[
                     0] * ns.user_config['bw_angles']  # retrieve 1-sum densities

    return angle_avg, angle_hist


# compare 2 models -- atomistic and CG models with plotting
def compare_models_lipids(ns, lipid_code, temp, updated_cg_itps, tpr_file='prod.tpr', xtc_file='prod.xtc'):

    # find if user has specified that we make use of the simulations available for this lipid
    # in the bottom-up component of the score
    bottom_up_active = False
    if ns.user_config['reference_AA_weight'][lipid_code] > 0:
        bottom_up_active = True

    # graphical parameters
    plt.rcParams['grid.color'] = 'k'  # plt grid appearance settings
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.5

    cg_map_itp = ns.cg_itps[
        lipid_code]  # access FROM THERE ONLY the stored AA mapped CG ITP for which we have geoms distributions already
    cg_iter_itp = {'geoms': {'constraint': {}, 'bond': {}, 'angle': {}}, 'rdfs_short': {},
                   'rdfs_long': {}}  # create storage for CG data from current iteration
    updated_cg_itp = updated_cg_itps[lipid_code]  # contains the parameters for the current eval

    # read CG traj of current simulation
    cg_iter_universe = mda.Universe(tpr_file, xtc_file, in_memory=True, refresh_offsets=True, guess_bonds=False)

    # select each molecule as an MDA atomgroup and make its coordinates whole, inplace, across the complete CG trajectory
    for _ in cg_iter_universe.trajectory:
        # make each lipid whole
        for i in range(cg_map_itp['meta']['nb_mols']):
            cg_mol = mda.AtomGroup(
                [i * cg_map_itp['meta']['nb_beads'] + bead_id for bead_id in range(cg_map_itp['meta']['nb_beads'])],
                cg_iter_universe)
            mda.lib.mdamath.make_whole(cg_mol, inplace=True)

    # 1. get APL
    # get dimensions of the box at each timestep to calculate CG APL
    x_boxdims = []
    for ts in cg_iter_universe.trajectory:
        x_boxdims.append(
            ts.dimensions[0])  # X-axis box size, Y is in principle identical and Z size is orthogonal to the bilayer
    x_boxdims = np.array(x_boxdims)

    cg_iter_itp['apl'] = {'avg': round(np.mean(x_boxdims ** 2 / (cg_map_itp['meta']['nb_mols'] / 2)) / 100, 4),
                          'med': round(np.median(x_boxdims ** 2 / (cg_map_itp['meta']['nb_mols'] / 2)) / 100, 4),
                          'std': round(np.std(x_boxdims ** 2 / (cg_map_itp['meta']['nb_mols'] / 2)) / 100, 4)}  # nm**2

    # 2. get Thickness
    # Dhh thickness definition: here we will use the Phosphates positions to get average Z-axis positive and negative values, then use the distance between those points
    # NOTE: we decided not to care about flip flops to calculate Z-axis positions, as this is probably done the same way in experimental calculations
    #       (anyway the impact would be very small but really, who knows ??)

    # get the id of the bead that should be used as reference for Dhh calculation + the delta for Dhh calculation, if any
    head_type = lipid_code[2:]  # NOTE: when we start incorporating lipids for which the code is not 4 letters this won't hold
    phosphate_bead_id = int(ns.user_config['phosphate_pos'][head_type]['CG']) - 1

    # to ensure thickness calculations are not affected by bilayer being split on Z-axis PBC
    # for each frame, calculate 7 thicknesses:
    # -- 1. without changing anything
    # -- 2. by shifting the upper half of the box below the lower half
    # -- 3. by shifting the upper third of the box below the lower 2 other thirds
    # -- 4. by shifting the upper quarter of the box below the lower 3 other quarters
    # -- 5. by shifting the lower half of the box over the upper half
    # -- 6. by shifting the lower third of the box over the upper 2 other thirds
    # -- 7. by shifting the lower quarter of the box over the upper 3 other quarters
    # (the 3-7. were added for WET because shifting only half causes problems for ~ small boxes, but we don't want to increase box size, to limit computation time)
    # then we select the minimum thickness value among those 4 definitions, so this in principle is very robust to any box sizes

    phos_z_dists = []

    for ts in cg_iter_universe.trajectory:

        z_all = np.empty(cg_map_itp['meta']['nb_mols'])
        for i in range(cg_map_itp['meta']['nb_mols']):
            id_phos = i * cg_map_itp['meta']['nb_beads'] + phosphate_bead_id
            z_phos = cg_iter_universe.atoms[id_phos].position[2]
            z_all[i] = z_phos

        # 1. without correction
        z_avg = np.mean(z_all)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all[i] > z_avg:
                z_pos.append(z_all[i])
            else:
                z_neg.append(z_all[i])
        phos_z_dists_1 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 2. with correction (half upper)
        z_all_corr = z_all.copy()
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > ts.dimensions[2] / 2:  # Z-axis box size
                z_all_corr[i] -= ts.dimensions[2]
        z_avg = np.mean(
            z_all_corr)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > z_avg:
                z_pos.append(z_all_corr[i])
            else:
                z_neg.append(z_all_corr[i])
        phos_z_dists_2 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 3. with correction (third upper)
        z_all_corr = z_all.copy()
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > ts.dimensions[2] * 2 / 3:  # Z-axis box size
                z_all_corr[i] -= ts.dimensions[2]
        z_avg = np.mean(
            z_all_corr)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > z_avg:
                z_pos.append(z_all_corr[i])
            else:
                z_neg.append(z_all_corr[i])
        phos_z_dists_3 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 4. with correction (quarter upper)
        z_all_corr = z_all.copy()
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > ts.dimensions[2] * 3 / 4:  # Z-axis box size
                z_all_corr[i] -= ts.dimensions[2]
        z_avg = np.mean(
            z_all_corr)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > z_avg:
                z_pos.append(z_all_corr[i])
            else:
                z_neg.append(z_all_corr[i])
        phos_z_dists_4 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 5. with correction (half lower)
        z_all_corr = z_all.copy()
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] < ts.dimensions[2] / 2:  # Z-axis box size
                z_all_corr[i] += ts.dimensions[2]
        z_avg = np.mean(
            z_all_corr)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > z_avg:
                z_pos.append(z_all_corr[i])
            else:
                z_neg.append(z_all_corr[i])
        phos_z_dists_5 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 6. with correction (third lower)
        z_all_corr = z_all.copy()
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] < ts.dimensions[2] * 1 / 3:  # Z-axis box size
                z_all_corr[i] += ts.dimensions[2]
        z_avg = np.mean(
            z_all_corr)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > z_avg:
                z_pos.append(z_all_corr[i])
            else:
                z_neg.append(z_all_corr[i])
        phos_z_dists_6 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 7. with correction (quarter lower)
        z_all_corr = z_all.copy()
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] < ts.dimensions[2] * 1 / 4:  # Z-axis box size
                z_all_corr[i] += ts.dimensions[2]
        z_avg = np.mean(
            z_all_corr)  # get average Z position, used as the threshold to define upper and lower phopshates' Z
        z_pos, z_neg = [], []
        for i in range(cg_map_itp['meta']['nb_mols']):
            if z_all_corr[i] > z_avg:
                z_pos.append(z_all_corr[i])
            else:
                z_neg.append(z_all_corr[i])
        phos_z_dists_7 = (np.mean(z_pos) - np.mean(z_neg)) / 10  # retrieve nm

        # 3. choose the appropriate thickness measurement
        phos_z_dists.append(
            min(phos_z_dists_1, phos_z_dists_2, phos_z_dists_3, phos_z_dists_4, phos_z_dists_5, phos_z_dists_6,
                phos_z_dists_7) + 2 * cg_map_itp['meta']['delta_Dhh'])

    cg_iter_itp['Dhh'] = {'avg': round(np.mean(phos_z_dists), 4), 'med': round(np.median(phos_z_dists), 4),
                          'std': round(np.std(phos_z_dists), 4)}  # nm

    # 3. get geoms distributions
    geoms_start = datetime.now().timestamp()

    # constraints
    for grp_constraint in range(cg_map_itp['meta']['nb_constraints']):
        try:
            cg_iter_itp['geoms']['constraint'][grp_constraint] = {'avg': None, 'hist': None}
            cg_iter_itp['geoms']['constraint'][grp_constraint]['avg'], \
            cg_iter_itp['geoms']['constraint'][grp_constraint]['hist'] = get_CG_bonds_distrib_lipids(ns, cg_map_itp[
                'constraint'][grp_constraint]['beads'], cg_iter_universe, ns.bins_constraints, ns.user_config['bw_constraints'])
            if bottom_up_active:
                cg_iter_itp['geoms']['constraint'][grp_constraint]['emd'] = emd(
                    cg_map_itp['constraint'][grp_constraint]['hist_' + temp],
                    cg_iter_itp['geoms']['constraint'][grp_constraint]['hist'],
                    ns.bins_constraints_dist_matrix) * ns.user_config['bonds2angles_scoring_factor']
        except IndexError:
            sys.exit(
                config.header_error + 'Most probably because you have constraints or constraints that exceed ' + str(
                    ns.constrainted_max_range) + ' nm, try to increase grid size')
    # bonds
    for grp_bond in range(cg_map_itp['meta']['nb_bonds']):
        try:
            cg_iter_itp['geoms']['bond'][grp_bond] = {'avg': None, 'hist': None}
            cg_iter_itp['geoms']['bond'][grp_bond]['avg'], cg_iter_itp['geoms']['bond'][grp_bond][
                'hist'] = get_CG_bonds_distrib_lipids(ns, cg_map_itp['bond'][grp_bond]['beads'], cg_iter_universe,
                                                      ns.bins_bonds, ns.user_config['bw_bonds'])
            if bottom_up_active:
                cg_iter_itp['geoms']['bond'][grp_bond]['emd'] = emd(cg_map_itp['bond'][grp_bond]['hist_' + temp],
                                                                    cg_iter_itp['geoms']['bond'][grp_bond]['hist'],
                                                                    ns.bins_bonds_dist_matrix) * ns.user_config['bonds2angles_scoring_factor']
        except IndexError:
            sys.exit(config.header_error + 'Most probably because you have bonds or constraints that exceed ' + str(
                ns.bonded_max_range) + ' nm, try to increase grid size')
    # angles
    for grp_angle in range(cg_map_itp['meta']['nb_angles']):
        cg_iter_itp['geoms']['angle'][grp_angle] = {'avg': None, 'hist': None}
        cg_iter_itp['geoms']['angle'][grp_angle]['avg'], cg_iter_itp['geoms']['angle'][grp_angle][
            'hist'] = get_CG_angles_distrib_lipids(ns, cg_map_itp['angle'][grp_angle]['beads'], cg_iter_universe)
        if bottom_up_active:
            cg_iter_itp['geoms']['angle'][grp_angle]['emd'] = emd(cg_map_itp['angle'][grp_angle]['hist_' + temp],
                                                                  cg_iter_itp['geoms']['angle'][grp_angle]['hist'],
                                                                  ns.bins_angles_dist_matrix)

    geoms_time = datetime.now().timestamp() - geoms_start

    # 4. get area compressibility
    apl_no_round = np.mean(x_boxdims ** 2 / (cg_map_itp['meta']['nb_mols'] / 2)) / 100
    area_compress = constants.Boltzmann * int(temp[:-1]) * apl_no_round / (cg_map_itp['meta']['nb_mols'] / 2 * np.mean(
        (x_boxdims ** 2 / (cg_map_itp['meta']['nb_mols'] / 2) / 100 - apl_no_round) ** 2)) * 10 ** 21  # retrieve mN/m

    #########################################
    # SCORING COMPONENTS FOR THIS ITERATION #
    #########################################

    # NOTE: DISABLED ATM -- storage of error estimations -- to directed swarm initialization on the next opti cycle
    error_data = {'constraints': collections.Counter(), 'bonds': collections.Counter(), 'angles': collections.Counter(),
                  'rdf_bead': collections.Counter(), 'rdf_pair': {}}
    geom_grps_counter = collections.Counter()  # to average error for each geom_grp, because each lipid might contain several instances of a geom_grp
    beads_types_counter = collections.Counter()

    ###############################
    # DISPLAY DISTRIBUTIONS PLOTS #
    ###############################

    larger_group = max(cg_map_itp['meta']['nb_constraints'], cg_map_itp['meta']['nb_bonds'],
                       cg_map_itp['meta']['nb_angles'])
    nrow, nrows, ncols = -1, 2, larger_group
    if cg_map_itp['meta']['nb_constraints'] > 0:
        nrows += 1

    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3), squeeze=False) # NOTE: this fucking line was responsible of the big memory leak (figures were not closing)
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    ax = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    # record x and y ranges for each geom type, to make scales consistent in the next code block
    pranges = {'x_range_max': {'constraint': 0, 'bond': 0, 'angle': 0},
               'x': {'constraint': {}, 'bond': {}, 'angle': {}},
               'y': {'constraint': [np.inf, -np.inf], 'bond': [np.inf, -np.inf], 'angle': [np.inf, -np.inf]}}

    # constraints
    if cg_map_itp['meta']['nb_constraints'] > 0:
        nrow += 1

        for grp_constraint in range(ncols):
            if grp_constraint < cg_map_itp['meta']['nb_constraints']:

                if bottom_up_active:
                    ax[nrow][grp_constraint].set_title(
                        'Constraint ' + str(grp_constraint + 1) + ' - Grp ' + cg_map_itp['constraint'][grp_constraint][
                            'geom_grp'] + ' - Δ ' + str(
                            round(cg_iter_itp['geoms']['constraint'][grp_constraint]['emd'], 2)))
                    ax[nrow][grp_constraint].plot(ns.bins_constraints[:-1] + ns.user_config['bw_constraints'] / 2,
                                                  cg_map_itp['constraint'][grp_constraint]['hist_' + temp], label='AA',
                                                  color=config.atom_color[0], alpha=config.line_alpha)
                    ax[nrow][grp_constraint].fill_between(ns.bins_constraints[:-1] + ns.user_config['bw_constraints'] / 2,
                                                          cg_map_itp['constraint'][grp_constraint]['hist_' + temp],
                                                          color=config.atom_color[0], alpha=config.fill_alpha)
                    ax[nrow][grp_constraint].plot(cg_map_itp['constraint'][grp_constraint]['avg_' + temp], 0,
                                                  color=config.atom_color[0], marker='D', alpha=config.line_alpha)
                else:
                    ax[nrow][grp_constraint].set_title(
                        'Constraint ' + str(grp_constraint + 1) + ' - Grp ' + cg_map_itp['constraint'][grp_constraint]['geom_grp'])

                ax[nrow][grp_constraint].plot(ns.bins_constraints[:-1] + ns.user_config['bw_constraints'] / 2,
                                              cg_iter_itp['geoms']['constraint'][grp_constraint]['hist'], label='CG',
                                              color=config.cg_color, alpha=config.line_alpha)
                ax[nrow][grp_constraint].fill_between(ns.bins_constraints[:-1] + ns.user_config['bw_constraints'] / 2,
                                                      cg_iter_itp['geoms']['constraint'][grp_constraint]['hist'],
                                                      color=config.cg_color, alpha=config.fill_alpha)
                ax[nrow][grp_constraint].plot(cg_iter_itp['geoms']['constraint'][grp_constraint]['avg'], 0,
                                              color=config.cg_color, marker='D', alpha=config.line_alpha)

                ax[nrow][grp_constraint].axvline(updated_cg_itp['constraint'][grp_constraint]['value'], ls='dashed',
                                                 color='black', alpha=0.5, label='Eq val')
                ax[nrow][grp_constraint].grid(zorder=0.5)
                # if grp_constraint == 0:
                ax[nrow][grp_constraint].legend()

                for i in range(len(ns.bins_constraints[:-1])):
                    if (bottom_up_active and cg_map_itp['constraint'][grp_constraint]['hist_' + temp][i] != 0) or \
                            cg_iter_itp['geoms']['constraint'][grp_constraint]['hist'][i] != 0:
                        pranges['x']['constraint'][grp_constraint] = [ns.bins_constraints[:-1][i], None]
                        break
                for i in range(len(ns.bins_constraints[:-1]) - 1, -1, -1):  # reverse
                    if (bottom_up_active and cg_map_itp['constraint'][grp_constraint]['hist_' + temp][i] != 0) or \
                            cg_iter_itp['geoms']['constraint'][grp_constraint]['hist'][i] != 0:
                        pranges['x']['constraint'][grp_constraint][1] = ns.bins_constraints[:-1][
                                                                            i] + ns.user_config['bw_constraints'] / 2
                        break
                pranges['x_range_max']['constraint'] = max(pranges['x_range_max']['constraint'],
                                                           pranges['x']['constraint'][grp_constraint][1] -
                                                           pranges['x']['constraint'][grp_constraint][0])
                pranges['y']['constraint'] = min(ax[nrow][grp_constraint].get_ylim()[0],
                                                 pranges['y']['constraint'][0]), max(
                    ax[nrow][grp_constraint].get_ylim()[1], pranges['y']['constraint'][1])

            else:
                ax[nrow][grp_constraint].set_visible(False)

        if ns.row_x_scaling:  # make scales consistent for each axis across all plots
            for grp_constraint in range(cg_map_itp['meta']['nb_constraints']):
                xmin, xmax = pranges['x']['constraint'][grp_constraint]
                xmin -= (pranges['x_range_max']['constraint'] - (xmax - xmin)) / 2
                xmax += (pranges['x_range_max']['constraint'] - (xmax - xmin)) / 2
                ax[nrow][grp_constraint].set_xlim(xmin,
                                                  xmax)  # NOTE: there is some error in the range, did not understand why but it's close enough to what we want and affect only plotting anyway
        if ns.row_y_scaling:
            for grp_constraint in range(cg_map_itp['meta']['nb_constraints']):
                ax[nrow][grp_constraint].set_ylim(*pranges['y']['constraint'])

    # bonds
    if cg_map_itp['meta']['nb_bonds'] > 0:
        nrow += 1

        for grp_bond in range(ncols):
            if grp_bond < cg_map_itp['meta']['nb_bonds']:

                if bottom_up_active:
                    ax[nrow][grp_bond].set_title(
                        'Bond ' + str(grp_bond + 1) + ' - Grp ' + cg_map_itp['bond'][grp_bond]['geom_grp'] + ' - Δ ' + str(
                            round(cg_iter_itp['geoms']['bond'][grp_bond]['emd'], 2)))
                    ax[nrow][grp_bond].plot(ns.bins_bonds[:-1] + ns.user_config['bw_bonds'] / 2,
                                            cg_map_itp['bond'][grp_bond]['hist_' + temp], label='AA',
                                            color=config.atom_color[0], alpha=config.line_alpha)
                    ax[nrow][grp_bond].fill_between(ns.bins_bonds[:-1] + ns.user_config['bw_bonds'] / 2,
                                                    cg_map_itp['bond'][grp_bond]['hist_' + temp],
                                                    color=config.atom_color[0], alpha=config.fill_alpha)
                    ax[nrow][grp_bond].plot(cg_map_itp['bond'][grp_bond]['avg_' + temp], 0, color=config.atom_color[0],
                                            marker='D', alpha=config.line_alpha)
                else:
                    ax[nrow][grp_bond].set_title(
                        'Bond ' + str(grp_bond + 1) + ' - Grp ' + cg_map_itp['bond'][grp_bond]['geom_grp'])

                ax[nrow][grp_bond].plot(ns.bins_bonds[:-1] + ns.user_config['bw_bonds'] / 2,
                                        cg_iter_itp['geoms']['bond'][grp_bond]['hist'], label='CG',
                                        color=config.cg_color, alpha=config.line_alpha)
                ax[nrow][grp_bond].fill_between(ns.bins_bonds[:-1] + ns.user_config['bw_bonds'] / 2,
                                                cg_iter_itp['geoms']['bond'][grp_bond]['hist'], color=config.cg_color,
                                                alpha=config.fill_alpha)
                ax[nrow][grp_bond].plot(cg_iter_itp['geoms']['bond'][grp_bond]['avg'], 0, color=config.cg_color,
                                        marker='D', alpha=config.line_alpha)

                ax[nrow][grp_bond].axvline(updated_cg_itp['bond'][grp_bond]['value'], ls='dashed', color='black', alpha=0.5,
                                           label='Eq val')
                ax[nrow][grp_bond].grid(zorder=0.5)
                # if grp_bond == 0:
                ax[nrow][grp_bond].legend()

                for i in range(len(ns.bins_bonds[:-1])):
                    if (bottom_up_active and cg_map_itp['bond'][grp_bond]['hist_' + temp][i] != 0) or cg_iter_itp['geoms']['bond'][grp_bond]['hist'][i] != 0:
                        pranges['x']['bond'][grp_bond] = [ns.bins_bonds[:-1][i], None]
                        break
                for i in range(len(ns.bins_bonds[:-1]) - 1, -1, -1):  # reverse
                    if (bottom_up_active and cg_map_itp['bond'][grp_bond]['hist_' + temp][i] != 0) or cg_iter_itp['geoms']['bond'][grp_bond]['hist'][i] != 0:
                        pranges['x']['bond'][grp_bond][1] = ns.bins_bonds[:-1][i] + ns.user_config['bw_bonds'] / 2
                        break
                pranges['x_range_max']['bond'] = max(pranges['x_range_max']['bond'],
                                                     pranges['x']['bond'][grp_bond][1] - pranges['x']['bond'][grp_bond][
                                                         0])
                pranges['y']['bond'] = min(ax[nrow][grp_bond].get_ylim()[0], pranges['y']['bond'][0]), max(
                    ax[nrow][grp_bond].get_ylim()[1], pranges['y']['bond'][1])

            else:
                ax[nrow][grp_bond].set_visible(False)

        if ns.row_x_scaling:  # make scales consistent for each axis across all plots
            for grp_bond in range(cg_map_itp['meta']['nb_bonds']):
                xmin, xmax = pranges['x']['bond'][grp_bond]
                xmin -= (pranges['x_range_max']['bond'] - (xmax - xmin)) / 2
                xmax += (pranges['x_range_max']['bond'] - (xmax - xmin)) / 2
                ax[nrow][grp_bond].set_xlim(xmin,
                                            xmax)  # NOTE: there is some error in the range, did not understand why but it's close enough to what we want and affect only plotting anyway
        if ns.row_y_scaling:
            for grp_bond in range(cg_map_itp['meta']['nb_bonds']):
                ax[nrow][grp_bond].set_ylim(*pranges['y']['bond'])

    # angles
    if cg_map_itp['meta']['nb_angles'] > 0:
        nrow += 1

        for grp_angle in range(ncols):
            if grp_angle < cg_map_itp['meta']['nb_angles']:

                if bottom_up_active:
                    ax[nrow][grp_angle].set_title(
                        'Angle ' + str(grp_angle + 1) + ' - Grp ' + cg_map_itp['angle'][grp_angle][
                            'geom_grp'] + ' - Δ ' + str(round(cg_iter_itp['geoms']['angle'][grp_angle]['emd'], 2)))
                    ax[nrow][grp_angle].plot(ns.bins_angles[:-1] + ns.user_config['bw_angles'] / 2,
                                             cg_map_itp['angle'][grp_angle]['hist_' + temp], label='AA',
                                             color=config.atom_color[0], alpha=config.line_alpha)
                    ax[nrow][grp_angle].fill_between(ns.bins_angles[:-1] + ns.user_config['bw_angles'] / 2,
                                                     cg_map_itp['angle'][grp_angle]['hist_' + temp],
                                                     color=config.atom_color[0], alpha=config.fill_alpha)
                    ax[nrow][grp_angle].plot(cg_map_itp['angle'][grp_angle]['avg_' + temp], 0, color=config.atom_color[0],
                                             marker='D', alpha=config.line_alpha)
                else:
                    ax[nrow][grp_angle].set_title(
                        'Angle ' + str(grp_angle + 1) + ' - Grp ' + cg_map_itp['angle'][grp_angle]['geom_grp'])

                ax[nrow][grp_angle].plot(ns.bins_angles[:-1] + ns.user_config['bw_angles'] / 2,
                                         cg_iter_itp['geoms']['angle'][grp_angle]['hist'], label='CG',
                                         color=config.cg_color, alpha=config.line_alpha)
                ax[nrow][grp_angle].fill_between(ns.bins_angles[:-1] + ns.user_config['bw_angles'] / 2,
                                                 cg_iter_itp['geoms']['angle'][grp_angle]['hist'],
                                                 color=config.cg_color, alpha=config.fill_alpha)
                ax[nrow][grp_angle].plot(cg_iter_itp['geoms']['angle'][grp_angle]['avg'], 0, color=config.cg_color,
                                         marker='D', alpha=config.line_alpha)

                ax[nrow][grp_angle].axvline(updated_cg_itp['angle'][grp_angle]['value'], ls='dashed', color='black',
                                            alpha=0.5, label='Eq val')
                ax[nrow][grp_angle].grid(zorder=0.5)
                ax[nrow][grp_angle].legend()

                for i in range(len(ns.bins_angles[:-1])):
                    if (bottom_up_active and cg_map_itp['angle'][grp_angle]['hist_' + temp][i] != 0) or \
                            cg_iter_itp['geoms']['angle'][grp_angle]['hist'][i] != 0:
                        pranges['x']['angle'][grp_angle] = [ns.bins_angles[:-1][i], None]
                        break
                for i in range(len(ns.bins_angles[:-1]) - 1, -1, -1):  # reverse
                    if (bottom_up_active and cg_map_itp['angle'][grp_angle]['hist_' + temp][i] != 0) or \
                            cg_iter_itp['geoms']['angle'][grp_angle]['hist'][i] != 0:
                        pranges['x']['angle'][grp_angle][1] = ns.bins_angles[:-1][i] + ns.user_config['bw_angles'] / 2
                        break
                pranges['x_range_max']['angle'] = max(pranges['x_range_max']['angle'],
                                                      pranges['x']['angle'][grp_angle][1] -
                                                      pranges['x']['angle'][grp_angle][0])
                pranges['y']['angle'] = min(ax[nrow][grp_angle].get_ylim()[0], pranges['y']['angle'][0]), max(
                    ax[nrow][grp_angle].get_ylim()[1], pranges['y']['angle'][1])

            else:
                ax[nrow][grp_angle].set_visible(False)

        if ns.row_x_scaling:  # make scales consistent for each axis across all plots
            for grp_angle in range(cg_map_itp['meta']['nb_angles']):
                xmin, xmax = pranges['x']['angle'][grp_angle]
                xmin -= (pranges['x_range_max']['angle'] - (xmax - xmin)) / 2
                xmax += (pranges['x_range_max']['angle'] - (xmax - xmin)) / 2
                ax[nrow][grp_angle].set_xlim(xmin,
                                             xmax)  # NOTE: there is some error in the range, did not understand why but it's close enough to what we want and affect only plotting anyway
                if cg_map_itp['angle'][grp_angle][
                    'value'] >= 150:  # ensure the vertical dashed line for equilbirium value will be visible for values at 180 degrees
                    ax[nrow][grp_angle].set_xlim(right=185)
        if ns.row_y_scaling:
            for grp_angle in range(cg_map_itp['meta']['nb_angles']):
                ax[nrow][grp_angle].set_ylim(*pranges['y']['angle'])

    # calculate score components + error estimations -- geoms
    delta_geoms = 0
    delta_geoms_per_grp = {}
    raw_delta_geoms = [[], []]  # bonds and angles separately

    if bottom_up_active:

        for grp_constraint in range(cg_map_itp['meta']['nb_constraints']):
            geom_grp = ns.cg_itps[lipid_code]['constraint'][grp_constraint]['geom_grp']
            geom_error = cg_iter_itp['geoms']['constraint'][grp_constraint]['emd']  # ns.constraints2angles_scoring_factor already applied
            geom_grps_counter[geom_grp] += 1
            if geom_grp in delta_geoms_per_grp:
                delta_geoms_per_grp[geom_grp].append(geom_error)
            else:
                delta_geoms_per_grp[geom_grp] = [geom_error]

            error_data['constraints'][geom_grp] += geom_error  # this is used for directed swarm initialization
            raw_delta_geoms[0].append(cg_iter_itp['geoms']['constraint'][grp_constraint]['emd'])  # currently unused

        for grp_bond in range(cg_map_itp['meta']['nb_bonds']):
            geom_grp = ns.cg_itps[lipid_code]['bond'][grp_bond]['geom_grp']
            geom_grps_counter[geom_grp] += 1
            geom_error = cg_iter_itp['geoms']['bond'][grp_bond]['emd']  # ns.bonds2angles_scoring_factor already applied
            geom_grps_counter[geom_grp] += 1
            if geom_grp in delta_geoms_per_grp:
                delta_geoms_per_grp[geom_grp].append(geom_error)
            else:
                delta_geoms_per_grp[geom_grp] = [geom_error]

            error_data['bonds'][geom_grp] += geom_error  # this is used for directed swarm initialization
            raw_delta_geoms[0].append(cg_iter_itp['geoms']['bond'][grp_bond]['emd'])  # currently unused

        for grp_angle in range(cg_map_itp['meta']['nb_angles']):
            geom_grp = ns.cg_itps[lipid_code]['angle'][grp_angle]['geom_grp']
            geom_grps_counter[geom_grp] += 1
            geom_error = cg_iter_itp['geoms']['angle'][grp_angle]['emd']
            geom_grps_counter[geom_grp] += 1
            if geom_grp in delta_geoms_per_grp:
                delta_geoms_per_grp[geom_grp].append(geom_error)
            else:
                delta_geoms_per_grp[geom_grp] = [geom_error]

            error_data['angles'][geom_grp] += geom_error  # this is used for directed swarm initialization
            raw_delta_geoms[1].append(cg_iter_itp['geoms']['angle'][grp_angle]['emd'])  # currently unused

        for geom_grp in delta_geoms_per_grp:
            delta_geoms += (np.sqrt(np.sum([delta_geom ** 2 for delta_geom in delta_geoms_per_grp[geom_grp]]) / len(delta_geoms_per_grp[geom_grp])) ** 2)
        delta_geoms = np.sqrt(delta_geoms / len(delta_geoms_per_grp))

    plot_filename = 'GEOMS_' + lipid_code + '_' + temp + '.png'
    if bottom_up_active:
        sup_title = lipid_code + ' ' + temp + ' -- Geoms: ' + str(round(delta_geoms, 2)) + ' -- APL: ' + str(
            round(cg_iter_itp['apl']['avg'], 2)) + ' -- Dhh: ' + str(round(cg_iter_itp['Dhh']['avg'], 2))
    else:
        sup_title = lipid_code + ' ' + temp + ' -- Geoms: Inactive -- APL: ' + str(
            round(cg_iter_itp['apl']['avg'], 2)) + ' -- Dhh: ' + str(round(cg_iter_itp['Dhh']['avg'], 2))
    plt.suptitle(sup_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_filename)
    plt.close(fig)

    ###########################
    # CALCULATE RDFs AND PLOT #
    ###########################

    # calculate radial distribution function from CG trajectory
    rdf_start = datetime.now().timestamp()
    cg_eval_rdfs_short, cg_eval_rdfs_long = {}, {}
    for pair_type in ns.cg_itps[lipid_code]['rdf_pairs']:

        bead_type_1, bead_type_2 = pair_type.split('_')
        ag1 = cg_iter_universe.atoms[ns.cg_itps[lipid_code]['rdf_pairs'][pair_type][0]]
        ag2 = cg_iter_universe.atoms[ns.cg_itps[lipid_code]['rdf_pairs'][pair_type][1]]

        irdf_short = rdf.InterRDF(ag1, ag2, nbins=round(ns.user_config['cutoff_rdfs'] / ns.user_config['bw_rdfs']),
                                  range=(0, ns.user_config['cutoff_rdfs'] * 10), exclusion_block=(
            len(ns.cg_itps[lipid_code]['beads_ids_per_beads_types_sing'][bead_type_1]),
            len(ns.cg_itps[lipid_code]['beads_ids_per_beads_types_sing'][bead_type_2])))

        # prevent the InterRDF segfault that happens when box explodes -- because there are no beads within cutoff and this is a MDA bug of InterRDF
        if cg_iter_itp['apl']['avg'] < 1 and cg_iter_itp['Dhh']['avg'] < 5:
            irdf_short.run(step=round(len(cg_iter_universe.trajectory) / ns.rdf_frames))
            # rdf_norm = irdf_short.rdf / np.sum(irdf_short.rdf) # norm
            # rdf_count = irdf_short.count / np.sum(irdf_short.count) * 100 # WITHOUT norm
            rdf_norm = irdf_short.count / ns.vol_shell
            rdf_count = irdf_short.count
            cg_eval_rdfs_short[pair_type] = rdf_count, rdf_norm

        else:
            cg_eval_rdfs_short[pair_type] = [[None], [None]]

    rdfs_time = datetime.now().timestamp() - rdf_start

    # Calculate the EMD / RDF scores according to the given cutoff radius
    nb_beads_types = len(ns.lipid_beads_types[lipid_code])

    if bottom_up_active:
        for i in range(nb_beads_types):  # matrix of LJ
            for j in range(nb_beads_types):

                if j >= i:
                    bead_type_1, bead_type_2 = ns.lipid_beads_types[lipid_code][i], ns.lipid_beads_types[lipid_code][j]
                    pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
                    aa_rdf, _ = ns.cg_itps[lipid_code]['rdf_' + temp + '_short'][pair_type]  # count, norm

                    # score
                    if cg_eval_rdfs_short[pair_type][0][0] is not None:

                        count_diff_penalty = (max(np.sum(aa_rdf), np.sum(cg_eval_rdfs_short[pair_type][0])) / min(
                            np.sum(aa_rdf), np.sum(cg_eval_rdfs_short[pair_type][0])) - 1) * 100
                        hist_AA_EMD = aa_rdf / ((np.sum(aa_rdf) + np.sum(cg_eval_rdfs_short[pair_type][0])) / 2)
                        hist_CG_EMD = cg_eval_rdfs_short[pair_type][0] / (
                                    (np.sum(aa_rdf) + np.sum(cg_eval_rdfs_short[pair_type][0])) / 2)

                        cg_iter_itp['rdfs_short'][pair_type] = min(round(emd(hist_AA_EMD, hist_CG_EMD, ns.bins_vol_matrix,
                                                                             extra_mass_penalty=0) * 100 + count_diff_penalty,
                                                                         3),
                                                                   100)  # distance matrix with radial shell volume using NOT normalized data
                    else:
                        cg_iter_itp['rdfs_short'][pair_type] = 100

    # calculate score components + error estimations -- rdfs
    delta_rdfs = 0
    delta_rdfs_per_grp = {}

    if bottom_up_active:
        for pair_type in cg_iter_itp['rdfs_short']:

            bead_type_1, bead_type_2 = pair_type.split('_')
            rdf_error = cg_iter_itp['rdfs_short'][pair_type]
            delta_rdfs += rdf_error ** 2
            delta_rdfs_per_grp[pair_type] = rdf_error  # the RMSE here will be done later, over all pairs of 2 given bead types

            error_data['rdf_pair'][pair_type] = rdf_error  # this is used for directed swarm initialization -- this does NOT need averaging
            error_data['rdf_bead'][bead_type_1] += rdf_error  # this is used for directed swarm initialization
            beads_types_counter[bead_type_1] += 1
            error_data['rdf_bead'][bead_type_2] += rdf_error  # this is used for directed swarm initialization
            beads_types_counter[bead_type_2] += 1

        # this is NOT directly used in the score, this quantity here only gives the error per lipid
        # that is stored in the logs and displayed in plots (analysis + RDFs plot per lipid)
        delta_rdfs = np.sqrt(delta_rdfs / len(cg_iter_itp['rdfs_short']))

    # RDF plots at the given cutoff
    fig = plt.figure(figsize=(nb_beads_types * 3, nb_beads_types * 3))
    ax = fig.subplots(nrows=nb_beads_types, ncols=nb_beads_types, squeeze=False)

    for i in range(nb_beads_types):  # matrix of LJ
        for j in range(nb_beads_types):

            if j >= i:
                bead_type_1, bead_type_2 = ns.lipid_beads_types[lipid_code][i], ns.lipid_beads_types[lipid_code][j]
                pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))

                # display RDF
                if cg_eval_rdfs_short[pair_type][0][0] is not None:
                    if bottom_up_active:
                        _, aa_rdf = ns.cg_itps[lipid_code]['rdf_' + temp + '_short'][pair_type]  # count, norm
                        hist_AA_display = aa_rdf / ((np.sum(aa_rdf) + np.sum(cg_eval_rdfs_short[pair_type][1])) / 2)
                        ax[i][j].plot(ns.bins_vol_shell - ns.user_config['bw_rdfs'] / 2, hist_AA_display, label='AA',
                                  alpha=config.rdf_alpha, color=config.atom_color[0])  # aa ref
                        hist_CG_display = cg_eval_rdfs_short[pair_type][1] / ((np.sum(aa_rdf) + np.sum(cg_eval_rdfs_short[pair_type][1])) / 2)
                    else:
                        hist_CG_display = cg_eval_rdfs_short[pair_type][1]
                    ax[i][j].plot(ns.bins_vol_shell - ns.user_config['bw_rdfs'] / 2, hist_CG_display, label='CG',
                                  alpha=config.rdf_alpha, color=config.cg_color)  # cg for this eval
                else:
                    ax[i][j].plot(0, 0, label='SIM. CRASHED')

                # display the sum of radii
                if ns.user_config['tune_radii']:
                    sig = ns.user_config['init_beads_radii'][ns.reverse_radii_mapping[bead_type_1]] + ns.user_config['init_beads_radii'][ns.reverse_radii_mapping[bead_type_2]]
                else:
                    sig = ns.user_config['init_nonbonded'][pair_type.replace('_', ' ')]['sig']  # pre-defined SIG in config
                ax[i][j].axvline(sig, label='Sig', ls='dashed', color='black', alpha=0.5)

                ax[i][j].set_xlim(0, ns.user_config['cutoff_rdfs'])
                if bottom_up_active:
                    ax[i][j].set_title(f"{bead_type_1} {bead_type_2} - RDF Δ {round(cg_iter_itp['rdfs_short'][pair_type], 2)}")
                else:
                    ax[i][j].set_title(f"{bead_type_1} {bead_type_2} - RDF Δ Inactive")
                ax[i][j].grid()
                ax[i][j].legend()
            else:
                ax[i][j].set_visible(False)

    if bottom_up_active:
        plt.suptitle(f"{lipid_code} {temp} -- Geoms: {round(delta_geoms, 2)} -- RDF: {round(delta_rdfs, 2)} -- APL: {round(cg_iter_itp['apl']['avg'], 2)} -- Dhh: {round(cg_iter_itp['Dhh']['avg'], 2)}")
    else:
        plt.suptitle(f"{lipid_code} {temp} -- Geoms: Inactive -- RDF: Inactive -- APL: {round(cg_iter_itp['apl']['avg'], 2)} -- Dhh: {round(cg_iter_itp['Dhh']['avg'], 2)}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('RDF_SHELL_NORM_' + lipid_code + '_' + temp + '_cutoff_' + str(ns.user_config['cutoff_rdfs']) + '_nm.png')
    plt.close(fig)

    # RDF plots at short cutoff = 1.5 nm
    fig = plt.figure(figsize=(nb_beads_types * 3, nb_beads_types * 3))
    ax = fig.subplots(nrows=nb_beads_types, ncols=nb_beads_types, squeeze=False)

    for i in range(nb_beads_types):  # matrix of LJ
        for j in range(nb_beads_types):

            if j >= i:
                bead_type_1, bead_type_2 = ns.lipid_beads_types[lipid_code][i], ns.lipid_beads_types[lipid_code][j]
                pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))

                # display RDF
                if cg_eval_rdfs_short[pair_type][0][0] is not None:
                    if bottom_up_active:
                        _, aa_rdf = ns.cg_itps[lipid_code]['rdf_' + temp + '_short'][pair_type]  # count, norm
                        hist_AA_display = aa_rdf / ((np.sum(aa_rdf) + np.sum(cg_eval_rdfs_short[pair_type][1])) / 2)
                        ax[i][j].plot(ns.bins_vol_shell - ns.user_config['bw_rdfs'] / 2, hist_AA_display, label='AA',
                                      alpha=config.rdf_alpha, color=config.atom_color[0])  # aa ref
                        hist_CG_display = cg_eval_rdfs_short[pair_type][1] / ((np.sum(aa_rdf) + np.sum(cg_eval_rdfs_short[pair_type][1])) / 2)
                    else:
                        hist_CG_display = cg_eval_rdfs_short[pair_type][1]
                    ax[i][j].plot(ns.bins_vol_shell - ns.user_config['bw_rdfs'] / 2, hist_CG_display, label='CG',
                                  alpha=config.rdf_alpha, color=config.cg_color)  # cg for this eval
                else:
                    ax[i][j].plot(0, 0, label='SIM. CRASHED')

                # display the sum of radii
                if ns.user_config['tune_radii']:
                    sig = ns.user_config['init_beads_radii'][ns.reverse_radii_mapping[bead_type_1]] + ns.user_config['init_beads_radii'][ns.reverse_radii_mapping[bead_type_2]]
                else:
                    sig = ns.user_config['init_nonbonded'][pair_type.replace('_', ' ')]['sig']  # pre-defined SIG in config
                ax[i][j].axvline(sig, label='Sig', ls='dashed', color='black', alpha=0.5)

                ax[i][j].set_xlim(0, 1.5)
                if bottom_up_active:
                    ax[i][j].set_title(f"{bead_type_1} {bead_type_2} - RDF Δ {round(cg_iter_itp['rdfs_short'][pair_type], 2)}")
                else:
                    ax[i][j].set_title(f"{bead_type_1} {bead_type_2} - RDF Δ Inactive")
                ax[i][j].grid()
                ax[i][j].legend()
            else:
                ax[i][j].set_visible(False)

    if bottom_up_active:
        plt.suptitle(f"{lipid_code} {temp} -- Geoms: {round(delta_geoms, 2)} -- RDF: {round(delta_rdfs, 2)} -- APL: {round(cg_iter_itp['apl']['avg'], 2)} -- Dhh: {round(cg_iter_itp['Dhh']['avg'], 2)}")
    else:
        plt.suptitle(
            f"{lipid_code} {temp} -- Geoms: Inactive -- RDF: Inactive -- APL: {round(cg_iter_itp['apl']['avg'], 2)} -- Dhh: {round(cg_iter_itp['Dhh']['avg'], 2)}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('RDF_SHELL_NORM_' + lipid_code + '_' + temp + '_cutoff_1.5_nm.png')
    plt.close(fig)

    if bottom_up_active:
        # average error by geom_grp -- how it is stored is a bit stupid but works OK
        for geom_grp in ns.all_constraints_types:
            if lipid_code in ns.all_constraints_types[geom_grp]:
                error_data['constraints'][geom_grp] /= geom_grps_counter[geom_grp]  # happens once per geom_grp, as we want
        for geom_grp in ns.all_bonds_types:
            if lipid_code in ns.all_bonds_types[geom_grp]:
                error_data['bonds'][geom_grp] /= geom_grps_counter[geom_grp]  # happens once per geom_grp, as we want
        for geom_grp in ns.all_angles_types:
            if lipid_code in ns.all_angles_types[geom_grp]:
                error_data['angles'][geom_grp] /= geom_grps_counter[geom_grp]

        # averaged error quantification for later bead radius variations
        for bead_type in error_data['rdf_bead']:
            error_data['rdf_bead'][bead_type] /= beads_types_counter[bead_type]

    return delta_geoms, delta_geoms_per_grp, raw_delta_geoms, cg_iter_itp['apl'], cg_iter_itp['Dhh'], delta_rdfs, delta_rdfs_per_grp, error_data, area_compress, geoms_time, rdfs_time