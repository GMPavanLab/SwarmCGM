import numpy as np
import config


def get_particle_score(ns, nb_eval_particle, swarm_res):

    part_apls = 0
    part_dhhs = 0
    part_geoms = 0
    part_rdfs = 0

    all_delta_geoms_per_grp = {}
    all_delta_rdfs_per_grp = {}

    # first we collect
    nb_lipids_and_temps = 0
    for lipid_code in ns.user_config['lipids_codes']:
        for temp in ns.user_config['lipids_codes'][lipid_code]:

            nb_lipids_and_temps += 1

            score_parts = swarm_res[nb_eval_particle][lipid_code][temp]['score_part']
            delta_geoms_per_grp = score_parts['delta_geoms_per_grp']  # key: geom_grp then: list of float
            delta_rdfs_per_grp = score_parts['delta_rdfs_per_grp']  # key: pair_type then: float

            for geom_grp in delta_geoms_per_grp:
                if geom_grp not in all_delta_geoms_per_grp:
                    all_delta_geoms_per_grp[geom_grp] = delta_geoms_per_grp[geom_grp].copy()
                else:
                    all_delta_geoms_per_grp[geom_grp].extend(delta_geoms_per_grp[geom_grp])

            for geom_grp in all_delta_geoms_per_grp:  # apply the error tolerance for unreliable AA trajs (down-weighting)
                for i in range(len(all_delta_geoms_per_grp[geom_grp])):
                    all_delta_geoms_per_grp[geom_grp][i] *= ns.user_config['reference_AA_weight'][lipid_code]

            for pair_type in delta_rdfs_per_grp:
                if pair_type not in all_delta_rdfs_per_grp:
                    all_delta_rdfs_per_grp[pair_type] = [delta_rdfs_per_grp[pair_type]]
                else:
                    all_delta_rdfs_per_grp[pair_type].append(delta_rdfs_per_grp[pair_type])

            for pair_type in all_delta_rdfs_per_grp:
                for i in range(len(all_delta_rdfs_per_grp[pair_type])):
                    all_delta_rdfs_per_grp[pair_type][i] *= ns.user_config['reference_AA_weight'][lipid_code]

            # if ns.user_config['exp_data'][lipid_code][temp]['apl'] is not None:
            part_apls += score_parts['perc_delta_apl_adapt'] ** 2

            # if ns.user_config['exp_data'][lipid_code][temp]['Dhh'] is not None:
            part_dhhs += score_parts['perc_delta_thick_adapt'] ** 2

    # then we aggregate
    for geom_grp in all_delta_geoms_per_grp:  # here by geom grp
        all_delta_geoms_per_grp[geom_grp] = np.sqrt(np.sum([delta_geom ** 2 for delta_geom in all_delta_geoms_per_grp[geom_grp]]) / len(all_delta_geoms_per_grp[geom_grp]))
    # here as the total geom deviation across all lipids
    part_geoms = np.sqrt(np.sum([all_delta_geoms_per_grp[geom_grp] ** 2 for geom_grp in all_delta_geoms_per_grp]) / len(all_delta_geoms_per_grp))

    for pair_type in all_delta_rdfs_per_grp:  # here by pair type
        all_delta_rdfs_per_grp[pair_type] = np.sqrt(np.sum([delta_rdf ** 2 for delta_rdf in all_delta_rdfs_per_grp[pair_type]]) / len(all_delta_rdfs_per_grp[pair_type]))
    # here as the total rdfs deviation across all lipids
    part_rdfs = np.sqrt(np.sum([all_delta_rdfs_per_grp[pair_type] ** 2 for pair_type in all_delta_rdfs_per_grp]) / len(all_delta_rdfs_per_grp))

    part_apls = np.sqrt(part_apls / nb_lipids_and_temps)
    part_dhhs = np.sqrt(part_dhhs / nb_lipids_and_temps)

    particle_score = np.sqrt((part_geoms ** 2 + part_rdfs ** 2 + part_apls ** 2 + part_dhhs ** 2) / 4)

    return particle_score

