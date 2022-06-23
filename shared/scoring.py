import numpy as np
import config


def get_particle_score(ns, nb_eval_particle, swarm_res):

    part_apls = 0
    part_dhhs = 0
    all_delta_geoms_per_grp = {}
    all_delta_rdfs_per_grp = {}

    # first we collect
    nb_lipids_and_temps = 0
    for lipid_code in ns.user_config['lipids_codes']:
        for temp in ns.user_config['lipids_codes'][lipid_code]:

            nb_lipids_and_temps += 1
            # TODO: divide by correct number of different components activated
            score_parts = swarm_res[nb_eval_particle][lipid_code][temp]['score_part']

            # if user has specified that we make use of the simulations available for this lipid for bottom-up scoring
            if ns.user_config['reference_AA_weight'][lipid_code] > 0:

                delta_geoms_per_grp = score_parts['delta_geoms_per_grp']  # key: geom_grp then: list of float
                delta_rdfs_per_grp = score_parts['delta_rdfs_per_grp']  # key: pair_type then: float

                for geom_grp in delta_geoms_per_grp:
                    weighted_delta_geoms = [dgeom * ns.user_config['reference_AA_weight'][lipid_code] for dgeom in delta_geoms_per_grp[geom_grp]]
                    if geom_grp not in all_delta_geoms_per_grp:
                        all_delta_geoms_per_grp[geom_grp] = weighted_delta_geoms
                    else:
                        all_delta_geoms_per_grp[geom_grp].extend(weighted_delta_geoms)

                for pair_type in delta_rdfs_per_grp:
                    weighted_delta_rdf = delta_rdfs_per_grp[pair_type] * ns.user_config['reference_AA_weight'][lipid_code]
                    if pair_type not in all_delta_rdfs_per_grp:
                        all_delta_rdfs_per_grp[pair_type] = [weighted_delta_rdf]
                    else:
                        all_delta_rdfs_per_grp[pair_type].append(weighted_delta_rdf)

            # if ns.user_config['exp_data'][lipid_code][temp]['apl'] is not None:
            part_apls += score_parts['perc_delta_apl_adapt'] ** 2
            # TODO: this here can crash because the top-down component would be missing (APL not defined in config for example)

            # if ns.user_config['exp_data'][lipid_code][temp]['Dhh'] is not None:
            part_dhhs += score_parts['perc_delta_thick_adapt'] ** 2

    part_apls = np.sqrt(part_apls / nb_lipids_and_temps)
    part_dhhs = np.sqrt(part_dhhs / nb_lipids_and_temps)

    if len(all_delta_geoms_per_grp) > 0 and len(all_delta_rdfs_per_grp) > 0:  # if bottom-up component exists
        for geom_grp in all_delta_geoms_per_grp:  # here aggregate by geom grp, total geom deviation across all lipids
            all_delta_geoms_per_grp[geom_grp] = np.sqrt(np.sum([delta_geom ** 2 for delta_geom in all_delta_geoms_per_grp[geom_grp]]) / len(all_delta_geoms_per_grp[geom_grp]))
        part_geoms = np.sqrt(np.sum([all_delta_geoms_per_grp[geom_grp] ** 2 for geom_grp in all_delta_geoms_per_grp]) / len(all_delta_geoms_per_grp))

        for pair_type in all_delta_rdfs_per_grp:  # here aggregate by pair of bead types, total rdfs deviation across all lipids
            all_delta_rdfs_per_grp[pair_type] = np.sqrt(np.sum([delta_rdf ** 2 for delta_rdf in all_delta_rdfs_per_grp[pair_type]]) / len(all_delta_rdfs_per_grp[pair_type]))
        part_rdfs = np.sqrt(np.sum([all_delta_rdfs_per_grp[pair_type] ** 2 for pair_type in all_delta_rdfs_per_grp]) / len(all_delta_rdfs_per_grp))

        if ns.user_config['score_rdfs']:
            particle_score = np.sqrt((part_geoms ** 2 + part_rdfs ** 2 + part_apls ** 2 + part_dhhs ** 2) / 4)
        else:
            particle_score = np.sqrt((part_geoms ** 2 + part_apls ** 2 + part_dhhs ** 2) / 3)

    else:  # make use exclusively of top-down component
        particle_score = np.sqrt((part_apls ** 2 + part_dhhs ** 2) / 2)

    return particle_score

