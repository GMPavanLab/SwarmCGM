import config


# rounding at given base
def base_round(val, base):
	return base * round(val/base)


# round the parameters given by FST-PSO to the required precision, given in config file
def get_rounded_parameters_set(ns, parameters_set):

	parameters_set_rounded = parameters_set.copy()
	pos_cursor = 0
	for param_dict in ns.all_params_opti:
		for param in param_dict:  # accessing each single key of each dict

			if param.startswith('B') and ns.tune_geoms:
				if param_dict[param] == 2:
					parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																	config.round_bases['bond_val'])
					pos_cursor += 1
				parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																config.round_bases['bond_fct'])
				pos_cursor += 1  # changed to tune force constants only for bonds

			elif param.startswith('A') and ns.tune_geoms:
				if param_dict[param] == 2:
					parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																	config.round_bases['angle_val'])
					pos_cursor += 1
				parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																config.round_bases['angle_fct'])
				pos_cursor += 1

			elif param.startswith('r_'):
				parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																config.round_bases['LJ_sig'])
				pos_cursor += 1

			elif param.startswith('LJ') and ns.tune_eps:

				parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor], config.round_bases['LJ_eps'])
				pos_cursor += 1

	return parameters_set_rounded