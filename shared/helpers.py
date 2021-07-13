import config


# rounding at given base
def base_round(val, base):
	return base * round(val/base)


# round the parameters given by FST-PSO to the required precision, given in config file
def get_rounded_parameters_set(ns, parameters_set):

	parameters_set_rounded = parameters_set.copy()
	pos_cursor = 0
	for param in ns.all_params_opti:

		if param.startswith('B') and param.endswith('val'):
			parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																config.round_bases['bond_val'])
		elif param.startswith('B') and param.endswith('fct'):
			parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
															config.round_bases['bond_fct'])
		elif param.startswith('A') and param.endswith('val'):
			parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
																config.round_bases['angle_val'])
		elif param.startswith('A') and param.endswith('fct'):
			parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
															config.round_bases['angle_fct'])
		elif param.startswith('r_'):
			parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor],
															config.round_bases['LJ_sig'])
		elif param.startswith('LJ'):
			parameters_set_rounded[pos_cursor] = base_round(parameters_set_rounded[pos_cursor], config.round_bases['LJ_eps'])

		pos_cursor += 1

	return parameters_set_rounded