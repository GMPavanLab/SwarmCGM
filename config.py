# lipids opti input
data_aa_storage_dir = 'STORAGE_READ_DATA_AA_TRAJ'
cg_models_data_dir = 'REF_MAPPINGS_LIPIDS'
cg_setups_data_dir = 'START_CG_SETUPS'
aa_data_dir = 'REF_AA_TRAJS_LIPIDS'

# parameters rounding
round_bases = {'bond_val': 0.001, 'bond_fct': 0.1, 'angle_val': 0.1, 'angle_fct': 0.1, 'LJ_sig': 0.001, 'LJ_eps': 0.01}

# plots display parameters
line_alpha = 0.50  # line alpha for the density plots
fill_alpha = 0.10  # fill alpha for the density plots
cg_color = '#1f77b4'
atom_color = ['#d62728', 'slategrey', 'darkturquoise', 'darkviolet']
rdf_alpha = 0.80

# optimization output filenames
iteration_sim_files_dirname = 'CG_SIMS_EVAL_'  # basename to be appended to with _NN = particle #iteration
best_fitted_model_dirname = 'Best_fitted_model'
distrib_plots_all_evals_dirname = 'Distributions_plots_all_evaluations'
log_files_all_evals_dirname = 'CG_sim_log_files_all_evaluations'
sim_files_all_evals_dirname = 'CG_sim_all_files_all_evaluations'
opti_perf_recap_file = '.internal/opti_recap_evals_perfs_and_params.csv'
opti_pairwise_distances_file = '.internal/opti_recap_evals_pairwise_distribs_diffs.csv'
ref_distrib_plots = 'Reference_model_AA-mapped_groups_distributions.png'
best_distrib_plots = 'Best_fitted_model_AA-mapped_vs_CG_groups_distributions.png'

# stdout display formatting
sep = ' --------------------------------------------------------------------------------------------- '
sep_close = '+---------------------------------------------------------------------------------------------+'
header_warning = '\n-- ! WARNING ! --\n'
header_error = '\n-- ! ERROR ! --\n'
header_gmx_error = f'{sep}\n  GMX ERROR MSG\n{sep}\n\n'
