import numpy as np
import sys, re, random, os, shutil, subprocess, time
import matplotlib
matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from scipy.spatial.distance import cdist
import MDAnalysis as mda
import collections
from datetime import datetime
import config
import math


# for summarizing an opti procedure in plots, forward_fill for simulations that crashed to make the display OK and more interpretable
def forward_fill(arr, cond_value):

	valid_val = None
	for i in range(len(arr)):
		if arr[i] != cond_value:
			valid_val = arr[i]
		else:
			j = i
			while valid_val is None and j < len(arr):
				j += 1
				try:
					if arr[j] is not cond_value:
						valid_val = arr[j]
						break
				except IndexError:
					sys.exit(config.header_error+'Unexpected read of the optimization results, please check that your simulations have not all been crashing')
			if valid_val is not None:
				arr[i] = valid_val
			else:
				sys.exit('All simulations crashed, nothing to display\nPlease check the setup and settings of your optimization run')
	return


# simple moving average
def sma(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')


# exponential moving average
def ewma(a, alpha, windowSize):
	wghts = (1-alpha)**np.arange(windowSize)
	wghts /= wghts.sum()
	out = np.full(len(a), np.nan)
	out = np.convolve(a, wghts, 'same')
	return out


# cast object as string, enclose by parentheses and return a string -- for arguments display in help
def par_wrap(string):
	return '('+str(string)+')'


# set MDAnalysis backend and number of threads
def set_MDA_backend(ns):

	# TODO: propagate number of threads to the functions calls of MDAnalysis, which means do a PR on MDAnalysis github
	# ns.mda_backend = 'serial' # atm force serial in case code is executed on clusters, because MDA will use all threads by default

	if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
		ns.mda_backend = 'OpenMP'
	else:
		# print('MDAnalysis was compiled without OpenMP support, calculation of bonds/angles/dihedrals distributions will use a single thread')
		ns.mda_backend = 'serial'

	return
	

# draw random float between given range and apply rounding to given digit
def draw_float(low, high, dg_rnd):
	
	return round(random.uniform(low, high), dg_rnd) # low and high included


# read one or more molecules from the AA TPR and trajectory
def load_aa_data(ns):

	ns.all_atoms = dict() # atom centered connectivity + atom type + heavy atom boolean + bead(s) to which the atom belongs (can belong to multiple beads depending on mapping)
	ns.all_aa_mols = [] # atom groups for each molecule of interest, in case we use several and average the distributions across many molecules, as we would do for membranes analysis

	if ns.molname_in is None:

		molname_atom_group = ns.aa_universe[0].atoms[0].fragment # select the AA connected graph for the first moltype found in TPR
		ns.all_aa_mols.append(molname_atom_group)
		# print(dir(molname_atom_group.atoms[0])) # for dev, display properties

		# atoms and their attributes
		for i in range(len(molname_atom_group)):

			atom_id = ns.aa_universe[0].atoms[i].id
			atom_type = ns.aa_universe[0].atoms[i].type[0] # TODO: using only first letter but do better with masses for exemple to discriminate/verify 2 letters atom types
			atom_charge = ns.aa_universe[0].atoms[i].charge
			atom_heavy = True
			if atom_type[0].upper() == 'H':
				atom_heavy = False

			ns.all_atoms[atom_id] = {'conn': set(), 'atom_type': atom_type, 'atom_charge': atom_charge, 'heavy': atom_heavy, 'beads_ids': set(), 'beads_types': set(), 'residue_names': set()}
			# print(ns.aa_universe.atoms[i].id, ns.aa_universe.atoms[i])

		# bonds
		for i in range(len(molname_atom_group.bonds)):

			atom_id_1 = molname_atom_group.bonds[i][0].id
			atom_id_2 = molname_atom_group.bonds[i][1].id
			ns.all_atoms[atom_id_1]['conn'].add(atom_id_2)
			ns.all_atoms[atom_id_2]['conn'].add(atom_id_1)			

	# TODO: read multiple instances of give moltype -- for membranes analysis -- USE RESINDEX property or MOLNUM, check for more useful properties
	else:
		pass

	# print(ns.aa_universe.atoms[1000].segindex, ns.aa_universe.atoms[134].resindex, ns.aa_universe.atoms[134].molnum)

	# for seg in ns.aa_universe.segments:

	# 	print(seg.segid)
	# 	sel = ns.aa_universe.select_atoms("segid "+str(seg.segid))
	# 	print(sel.atoms)

	# print(ns.aa_universe.atoms[0].segid)

	# sel = ns.aa_universe.select_atoms("moltype SOL")
	# for atom in sel.atoms:
	# 	print(atom.id)
	# 	print("  ", sel.atoms[atom.id].fragment)

	# TODO: print this charge, if it is not null then we need to check for Q-type beads and for the 2 Q-types that have no defined charge value, raise a warning to tell the user he has to edit the file manually
	# net_charge = molname_atom_group.total_charge()
	# print('Net charge of the reference all atom model:', round(net_charge, 4))

	return


# read coarse-grain ITP
def read_cg_itp_file_grp_comments(ns, itp_lines):

	print('  Reading CG ITP file')
	ns.cg_itp = {'moleculetype': {'molname': '', 'nrexcl': 0}, 'atoms': [], 'constraint': [], 'bond': [], 'angle': [], 'dihedral': [], 'exclusion': []}
	ns.nb_constraints, ns.nb_bonds, ns.nb_angles, ns.nb_dihedrals = -1, -1, -1, -1
	ns.nb_beads_itp = -1

	# for the input of RDF calculations, while allowing nrexcl = 1
	ns.cg_itp['beads_ids_per_beads_types_sing'] = {}
	ns.cg_itp['conn_per_beads_pair_types'] = {}

	# for building the RDF request and specifying beads pairs without neighbors

	for i in range(len(itp_lines)):
		itp_line = itp_lines[i]
		if itp_line != '' and not itp_line.startswith(';'):

			if bool(re.search(r'\[.*moleculetype.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = True, False, False, False, False, False, False
			elif bool(re.search(r'\[.*atoms.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, True, False, False, False, False, False
			elif bool(re.search(r'\[.*constraint.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, False, True, False, False, False, False
			elif bool(re.search(r'\[.*bond.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, False, False, True, False, False, False
			elif bool(re.search(r'\[.*angle.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, False, False, False, True, False, False
			elif bool(re.search(r'\[.*dihedral.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, False, False, False, False, True, False
			elif bool(re.search(r'\[.*exclusion.*\]', itp_line)):
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, False, False, False, False, False, True
			elif bool(re.search(r'\[.*\]', itp_line)):  # all other sections
				r_moleculetype, r_atoms, r_constraints, r_bonds, r_angles, r_dihedrals, r_exclusion = False, False, False, False, False, False, False

			else:
				sp_itp_line = itp_line.split()

				if r_moleculetype:

					ns.cg_itp['moleculetype']['molname'], ns.cg_itp['moleculetype']['nrexcl'] = sp_itp_line[0], int(sp_itp_line[1])

				elif r_atoms:

					bead_id, bead_type, resnr, residue, atom, cgnr, charge = sp_itp_line[:7]
					ns.nb_beads_itp = max(ns.nb_beads_itp, int(bead_id))
					# NOTE: masses are taken from the config and if there are some in the ITP they are ignored
					ns.cg_itp['atoms'].append({'bead_id': int(bead_id)-1, 'bead_type': bead_type, 'resnr': int(resnr), 'residue': residue, 'atom': atom, 'cgnr': int(cgnr), 'charge': float(charge), 'mass': ns.beads_masses[bead_type]}) # retrieve indexing from 0 for CG beads IDS for MDAnalysis

					# record beads ids per beads types, for later RDF calculation
					if bead_type in ns.cg_itp['beads_ids_per_beads_types_sing']:
						ns.cg_itp['beads_ids_per_beads_types_sing'][bead_type].append(int(bead_id)-1)
					else:
						ns.cg_itp['beads_ids_per_beads_types_sing'][bead_type] = [int(bead_id)-1]

				elif r_constraints:

					ns.nb_constraints += 1
					ns.cg_itp['constraint'].append({}) # initialize storage for this new group
				
					try:
						ns.cg_itp['constraint'][ns.nb_constraints]['beads'] = [[int(bead_id)-1 for bead_id in sp_itp_line[0:2]]] # retrieve indexing from 0 for CG beads IDS for MDAnalysis
						# take into account multiple instances of the same molecule in trajectory
						for j in range(1, ns.nb_mol_instances):
							ns.cg_itp['constraint'][ns.nb_constraints]['beads'].append([j*ns.nb_beads_initial+int(bead_id)-1 for bead_id in sp_itp_line[0:2]])
					except ValueError:
						sys.exit(config.header_error+'Incorrect reading of the CG ITP file within [constraints] section, please check this file')

					ns.cg_itp['constraint'][ns.nb_constraints]['funct'] = int(sp_itp_line[2])
					ns.cg_itp['constraint'][ns.nb_constraints]['value'] = float(sp_itp_line[3])

					try:
						ns.cg_itp['constraint'][ns.nb_constraints]['geom_grp'] = sp_itp_line[5]
					except IndexError:
						sys.exit('Cannot find geom group in ITP for constraint nb '+str(ns.nb_bonds+1))

					# record connections between beads types, for later RDF calculation
					bead_type_1, bead_type_2 = [ns.cg_itp['atoms'][int(bead_id)-1]['bead_type'] for bead_id in sp_itp_line[0:2]]
					pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
					pair_beads_ids = sorted([int(bead_id)-1 for bead_id in sp_itp_line[0:2]])
					if pair_type in ns.cg_itp['conn_per_beads_pair_types']:
						ns.cg_itp['conn_per_beads_pair_types'][pair_type].append(pair_beads_ids)
					else:
						ns.cg_itp['conn_per_beads_pair_types'][pair_type] = [pair_beads_ids]

				elif r_bonds:

					ns.nb_bonds += 1
					ns.cg_itp['bond'].append({}) # initialize storage for this new group
				
					try:
						ns.cg_itp['bond'][ns.nb_bonds]['beads'] = [[int(bead_id)-1 for bead_id in sp_itp_line[0:2]]] # retrieve indexing from 0 for CG beads IDS for MDAnalysis
						# take into account multiple instances of the same molecule in trajectory
						for j in range(1, ns.nb_mol_instances):
							ns.cg_itp['bond'][ns.nb_bonds]['beads'].append([j*ns.nb_beads_initial+int(bead_id)-1 for bead_id in sp_itp_line[0:2]])
					except ValueError:
						sys.exit(config.header_error+'Incorrect reading of the CG ITP file within [bonds] section, please check this file')

					ns.cg_itp['bond'][ns.nb_bonds]['funct'] = int(sp_itp_line[2])
					ns.cg_itp['bond'][ns.nb_bonds]['value'] = float(sp_itp_line[3])
					ns.cg_itp['bond'][ns.nb_bonds]['fct'] = float(sp_itp_line[4])

					try:
						ns.cg_itp['bond'][ns.nb_bonds]['geom_grp'] = sp_itp_line[6]
					except IndexError:
						sys.exit('Cannot find geom group in ITP for bond nb '+str(ns.nb_bonds+1))

					# record connections between beads types, for later RDF calculation
					bead_type_1, bead_type_2 = [ns.cg_itp['atoms'][int(bead_id)-1]['bead_type'] for bead_id in sp_itp_line[0:2]]
					pair_type = '_'.join(sorted([bead_type_1, bead_type_2]))
					pair_beads_ids = sorted([int(bead_id)-1 for bead_id in sp_itp_line[0:2]])
					if pair_type in ns.cg_itp['conn_per_beads_pair_types']:
						ns.cg_itp['conn_per_beads_pair_types'][pair_type].append(pair_beads_ids)
					else:
						ns.cg_itp['conn_per_beads_pair_types'][pair_type] = [pair_beads_ids]

				elif r_angles:

					ns.nb_angles += 1
					ns.cg_itp['angle'].append({}) # initialize storage for this new group
				
					try:
						ns.cg_itp['angle'][ns.nb_angles]['beads'] = [[int(bead_id)-1 for bead_id in sp_itp_line[0:3]]] # retrieve indexing from 0 for CG beads IDS for MDAnalysis
						# take into account multiple instances of the same molecule in trajectory
						for j in range(1, ns.nb_mol_instances):
							ns.cg_itp['angle'][ns.nb_angles]['beads'].append([j*ns.nb_beads_initial+int(bead_id)-1 for bead_id in sp_itp_line[0:3]])
					except ValueError:
						sys.exit(config.header_error+'Incorrect reading of the CG ITP file within [angles] section, please check this file')

					ns.cg_itp['angle'][ns.nb_angles]['funct'] = int(sp_itp_line[3])
					ns.cg_itp['angle'][ns.nb_angles]['value'] = float(sp_itp_line[4])
					ns.cg_itp['angle'][ns.nb_angles]['fct'] = float(sp_itp_line[5])

					try:
						ns.cg_itp['angle'][ns.nb_angles]['geom_grp'] = sp_itp_line[7]
					except IndexError:
						sys.exit('Cannot find geom group in ITP for angle nb '+str(ns.nb_angles+1))

				elif r_dihedrals:

					ns.nb_dihedrals += 1
					ns.cg_itp['dihedral'].append({}) # initialize storage for this new group

					try:
						ns.cg_itp['dihedral'][ns.nb_dihedrals]['beads'] = [[int(bead_id)-1 for bead_id in sp_itp_line[0:4]]] # retrieve indexing from 0 for CG beads IDS for MDAnalysis
						# take into account multiple instances of the same molecule in trajectory
						for j in range(1, ns.nb_mol_instances):
							ns.cg_itp['dihedral'][ns.nb_dihedrals]['beads'].append([j*ns.nb_beads_initial+int(bead_id)-1 for bead_id in sp_itp_line[0:4]])
					except ValueError:
						sys.exit(config.header_error+'Incorrect reading of the CG ITP file within [dihedrals] section, please check this file')

					func = int(sp_itp_line[4])
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['funct'] = func
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['value'] = float(sp_itp_line[5]) # issue happens here for functions that are not handled
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['fct'] = float(sp_itp_line[6])

					# handle multiplicity if function assumes multiplicity
					if func in config.dihedral_func_with_mult:
						try:
							ns.cg_itp['dihedral'][ns.nb_dihedrals]['mult'] = int(sp_itp_line[7])
						except (IndexError, ValueError): # incorrect read of multiplicity -- or it was expected but not provided
							sys.exit('  Cannot read multiplicity for dihedral at ITP line '+str(i+1))

						try:
							ns.cg_itp['dihedral'][ns.nb_dihedrals]['geom_grp'] = sp_itp_line[9]
						except IndexError:
							sys.exit('Cannot find geom group in ITP for dihedral nb '+str(ns.nb_dihedrals+1))

					else: # no multiplicity parameter is expected
						ns.cg_itp['dihedral'][ns.nb_dihedrals]['mult'] = ''

						try:
							ns.cg_itp['dihedral'][ns.nb_dihedrals]['geom_grp'] = sp_itp_line[8]
						except IndexError:
							sys.exit('Cannot find geom group in ITP for dihedral nb '+str(ns.nb_dihedrals+1))

				elif r_exclusion:

					ns.cg_itp['exclusion'].append([int(bead_id)-1 for bead_id in sp_itp_line])
	
	ns.nb_constraints += 1
	ns.nb_bonds += 1
	ns.nb_angles += 1
	ns.nb_dihedrals += 1
	print('    Found '+str(ns.nb_constraints)+' constraints groups', flush=True)
	print('    Found '+str(ns.nb_bonds)+' bonds groups', flush=True)
	print('    Found '+str(ns.nb_angles)+' angles groups', flush=True)
	print('    Found '+str(ns.nb_dihedrals)+' dihedrals groups', flush=True)

	return


# load CG beads from NDX-like file
def read_ndx_atoms2beads(ns):

	with open(ns.cg_map_filename, 'r') as fp:

		ndx_lines = fp.read().split('\n')
		ndx_lines = [ndx_line.strip().split(';')[0] for ndx_line in ndx_lines] # split for comments

		ns.atoms_occ_total = collections.Counter()
		ns.all_beads = dict() # atoms id mapped to each bead -- for all residues
		ns.simple_beads = dict() # atoms id mapped to each bead -- for a single residue
		bead_id = 0

		for ndx_line in ndx_lines:
			if ndx_line != '':

				if bool(re.search('\[.*\]', ndx_line)):
					ns.all_beads[bead_id] = {'atoms_id': []}
					ns.simple_beads[bead_id] = []
					lines_read = 0 # error handling, ensure only 1 line is read for each NDX file section/bead

				else:
					try:
						lines_read += 1
						if lines_read > 1:
							sys.exit(config.header_error+'Some sections of the CG beads mapping file have multiple lines, please correct the mapping')
						bead_atoms_id = [int(atom_id)-1 for atom_id in ndx_line.split()] # retrieve indexing from 0 for atoms IDs for MDAnalysis
						ns.all_beads[bead_id]['atoms_id'].extend(bead_atoms_id) # all atoms included in current bead
						ns.simple_beads[bead_id].extend(bead_atoms_id)

						for atom_id in bead_atoms_id: # bead to which each atom belongs (one atom can belong to multiple beads if there is split-mapping)
							ns.atoms_occ_total[atom_id] += 1
						bead_id += 1

					except NameError:
						sys.exit(config.header_error+'The CG beads mapping file does NOT seem to contain CG beads sections, please verify the input mapping')
					except ValueError: # non-integer atom ID provided
						sys.exit(config.header_error+'Incorrect reading of the sections\' content in the CG beads mapping file, please verify the input mapping')

		# take into account multiple instances of the same molecule in trajectory
		# WARNING: we assume it's all the same molecules
		ns.nb_beads_initial = len(ns.all_beads.keys())

		for i in range(1, ns.nb_mol_instances):

			for bead_id_initial in range(ns.nb_beads_initial):
				new_bead_atoms_id = [i*ns.nb_mol_atoms+atom_id for atom_id in ns.all_beads[bead_id_initial]['atoms_id']]
				ns.all_beads[bead_id] = {'atoms_id': new_bead_atoms_id}
				# print('bead id:', bead_id, '-- atoms ids:', ns.all_beads[bead_id]['atoms_id'])

				for atom_id in new_bead_atoms_id: # bead to which each atom belongs (one atom can belong to multiple beads if there is split-mapping)
					ns.atoms_occ_total[atom_id] += 1
				bead_id += 1

	return


# calculate weight ratio of atom ID in given CG bead
def get_atoms_weights_in_beads(ns):

	# print('Calculating atoms weights in respect to CG beads mapping')
	ns.atom_w = dict()
	for bead_id in ns.all_beads:
		# print()
		# print('  Weighting bead_id', bead_id)
		ns.atom_w[bead_id] = []
		beads_atoms_counts = collections.Counter(ns.all_beads[bead_id]['atoms_id'])
		# for atom_id in beads_atoms_counts:
		for atom_id in ns.all_beads[bead_id]['atoms_id']:
			ns.atom_w[bead_id].append(round(beads_atoms_counts[atom_id] / ns.atoms_occ_total[atom_id], 3))
			# print('    Weight ratio is', ns.atom_w[bead_id][atom_id], 'for atom ID', atom_id, 'attributed to CG bead ID', bead_id)

	return


# for each CG bead, create atom groups for trajectory geoms calculation using mass and atom weights across beads
def get_beads_MDA_atomgroups(ns, aa_universe):

	mda_beads_atom_grps, mda_weights_atom_grps = dict(), dict()
	for bead_id in ns.atom_w:

		if ns.map_center == 'COM':
			# print('Created bead_id', bead_id, 'using atoms', [atom_id for atom_id in ns.all_beads[bead_id]['atoms_id']])
			# print('and weights:\n', [ns.atom_w[bead_id][i] * aa_universe.atoms[ns.all_beads[bead_id]['atoms_id'][i]].mass for i in range(len(ns.all_beads[bead_id]['atoms_id']))])
			# print()
			mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in ns.all_beads[bead_id]['atoms_id']], aa_universe)
			mda_weights_atom_grps[bead_id] = np.array([ns.atom_w[bead_id][i] * aa_universe.atoms[ns.all_beads[bead_id]['atoms_id'][i]].mass for i in range(len(ns.all_beads[bead_id]['atoms_id']))])
			# mda_weights_atom_grps[bead_id] = np.array([ns.atom_w[bead_id][atom_id]*ns.all_atoms[atom_id]['atom_mass'] for atom_id in ns.atom_w[bead_id]])

		elif ns.map_center == 'COG':
			mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in ns.all_beads[bead_id]['atoms_id']], aa_universe)
			mda_weights_atom_grps[bead_id] = np.array([1 for _ in ns.all_beads[bead_id]['atoms_id']])

	return mda_beads_atom_grps, mda_weights_atom_grps


# read 1 column of xvg file and return as array
# column is 0-indexed
def read_xvg_col(xvg_file, col):
	with open(xvg_file, 'r') as fp:
		lines = [line.strip() for line in fp.read().split('\n')]
		data = []
		for line in lines:
			if not line.startswith(('#', '@')) and line != '':
				sp_lines = list(map(float, line.split()))
				data.append(sp_lines[col])
	return data


# set dimensions of the search space according to the type of optimization (= geom type(s) to optimize)
def get_search_space_boundaries(ns):
	
	search_space_boundaries = []

	# print('Setting search space boundaries')
	nb_LJ = 0

	for param_dict in ns.all_params_opti:  # list of dict having unique keys
		for param in param_dict:  # accessing each single key of each dict

			# bonds, tune both value and force constants
			if param.startswith('B') and ns.tune_geoms:
				if param_dict[param] == 2:
					search_space_boundaries.extend([ns.params_val[param]['range']])  # val
				search_space_boundaries.extend([[ns.user_config.min_fct_bonds, ns.user_config.max_fct_bonds]])  # fct

			# angles, tune either: (1) both value and force constants or (2) fct only if angle was initially set to 90°, 120° or 180°
			elif param.startswith('A') and ns.tune_geoms:
				if param_dict[param] == 2:
					search_space_boundaries.extend([ns.params_val[param]['range']])  # val
				search_space_boundaries.extend([[ns.user_config.min_fct_angles, ns.user_config.max_fct_angles]])  # fct

			elif param.startswith('r_'):  # now it's bead radius that we tune for each bead type
				search_space_boundaries.extend([[ns.user_config.min_radius, ns.user_config.max_radius]])

			elif param.startswith('LJ') and ns.tune_eps:
				if ns.eps_perc_range == None and ns.eps_flat_range == None:
					search_space_boundaries.extend([[ns.user_config.min_epsilon, ns.user_config.max_epsilon]])
				else:
					search_space_boundaries.extend([ns.eps_ranges[nb_LJ]])
					nb_LJ += 1

	return search_space_boundaries


# build initial guesses for particles initialization, by generating some noise around the best solution
def get_cycle_restart_guess_list(ns):
	path_log_file_1 = f'{ns.exec_folder}/{ns.opti_moves_file}'
	best_per_sw_iter = {}  # find best score and parameters -- per swarm iteration
	best_score, best_sw_iter = np.inf, None  # find best score -- within the previous opti cycle

	with open(path_log_file_1, 'r') as fp:
		lines = fp.read().split('\n')
		params_names = lines[0].split()[4:]

		for i in range(1, len(lines)):
			if lines[i] != '':
				sp_line = lines[i].split()
				n_cycle = int(sp_line[0])
				n_sw_iter = int(sp_line[1])
				n_particle = int(sp_line[2])
				iter_score = float(sp_line[3])
				params_values = sp_line[4:]

				# find the best particle per swarm iteration
				if ns.n_cycle - 1 == n_cycle:  # pick ONLY from the previous cycle
					if n_sw_iter not in best_per_sw_iter or iter_score < best_per_sw_iter[n_sw_iter]['iter_score']:
						best_per_sw_iter[n_sw_iter] = {
							'iter_score': iter_score,
							'n_particle': n_particle,
							'params': {}
						}
						for j, param_name in enumerate(params_names):
							best_per_sw_iter[n_sw_iter]['params'][param_name] = float(params_values[j])
					if iter_score < best_score:
						best_score = iter_score
						best_sw_iter = n_sw_iter

	# initialize particles for the new opti cycle
	cycle_restart_guess_list = []

	# first particle == previous best in previous opti cycle
	input_guess = []

	for param_dict in ns.all_params_opti:  # list of dict having unique keys
		for param in param_dict:  # accessing each single key of each dict

			if param.startswith('B') and ns.tune_geoms:
				if param_dict[param] == 2:
					input_guess.append(min(max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_val'], ns.params_val[param]['range'][0]), ns.params_val[param]['range'][1]))  # val
				input_guess.append(min(max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_fct'], ns.user_config.min_fct_bonds), ns.user_config.max_fct_bonds))  # fct

			elif param.startswith('A') and ns.tune_geoms:
				if param_dict[param] == 2:
					input_guess.append(min(max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_val'], ns.params_val[param]['range'][0]), ns.params_val[param]['range'][1]))  # val
				input_guess.append(min(max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_fct'], ns.user_config.min_fct_angles), ns.user_config.max_fct_angles))  # fct

			elif param.startswith('r_'):
				input_guess.append(min(max(best_per_sw_iter[best_sw_iter]['params'][param], ns.user_config.min_radius), ns.user_config.max_radius))

			elif param.startswith('LJ') and ns.tune_eps:
				input_guess.append(min(max(best_per_sw_iter[best_sw_iter]['params'][param], ns.user_config.min_epsilon), ns.user_config.max_epsilon))

	cycle_restart_guess_list.append(input_guess)

	# other particles, noise around the previous best in previous opti cycle
	for i in range(1, ns.nb_particles):
		input_guess = []
		nb_LJ = 0

		for param_dict in ns.all_params_opti:  # list of dict having unique keys
			for param in param_dict:  # accessing each single key of each dict

				if param.startswith('B') and ns.tune_geoms:
					if param_dict[param] == 2:
						draw_low = max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_val'] - config.bond_value_guess_variation, ns.params_val[param]['range'][0])
						draw_high = min(best_per_sw_iter[best_sw_iter]['params'][f'{param}_val'] + config.bond_value_guess_variation, ns.params_val[param]['range'][1])
						input_guess.append(draw_float(draw_low, draw_high, 3))  # val
					draw_low = max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_fct'] - config.bond_fct_guess_variation, ns.user_config.min_fct_bonds)
					draw_high = min(best_per_sw_iter[best_sw_iter]['params'][f'{param}_fct'] + config.bond_fct_guess_variation, ns.user_config.max_fct_bonds)
					input_guess.append(draw_float(draw_low, draw_high, 3))  # fct

				elif param.startswith('A') and ns.tune_geoms:
					if param_dict[param] == 2:
						draw_low = max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_val'] - config.angle_value_guess_variation, ns.params_val[param]['range'][0])
						draw_high = min(best_per_sw_iter[best_sw_iter]['params'][f'{param}_val'] + config.angle_value_guess_variation, ns.params_val[param]['range'][1])
						input_guess.append(draw_float(draw_low, draw_high, 3))  # val
					draw_low = max(best_per_sw_iter[best_sw_iter]['params'][f'{param}_fct'] - config.angle_fct_guess_variation, ns.user_config.min_fct_angles)
					draw_high = min(best_per_sw_iter[best_sw_iter]['params'][f'{param}_fct'] + config.angle_fct_guess_variation, ns.user_config.max_fct_angles)
					input_guess.append(draw_float(draw_low, draw_high, 3))  # fct

				elif param.startswith('r_'):
					draw_low = max(best_per_sw_iter[best_sw_iter]['params'][param] - config.radius_guess_variation, ns.user_config.min_radius)
					draw_high = min(best_per_sw_iter[best_sw_iter]['params'][param] + config.radius_guess_variation, ns.user_config.max_radius)
					input_guess.append(draw_float(draw_low, draw_high, 3))  # radius

				elif param.startswith('LJ') and ns.tune_eps:
					if ns.eps_perc_range is None and ns.eps_flat_range is None:  # EPS ranges according to config file values
						draw_low = max(best_per_sw_iter[best_sw_iter]['params'][param] - config.eps_guess_variation, ns.user_config.min_epsilon)
						draw_high = min(best_per_sw_iter[best_sw_iter]['params'][param] + config.eps_guess_variation, ns.user_config.max_epsilon)
					else:  # EPS ranges according to either flat or perc around starting values OF THE VERY FIRST OPTI CYCLE
						draw_low = ns.eps_ranges[nb_LJ][0]
						draw_high = ns.eps_ranges[nb_LJ][1]
					input_guess.append(draw_float(draw_low, draw_high, 3))  # eps
					nb_LJ += 1

		cycle_restart_guess_list.append(input_guess)

	return cycle_restart_guess_list


# build initial guesses for particles initialization, as variations around initial parameters from either the ITPs or config file
def get_initial_guess_list(ns):

	initial_guess_list = []  # array of arrays (inner arrays are the values used for particles initialization)

	# initialize first particle
	input_guess = []
	nb_LJ = 0  # to find ordered LJ pairs within ns.all_params_opti when geoms+LJ are present
	for param_dict in ns.all_params_opti:  # list of dict having unique keys
		for param in param_dict:  # accessing each single key of each dict

			# bonds, tune both value and force constants
			if param.startswith('B') and ns.tune_geoms:
				if param_dict[param] == 2:
					if ns.bonds_equi_val_from_config and ns.n_cycle == 1:
						input_guess.append(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['val'])  # val
					else:
						input_guess.append(ns.params_val[param]['avg'])  # val
				input_guess.append(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'])  # fct

			# angles, tune either: (1) both value and force constants or (2) fct only if angle was initially set to 90°, 120° or 180°
			elif param.startswith('A') and ns.tune_geoms:
				if param_dict[param] == 2:
					input_guess.append(ns.params_val[param]['avg'])  # val
				input_guess.append(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'])  # fct

			elif param.startswith('r_'):
				radii_grp = param.split('_')[1]
				r_default = ns.user_config.init_beads_radii[radii_grp]
				input_guess.append(r_default)

			elif param.startswith('LJ') and ns.tune_eps:
				eps_default = ns.user_config.init_nonbonded[' '.join(ns.all_beads_pairs[nb_LJ])]['eps'] # get initial LJ from existing FF data
				input_guess.append(eps_default)
				nb_LJ += 1

	initial_guess_list.append(input_guess)

	# for the other particles we generate variations of the input CG ITP, still within defined boundaries for optimization
	# boundaries are defined:
	#   for constraints/bonds length and angles/dihedrals values, according to atomistic mapped trajectory and maximum searchable 
	#   for force constants, according to default or user provided maximal ranges (see config file for defaults)
	for i in range(1, ns.nb_particles):

		input_guess = []
		nb_LJ = 0
		for param_dict in ns.all_params_opti: # list of dict having unique keys
			for param in param_dict: # accessing each single key of each dict

				# bonds, tune both value and force constants
				if param.startswith('B') and ns.tune_geoms:

					if param_dict[param] == 2:
						if ns.bonds_equi_val_from_config and ns.n_cycle == 1:
							draw_low = max(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['val'] - config.bond_value_guess_variation, ns.params_val[param]['range'][0])
							draw_high = min(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['val'] + config.bond_value_guess_variation, ns.params_val[param]['range'][1])
							input_guess.append(draw_float(draw_low, draw_high, 3)) # val
						else:
							draw_low = max(ns.params_val[param]['avg'] - config.bond_value_guess_variation, ns.params_val[param]['range'][0])
							draw_high = min(ns.params_val[param]['avg'] + config.bond_value_guess_variation, ns.params_val[param]['range'][1])
							input_guess.append(draw_float(draw_low, draw_high, 3)) # val

					draw_low = max(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] - config.bond_fct_guess_variation, ns.user_config.min_fct_bonds)
					draw_high = min(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] + config.bond_fct_guess_variation, ns.user_config.max_fct_bonds)

					input_guess.append(draw_float(draw_low, draw_high, 3)) # fct

				# angles, tune either: (1) both value and force constants or (2) fct only if angle was initially set to 90°, 120° or 180°
				elif param.startswith('A') and ns.tune_geoms:

					if param_dict[param] == 2:
						draw_low = max(ns.params_val[param]['avg'] - config.angle_value_guess_variation, ns.params_val[param]['range'][0])
						draw_high = min(ns.params_val[param]['avg'] + config.angle_value_guess_variation, ns.params_val[param]['range'][1])
						input_guess.append(draw_float(draw_low, draw_high, 3)) # val

					draw_low = max(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] - config.angle_fct_guess_variation, ns.user_config.min_fct_angles)
					draw_high = min(ns.opti_config.init_bonded[ns.mapping_type][ns.solv][param]['fct'] + config.angle_fct_guess_variation, ns.user_config.max_fct_angles)

					input_guess.append(draw_float(draw_low, draw_high, 3)) # fct

				elif param.startswith('r_'):

					radii_grp = param.split('_')[1]
					r_default = ns.user_config.init_beads_radii[radii_grp] # got actualized with best previous result at end of opti cycle

					# random variations in config defined range
					draw_low = max(r_default-config.radius_guess_variation, ns.user_config.min_radius)
					draw_high = min(r_default+config.radius_guess_variation, ns.user_config.max_radius)

					input_guess.append(draw_float(draw_low, draw_high, 3)) # radius

				elif param.startswith('LJ') and ns.tune_eps:

					eps_default = ns.user_config.init_nonbonded[' '.join(ns.all_beads_pairs[nb_LJ])]['eps']  # get initial LJ from existing FF data -- got actualized with best previous result at end of opti cycle

					# EPS ranges according to config file values
					if ns.eps_perc_range is None and ns.eps_flat_range is None:
						draw_low = max(eps_default - config.eps_guess_variation, ns.user_config.min_epsilon)
						draw_high = min(eps_default + config.eps_guess_variation, ns.user_config.max_epsilon)
					# EPS ranges according to either flat or perc around starting values
					else:
						draw_low = ns.eps_ranges[nb_LJ][0]
						draw_high = ns.eps_ranges[nb_LJ][1]

					input_guess.append(draw_float(draw_low, draw_high, 3)) # EPS

					nb_LJ += 1

		initial_guess_list.append(input_guess) # register new particle, built during this loop

	return initial_guess_list


# use selected whole molecules as MDA atomgroups and make their coordinates whole, inplace, across the complete tAA rajectory
def make_aa_traj_whole_for_selected_mols(ns):

	for i in range(len(ns.active_aa_trajs)):
		if ns.active_aa_trajs[i] == True:
			for _ in ns.aa_universe[i].trajectory:
				for aa_mol in ns.all_aa_mols:
					mda.lib.mdamath.make_whole(aa_mol, inplace=True)
	return


# get bins and distance matrix for pairwise distributions comparison using Earth Mover's Distance (EMD)
def create_bins_and_dist_matrices(ns, constraints=True):

	# bins for histogram distributions of bonds/angles
	if constraints:
		ns.bins_constraints = np.arange(0, ns.bonded_max_range+ns.bw_constraints, ns.bw_constraints)
	ns.bins_bonds = np.arange(0, ns.bonded_max_range+ns.bw_bonds, ns.bw_bonds)
	ns.bins_angles = np.arange(0, 180+2*ns.bw_angles, ns.bw_angles) # one more bin for angle/dihedral because we are later using a strict inferior for bins definitions
	# ns.bins_dihedrals = np.arange(-180, 180+2*ns.bw_dihedrals, ns.bw_dihedrals)

	# bins distance for Earth Mover's Distance (EMD) to calculate histograms similarity
	if constraints:
		bins_constraints_reshape = np.array(ns.bins_constraints).reshape(-1,1)
		ns.bins_constraints_dist_matrix = cdist(bins_constraints_reshape, bins_constraints_reshape)
	bins_bonds_reshape = np.array(ns.bins_bonds).reshape(-1,1)
	ns.bins_bonds_dist_matrix = cdist(bins_bonds_reshape, bins_bonds_reshape)
	bins_angles_reshape = np.array(ns.bins_angles).reshape(-1,1)
	ns.bins_angles_dist_matrix = cdist(bins_angles_reshape, bins_angles_reshape)
	# bins_dihedrals_reshape = np.array(ns.bins_dihedrals).reshape(-1,1)
	# bins_dihedrals_dist_matrix = cdist(bins_dihedrals_reshape, bins_dihedrals_reshape) # 'classical' distance matrix
	# ns.bins_dihedrals_dist_matrix = np.where(bins_dihedrals_dist_matrix > max(bins_dihedrals_dist_matrix[0])/2, max(bins_dihedrals_dist_matrix[0])-bins_dihedrals_dist_matrix, bins_dihedrals_dist_matrix) # periodic distance matrix

	# RDFs
	ns.bins_vol_shell = np.arange(ns.bw_rdfs, ns.cut_rdfs_short + ns.bw_rdfs, ns.bw_rdfs)

	# volume of the radial shell
	vol_real = 4/3.0 * np.pi * np.power(ns.bins_vol_shell, 3)
	ns.vol_shell = np.copy(vol_real)
	for i in range(1, len(ns.vol_shell)):
		ns.vol_shell[i] = vol_real[i] - vol_real[i-1]

	# create the volume-based distance matrix for EMD comparison of RDFs -- new approach based on counts
	ns.bins_vol_matrix = np.empty([len(ns.vol_shell), len(ns.vol_shell)], dtype=np.float)

	for i in range(len(ns.vol_shell)):
		for j in range(len(ns.vol_shell)):

			if i == j:
				ns.bins_vol_matrix[i, j] = 0
			else:
				if j > i:
					ns.bins_vol_matrix[i, j] = ns.vol_shell[j] / ns.vol_shell[i]
				else:
					ns.bins_vol_matrix[i, j] = ns.vol_shell[i] / ns.vol_shell[j]

	return


def initialize_cg_traj(ns):

	masses = np.array([val['mass'] for _ in range(ns.nb_mol_instances) for val in ns.cg_itp['atoms']])
	names = np.array([val['atom'] for _ in range(ns.nb_mol_instances) for val in ns.cg_itp['atoms']])
	
	# NOTE: this is only correct with a single residue in each molecular specie
	resnames = np.array([ns.cg_itp['atoms'][0]['residue'] for _ in range(ns.nb_mol_instances)])
	resid = np.array([molnum for molnum in range(ns.nb_mol_instances) for _ in ns.cg_itp['atoms']])
	ressig = np.array([0 for _ in range(ns.nb_mol_instances)])

	aa2cg_universe = mda.Universe.empty(ns.nb_mol_instances*len(ns.cg_itp['atoms']), n_residues=ns.nb_mol_instances, atom_resindex=resid, n_segments=ns.nb_mol_instances, residue_segindex=ressig, trajectory=True)
	
	aa2cg_universe.add_TopologyAttr('masses', np.array(masses))
	aa2cg_universe.add_TopologyAttr('names', names)
	aa2cg_universe.add_TopologyAttr('resnames', resnames)

	return aa2cg_universe


def map_aa2cg_traj(ns, aa_universe, aa2cg_universe, mda_beads_atom_grps, mda_weights_atom_grps):

	if ns.map_center == 'COM':
		print('    Interpretation: Center of Mass (COM)')
	elif ns.map_center == 'COG':
		print('    Interpretation: Center of Geometry (COG)')

	# regular beads are mapped using center of mass of groups of atoms
	coord = np.empty((len(aa_universe.trajectory), len(ns.all_beads), 3))

	for bead_id in range(len(ns.all_beads)):

		bead_id_sing = bead_id % ns.nb_beads_itp

		if not ns.cg_itp['atoms'][bead_id_sing]['bead_type'].startswith('v'):  # bead is NOT a virtual site
			traj = np.empty((len(aa_universe.trajectory), 3))
			for ts in aa_universe.trajectory:
				traj[ts.frame] = mda_beads_atom_grps[bead_id].center(
					mda_weights_atom_grps[bead_id], pbc=None, compound='group'
				)  # no need for PBC handling, trajectories were made wholes for the molecule
			
			coord[:, bead_id, :] = traj

	aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

	# VS treatment below has NOT been actualized with respect to Swarm-CG 1 for here handling multiple molecules in trajectory -- so currently commented

	# # virtual sites are mapped using previously defined regular beads positions and appropriate virtual sites functions
	# # it is also possible to use a VS for defining another VS position, if the VS used for definition is defined before
	# # no need to check if the functions used for VS definition are correct here, this has been done already
	# for bead_id in range(len(ns.cg_itp['atoms'])):
	# 	if ns.cg_itp['atoms'][bead_id]['bead_type'].startswith('v'):

	# 		traj = np.empty((len(aa2cg_universe.trajectory), 3))

	# 		if ns.cg_itp['atoms'][bead_id]['vs_type'] == 2:
	# 			vs_def_beads_ids = ns.cg_itp['virtual_sites2'][bead_id]['vs_def_beads_ids']
	# 			vs_params = ns.cg_itp['virtual_sites2'][bead_id]['vs_params']

	# 			if ns.cg_itp['virtual_sites2'][bead_id]['func'] == 1:
	# 				vsf.vs2_func_1(ns, traj, vs_def_beads_ids, vs_params)
	# 			elif ns.cg_itp['virtual_sites2'][bead_id]['func'] == 2:
	# 				vsf.vs2_func_2(ns, traj, vs_def_beads_ids, vs_params)

	# 		if ns.cg_itp['atoms'][bead_id]['vs_type'] == 3:
	# 			vs_def_beads_ids = ns.cg_itp['virtual_sites3'][bead_id]['vs_def_beads_ids']
	# 			vs_params = ns.cg_itp['virtual_sites3'][bead_id]['vs_params']

	# 			if ns.cg_itp['virtual_sites3'][bead_id]['func'] == 1:
	# 				vsf.vs3_func_1(ns, traj, vs_def_beads_ids, vs_params)
	# 			elif ns.cg_itp['virtual_sites3'][bead_id]['func'] == 2:
	# 				vsf.vs3_func_2(ns, traj, vs_def_beads_ids, vs_params)
	# 			elif ns.cg_itp['virtual_sites3'][bead_id]['func'] == 3:
	# 				vsf.vs3_func_3(ns, traj, vs_def_beads_ids, vs_params)
	# 			elif ns.cg_itp['virtual_sites3'][bead_id]['func'] == 4:
	# 				vsf.vs3_func_4(ns, traj, vs_def_beads_ids, vs_params)

	# 		# here it's normal there is only function 2, that's the only one that exists in gromacs for some reason
	# 		if ns.cg_itp['atoms'][bead_id]['vs_type'] == 4:
	# 			vs_def_beads_ids = ns.cg_itp['virtual_sites4'][bead_id]['vs_def_beads_ids']
	# 			vs_params = ns.cg_itp['virtual_sites4'][bead_id]['vs_params']

	# 			if ns.cg_itp['virtual_sites4'][bead_id]['func'] == 2:
	# 				vsf.vs4_func_2(ns, traj, vs_def_beads_ids, vs_params)

	# 		if ns.cg_itp['atoms'][bead_id]['vs_type'] == 'n':
	# 			vs_def_beads_ids = ns.cg_itp['virtual_sitesn'][bead_id]['vs_def_beads_ids']
	# 			vs_params = ns.cg_itp['virtual_sitesn'][bead_id]['vs_params']

	# 			if ns.cg_itp['virtual_sitesn'][bead_id]['func'] == 1:
	# 				vsf.vsn_func_1(ns, traj, vs_def_beads_ids)
	# 			elif ns.cg_itp['virtual_sitesn'][bead_id]['func'] == 2:
	# 				vsf.vsn_func_2(ns, traj, vs_def_beads_ids, bead_id)
	# 			elif ns.cg_itp['virtual_sitesn'][bead_id]['func'] == 3:
	# 				vsf.vsn_func_3(ns, traj, vs_def_beads_ids, vs_params)

	# 		coord[:, bead_id, :] = traj

	# aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

	# copy dimensions from the AA original universe
	for ts in aa_universe.trajectory:
		aa2cg_universe.trajectory[ts.frame].dimensions = ts.dimensions.copy()
		# print(aa2cg_universe.trajectory[ts.frame].dimensions)

	return aa2cg_universe


# calculate bonds distribution from AA trajectory
def get_AA_bonds_distrib(ns, beads_ids, grp_type, uni):

	# bond_values = np.empty(len(ns.aa_universe.trajectory) * len(beads_ids))
	bond_values = np.empty(len(uni.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in uni.trajectory:
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2 = beads_ids[i]
			bead_pos_1[i] = uni.atoms[bead_id_1].position
			bead_pos_2[i] = uni.atoms[bead_id_2].position

		mda.lib.distances.calc_bonds(bead_pos_1, bead_pos_2, backend=ns.mda_backend, box=None, result=frame_values)
		bond_values[len(beads_ids)*ts.frame:len(beads_ids)*(ts.frame+1)] = frame_values / 10 # retrieved nm

	bond_avg = round(np.average(bond_values), 3)
	if grp_type == 'constraint':
		bond_hist = np.histogram(bond_values, ns.bins_constraints, density=True)[0] * ns.bw_constraints  # retrieve 1-sum densities
	elif grp_type == 'bond':
		bond_hist = np.histogram(bond_values, ns.bins_bonds, density=True)[0] * ns.bw_bonds  # retrieve 1-sum densities
	else:
		sys.exit('Coding error get_AA_bonds_distrib')

	return bond_avg, bond_hist, bond_values


# calculate angles distribution from AA trajectory
def get_AA_angles_distrib(ns, beads_ids, uni):

	angle_values_rad = np.empty(len(uni.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in uni.trajectory:
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2, bead_id_3 = beads_ids[i]
			bead_pos_1[i] = uni.atoms[bead_id_1].position
			bead_pos_2[i] = uni.atoms[bead_id_2].position
			bead_pos_3[i] = uni.atoms[bead_id_3].position

		mda.lib.distances.calc_angles(bead_pos_1, bead_pos_2, bead_pos_3, backend=ns.mda_backend, box=None, result=frame_values)
		angle_values_rad[len(beads_ids)*ts.frame:len(beads_ids)*(ts.frame+1)] = frame_values

	angle_values_deg = np.rad2deg(angle_values_rad)
	angle_avg = round(np.mean(angle_values_deg), 3)
	angle_hist = np.histogram(angle_values_deg, ns.bins_angles, density=True)[0]*ns.bw_angles # retrieve 1-sum densities

	return angle_avg, angle_hist, angle_values_deg
