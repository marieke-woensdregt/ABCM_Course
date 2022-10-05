import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


###################### PARAMETER SETTINGS: ######################
n_runs = 10  # int: number of independent simulation runs. Cuskley et al. (2018) used 100
pop_sizes = [10, 50]  # list of ints: initial pop sizes. Cuskley et al. (2018) used 20 for small and 100 for large pop
n_lemmas = 14  # int: number of lemmas. Cuskley et al. (2018) used 28
n_tokens = 250  # int: number of tokens in vocabulary. Cuskley et al. seem to have used 500 (in C++ implementation)
n_inflections = 6  # int: number of inflections. Cuskley et al. (2018) used 12
zipf_exponent = 2  # int: exponent used to create Zipfian frequency distribution. Cuskley et al., 2018 used 2
k_proficiency = 750  # int: token threshold that determines proficiency. Cuskley et al. (2018) used 1500
r_replacement = 0.001  # float: replacement rate for turnover condition. Cuskley et al. (2018) used 0.001.
# At every interaction, there is an r chance that a randomly selected learner will be replaced by a new learner
g_growth = 0.001  # float: growth rate for growth condition. Cuskley et al. (2018) used 0.001.
# At every interaction, there's a g chance that a new learner will be *added* to the population
replacement = True  # Boolean: determines whether this simulation includes replacement (turnover)
growth = False  # Boolean; determines whether this simulation includes growth
t_timesteps = 3000  # int: number of timesteps to run per simulation. Cuskley et al. (2018) used 10,000
d_memory = 100  # int: no. of timesteps after which agent forgets lemma-inflection pairing. Cuskley et al. used 100


def generate_vocab(n_lemmas, zipf_exponent, n_tokens):
	"""
	Generates a vocabulary (numpy array of n_tokens tokens of n_lemmas types, with Zipfian frequency distribution)
	:param n_lemmas: int: number of lemmas
	:param zipf_exponent: int: exponent used to create Zipfian frequency distribution. Cuskley et al., 2018 used 2
	:param n_tokens: int: number of tokens in vocabulary. Cuskley et al. seem to have used 500 (in C++ implementation)
	:return: (1) numpy array containing n_tokens tokens of n_lemmas types; (2) numpy array with log frequency per lemma
	"""
	lemma_indices = np.arange(n_lemmas)  # create numpy array with index for each lemma
	zipf_dist = np.random.zipf(zipf_exponent, size=n_lemmas)  # create Zipfian frequency distribution for lemmas
	zipf_dist_in_probs = np.divide(zipf_dist, np.sum(zipf_dist))
	zipf_dist_for_n_tokens = np.multiply(zipf_dist_in_probs, n_tokens)
	zipf_dist_for_n_tokens = np.ceil(zipf_dist_for_n_tokens)  # Round UP, so that we get *at least* n_tokens in vocab
	vocabulary = np.array([])
	for i in range(len(lemma_indices)):
		lemma_index = lemma_indices[i]
		lemma_freq = zipf_dist_for_n_tokens[i]
		vocabulary = np.concatenate((vocabulary, np.array([lemma_index for x in range(int(lemma_freq))])))
	for j in range(2):  # doing this twice because sth weird w/ np.delete() function: doesn't always delete all indices
		# (possibly to do with later index going out of bounds once previous indices have been deleted)
		if vocabulary.shape[0] > n_tokens:  # if vocab is larger than n_tokens, randomly remove excess tokens
			random_indices = np.random.choice(np.arange(vocabulary.shape[0]), size=(vocabulary.shape[0] - n_tokens))
			vocabulary = np.delete(vocabulary, random_indices)
	np.random.shuffle(vocabulary)  # finally, shuffle the array so that tokens of lemmas occur in random order
	vocabulary = vocabulary.astype(int)
	return vocabulary, np.log(zipf_dist_in_probs)


class Inflection:
	"""
	Class which defines an inflection as paired with a lemma
	"""
	def __init__(self, interactions=0, successes=0, weight=np.nan, last_interaction=np.nan):
		"""
		Initialises Inflection object
		:param interactions: int: number of interactions agents has had about this lemma
		:param successes: int: no. of successful interactions agent has had about this lemma
		:param weight: float: no. of successes / no. of interactions. Initialised as NAN for pairing is non-existent
		:param last_interaction: int: timestep when the pairing was last encountered, later compared against d_memory
		"""
		self.interactions = interactions
		self.successes = successes
		self.weight = weight
		self.last_interaction = last_interaction

	def empty_inflection(self):
		"""
		Empties the inflection by resetting each of its attributes; used by Lemma.purge() method when memory window
		(i.e., d_memory timesteps) has elapsed since last interaction with this inflection
		:return: resets each of the inflection object's attributes; doesn't return anything
		"""
		self.interactions = 0
		self.successes = 0
		self.weight = np.nan
		self.last_interaction = np.nan


class Lemma:
	"""
	Lemma class
	"""
	def __init__(self, lemma_index, tokens, seen, inflections):
		"""
		Initialises Lemma object
		:param lemma_index: int: index of the lemma
		:param tokens: int: number of times the agent has encountered this lemma
		:param seen: Boolean: whether the agent has encountered this lemma before
		:param inflections: dictionary with keys: "interactions", "successes", "weight", "last_interaction"
		"""
		self.index = lemma_index
		self.tokens = tokens
		self.seen = seen
		self.inflections = inflections

	def reset_lemma(self):  # TODO: Check whether this method can be made obsolete by just having default input arguments
		"""
		Initialises/resets all attributes of the lemma object
		:param self:
		:return:
		"""
		self.tokens = 0
		self.seen = False
		self.inflections = [Inflection() for i in range(n_inflections)]  # n_inflections is global variable

	def add_inflection(self, infl_index, outcome, timestep):
		"""
		Adds an inflection to the lemma (as a result of an interaction in which that inflection was used),
		with weight depending on the outcome of the interaction (success or failure) and the number of
		previous interaction in which this inflection was used
		:param infl_index: int: index of the inflection in self.inflections
		:param outcome: int: 1 if success (i.e., if receiver has lemma-inflection pairing in inventory), 0 if failure
		:param timestep: int: timestep of current interaction
		:return: updates attributes of lemma object; doesn't return anything
		"""
		self.seen = True
		self.tokens = 1
		self.inflections[infl_index].interactions = 1
		self.inflections[infl_index].successes = outcome
		self.inflections[infl_index].weight = float(outcome) / float(self.inflections[infl_index].interactions)
		self.inflections[infl_index].last_interaction = timestep

	def update_inflection(self, infl_index, outcome, timestep):
		"""
		Updates a lemma-inflection pairing based on the outcome of an interaction
		:param infl_index: int: index of the inflection in self.inflections
		:param outcome: int: 1 if success (i.e., if receiver has lemma-inflection pairing in inventory), 0 if failure
		:param timestep: int: timestep of current interaction
		:return: updates attributes of lemma object; doesn't return anything
		"""
		self.tokens += 1
		self.inflections[infl_index].interactions += 1
		self.inflections[infl_index].successes += outcome
		self.inflections[infl_index].weight = float(self.inflections[infl_index].successes) / float(
			self.inflections[infl_index].interactions)
		self.inflections[infl_index].last_interaction = timestep

	def has_inflection(self, infl_index):
		"""
		Checks whether agent already has a specific inflection (indicated by infl_index) for this lemma
		:param infl_index: int: index of the inflection in self.inflections
		:return: Boolean: True if agent already has this specific inflection for this lemma, False if not
		"""
		if self.inflections[infl_index].interactions > 0:
			return True
		else:
			return False

	def get_best(self):  # TODO: Looks like the Cuskley et al. implementation in C++ just selects the first inflection in the array that has the highest weight
		"""
		Finds indices of inflections with highest weight for this lemma
		:return: int: index of highest-weighted inflection. If multiple with max weight, one is selected randomly
		"""
		weight_array = np.array([self.inflections[i].weight for i in range(len(self.inflections))])
		if np.isnan(weight_array).all() == True:  # if only NANs in the array, just choose an index randomly
			max_index = np.random.choice(np.arange(
				len(self.inflections)))  # TODO: Looks like in the Cuskley et al. implementation in C++, this method returns -1 as an index if none of the weights have been set yet...
		else:
			max_weight = np.nanmax(weight_array)
			max_indices = np.where(weight_array == max_weight)[0]
			max_index = np.random.choice(max_indices)
		return max_index

	def has_any_inflection(self):
		"""
		Checks whether this lemma has any inflections yet (this is considered to be the case if any of the possible
		inflections have come up in an interaction about this lemma before).  # TODO: Check whether it really makes sense to assume that if self.inflections[i].interactions > 0, this means that this lemma has an existing inflection.
		:return: Boolean: False if lemma object doesn't have any inflections yet; True if it does
		"""
		interactions_per_inflection = np.array([self.inflections[i].interactions for i in range(len(self.inflections))])
		if np.sum(interactions_per_inflection) == 0:
			return False
		elif np.sum(interactions_per_inflection) > 0:
			return True

	def purge(self, timestep):
		"""
		Resets lemma-inflection pairing if memory window (d_memory) has elapsed
		:param timestep: int: timestep of current interaction
		:return: updates self.inflections attribute of lemma object; doesn't return anything
		"""
		for i in range(len(self.inflections)):
			if (timestep - self.inflections[
				i].last_interaction) > d_memory:  # d_memory is global variable; see parameter settings
				self.inflections[i].empty_inflection()



class Agent:
	"""
	Agent class
	"""
	def __init__(self, tokens=0, k_threshold=k_proficiency, memory_window=d_memory, type_generalise=False,
				 is_active=False):
		"""
		Initialises Agent object
		:param tokens: int: number of tokens seen by the agent in total. Initial value: 0
		:param k_threshold: int: token threshold that determines proficiency. Default: k_proficiency (=global variable)
		:param memory_window: int: no. of timesteps after which agent forgets pairing. Default: d_memory (=global var.)
		:param type_generalise: Boolean: False = agent is token-generaliser; True = type-generaliser. Initial: False
		:param is_active: Boolean: Initial value: False. Agent's status gets changed to active when it gets added to pop
		"""
		self.tokens = tokens
		self.k_threshold = k_threshold
		self.memory_window = memory_window
		self.type_generalise = type_generalise
		self.is_active = is_active
		empty_inflections = [Inflection() for i in range(n_inflections)]  # used for initiliasing empty vocab below
		self.vocabulary = [Lemma(0, 0, False, empty_inflections) for x in range(n_lemmas)]  # initialise empty vocab

	def reset_agent(self):
		"""
		Resets agent's attributes to initial/empty
		:return: resets agent's attributes; doesn't return anything
		"""
		self.is_active = True
		self.tokens = 0
		self.type_generalise = False
		for lemma in self.vocabulary:
			lemma.reset_lemma()

	def has_inflections(self, lemma_index):
		"""
		Checks whether agent has any inflections for a particular lemma (indicated by lemma_index)
		:param lemma_index: int: index of particular Lemma object in self.vocabulary
		:return: Boolean: True if agent has any inflections for this particular lemma, False if not
		"""
		return self.vocabulary[lemma_index].has_any_inflection()

	def update_lemma(self, lemma_index, infl_index, outcome, timestep):
		"""
		Update the entry for a particular lemma
		:param lemma_index: int: index of the lemma (in the agent's self.vocabulary attribute)
		:param infl_index: int: index of the inflection
		:param outcome: int: 1 if success (i.e., if receiver has lemma-inflection pairing in inventory), 0 if failure
		:param timestep: int: timestep of current interaction
		:return: updates lemma in agent's vocabulary; doesn't return anything
		"""
		self.tokens += 1
		# If lemma-inflection pairing exists, update the weighting according to the outcome of the interaction:
		if self.vocabulary[lemma_index].has_inflection(infl_index):
			self.vocabulary[lemma_index].update_inflection(infl_index, outcome, timestep)
		# If lemma-inflection pairing doesn't exist yet, create it:
		else:
			self.vocabulary[lemma_index].add_inflection(infl_index, outcome, timestep)
		# Purge the inflections of the lemma (i.e., remove inflections that haven't been used for d_memory timesteps)
		self.vocabulary[lemma_index].purge(timestep)
		# Finally, set agent's self.type_generalise attribute depending on how many tokens it has encountered in total
		if self.tokens > self.k_threshold:
			self.type_generalise = True
		else:
			self.type_generalise = False

	def get_best(self, lemma_index):
		"""
		Get best (i.e., heighest-weighted) inflection for this lemma
		:param lemma_index: int: index of the lemma (in the agent's self.vocabulary attribute)
		:return: int: index of best (i.e., heighest-weighted) inflection for this lemma
		"""
		return self.vocabulary[lemma_index].get_best()

	def get_token_best(self):
		"""
		Token-generalise: Look across vocab and extend rule that was used most frequently across all tokens of any type
		:return: int: index of inflection used across most *tokens*
		"""
		max_tokens = np.zeros(n_inflections)  # TODO: Figure out why C++ code of Cuskley et al. (2018) uses 10 instead of 12 here
		for lemma_index in range(len(self.vocabulary)):
			for i in range(n_inflections):
				max_tokens[i] += self.vocabulary[lemma_index].inflections[i].successes
		max_successes = np.amax(max_tokens)
		max_token_indices = np.where(max_tokens == max_successes)[0]
		max_index = np.random.choice(max_token_indices)
		return max_index

	def get_type_best(self):
		"""
		Type-generalise: Look across vocab and extend the rule which applies to the most types in agent's vocab
		:return: int: index of inflection used across most *types*
		"""
		print('')
		print('')
		print('WOOOOHOOOO!! This is the get_type_best() method of the Agent class:')
		max_types = np.zeros(n_inflections)  # TODO: Figure out why C++ code of Cuskley et al. (2018) uses 10 instead of 12 here
		for lemma_index in range(len(self.vocabulary)):
			best_inflection = self.vocabulary[lemma_index].get_best()
			max_types[best_inflection] += 1
		print("max_types are:")
		print(max_types)
		max_values = np.amax(max_types)
		print("max_values is:")
		print(max_values)
		max_token_indices = np.where(max_types == max_values)[0]
		print("max_token_indices is:")
		print(max_token_indices)
		max_index = np.random.choice(max_token_indices)
		print("max_index is:")
		print(max_index)
		return max_index

	def generate_inflection(self):
		"""
		If a lemma has no inflections, generate an inflection based on generalisation processes
		:return: int: index of newly generated (/generalised) inflection
		"""
		inflection_utterance = np.nan
		# If self.type_generalise is True (= when agent has exceeded k_threshold), find inflection used for most types
		if self.type_generalise:
			print('')
			print('')
			print("This is the generate_inflection() method of the Agent class")
			print("self.type_generalise is True!")
			inflection_utterance = self.get_type_best()
			print("inflection_utterance after type-generalising is:")
			print(inflection_utterance)
			# If preferred generalisation process doesn't provide inflection, try other method (token-generalise)
			if np.isnan(inflection_utterance):
				print("apparently inflection after type-generalising is NAN")
				inflection_utterance = self.get_token_best()
				print("inflection_utterance after token-generalising is:")
				print(inflection_utterance)
		# If self.type_generalise is False (=agent hasn't reached k_threshold yet) find inflection used for most tokens
		else:
			inflection_utterance = self.get_token_best()
			# If preferred generalisation process doesn't provide inflection, try other method (type-generalise)
			if np.isnan(inflection_utterance):
				inflection_utterance = self.get_type_best()
		# If agent has no inflections in vocabulary, they will choose a random inflection from the predefined set
		if np.isnan(inflection_utterance):
			inflection_utterance = np.random.choice(np.arange(n_inflections))
		return inflection_utterance

	def receive(self, lemma_index, infl_index, timestep):
		"""
		Take inflection in as receiver and update lemmas in vocabulary accordingly
		:param lemma_index: int: index of the lemma (in the agent's self.vocabulary attribute)
		:param infl_index: int: index of the inflection
		:param timestep: int: timestep of current interaction
		:return: Boolean: 1 if interaction is success (= lemma-inflection pairing in receiver's vocab), 0 otherwise
		"""
		# If agent has any inflections for this lemma:
		if self.has_inflections(lemma_index):
			# If the agent has this particular inflection for this lemma --- no matter the weight --- return success
			if self.vocabulary[lemma_index].has_inflection(infl_index):
				self.update_lemma(lemma_index, infl_index, 1, timestep)
				return 1
			else:
				self.update_lemma(lemma_index, infl_index, 0, timestep)
				return 0
		# If agent doesn't have any inflections for this lemma, generate inflection based on generalisation processes
		else:
			guess = self.generate_inflection()  # TODO: Check whether this makes sense
			# If the newly generated inflection matches the inflection in question, return success:
			if guess == infl_index:
				self.update_lemma(lemma_index, infl_index, 1, timestep)
				return 1
			else:
				self.update_lemma(lemma_index, infl_index, 0, timestep)
				return 0



class Simulation:
	"""
	Simulation class
	"""
	def __init__(self, pop_size):
		"""
		Initialises simulation object with self.pop_size, self.population and self.running_popsize
		:param pop_size: int: population size
		"""
		self.pop_size = pop_size
		self.population = [Agent() for x in range(3000)]  # Create initial pop, plus "dormant" agents to allow for growth
		self.running_popsize = pop_size  # int: keeps track of the changing population size in the growth condition
		self.n_interactions = pop_size  # int: no. of interactions per timestep. Cuskley et al. used same as pop size
		self.vocabulary, self.log_freqs_per_lemma = generate_vocab(n_lemmas, zipf_exponent, n_tokens)  # generate vocab
		self.all_tokens = 0  # Keeps track of total number of tokens that have come up in interactions across timesteps
		self.global_inflections = np.zeros(n_inflections)  # Keeps track of frequency of each inflection throughout the simulation
		self.global_counts = np.zeros(n_lemmas)  # Keeps track of the frequency of each lemma throughout the simulation
		self.pop_size_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.r_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.tstep_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.lemma_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.log_freq_column = np.zeros(n_runs * t_timesteps * n_lemmas)
		self.infl_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.vocab_entropy_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.meaning_entropy_column = np.zeros(n_runs*t_timesteps*n_lemmas)

	def interaction(self, producer, receiver, lemma, current_timestep):
		"""
		Run single interaction between producer and receiver
		:param producer: int: index of producer agent in self.population
		:param receiver: int: index of receiver agent in self.population
		:param lemma: int: index of lemma in agent.vocabulary
		:param current_timestep: int: current timestep
		:return: updates the lemma in the producer's and receiver's vocabulary based on how the interaction goes,
		and self.global_inflections, which keeps track of frequency of each inflection throughout the simulation;
		doesn't return anything
		"""
		if self.population[producer].has_inflections(lemma):
			utterance = self.population[producer].get_best(lemma)
			result = self.population[receiver].receive(lemma, utterance, current_timestep)
		else:
			utterance = self.population[producer].generate_inflection()
			result = self.population[receiver].receive(lemma, utterance, current_timestep)
		self.population[producer].update_lemma(lemma, utterance, result, current_timestep)
		self.global_inflections[utterance] += 1

	def replace_agent(self):
		"""
		Replace an agent in turnover condition. Randomly selects an agent from the population and resets its attributes
		(equivalent to removing the selected agent and adding a new agent)
		:return: updates self.population by resetting the attributes of the selected agent; doesn't return anything
		"""
		# Generate random float from uniform dist. [0.0, 1.0); if float <= r_replacement probability: reset random agent
		if np.random.random() <= r_replacement:
			chosen_one_index = np.random.choice(np.arange(self.running_popsize - 1))
			self.population[chosen_one_index].reset_agent()

	def add_agent(self):
		"""
		Add agent to population in growth condition by setting one of the "dormant" agents' .is_active attribute to True
		:return: updates self.population by adding a new agent (by setting .is_active to True); doesn't return anything
		"""
		if np.random.random() <= g_growth:
			self.running_popsize += 1
			# Take next of "dormant" agents in line and turn its .is_active attribute to True:
			self.population[self.running_popsize-1].is_active = True

	def timestep(self, current_timestep):
		"""
		Runs through 1 timestep in simulation. Each timestep consists of self.n_interactions interactions.
		Cuskley et al. (2018) used n_interactions = pop_size
		:param current_timestep: int: current timestep
		:return: Updates attributes of population and its agents based on the interactions they go through,
		and whether the replacement and growth conditions are turned on or off (see global variables)
		"""
		vocab_index = 0
		for i in range(self.n_interactions):
			# Randomly select producer and receiver agent:
			producer_index = np.random.choice(np.arange(self.running_popsize-1))
			receiver_index = np.random.choice(np.arange(self.running_popsize-1))
			# Make sure producer and receiver are not the same agent:
			while producer_index == receiver_index:
				receiver_index = np.random.choice(np.arange(self.running_popsize-1))
			# If we've reached the end of the vocabulary array, re-shuffle it:
			if vocab_index >= (n_tokens-1):  # TODO: Why not compare to length of self.vocabulary directly here, rather than n_tokens?
				np.random.shuffle(self.vocabulary)
				vocab_index = 0
			topic = self.vocabulary[vocab_index]
			self.interaction(producer_index, receiver_index, topic, current_timestep)
			if growth:  # growth is global variable (Boolean)
				self.add_agent()
			if replacement:  # growth is global variable (Boolean)
				self.replace_agent()
			self.global_counts[topic] += 1
			self.all_tokens += 1
			vocab_index += 1

	def inflections_in_vocab(self):
		"""
		Counts total number of inflections present in population
		:return: int: total number of inflections present in population
		"""
		infl_counts = np.zeros(n_inflections)
		total_inflections = 0.
		# First, create array which counts for each inflection how many agents in population have that inflection
		for l in range(n_lemmas):
			for a in range(self.running_popsize):
				if self.population[a].has_inflections(l):
					best_infl = self.population[a].get_best(l)
					infl_counts[best_infl] += 1
		# Then, get the total number of inflections which has a count >0 (i.e. that is used by at least 1 agent in pop)
		for i in range(n_inflections):
			if infl_counts[i] > 0:
				total_inflections += 1
		return total_inflections

	def get_entropy(self, probability_array):
		"""
		Calculates the entropy from a list of probablities/frequencies
		:param probability_array: 1D numpy array containing probabilities (i.e., should sum to 1.0)
		:return: float: entropy
		"""
		entropy = 0.
		for p in probability_array:
			if p > 0.:
				entropy += p * np.log2(1./p)
		return entropy

	def vocabulary_entropy(self):
		"""
		Calculates entropy of inflection across the vocabulary, H_v
		:return: float: H_v
		"""
		# how predictable is the inflection of any given lemma?
		# for each lemma
		inflection_probs = np.zeros(n_inflections)
		denominator = 0.
		for l in range(n_lemmas):
			for a in range(self.running_popsize):
				if self.population[a].vocabulary[l].has_any_inflection():
					denominator += 1
					best_infl = self.population[a].get_best(l)
					inflection_probs[best_infl] += 1
		inflection_probs = np.divide(inflection_probs, denominator)
		return self.get_entropy(inflection_probs)

	def meaning_entropy(self, lemma):
		"""
		Calculates entropy of the inflection for a specific lemma, H_l
		:param lemma: int: index of lemma that should be conditioned on
		:return: float: H_l
		"""
		# what is the probability of each inflection given this lemma?
		inflections = np.zeros(n_inflections)
		lemma_count = 0.
		for a in range(self.running_popsize):
			if self.population[a].has_inflections(lemma):
				best_infl = self.population[a].get_best(lemma)
				inflections[best_infl] += 1.
				lemma_count += 1.
		inflection_probs = np.divide(inflections, lemma_count)
		return self.get_entropy(inflection_probs)

	def single_run(self, run_number, counter):
		"""
		Runs a single simulation. Each run is t_timesteps long (Cuskley et al., 2018 used 10,000)
		:param run_number: int: index of current run
		:return: Updates the Simulation object's attributes (specifically the results arrays); doesn't return anything
		"""
		for t in range(t_timesteps):
			if t % 500 == 0:  # after every 50 timesteps, print the current timestep, so we know where we're at:
				print("t: "+str(t))
			self.timestep(t)
			total_inflections = self.inflections_in_vocab()
			if t == t_timesteps-1:
				vocab_entropy = self.vocabulary_entropy()
			for lemma_index in range(n_lemmas):
				self.pop_size_column[counter] = self.pop_size
				self.r_column[counter] = run_number
				self.tstep_column[counter] = t
				self.lemma_column[counter] = lemma_index
				self.log_freq_column[counter] = self.log_freqs_per_lemma[lemma_index]
				self.infl_column[counter] = total_inflections
				if t == t_timesteps-1:
					self.vocab_entropy_column[counter] = vocab_entropy
					self.meaning_entropy_column[counter] = self.meaning_entropy(lemma_index)
				else:
					self.vocab_entropy_column[counter] = np.nan
					self.meaning_entropy_column[counter] = np.nan
				counter += 1
		print("self.running_popsize at end of simulation:")
		print(self.running_popsize)
		return counter

	def multi_runs(self):
		"""
		Runs multiple runs of the simulation
		:return: pandas dataframe containing all results
		"""
		for i in range(self.pop_size):
			self.population[i].is_active = True
		counter = 0
		for r in range(n_runs):
			print('')
			print("r: "+str(r))
			# First, reset self.all_tokens, self.global_inflections, and self.global_counts before starting a new run:
			self.all_tokens = 0
			self.global_inflections = np.zeros(n_inflections)
			self.global_counts = np.zeros(n_lemmas)
			# Then, run a new run:
			counter = self.single_run(r, counter)
		# After all runs have finished, turn the numpy arrays with results into a pandas dataframe:
		results_dict = {"pop_size": self.pop_size_column,
						"run": self.r_column,
						"timestep": self.tstep_column,
						"lemma": self.lemma_column,
						"log_freq": self.log_freq_column,
						"n_inflections": self.infl_column,
						"vocab_entropy": self.vocab_entropy_column,
						"meaning_entropy": self.meaning_entropy_column}
		results_dataframe = pd.DataFrame(results_dict)
		return results_dataframe


def run_multi_sizes(pop_sizes):
	"""
	Runs simulations for each pop_size in pop_sizes
	:param pop_sizes: list of ints specifying the different population sizes that should be run
	:return: pandas dataframe containing simulation results for all pop_sizes in pop_sizes
	"""
	start_time = time.time()
	frames = []
	# First run simulations for each of the pop_sizes:
	for pop_size in pop_sizes:
		print('')
		print("pop_size is:")
		print(pop_size)
		simulation = Simulation(pop_size)
		results_dataframe = simulation.multi_runs()
		frames.append(results_dataframe)
		print("Simulation(s) took %s minutes to run" % round(((time.time() - start_time) / 60.), 2))
	# Then combine the results for each of the pop_sizes into one big dataframe, so that they can be plotted together:
	combined_dataframe = pd.concat(frames, ignore_index=True)
	combined_dataframe.to_pickle("./results_"+"n_runs_"+str(n_runs)+"_tsteps_" + str(t_timesteps) +"_replacement_"+str(replacement)+"_growth_"+str(growth)+"_n_lem_"+str(n_lemmas)+"_n_infl_"+str(n_inflections)+"_n_tok_"+str(n_tokens)+".pkl")
	return combined_dataframe


def plot_vocab_entropy(results_df):
	with sns.color_palette("deep", 2):
		sns.displot(data=results_df, x="vocab_entropy", hue="pop_size", kind="kde", fill=True)
	plt.savefig("./Hv_plot_"+"n_runs_"+str(n_runs)+"_tsteps_" + str(t_timesteps) +"_replacement_"+str(replacement)+"_growth_"+str(growth)+"_n_lem_"+str(n_lemmas)+"_n_infl_"+str(n_inflections)+"_n_tok_"+str(n_tokens)+".pdf")
	plt.show()


def plot_meaning_entropy_by_freq(results_df):
	with sns.color_palette("deep", 2):
		sns.lineplot(data=results_df, x="log_freq", y="meaning_entropy", hue="pop_size")
	plt.savefig("./Hl_plot_"+"n_runs_"+str(n_runs)+"_tsteps_" + str(t_timesteps) +"_replacement_"+str(replacement)+"_growth_"+str(growth)+"_n_lem_"+str(n_lemmas)+"_n_infl_"+str(n_inflections)+"_n_tok_"+str(n_tokens)+".pdf")
	plt.show()


def plot_active_inflections_over_time(results_df):
	with sns.color_palette("deep", 2):
		sns.lineplot(data=results_df, x="timestep", y="n_inflections", hue="pop_size")
	plt.savefig("./Inflections_plot_"+"n_runs_"+str(n_runs)+"_tsteps_" + str(t_timesteps) +"_replacement_"+str(replacement)+"_growth_"+str(growth)+"_n_lem_"+str(n_lemmas)+"_n_infl_"+str(n_inflections)+"_n_tok_"+str(n_tokens)+".pdf")
	plt.show()


combined_dataframe = run_multi_sizes(pop_sizes)
print('')
print("combined_dataframes is:")
print(combined_dataframe)

final_timestep_results = combined_dataframe[combined_dataframe["timestep"]==t_timesteps-1]
print("final_timestep_results are:")
print(final_timestep_results)


plot_vocab_entropy(final_timestep_results)

plot_meaning_entropy_by_freq(final_timestep_results)

plot_active_inflections_over_time(combined_dataframe)


