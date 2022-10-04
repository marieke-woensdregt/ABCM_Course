import numpy as np
import pandas as pd
import time
import seaborn as sns


###################### PARAMETER SETTINGS: ######################
n_runs = 2  # int: number of independent simulation runs. Cuskley et al. (2018) used 100
pop_size = 20  # int: initial population size. Cuskley et al. (2018) used 20 for small population and 100 for large pop
n_lemmas = 28  # int: number of lemmas. Cuskley et al. (2018) used 28
n_tokens = 500  # int: number of tokens in vocabulary. Cuskley et al. seem to have used 500 (in C++ implementation)
n_inflections = 12  # int: number of inflections. Cuskley et al. (2018) used 12
zipf_exponent = 2  # int: exponent used to create Zipfian frequency distribution. Cuskley et al., 2018 used 2)
k_proficiency = 1500  # int: token threshold that determines proficiency. Cuskley et al. (2018) used 1500
r_replacement = 0.001  # float: replacement rate for turnover condition. Cuskley et al. (2018) used 0.001.
# At every interaction, there is an r chance that a randomly selected learner will be replaced by a new learner
g_growth = 0.001  # float: growth rate for growth condition. Cuskley et al. (2018) used 0.001.
# At every interaction, there's a g chance that a new learner will be *added* to the population
replacement = True  # Boolean: determines whether this simulation includes replacement (turnover)
growth = False  # Boolean; determines whether this simulation includes growth
t_timesteps = 3000  # int: number of timesteps to run per simulation. Cuskley et al. (2018) used 10,000
n_interactions = pop_size  # int: number of interactions per timestep. Cuskley et al. used same as population size
d_memory = 100  # int: no. of timesteps after which agent forgets lemma-inflection pairing. Cuskley et al. used 100


def generate_vocab(n_lemmas, zipf_exponent, n_tokens):  # TODO: Turn this into method of Simulation class?
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
	return vocabulary


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


# my_inflections_list = [Inflection() for i in range(n_inflections)]
# lemmas_list = []
# for i in range(n_lemmas):
#   lemma = Lemma(0, 0, False, my_inflections_list)
#   lemmas_list.append(lemma)
#
#
# for i in range(300):
#   lemma_index = np.random.choice(vocabulary)
#   lemma = lemmas_list[lemma_index]
#   inflection_frequencies = np.array([0.01, 0.3, 0.2, 3, 0.5, 4, 0.2, 1, 0.02, 2, 0.4, 0.08])
#   # np.random.shuffle(inflection_frequencies)
#   inflection_probs = np.divide(inflection_frequencies.astype(float), np.sum(inflection_frequencies))
#   infl_index = np.random.choice(np.arange(n_inflections), p=inflection_probs)
#   outcome = np.random.randint(2)
#   timestep = i
#   lemma.update_inflection(infl_index, outcome, timestep)
#
#
# print('')
# print('')
# random_lemma = np.random.choice(lemmas_list)
# print("random_lemma.__dict__ is:")
# print(random_lemma.__dict__)
#
# best_infl_index = random_lemma.get_best()
# print('')
# print('')
# print("best_infl_index is:")
# print(best_infl_index)
#
# has_inflection = random_lemma.has_any_inflection()
# print('')
# print('')
# print("has_inflection is:")
# print(has_inflection)
#
# timestep = 300
# random_lemma.purge(timestep)
# print("random_lemma.__dict__ is:")
# print(random_lemma.__dict__)


class Agent:
	"""
	Agent class
	"""
	def __init__(self, tokens=0, k_threshold=k_proficiency, memory_window=d_memory, type_generalise=False,
				 is_active=False):
		"""
		Initialises Agent object
		:param tokens: int: number of tokens. Initial value: 0  # TODO: figure out what this attribute is/does exactly.
		:param k_threshold: int: token threshold that determines proficiency. Default: k_proficiency (=global variable)
		:param memory_window: int: no. of timesteps after which agent forgets pairing. Default: d_memory (=global var.)
		:param type_generalise: Boolean: False = agent is token-generaliser; True = type-generaliser. Initial: False
		:param is_active: Boolean: Initial value: False  # TODO figure out what this attribute is/does exactly. When updated?
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
		print('')
		print('')
		print('This is the get_token_best() method of the Agent class:')
		max_tokens = np.zeros(10)  # TODO: Figure out what the idea behind this max_tokens array is
		for lemma_index in range(len(self.vocabulary)):
			for i in range(10):  # TODO: Where does range(len(10)) come from? Shouldn't this loop over all inflections?
				max_tokens[i] += self.vocabulary[lemma_index].inflections[i].successes
		print("max_tokens are:")
		print(max_tokens)
		max_successes = np.amax(max_tokens)
		print("max_successes is:")
		print(max_successes)
		max_token_indices = np.where(max_tokens == max_successes)[0]
		print("max_token_indices is:")
		print(max_token_indices)
		max_index = np.random.choice(max_token_indices)
		print("max_index is:")
		print(max_index)
		return max_index

	def get_type_best(self):
		"""
		Type-generalise: Look across vocab and extend the rule which applies to the most types in agent's vocab
		:return: int: index of inflection used across most *types*
		"""
		print('')
		print('')
		print('This is the get_type_best() method of the Agent class:')
		max_types = np.zeros(10)  # TODO: Figure out what the idea behind this max_types array is
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
			inflection_utterance = self.get_type_best()
			# If preferred generalisation process doesn't provide inflection, try other method (token-generalise)
			if np.isnan(inflection_utterance):
				inflection_utterance = self.get_token_best()
		# If self.type_generalise is False (=agent hasn't reached k_threshold yet) find inflection used for most tokens
		else:
			inflection_utterance = self.get_token_best()
			# If preferred generalisation process doesn't provide inflection, try other method (type-generalise)
			if np.isnan(inflection_utterance):
				inflection_utterance = self.get_type_best()
		# If agent has no inflections in vocabulary, they will choose a random inflection from the predefined set of 12
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
			guess = self.generate_inflection()
			print('')
			print('')
			print("This is the .receive() method of the Agent class:")
			print("guess is:")
			print(guess)
			# If the newly generated inflection matches the inflection in question, return success:
			if guess == infl_index:
				self.update_lemma(lemma_index, infl_index, 1, timestep)
				return 1
			else:
				self.update_lemma(lemma_index, infl_index, 0, timestep)
				return 0


# my_agent = Agent()
# print('')
# print('')
# print("my_agent.__dict__ is:")
# print(my_agent.__dict__)
#
# my_agent.reset_agent()
# print('')
# print('')
# print("my_agent.__dict__ AFTER RESETTING is:")
# print(my_agent.__dict__)
#
# timestep = 10
# for lemma_index in range(n_lemmas):
# 	print('')
# 	print("lemma_index is:")
# 	print(lemma_index)
# 	lemma = my_agent.vocabulary[lemma_index]
# 	print("lemma.__dict__ is:")
# 	print(lemma.__dict__)
# 	has_inflections = my_agent.has_inflections(lemma_index)
# 	print("has_inflections is:")
# 	print(has_inflections)
# 	for infl_index in range(n_inflections):
# 		my_agent.receive(lemma_index, infl_index, timestep)
# print("my_agent.__dict__ AFTER UPDATING is:")
# print(my_agent.__dict__)


class Simulation:
	"""
	Simulation class
	"""
	def __init__(self):
		"""
		Initialises simulation object with self.population and self.running_popsize
		"""
		self.population = [Agent() for x in range(3000)]  # Create initial pop, plus "dormant" agents to allow for growth
		self.running_popsize = pop_size  # int: keeps track of the changing population size in the growth condition
		self.vocabulary = generate_vocab(n_lemmas, zipf_exponent, n_tokens)  # generate vocabulary (numpy array)
		self.all_tokens = 0  # Keeps track of total number of tokens that have come up in interactions across timesteps
		self.global_inflections = np.zeros(12)  # Keeps track of frequency of each inflection throughout the simulation
		self.global_counts = np.zeros(28)  # Keeps track of the frequency of each lemma throughout the simulation
		self.r_column = np.zeros(n_runs*t_timesteps*n_lemmas)
		self.tstep_column = np.zeros(n_runs*t_timesteps*n_lemmas)
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
			print('')
			print('')
			print("This is the replace_agent() method of the Simulation class:")
			print('YAY! np.random.random() <= r_replacement')
			chosen_one_index = np.random.choice(np.arange(self.running_popsize - 1))
			print("self.population[chosen_one_index].__dict__ BEFORE RESETTING is:")
			print(self.population[chosen_one_index].__dict__)
			self.population[chosen_one_index].reset_agent()
			print("self.population[chosen_one_index].__dict__ AFTER RESETTING is:")
			print(self.population[chosen_one_index].__dict__)

	def add_agent(self):
		"""
		Add agent to population in growth condition by setting one of the "dormant" agents' .is_active attribute to True
		:return: updates self.population by adding a new agent (by setting .is_active to True); doesn't return anything
		"""
		if np.random.random() <= g_growth:
			print('')
			print('')
			print("This is the add_agent() method of the Simulation class:")
			print('YAY! np.random.random() <= g_growth')
			self.running_popsize += 1
			print("self.population[self.running_popsize - 1].__dict__ BEFORE UPDATING")
			print(self.population[self.running_popsize - 1].__dict__)
			self.population[self.running_popsize-1].is_active = True
			print("self.population[self.running_popsize - 1].__dict__ AFTER UPDATING")
			print(self.population[self.running_popsize - 1].__dict__)

	def timestep(self, current_timestep):
		"""
		Runs through 1 timestep in simulation. Each timestep consists of n_interactions interactions.
		Cuskley et al. (2018) used n_interactions = pop_size
		:param current_timestep: int: current timestep
		:return: Updates attributes of population and its agents based on the interactions they go through,
		and whether the replacement and growth conditions are turned on or off (see global variables)
		"""
		vocab_index = 0
		for i in range(n_interactions):
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

	def inflections_in_vocab(self):  # counts inflections for whole population
		pass
		return

	def get_entropy(self, probability_array):
		pass

	def vocabulary_entropy(self):
		pass
		return

	def meaning_entropy(self, lemma):
		pass
		return

	def single_run(self, run_number):
		"""
		Runs a single simulation. Each run is t_timesteps long (Cuskley et al., 2018 used 10,000)
		:param run_number: int: index of current run
		:return: Updates the Simulation object's attributes (specifically the results arrays); doesn't return anything
		"""
		print('This is the single_run method of the Simulation class:')
		for t in range(t_timesteps):
			print("t is:")
			print(t)
			self.timestep(t)
			for lemma_index in range(n_lemmas):
				self.r_column[run_number + t + lemma_index] = run_number
				self.tstep_column[run_number + t + lemma_index] = t
				# self.infl_column[run_number + t + lemma_index] = self.inflections_in_vocab()
				# self.vocab_entropy_column[run_number + t + lemma_index] = self.vocabulary_entropy()
				# self.meaning_entropy_column[run_number + t + lemma_index] = self.meaning_entropy(lemma_index)
		print("self.running_popsize at end of simulation:")
		print(self.running_popsize)

	def multi_runs(self):
		"""
		Runs multiple runs of the simulation
		:return: pandas dataframe containing all results
		"""
		print('')
		print('')
		print('This is the multi_runs method of the Simulation class:')
		for i in range(pop_size):  # pop_size is global variable
			self.population[i].is_active = True
		for r in range(n_runs):
			print("r is:")
			print(r)
			# First, reset self.all_tokens, self.global_inflections, and self.global_counts before starting a new run:
			self.all_tokens = 0
			self.global_inflections = np.zeros(12)
			self.global_counts = np.zeros(28)
			# Then, run a new run:
			self.single_run(r)
		# After all runs have finished, turn the numpy arrays with results into a pandas dataframe:
		results_dict = {"run": self.r_column,
						"timestep": self.tstep_column,
						"n_inflections": self.infl_column,
						"vocab_entropy": self.vocab_entropy_column,
						"meaning_entropy": self.meaning_entropy_column}
		results_dataframe = pd.DataFrame(results_dict)
		return results_dataframe

start_time = time.time()

simulation = Simulation()
# print('')
# print('')
# print("simulation.__dict__ is:")
# print(simulation.__dict__)

results_dataframe = simulation.multi_runs()
print('')
print('')
print("results_dataframe is:")
print(results_dataframe)


print("--- %s seconds ---" % (time.time() - start_time))