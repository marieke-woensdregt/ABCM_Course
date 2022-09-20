import tomsup as ts
import random
import numpy as np
import time
import matplotlib.pyplot as plt
random.seed(1995)
np.random.seed(1995)

start_time = time.time()

# penny = ts.PayoffMatrix(name='penny_competitive')
#
# print(penny)
#
# # define the random bias agent, which chooses 1 70 percent of the time, and call the agent "jung":
# jung = ts.RB(bias=0.7)
#
# # Examine agent:
# print(f"jung is a class of type: {type(jung)}")
# if isinstance(jung, ts.Agent):
#     print(f"but jung is also an instance of the parent class ts.Agent")
#
# # let us have Jung make a choice:
# choice = jung.compete()
#
# print(f"jung chose {choice} and his probability for choosing 1 was {jung.get_bias()}.")
#
#
# # create a reinforcement learning agent:
# skinner = ts.create_agents(agents='QL', start_params={'save_history':True})
#
# # have the agents compete for 30 rounds
# results = ts.compete(jung, skinner, p_matrix=penny, n_rounds=5)
#
# # examine results
# print(results) #inspect the first 5 rows of the dataframe


# simply summing the pay-off column would determine the winner:

def who_won(results_df):
    pay_off_agent_0 = np.sum(results_df["payoff_agent0"])
    print(pay_off_agent_0)
    pay_off_agent_1 = np.sum(results_df["payoff_agent1"])
    print(pay_off_agent_1)
    if pay_off_agent_0 > pay_off_agent_1:
        return("agent0")
    elif pay_off_agent_0 < pay_off_agent_1:
        return("agent1")
    else:
        return("tie")

print(who_won(results))


# ##################################################################
# # Using k-ToM agents:
#
#
# # Creating a simple 1-ToM with default parameters
# tom_1 = ts.TOM(level=1, dilution=None, save_history=True)
#
# # print the parameters
# tom_1.print_parameters()
#
# tom_2 = ts.TOM(level=2, volatility=-2, b_temp=-2, # more deterministic
# bias=0, dilution=None, save_history=True)
# choice = tom_2.compete(p_matrix=penny, agent=0, op_choice=None)
# print("tom_2 chose:", choice)
#
#
# # Now let's have tom_1 and tom_2 compete for several rounds:
#
# results_tom = ts.compete(tom_1, tom_2, p_matrix=penny, n_rounds=4)
# print(results_tom)
#
# tom_2.print_internal(keys=["p_k", "p_op"], # print these two states
#                      level=[0, 1])          # for the agent simulated opponentes
#                                             # 0-ToM and 1-ToM
#
# help(tom_2.print_internal)


# SIMULATING MULTIPLE AGENTS


games = ['staghunt', 'penny_competitive', 'penny_cooperative', 'party', 'sexes', 'chicken', 'deadlock', 'prisoners_dilemma']


for i in range(len(games)):

    game = ts.PayoffMatrix(name=games[i])

    # Create a list of agents
    agents = ['RB', 'QL', 'WSLS', '1-TOM', '2-TOM']
    # And set their starting parameters. An empty dictionary, denoted by '{}' gives default values
    start_params = [{'bias':0.7}, {'learning_rate':0.5}, {}, {}, {}]

    group = ts.create_agents(agents, start_params) # create a group of agents

    # Specify the environment
    # round_robin e.g. each agent will play against all other agents
    group.set_env(env='round_robin')

    # Finally, we make the group compete 20 simulations of 30 rounds
    results = group.compete(p_matrix=penny, n_rounds=30,
                            n_sim=20, save_history=True)

    res = group.get_results()
    print(res.head(1)) # print the first row

    # plot a heatmap of the rewards for all agents in the tournament
    group.plot_heatmap(cmap="RdBu", show=False)
    plt.savefig("heatmap_different_agent_rewards.pdf")

    # plot the choices of the 1-ToM agent when competing against the WSLS agent
    group.plot_choice(agent0="WSLS", agent1="1-TOM", agent=1, show=False)
    # plot the choices of the 1-ToM agent when competing against the WSLS agent
    group.plot_choice(agent0="RB", agent1="1-TOM", agent=1, show=False)

    # plot the score of the 1-ToM agent when competing against the WSLS agent
    group.plot_score(agent0="WSLS", agent1="1-TOM", agent=1, show=False)
    plt.savefig("score_1-TOM_vs_WSLS.pdf")

    # plot the score of the 2-ToM agent when competing against the WSLS agent
    group.plot_score(agent0="WSLS", agent1="2-TOM", agent=1, show=False)
    plt.savefig("score_2-TOM_vs_WSLS.pdf")


    # plot 2-ToM estimate of its opponent sophistication level
    group.plot_p_k(agent0="1-TOM", agent1="2-TOM", agent=1, level=0, show=False)
    plt.savefig("2-TOM_prob_of_opp_k_0.pdf")
    group.plot_p_k(agent0="1-TOM", agent1="2-TOM", agent=1, level=1, show=False)
    plt.savefig("2-TOM_prob_of_opp_k_1.pdf")

    # plot 2-ToM estimate of its opponent's volatility while believing the opponent to be level 1.
    group.plot_tom_op_estimate(agent0="1-TOM", agent1="2-TOM", agent=1,
                               estimate="volatility", level=1,
                               plot="mean", show=False)
    plt.savefig("2-TOM_estimate_of_1-TOM_volatility.pdf")


    # plot 2-ToM estimate of its opponent's bias while believing the opponent to be level 1.
    group.plot_tom_op_estimate(agent0="1-TOM", agent1="2-TOM", agent=1,
                               estimate="bias", level=1, plot="mean", show=False)
    plt.savefig("2-TOM_estimate_of_1-TOM_bias.pdf")



print("")
print("--- %s seconds ---" % (time.time() - start_time))