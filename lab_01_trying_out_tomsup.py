import tomsup as ts
import random
import numpy as np
import time
random.seed(1995)
np.random.seed(1995)

start_time = time.time()

penny = ts.PayoffMatrix(name='penny_competitive')

print(penny)

# define the random bias agent, which chooses 1 70 percent of the time, and call the agent "jung":
jung = ts.RB(bias=0.7)

# Examine agent:
print(f"jung is a class of type: {type(jung)}")
if isinstance(jung, ts.Agent):
    print(f"but jung is also an instance of the parent class ts.Agent")

# let us have Jung make a choice:
choice = jung.compete()

print(f"jung chose {choice} and his probability for choosing 1 was {jung.get_bias()}.")


# create a reinforcement learning agent:
skinner = ts.create_agents(agents='QL', start_params={'save_history':True})

# have the agents compete for 30 rounds
results = ts.compete(jung, skinner, p_matrix=penny, n_rounds=5)

# examine results
print(results) #inspect the first 5 rows of the dataframe


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


##################################################################
# Using k-ToM agents:




print("")
print("--- %s seconds ---" % (time.time() - start_time))