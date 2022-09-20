import tomsup as ts
import random
import numpy as np
import time

start_time = time.time()


# EXERCISE 01:
## Use the command help(ts.PayoffMatrix) (see p. 11 of Waade et al., 2022) to explore what Game Theory games are pre-specified in the tomsup package. Print and investigate each of these pay-off matrices. For each one: Write down whether they are competitive or cooperative in nature. Also explain why.

# ENVISIONED ANSWER TO EXERCISE 01:

help(ts.PayoffMatrix)

print(ts.PayoffMatrix(name='staghunt'))
# Staghunt is cooperative because the cell in the pay-off matrix where the two agents get the highest pay-off is the same for both agents (i.e., when they both choose action 1

print(ts.PayoffMatrix(name='penny_competitive'))
# Penny competitive is competitive in nature because the pay-off matrices for the two agents are each others' mirror image

print(ts.PayoffMatrix(name='penny_cooperative'))
# Penny cooperative is cooperative in nature because the pay-off matrices of the two agents are identical

print(ts.PayoffMatrix(name='party'))
# Party is cooperative because the pay-off matrices of the two agents are identical

print(ts.PayoffMatrix(name='sexes'))
# Sexes is competitive in nature, because the cells in which the agents get the highest pay-off are each other's mirror image

print(ts.PayoffMatrix(name='chicken'))
# Chicken is competitive because the cells in which the agents get the highest pay-off are each other's mirror image. However, there is also a very high cost for both agents if neither of them "yields" (i.e., chooses action 1)

print(ts.PayoffMatrix(name='deadlock'))
# Deadlock is competitive because the cells in which the agents get the highest pay-off are each other's mirror image

print(ts.PayoffMatrix(name='prisoners_dilemma'))
# Prisoner's dilemma is competitive because the cells in which the agents get the highest pay-off are each other's mirror image


# EXERCISE 02:
# penny_competitive is an example of a zero-sum game. The definition of a zero-sum game is as follows:
# "games in which choices by players can neither increase nor decrease the available resources. In zero-sum games, the total benefit that goes to all players in a game, for every combination of strategies, always adds to zero (more informally, a player benefits only at the equal expense of others)"
# Can you find any other example of a zero-sum game among the predefined games in the tomsup package?

# ENVISIONED ANSWER TO EXERCISE 02:
# No, there are no other examples of zero-sum games among the predefined games in the package.


# EXERCISE 03:
# prisoner's dilemma is an example of a game that has a Nash equilibrium that is suboptimal for both agents. That is, when both agents decide to betray each other (i.e, both choose action 0), they are worse off than if they both remain silent (i,e., both choose action 1). However, if they are in a state where they both choose action 0, neither agent can improve their own pay-off by changing strategy, making this state a Nash equilibrium. Can you find any other games among the predefined games that have such a Nash equilibrium that is suboptimal for both agents? If so, explain why.

# ENVISIONED ANSWER TO EXERCISE 03:
# party is another example: When both agents are in the state of choosing action 0, this is suboptimal (they would both be better off if they each chose action 1). However, if either agent chooses to change strategy to action 1 individually this will decrease their pay-off.



# EXERCISE 04:
# Write a function that takes a results dataframe (as resulting from the ts.compete() function) as input, and returns a string stated which agent has won. Also make sure the function returns something if it's a tie (i.e., if both agents have an equal number of points).

# ENVISIONED ANSWER TO EXERCISE 04:
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


# EXERCISE 05:







print("")
print("--- %s seconds ---" % (time.time() - start_time))