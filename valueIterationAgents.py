# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
          ThisValues = self.values.copy() #WTF WHY THIS TOOK HOURS
          for AgentState in self.mdp.getStates():
            values = [float("-inf")]
            if not self.mdp.isTerminal(AgentState):
              for PossibleAction in self.mdp.getPossibleActions(AgentState):
                values += [self.computeQValueFromValues(AgentState,PossibleAction)]
              ThisValues[AgentState] = max(values)
          self.values = ThisValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        action_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        Sum_value = 0
        for new_state, probability in action_and_probs:
            next_reward = self.mdp.getReward(state, action, new_state)
            Sum_value += probability * (next_reward + self.discount * self.values[new_state])
        return Sum_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        optimized_action = None
        maxValues = float("-inf")
        for PossibleAction in self.mdp.getPossibleActions(state):
          value_q = self.computeQValueFromValues(state, PossibleAction)
          if value_q > maxValues:
            maxValues = value_q
            optimized_action = PossibleAction
        return optimized_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        AllStates = self.mdp.getStates()
        States_count = len(AllStates)
        for i in range(self.iterations):
          This_state = AllStates[i % States_count]
          if not self.mdp.isTerminal(This_state):
            list_values = []
            for PossibleAction in self.mdp.getPossibleActions(This_state):
              value_q = self.computeQValueFromValues(This_state, PossibleAction)
              list_values.append(value_q)
            self.values[This_state] = max(list_values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        priorityQueue = util.PriorityQueue()
        predecessors = {}
        for Gamestate in self.mdp.getStates():
          if not self.mdp.isTerminal(Gamestate):
            for PossibleAction in self.mdp.getPossibleActions(Gamestate):
              for newState, probability in self.mdp.getTransitionStatesAndProbs(Gamestate, PossibleAction):
                if newState in predecessors:
                  predecessors[newState].add(Gamestate)
                else:
                  predecessors[newState] = {Gamestate}

        for Gamestate in self.mdp.getStates():
          if not self.mdp.isTerminal(Gamestate):
            list_values = []
            for PossibleAction in self.mdp.getPossibleActions(Gamestate):
              value_q = self.computeQValueFromValues(Gamestate, PossibleAction)
              list_values.append(value_q)
            difference = abs(max(list_values) - self.values[Gamestate])
            priorityQueue.update(Gamestate, - difference)

        for i in range(self.iterations):
          if priorityQueue.isEmpty():
            break
          state_temp = priorityQueue.pop()
          if not self.mdp.isTerminal(state_temp):
            list_values = []
            for PossibleAction in self.mdp.getPossibleActions(state_temp):
              value_q = self.computeQValueFromValues(state_temp, PossibleAction)
              list_values.append(value_q)
            self.values[state_temp] = max(list_values)

          for predecessor in predecessors[state_temp]:
            if not self.mdp.isTerminal(predecessor):
              list_values = []
              for PossibleAction in self.mdp.getPossibleActions(predecessor):
                value_q = self.computeQValueFromValues(predecessor, PossibleAction)
                list_values.append(value_q)
              difference = abs(max(list_values) - self.values[predecessor])
              if difference > self.theta:
                priorityQueue.update(predecessor, -difference)


