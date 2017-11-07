"""
In a grid world of 4x4, left-up corner and right-button corner are terminate states,
each move cost 1 point, in every state, there are four actions: up, down, left, right.
if hit the board of the grid world, stay still. 
"""
import numpy as np
import itertools
import enum
import random

WORLD_SIZE = 4

def initValueMatrix():
    return np.zeros((WORLD_SIZE, WORLD_SIZE))

class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def getReward(i, j):
    if i == j == 0 or i == j == WORLD_SIZE - 1:
        return 0
    else:
        return -1

def next(i ,j, action):
    if i == j == 0 or i == j == WORLD_SIZE - 1:
        # Start from terminate states.
        return i, j
    if action == Action.UP and i > 0:
        i -= 1
    elif action == Action.DOWN and i < WORLD_SIZE - 1:
        i += 1
    elif action == Action.LEFT and j > 0:
        j -= 1
    elif action == Action.RIGHT and j < WORLD_SIZE - 1:
        j += 1
    return i, j

def mcEvaluatePolicy():
    """Evaluate a policy using first-visit M.C. method."""
    values = initValueMatrix()
    nEpisode = 10000
    counter = np.zeros_like(values, dtype=np.uint32)
    for n in range(nEpisode):
        nStep = 0
        scoreHistory = []
        # Initialize state.
        i, j = random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE)
        # This record number of step have taken when a state being visited the first time.
        firstVisit = - np.ones_like(values, dtype=np.int32)
        while not (i == j == 0 or i == j == WORLD_SIZE - 1):
            # Record current state.
            if firstVisit[i, j] == -1:
                firstVisit[i, j] = nStep
            # Go to next state.
            i, j = next(i, j, random.choice((Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)))
            # Update loop variable.
            nStep += 1
            scoreHistory.append(-1)

        # Calculate 'scores' matrix in this episode and update `counter` matrix.
        assert len(scoreHistory) == nStep
        # `score` stores the reward of states in one episode.
        scores = np.zeros_like(values, dtype=np.int32)
        for i, j in itertools.product(range(firstVisit.shape[0]), range(firstVisit.shape[1])):
            if firstVisit[i, j] > -1:
                scores[i, j] = sum(scoreHistory[firstVisit[i, j]:])
                # `counter` stores number of first visit of a state across episodes.
                counter[i, j] += 1

        # Update value matrix.
        updateIndex = firstVisit > -1
        values[updateIndex] += (scores[updateIndex] - values[updateIndex]) / counter[updateIndex]
    print(values)
    return values


def tdEvaluatePolicy(lambda_=0, alpha=.1):
    """Evaluate a policy using TD(lambda)"""
    values = initValueMatrix()
    nStep = 500000
    for n in range(nStep):
        # Start from a random state.
        rewards = []
        i, j = random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE)
        i_, j_ = i, j
        # Move `lambda_` + 1 step.
        for k in range(lambda_ + 1):
            # Collect reward.
            rewards.append(getReward(i_, j_))
            # Move one step.
            i_, j_ = next(i_, j_, random.choice((Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)))
        tdTarget = values[i_, j_] + sum(rewards)
        # Update values matrix in state (i, j).
        values[i, j] += alpha * (tdTarget - values[i, j])
    print(values)
    return values

def tdlambdaEvaluatePolicy(lambda_):
    """Evaluate a policy using TD(lambda)"""
    values = initValueMatrix()
    nEpisode = 10000
    for n in range(nEpisode):
        # Initialize state.
        i, j = random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE)
        eligibility = np.zeros_like(values, dtype=np.float64)
        while not (i == j == 0 or i == j == WORLD_SIZE - 1):
            # Update eligibility matrix.
            eligibility *= lambda_
            eligibility[i, j] += 1
            

        

def evaluatePolicy():
    """Evaluate a policy using synchronous DP"""
    values = initValueMatrix()
    niter = 0
    while(True):
        niter += 1
        newValues = values.copy()
        for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
            newValues[i, j] = getReward(i, j) + np.array([
                values[i, j] for (i, j) in [
                    next(i, j, Action.UP),
                    next(i, j, Action.DOWN),
                    next(i, j, Action.LEFT),
                    next(i, j, Action.RIGHT)
                    ]
            ]).dot(
                np.array([.25, .25, .25, .25])  # The probability to choose each action in state i, j.
            )
        if (newValues == values).all():
            break
        else:
            values = newValues
    print(niter, values)
    return values

def _testNext():
    inputs = ((0, 0, Action.RIGHT),
              (0, 1, Action.RIGHT),
              (0, 1, Action.DOWN),
              (0, 1, Action.UP),
              (0, 1, Action.LEFT))
    for input_ in inputs:
        print(next(*input_))


def main():
    tdEvaluatePolicy()
   

if __name__ == '__main__':
    main()

