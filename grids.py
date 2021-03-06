"""
In a grid world of 4x4, left-up corner and right-button corner are terminate states,
each move cost 1 point, in every state, there are four actions: up, down, left, right.
if hit the board of the grid world, stay still. 
"""
import numpy as np
import itertools
import enum
import random

WORLD_SIZE = 5

def initValueMatrix():
    return np.zeros((WORLD_SIZE, WORLD_SIZE))

def isTerminateState(i, j):
    if i == j == 0 or i == j == WORLD_SIZE - 1:
        return True
    else:
        return False

class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def initQMatrix():
    return np.zeros((WORLD_SIZE, WORLD_SIZE, len(Action)))

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

def tdlambdaEvaluatePolicy(lambda_=.9, alpha=.1):
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
            # Go to next state.
            i_, j_ = next(i, j, random.choice((Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)))
            # Calculate td delta.
            tdDelta = (getReward(i, j) + values[i_, j_]) - values[i, j]
            # Update `values` matrix.
            values += alpha * tdDelta * eligibility
            i, j = i_, j_
    print(values)
    return values

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

def getGreedyPolicy(values):
    """Given a value funtion of states, return the greedy policy."""
    policy = np.empty_like(values, dtype=object)
    for i, j in itertools.product(range(policy.shape[0]), range(policy.shape[1])):
        bestAction = None
        bestValue = None
        for action in Action:
            value = getReward(i, j) + values[next(i, j, action)]
            if bestValue is None:
                bestValue, bestAction = value, action
            elif value > bestValue:
                bestValue, bestAction = value, action
        policy[i, j] = bestAction
    return policy

def _testGetGreedyPolicy():
    values = np.array([[0, -1, -1, -1],
                       [0, 0, 0, 0],
                       [0, 0, 4, 0],
                       [0, 0, 0, 0]])
    print('values')
    print(values)
    print('policy')
    print(getGreedyPolicy(values))

def valueIteration():
    """Find the optimal values matrix and optimal policy using synchronous value iteration."""
    values = initValueMatrix()
    policy = getGreedyPolicy(values)
    niter = 0
    while True:
        niter += 1
        newValues = values.copy()
        newPolicy = None
        for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
            newValues[i, j] = getReward(i, j) + values[next(i, j, policy[i, j])]
            newPolicy = getGreedyPolicy(newValues)
        if (newPolicy == policy).all():
            break
        else:
            values = newValues
            policy = newPolicy
    print("niter: ", niter)
    print(policy)
    print(values)
    return values, policy

def chooseAction(i, j, qMatrix, epsion=.1):
    """Choose action in state i, j using epsion-greedy policy accroding to qMatrix."""
    if random.random() > epsion:
        # Exploitation.
        return Action(np.argmax(qMatrix[i, j]))
    else:
        # Exploration
        return random.choice(list(Action))

def mcFindOptimalPolicy():
    """Find the optimal policy using M.C. search and epison greedy policy
    Note:
        This function seems not converge to optimal policy when world size is 5,
        nEpisode is 50,000, epsion = 1 / (n + 1), qMatrix using stationary mean.
    """
    qMatrix = initQMatrix()
    nEpisode = 50000
    counter = np.zeros_like(qMatrix, dtype=np.uint32)
    epsion = 1
    for n in range(nEpisode):
        nStep = 0
        scoreHistory = []
        # Initialize state.
        i, j = random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE)
        # This record number of step have taken when a state being visited the first time.
        firstVisit = - np.ones_like(qMatrix, dtype=np.int32)
        while not (i == j == 0 or i == j == WORLD_SIZE - 1):
            # print("current state: ", i, j)
            action = chooseAction(i, j, qMatrix, epsion=epsion/(1+n))
            # print("action:", action)
            # Record current state action pair.
            if firstVisit[i, j, action.value] == -1:
                firstVisit[i, j, action.value] = nStep
            # Go to next state.
            i, j = next(i, j, action)
            # Update loop variable.
            nStep += 1
            scoreHistory.append(-1)

        # Calculate 'scores' matrix in this episode and update `counter` matrix.
        assert len(scoreHistory) == nStep
        # `score` stores the reward of states in one episode.
        scores = np.zeros_like(qMatrix, dtype=np.int32)
        for i, j, k in itertools.product(range(firstVisit.shape[0]), range(firstVisit.shape[1]), range(firstVisit.shape[2])):
            if firstVisit[i, j, k] > -1:
                scores[i, j, k] = sum(scoreHistory[firstVisit[i, j, k]:])
                # `counter` stores number of first visit of a state across episodes.
                counter[i, j, k] += 1

        
        # Update qMatrix.
        updateIndex = firstVisit > -1
        # print("qMatrix:", qMatrix)
        qMatrix[updateIndex] += (scores[updateIndex] - qMatrix[updateIndex]) / counter[updateIndex]
        # qMatrix[updateIndex] += .05 * (scores[updateIndex] - qMatrix[updateIndex])
        # print("first visit:", firstVisit)
        # print("qMatrix:", qMatrix)

    optimalValue = qMatrix.max(axis=2)
    optimalPolicyIndex = qMatrix.argmax(axis=2)
    optimalPolicy = np.vectorize(Action)(optimalPolicyIndex)
    print('counter')
    print(counter.sum(axis=2))
    print('value')
    print(optimalValue)
    print('policy')
    print(optimalPolicy)
    return optimalValue, optimalPolicy

def sarsaFindOpitmalPolicy(alpha=.1):
    qMatrix = initQMatrix()
    nEpsion = 10000
    nStep = 0
    stepsPerEpisode = []
    for n in range(nEpsion):
        # Init state randomly.
        i, j = (random.randrange(WORLD_SIZE) for i in range(2))
        # Choose a action.
        action = chooseAction(i, j, qMatrix, epsion=1/(n+1))
        while not isTerminateState(i, j):
            reward = getReward(i, j)
            nextState = next(i, j, action)
            nextAction = chooseAction(*nextState, qMatrix, epsion=1/(n+1))
            tdTarget = reward + qMatrix[nextState + (nextAction.value,)]
            tdDelta = tdTarget - qMatrix[i, j, action.value]
            # Updatea qMatrix
            qMatrix[i, j, action.value] += alpha * tdDelta
            # Update loop variable.
            i, j = nextState
            action = nextAction
            nStep += 1
        stepsPerEpisode.append(nStep)
    
    optimalValue = qMatrix.max(axis=2)
    optimalPolicyIndex = qMatrix.argmax(axis=2)
    optimalPolicy = np.vectorize(Action)(optimalPolicyIndex)
    print("nStep:", nStep)
    print('value')
    print(optimalValue)
    print('policy')
    print(optimalPolicy)
    print('len:', len(stepsPerEpisode))
    print(np.array(stepsPerEpisode[500: 600]) - np.array(stepsPerEpisode[499: 599]))
    return optimalValue, optimalPolicy

def sarsaLambdaFindOpitmalPolicy(alpha=.1, lambda_=.9):
    qMatrix = initQMatrix()
    nEpsion = 10000
    nStep = 0
    stepsPerEpisode = []
    for n in range(nEpsion):
        # Init state randomly.
        i, j = (random.randrange(WORLD_SIZE) for i in range(2))
        eligibility = np.zeros_like(qMatrix)
        # Choose a action.
        action = chooseAction(i, j, qMatrix, epsion=1/(n+1))
        while not isTerminateState(i, j):
            # Update the `eligibility` matrix.
            eligibility *= lambda_
            eligibility[i, j, action.value] += 1
            reward = getReward(i, j)
            nextState = next(i, j, action)
            nextAction = chooseAction(*nextState, qMatrix, epsion=1/(n+1))
            tdTarget = reward + qMatrix[nextState + (nextAction.value,)]
            tdDelta = tdTarget - qMatrix[i, j, action.value]
            # Updatea qMatrix
            qMatrix += alpha * tdDelta * eligibility
            # Update loop variable.
            i, j = nextState
            action = nextAction
            nStep += 1
        stepsPerEpisode.append(nStep)
    
    optimalValue = qMatrix.max(axis=2)
    optimalPolicyIndex = qMatrix.argmax(axis=2)
    optimalPolicy = np.vectorize(Action)(optimalPolicyIndex)
    print("nStep:", nStep)
    print('value')
    print(optimalValue)
    print('policy')
    print(optimalPolicy)
    print('len:', len(stepsPerEpisode))
    print(np.array(stepsPerEpisode[500: 600]) - np.array(stepsPerEpisode[499: 599]))
    return optimalValue, optimalPolicy

def _testNext():
    inputs = ((0, 0, Action.RIGHT),
              (0, 1, Action.RIGHT),
              (0, 1, Action.DOWN),
              (0, 1, Action.UP),
              (0, 1, Action.LEFT))
    for input_ in inputs:
        print(next(*input_))


def main():
    sarsaLambdaFindOpitmalPolicy()
   

if __name__ == '__main__':
    main()

