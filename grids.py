"""
In a grid world of 4x4, left-up corner and right-button corner are terminate states,
each move cost 1 point, in every state, there are four actions: up, down, left, right.
if hit the board of the grid world, stay still. 
"""
import numpy as np
import itertools
import enum

WORD_SIZE = 4

def initValueMatrix():
    return np.zeros((WORD_SIZE, WORD_SIZE))

class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def getReward(i, j):
    if i == j == 0 or i == j == WORD_SIZE - 1:
        return 0
    else:
        return -1

def next(i ,j, action):
    if i == j == 0 or i == j == WORD_SIZE - 1:
        # Start from terminate states.
        return i, j
    if action == Action.UP and i > 0:
        i -= 1
    if action == Action.DOWN and i < WORD_SIZE - 1:
        i += 1
    if action == Action.LEFT and j > 0:
        j -= 1
    if action == Action.RIGHT and j < WORD_SIZE - 1:
        j += 1
    return i, j

def evaluatePolicy():
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
    print(evaluatePolicy())
   

if __name__ == '__main__':
    main()

