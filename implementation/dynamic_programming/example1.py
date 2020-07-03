# https://towardsdatascience.com/reinforcement-learning-dynamic-programming-2b89da6ea1b

import numpy as np

class Grid:  # Environment
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        """
        :param rewards: (dict) (row, col) reward
        :param actions: (dict) (row, col) list of actions
        """
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        """
        :param s: (tuple): (int, int)
        """
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def move(self, action):
        if action in self.actions[self.i, self.j]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        return self.rewards.get((self.i, self.j), 0)

    def all_states(self):
        """
        Gives us a set of all possible states in grid world
        """
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def grid():
    grd = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {(0, 0): ('D', 'R'),
               (0, 1): ('L', 'R'),
               (0, 2): ('L', 'D', 'R'),
               (1, 0): ('U', 'D'),
               (1, 2): ('U', 'D', 'R'),
               (2, 0): ('U', 'R'),
               (2, 1): ('L', 'R'),
               (2, 2): ('L', 'R', 'U'),
               (2, 3): ('L', 'U')}
    grd.set(rewards, actions)
    return grd


thresh = 1e-4
GAMMA = 0.9
ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    grid = grid()

    policy = {}

    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTIONS)

    # initialize V(s) randomly between 0 and 1
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state so we set Value to 0
            V[s] = 0

    # Repeat until converged
    # V[s] = max[a]{sum[s', r] {p(s',r|s,a)[r+GAMMA * V[s']}}
    while True:
        max_change = 0
        for s in states:
            old_vs = V[s]

            # V[s] only has policy if not a terminal state
            if s in policy:
                new_v = float('-inf')

                # find max[a]
                for a in ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                biggest_change = max(max_change, np.abs(old_vs - V[s]))
                max_change = biggest_change

        # When the value function converges break out of the loop
        if max_change < thresh:
            break

    # Find a policy that leads to optimal value function
    for s in policy.keys():
        best_act = None
        best_value = float('-inf')
        for a in ACTIONS:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]
            if v > best_value:
                best_value = v
                best_act = a
        policy[s] = best_act
    print(policy)

