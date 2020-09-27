import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class WindyGridworldEnv(discrete.DiscreteEnv):
    def __init__(self):
        self.shape = (7, 10)
        nS = self.shape[0] * self.shape[1]
        nA = 4
        # Wind locations
        winds = np.zeros(self.shape)
        winds[:, [3,4,5,8]] = 1
        winds[:, [6, 7]] = 2
        self.goal = (3, 7)
        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a:[] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)
        # Calculate initial state distribution
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1
        super().__init__(nS, nA, P, isd)
    
    
    def _calculate_transition_prob(self, current, delta, winds):
        """Determine the outcome for an action. Transition prob is always 1.0.
        Args:
            current: current position
            delta: position change
            winds: wind effect
        Returns:
            [(1.0, new_state, reward, is_done)]
        """

        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.goal
        return [(1.0, new_state, -1, is_done)]
    

    def _limit_coordinates(self, position):
        position[0] = min(position[0], self.shape[0]-1)
        position[0] = max(0, position[0])
        position[1] = min(position[1], self.shape[1]-1)
        position[1] = max(0, position[1])
        return position
    

    def render(self):
        outfile = sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = ' x '
            elif position == self.goal:
                output = ' T '
            else:
                output = ' o '
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1]-1:
                output = output.rstrip()
                output += '\n'
            outfile.write(output)
        outfile.write('\n')