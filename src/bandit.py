# --- File: bandit.py ---

import numpy as np
from sklearn.linear_model import Ridge

class LinUCB:
    def __init__(self, num_actions, d, alpha=1.0):
        """
        LinUCB contextual bandit algorithm.

        Args:
            num_actions (int): The number of arms/actions (our aggregation rules).
            d (int): The dimension of the context/state vector.
            alpha (float): The exploration parameter.
        """
        self.num_actions = num_actions
        self.d = d
        self.alpha = alpha
        
        # We use Ridge regression for a more stable inverse calculation.
        # One model per action.
        self.models = [Ridge(alpha=1.0, fit_intercept=False) for _ in range(num_actions)]
        
        # Store historical data for retraining
        self.history = [[] for _ in range(num_actions)]

    def choose_action(self, context):
        """
        Chooses an action based on the current context using the UCB formula.

        Args:
            context (np.array): The state vector S_t.

        Returns:
            int: The index of the chosen action.
        """
        context = context.reshape(1, -1) # Reshape for sklearn
        
        ucb_scores = []
        for i in range(self.num_actions):
            # Fit the model if there is data, otherwise predict zeros
            if len(self.history[i]) > 0:
                X = np.array([h[0] for h in self.history[i]])
                y = np.array([h[1] for h in self.history[i]])
                self.models[i].fit(X, y)
                
                # Predict expected reward
                mu = self.models[i].predict(context)[0]
            else:
                mu = 0.0
            
            # Simple UCB calculation (a more complex version would use matrix inverses)
            # For simplicity, we use a count-based exploration bonus.
            exploration_bonus = self.alpha * np.sqrt(1 / (len(self.history[i]) + 1))
            ucb_scores.append(mu + exploration_bonus)
            
        # Choose the action with the highest UCB score
        return np.argmax(ucb_scores)

    def update(self, action, context, reward):
        """
        Updates the model for the chosen action with the observed reward.

        Args:
            action (int): The action that was taken.
            context (np.array): The context in which the action was taken.
            reward (float): The observed reward.
        """
        self.history[action].append((context, reward))