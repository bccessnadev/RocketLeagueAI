import os
from stable_baselines3 import PPO

class Agent:
    def __init__(self):
        # Load the trained model
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, 'MyModel.zip')
        self.model = PPO.load(model_path)
        print("Loading Model")

    def act(self, state):
        # Evaluate your model here
        action, _ = self.model.predict(state, deterministic=True)
        return action
