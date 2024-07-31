import torch
from torch import nn
import torch.nn.functional as F
   
# Actor Network
class TD3_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, action_low=None, action_high=None):
        super(TD3_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Ensure action_low and action_high are specified
        if action_low is None or action_high is None:
            raise ValueError("action_low and action_high must be specified")
        
        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(self.device)

        self.max_action = (self.action_high - self.action_low) / 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # # Each action needs to be scaled to different ranges

        # # Scale the output to the desired action ranges
        # action_range = (self.action_high - self.action_low) / 2
        # action_mid = (self.action_high + self.action_low) / 2

        # # Ensure action_mid and action_range are on the same device as x
        # action_mid = action_mid.to(x.device)
        # action_range = action_range.to(x.device)

        # # Scale the output to the desired action ranges
        # return action_mid + x * action_range
        return self.max_action * torch.tanh(self.fc3(x))

# Critic Network (Twin Critic Networks)
class TD3_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, hidden_dim_1=400, hidden_dim_2=300):
        super(TD3_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)

        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc5 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc6 = nn.Linear(hidden_dim_2, 1)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, x, u):
        xu = torch.cat([x, u], 1).to(self.device)
        q1 = F.relu(self.fc1(xu))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(xu))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1).to(self.device)
        q1 = F.relu(self.fc1(xu))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
    
if __name__ == '__main__':
    state_dim = 12                                          # define number of input variables (the state)
    action_dim = 2                                          # define the number of possible outputs (the action)
    action_high = [1]
    action_low = [-1]                                      # define the maximum action value (continuous action space)
    net = TD3_Actor(state_dim, action_dim, action_high, action_low)      # define the network with the state dimensions (12) and action dimensions (2)
    state = torch.randn(1, state_dim)                       # create some random input
    output = net(state)                                     # send some random input into the network
    print(output)                                           # print the output