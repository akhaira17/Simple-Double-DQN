class EpsilonStrategy:
    def __init__(self,
                 init_epsilon,
                 end_epsilon,
                 decay_rate=10000):
        self.init_epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.decay_rate = decay_rate
        # epsilon will start at initial epsilon 
        self.epsilon = init_epsilon 
        self.t = 0
        
    def decay_epsilon(self):
        self.epsilon = (self.init_epsilon - self.end_epsilon) * (np.exp(-self.t/self.decay_rate)) + self.end_epsilon
        self.t+=1
        return self.epsilon
    
    def select_action(self, q_values):
        # first check if q_values are tensors. If they are tensor then change to numpy
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().detach().numpy().squeeze() # detach and cpu can be interchangeable 
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.choice(len(q_values))
        self.decay_epsilon()
        return action 
        
        