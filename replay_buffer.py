def combine_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)
    

class ReplayBuffer:
    def __init__(self, 
                obs_dim,
                act_dim,
                size):
        
        self.obs_buf = np.zeros(combine_shape(size, obs_dim),dtype=np.float32)
        self.obs2_buf = np.zeros(combine_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        # initialise max size too
        self.max_size = size
        # initialise a pointer to keep count
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        
    def store(self,
            obs,
            obs2,
            act,
            rew,
            done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        # update tracking metrics
        self.ptr = (self.ptr+1)%self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch_dict = {
            'obs':self.obs_buf[idxs],
            'obs2':self.obs2_buf[idxs],
            'act':self.act_buf[idxs],
            'rew':self.rew_buf[idxs],
            'done':self.done_buf[idxs]
        }
        # return as tensor dictionary for ingestion into network
        return {
            k:torch.tensor(v, dtype=torch.long, device=self.device) if k == 'act' else 
            torch.tensor(v, dtype=torch.float32, device=self.device) for k,v in batch_dict.items()
        }        
        ### MAY HAVE TO CHECK DEVICE AT SOME POINT 
        
    def __len__(self):
        return self.size