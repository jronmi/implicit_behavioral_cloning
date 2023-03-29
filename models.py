import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
        general framework for the (explicit) MLP model which was described in Appendix E (pg.26) of Implicit Behavioral Cloning
    """

    def __init__(self, obs_dim, act_dim, hidden_dim, n_hidden):
        """
            params: 
                obs_dim (int): the dimensionality of the observation space (X)
                act_dim (int): the dimensionality of the action space (Y)
                hidden_dim (int): the number of hidden units in each hidden layer
                n_hidden (int): the number of hidden layers
        """

        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.ReLU())
            
        for _ in range(n_hidden-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, act_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        estimate = self.mlp(x)
        return estimate

class ConvMaxPool(nn.Module):
    """
        general framework for the (explicit) ConvMaxPool submodel as described in Appendix E (pg.26) of Implicit Behavioral Cloning
    """

    def __init__(self, input_channels, n_channels):
        """
            params:
                input_channels (int): number of channels in input image
                n_channels (list[int]): list of the number of channels in arbitrary number of conv layers
        """
        layers = []
        layers.append(nn.Conv2d(input_channels, n_channels[0]))
        layers.append(nn.MaxPool2d(kernel_size = 3))

        for n in range(1, len(n_channels)):
            layers.append(nn.Conv2d(n_channels[n-1], n_channels[n], kernel_size = 3))
            layers.append(nn.MaxPool2d(kernel_size = 3))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        estimate = self.cnn(x)
        return estimate

class ExplicitConvMLP(nn.Module):
    """
        general framework for the (explicit) ConvMLP model as described in Appendex E (pg.26) of Implicit Behavioral Cloning
    """
    
    def __init__(self, input_channels, n_channels, act_dim, hidden_dim, n_hidden):
        """
            params mirror ConvMaxPool and MLP respectively
        """
        self.cnn = ConvMaxPool(input_channels, n_channels)
        obs_dim = None # calculate dimensions of flattened image
        self.mlp = MLP(obs_dim, act_dim, hidden_dim, n_hidden)

    def forward(self, x):
        estimate = self.cnn(x)
        estimate = estimate.flatten()


class ImplicitMLP:
    """
        Abstracts the implicit model described in section 2 of Implicit Behavioral Cloning
    """
    # the ebm will take in a obs_dim + act_dim input and output a 1 dimensional energy
    # output function needs to be ReLU to restrict energies to >=0
    
    def __init__(self, obs_dim, act_dim, hidden_dim, n_hidden, output_bounds):
        """
            params: 
                obs_dim (int): the dimensionality of the observation space (X)
                    embeddings of the observation space can also be fed into here
                act_dim (int): the dimensionality of the action space (Y)
                output_bounds (tuple[list[int]]): a 2-tuple which contains a list of output lower bounds and a list of output upper bounds
                    the size of each list must have a length equal to act_dim
        """
        self.ebm = nn.Sequential(
                MLP(obs_dim + act_dim, 1, hidden_dim, n_hidden),
                nn.ReLU()
                )

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lower_bounds = torch.tensor(output_bounds[0])
        self.upper_bounds = torch.tensor(output_bounds[1])

    def inference_b1(self, x, sigma_init = 0.33, k = 0.5, n_samples = 16_384, n_iters = 3):
        sigma = sigma_init
        ys = torch.rand((n_samples, self.act_dim)) # each row is a y sample
        ys = ys*(self.upper_bounds - self.lower_bounds) + self.lower_bounds # should naturally broadcast
        x_mat = x.unsqueeze(0).expand(n_samples, -1) # add dimension 0 of size n_samples without repeating memory

        for iter in range(n_iters): 
            inputs = torch.cat((ys, x_mat), 1) # concatenate x and y on 1st axis
            energies = self.ebm(inputs).flatten() # nn.Module can handle batch predictions
            probs = torch.exp(-1*energies)
            probs /= torch.sum(probs) # manual softmax calculation

            if iter < n_iters-1:
                # sample from multinomial
                indices = torch.multinomial(probs, n_samples, replacement = True)
                ys = ys[indices, :]

                # add gaussian noise
                # pg. 15 describes the noise as a vector of n_sample independent vals
                noise = torch.normal(0, sigma, (n_samples,1))
                ys += noise 
                
                # clamp to within bounds
                ys = torch.clamp(ys, self.lower_bounds, self.upper_bounds)

                # shrink variance
                sigma *= k
            else: 
                index = torch.argmax(probs).item()
                prediction = ys[index]
                return prediction

if __name__ == "__main__":
    pass



