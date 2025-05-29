import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore


# -------------------------
# TorchScript-Compatible Surrogate Spike Function
# -------------------------
@torch.jit.script
def surrogate_spike(mem: torch.Tensor, threshold: float, slope: float = 10.0) -> torch.Tensor:
    # Smooth approximation to a step function.
    return torch.sigmoid(slope * (mem - threshold))


class FraudSNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_steps: int,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        """
        input_size: Number of features.
        hidden_size: Number of neurons in the hidden layer.
        time_steps: Number of time steps (manually unrolled, fixed to 10).
        """
        super().__init__()
        self.time_steps = time_steps

        # Fully connected layers.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        # Build custom LIF neurons using surrogate_spike.
        self.lif1 = self.build_lif(beta, threshold)
        self.lif2 = self.build_lif(beta, threshold)

        # Final sigmoid activation.
        self.sigmoid = nn.Sigmoid()

    def build_lif(self, beta, threshold):
        # Define a TorchScript-friendly LIF neuron using the surrogate spike.
        class LIFNeuronWithSTE(nn.Module):
            def __init__(self, beta, threshold):
                super().__init__()
                self.beta = beta
                self.threshold = threshold

            def forward(self, input_current: torch.Tensor, mem: torch.Tensor):
                mem = self.beta * mem + input_current
                spike = surrogate_spike(mem, self.threshold)
                mem = mem - spike * self.threshold
                return spike, mem

        return LIFNeuronWithSTE(beta, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 2, input_size) since time_steps==2
        batch_size = x.size(0)
        # Initialize membrane potentials.
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, 1, device=x.device)

        # Time step 0.
        cur1_0 = self.fc1(x[:, 0, :])
        spk1_0, mem1 = self.lif1(cur1_0, mem1)
        cur2_0 = self.fc2(spk1_0)
        spk2_0, mem2 = self.lif2(cur2_0, mem2)

        # Time step 1.
        cur1_1 = self.fc1(x[:, 1, :])
        spk1_1, mem1 = self.lif1(cur1_1, mem1)
        cur2_1 = self.fc2(spk1_1)
        spk2_1, mem2 = self.lif2(cur2_1, mem2)
        
        # Time step 2.
        cur1_2 = self.fc1(x[:, 2, :])
        spk1_2, mem1 = self.lif1(cur1_2, mem1)
        cur2_2 = self.fc2(spk1_2)
        spk2_2, mem2 = self.lif2(cur2_2, mem2)
        
        # Time step 3.
        cur1_3 = self.fc1(x[:, 3, :])
        spk1_3, mem1 = self.lif1(cur1_3, mem1)
        cur3_3 = self.fc2(spk1_3)
        spk2_3, mem2 = self.lif2(cur3_3, mem2)
        
        # Time step 4.
        cur1_4 = self.fc1(x[:, 4, :])
        spk1_4, mem1 = self.lif1(cur1_4, mem1)
        cur4_4 = self.fc2(spk1_4)   
        spk2_4, mem2 = self.lif2(cur4_4, mem2)
        
        # Time step 5.
        cur1_5 = self.fc1(x[:, 5, :])
        spk1_5, mem1 = self.lif1(cur1_5, mem1)
        cur5_5 = self.fc2(spk1_5)
        spk2_5, mem2 = self.lif2(cur5_5, mem2)
            
        # Time step 6.
        cur1_6 = self.fc1(x[:, 6, :])
        spk1_6, mem1 = self.lif1(cur1_6, mem1)
        cur6_6 = self.fc2(spk1_6)
        spk2_6, mem2 = self.lif2(cur6_6, mem2)
        
        # Time step 7.
        cur1_7 = self.fc1(x[:, 7, :])
        spk1_7, mem1 = self.lif1(cur1_7, mem1)
        cur7_7 = self.fc2(spk1_7)
        spk2_7, mem2 = self.lif2(cur7_7, mem2)
        
        # Time step 8.
        cur1_8 = self.fc1(x[:, 8, :])
        spk1_8, mem1 = self.lif1(cur1_8, mem1)
        cur8_8 = self.fc2(spk1_8)
        spk2_8, mem2 = self.lif2(cur8_8, mem2)
        
        # Time step 9.
        cur1_9 = self.fc1(x[:, 9, :])
        spk1_9, mem1 = self.lif1(cur1_9, mem1)
        cur9_9 = self.fc2(spk1_9)
        spk2_9, mem2 = self.lif2(cur9_9, mem2)
        
        spk_sum = spk2_0 + spk2_1 + spk2_2 + spk2_3 + spk2_4 + spk2_5 + spk2_6 + spk2_7 + spk2_8 + spk2_9
        spk_mean = spk_sum / 10.0   
        out = self.sigmoid(spk_mean)
        return out


# -------------------------
# Modified Training Function to Record Loss Curves
# -------------------------

