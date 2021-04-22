import torch
import torch.nn
from planar_flow import PlanarFlow
from loss import NfLoss
from test_energy_function import TargetDistribution
from utils import plot_comparison


def main():
    
    EPOCHS = 5000
    BATCH_SIZE = 64
    lr = 6e-4
    flow_length = 32

    target_distr = "U_1"
    nf_model = PlanarFlow(dim=2, K=flow_length)
    target_dist = TargetDistribution(target_distr)
    loss_fn = NfLoss(target_dist)
    optimiser = torch.optim.Adam(nf_model.parameters(), lr=lr)
    

    for e in range(1, EPOCHS):
        z_0 = torch.zeros(size=(BATCH_SIZE, 2)).normal_(mean=0, std=1)
        zk, ln_jacobians = nf_model(z_0)
        loss = loss_fn(z_0, zk, ln_jacobians)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (e % 1000) == 0:
            print(f"(epoch_num {e:05d}/{EPOCHS}) loss: {loss}")
            plot_comparison(nf_model, target_distr, flow_length)


if __name__ == "__main__":
    main()