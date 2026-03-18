from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from kldm.data import FormulaDataset, StoredCrystalDataset
from kldm.scoreNetwork import SimpleScoreGNN


class ModelKLDM(nn.Module):
    """
    A model consisting of Algorithm (1-4) in KLDM Paper.

    1. Initialise score network (CSPVNet) an equvirant GNN
    which is found in kldm.scoreNetwork, the GNN is out of scope,
    please refer to the KLDM paper for more information about it.

    2. Intialise

            -  trivialised diffusion model for velocity
            -  continious variance exploding SDE lattice
            -  Discrete/Analogbits/Continous for atom types

    3. Perform algorithm (1), training targets

    4. Perform algorithm (2)......
    """


    def __init__(
        self,
    ) -> None:
        super().__init__()


    ################################
    # Algorithm 1 - Training targets
    ################################

    def training_targets_dng(
        self,
        initial_sample: Data,  # sample from StoredCrystalDataset / DNGTask.fit_dataset
        timestep: int | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Training targets for DNG (de-novo generation).

        Expects a DNG training sample from StoredCrystalDataset, which carries:
          pos (fractional coords), h (atomic numbers), l (6D lattice),
          edge_node_index.

        From KLDM: Sampling f_t, v_t, l_t, a_t from the transition kernel
        and the corresponding target scores.
        """



        ##########################
        # Sample velocity / coord #
        ##########################

        #Sample epsilon_v ~ N_v(0, I),

        #Sample epsilon_r_t ~ N(0, I),

        #If initial velocities are zero, (which they are by design)
            #target_v = -v_t / sigma^2_v(t)

        #Calculate displacement r_t = w(mu_r_t, (t, v0, vt) + sigma_r_t(t,v0,vt) * epsilon_r_t)

        #Walk on the manifold: f_t = w(f_0 + r_t)

        #Energy center: f_t = center(f_t), such that mean = 0

        #Calculate target_s = eq (26)

        #Target_v = target_v + target_s

        ##########################
        # Sample lattice wrt time #
        ##########################

        #sample epsilon_l ~N(0,I)

        #l_t = alpha(t)l_0+ sigma(t)*epsilon_l

        #target_l = epsilon_l

        ##########################
        # Sample atom types (DNG) #
        ##########################

        #sample a_t

        #Sample epsilon_a ~N(0, I)
        #a_t = alpha(t)a_0 + sigma(t)*epsilon_a
        #target_a = epsilon_a

        #return (v_t, f_t, l_t, a_t), (target_v, target_l, target_a)

        return

    def training_targets_csp(
        self,
        initial_sample: Data,  # sample from FormulaDataset / CSPTask.sample_loader
        timestep: int | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Training targets for CSP (crystal structure prediction).

        Expects a CSP sampling sample from FormulaDataset, which carries:
          pos (placeholder coords), h (atomic numbers from formula),
          l (default zero lattice), edge_node_index.
          Atom types are fixed — not diffused in CSP.

        From KLDM: Sampling f_t, v_t, l_t from the transition kernel
        and the corresponding target scores.
        """

        print("Input data type: CSP")



        ##########################
        # Sample velocity / coord #
        ##########################

        #Sample epsilon_v ~ N_v(0, I),

        #Sample epsilon_r_t ~ N(0, I),

        #If initial velocities are zero, (which they are by design)
            #target_v = -v_t / sigma^2_v(t)

        #Calculate displacement r_t = w(mu_r_t, (t, v0, vt) + sigma_r_t(t,v0,vt) * epsilon_r_t)

        #Walk on the manifold: f_t = w(f_0 + r_t)

        #Energy center: f_t = center(f_t), such that mean = 0

        #Calculate target_s = eq (26)

        #Target_v = target_v + target_s

        ##########################
        # Sample lattice wrt time #
        ##########################

        #sample epsilon_l ~N(0,I)

        #l_t = alpha(t)l_0+ sigma(t)*epsilon_l

        #target_l = epsilon_l

        #return (v_t, f_t, l_t), (target_v, target_l)


        return




if __name__ == "__main__":
    model = ModelKLDM()


    #TRAINING TARGET:
