from __future__ import annotations

import math

import numpy as np
import torch
from torch_geometric.data import Data

from kldm.data import (
    ConcatFeatures,
    ContinuousIntervalAngles,
    ContinuousIntervalLengths,
    FullyConnectedGraph,
    KLDMState,
    OneHot,
    SampleDatasetCSP,
    SampleDatasetDNG,
)


def _reference_fully_connected_edges(num_nodes: int) -> torch.Tensor:
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_fully_connected_graph_matches_reference():
    data = Data(pos=torch.zeros(4, 3))
    transformed = FullyConnectedGraph()(data)
    expected = _reference_fully_connected_edges(4)
    assert torch.equal(transformed.edge_node_index, expected)


def test_continuous_interval_lengths_matches_reference():
    data = Data(pos=torch.zeros(8, 3), lengths=torch.tensor([[2.0, 4.0, 8.0]]))
    transformed = ContinuousIntervalLengths()(data)
    expected = torch.log(torch.tensor([[2.0, 4.0, 8.0]]))
    assert torch.allclose(transformed.lengths, expected)


def test_continuous_interval_angles_matches_reference():
    data = Data(angles=torch.tensor([[60.0, 90.0, 120.0]]))
    transformed = ContinuousIntervalAngles()(data)
    radians = torch.deg2rad(torch.tensor([[60.0, 90.0, 120.0]]))
    expected = torch.tan(radians - torch.pi / 2.0)
    assert torch.allclose(transformed.angles, expected)


def test_one_hot_matches_reference_without_noise():
    data = Data(h=torch.tensor([1, 8, 1], dtype=torch.long))
    transformed = OneHot(values=[1, 6, 8], key="h", expand_as_vector=True, noise_std=0.0)(data)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(transformed.h, expected)


def test_concat_features_matches_reference():
    data = Data(a=torch.tensor([[1.0, 2.0]]), b=torch.tensor([[3.0, 4.0]]))
    transformed = ConcatFeatures(in_keys=["a", "b"], out_key="ab")(data)
    assert torch.equal(transformed.ab, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_kldm_state_attaches_expected_fields():
    data = Data(
        pos=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        h=torch.tensor([6, 8], dtype=torch.long),
        lengths=torch.tensor([[3.0, 4.0, 5.0]]),
        angles=torch.tensor([[90.0, 91.0, 92.0]]),
    )
    transformed = KLDMState(atom_values=[1, 6, 8])(data)
    assert torch.equal(transformed.f0, data.pos)
    assert torch.equal(transformed.v0, torch.zeros_like(data.pos))
    assert torch.equal(transformed.l0, torch.tensor([[3.0, 4.0, 5.0, 90.0, 91.0, 92.0]]))
    assert torch.equal(
        transformed.a0,
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_sample_dataset_dng_matches_reference_distribution_sampling():
    empirical_distribution = np.array([0.0, 0.2, 0.3, 0.5], dtype=np.float64)
    dataset = SampleDatasetDNG(empirical_distribution=empirical_distribution, n_samples=6, seed=7)
    reference_rng = np.random.RandomState(7)
    expected_num_atoms = reference_rng.choice(len(empirical_distribution), 6, p=empirical_distribution)
    assert np.array_equal(dataset.num_atoms, expected_num_atoms)

    torch.manual_seed(11)
    sample = dataset[0]
    torch.manual_seed(11)
    expected = Data(
        pos=torch.randn(int(expected_num_atoms[0]), 3),
        h=torch.full((int(expected_num_atoms[0]),), 6, dtype=torch.long),
    )
    assert torch.equal(sample.h, expected.h)
    assert torch.allclose(sample.pos, expected.pos)


def test_sample_dataset_csp_matches_reference_formula_expansion():
    dataset = SampleDatasetCSP(formulas=["SiO2"], n_samples_per_formula=1)

    torch.manual_seed(13)
    sample = dataset[0]
    torch.manual_seed(13)
    expected_h = torch.tensor([14, 8, 8], dtype=torch.long)
    expected_pos = torch.randn(3, 3)

    assert torch.equal(sample.h, expected_h)
    assert torch.allclose(sample.pos, expected_pos)


def test_angle_transform_inversion_round_trip():
    transform = ContinuousIntervalAngles()
    angles = np.array([60.0, 90.0, 120.0], dtype=np.float64)
    transformed = np.tan(np.deg2rad(angles) - math.pi / 2.0)
    recovered = transform.invert_one(transformed)
    assert np.allclose(np.array(recovered), angles)
