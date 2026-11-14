import copy
import unittest

import numpy as np
import torch

from kenggo.encoding import Context, INPUT_KEYS, encode
from kenggo.go import Game, PASS
from kenggo.learner import ModelConfig, PopulationModel


def model_inputs(games=None, context=Context()):
    games = [Game.new(), Game.new().play("B", 0)] if games is None else games
    rows = [encode(game, context) for game in games]
    return {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in INPUT_KEYS}


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_default_network_produces_362_legal_actions_and_bounded_mover_value(self):
        model = PopulationModel()
        self.assertEqual(model.config, ModelConfig(blocks=8, channels=128, rank_conditioned=True))
        inputs = model_inputs()
        with torch.no_grad():
            logits, value = model(inputs)
        self.assertEqual(logits.shape, (2, 362))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.isfinite(logits[inputs["legal"]]).all())
        self.assertTrue(torch.isneginf(logits[~inputs["legal"]]).all())
        probabilities = logits.softmax(dim=1)
        self.assertTrue((probabilities[~inputs["legal"]] == 0).all())
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
        self.assertTrue((probabilities[:, PASS] > 0).all())
        self.assertTrue((value.abs() <= 1).all())

    def test_masked_nan_metadata_and_invalid_rank_payloads_do_not_change_predictions(self):
        model = PopulationModel(ModelConfig(blocks=1, channels=16)).eval()
        inputs = model_inputs()
        changed = copy.deepcopy(inputs)
        changed["scalars"][~changed["scalar_valid"]] = torch.nan
        changed["ranks"][:] = -32768
        with torch.no_grad():
            first, second = model(inputs), model(changed)
        for expected, actual in zip(first, second):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_unconditioned_model_ignores_rank_values_and_validity_with_the_same_architecture(self):
        conditioned = PopulationModel(ModelConfig(blocks=1, channels=16))
        model = PopulationModel(ModelConfig(blocks=1, channels=16, rank_conditioned=False))
        model.load_state_dict(conditioned.state_dict())
        known = model_inputs(context=Context("5k", "1d"))
        changed = copy.deepcopy(known)
        changed["rank_valid"][:] = False
        changed["ranks"][:] = 32767
        with torch.no_grad():
            first, second = model(known), model(changed)
            conditional = conditioned(known)
        for expected, actual in zip(first, second):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertFalse(torch.equal(conditional[0], first[0]))

    def test_unknown_mover_and_opponent_embeddings_each_receive_gradients(self):
        for side in (0, 1):
            with self.subTest(side=side):
                model = PopulationModel(ModelConfig(blocks=1, channels=16))
                inputs = model_inputs(context=Context("5k", "1d"))
                inputs["rank_valid"][:, side] = False
                inputs["ranks"][:, side] = 32767
                logits, value = model(inputs)
                (logits[inputs["legal"]].mean() + value.mean()).backward()
                self.assertGreater(float(model.rank_embedding.weight.grad[0].abs().sum()), 0)

    def test_rejects_all_illegal_history_extra_fields_and_visible_bad_payloads(self):
        model = PopulationModel(ModelConfig(blocks=1, channels=16))
        mutations = [lambda x: x["legal"].fill_(False),
                     lambda x: x["scalars"].__setitem__((0, 0), torch.nan),
                     lambda x: x.update(metadata=torch.zeros(2)),
                     lambda x: x.update(history_actions=torch.zeros((2, 8), dtype=torch.int16)),
                     lambda x: x["ranks"].__setitem__((0, 0), 0)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                inputs = model_inputs(context=Context("5k", "1d"))
                mutate(inputs)
                with self.assertRaises(ValueError):
                    model(inputs)

    def test_model_configuration_is_bounded(self):
        for config in (dict(blocks=0), dict(blocks=True), dict(blocks=17), dict(channels=15),
                       dict(channels=257), dict(channels=23), dict(rank_conditioned=1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ModelConfig(**config)


if __name__ == "__main__":
    unittest.main()
