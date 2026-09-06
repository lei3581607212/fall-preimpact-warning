import unittest

import torch

from scripts.model import PureTCNBaseline


class PureTCNBaselineTests(unittest.TestCase):
    def test_output_contract(self):
        model = PureTCNBaseline(input_dim=80, window_size=64).eval()
        risk, time, action = model(torch.zeros(2, 64, 80))
        self.assertEqual(tuple(risk.shape), (2, 3))
        self.assertEqual(tuple(time.shape), (2,))
        self.assertEqual(tuple(action.shape), (2, 8))

    def test_last_embedding_is_causal(self):
        model = PureTCNBaseline(input_dim=80, window_size=64).eval()
        prefix = torch.randn(1, 48, 80)
        first = torch.cat((prefix, torch.zeros(1, 16, 80)), dim=1)
        second = torch.cat((prefix, torch.randn(1, 16, 80)), dim=1)
        # At t=47 the two streams are identical; reading a prefix must therefore
        # produce identical outputs even though their later frames differ.
        with torch.no_grad():
            out_a = model.tcn(model.input(first).transpose(1, 2))[:, :, 47]
            out_b = model.tcn(model.input(second).transpose(1, 2))[:, :, 47]
        self.assertTrue(torch.equal(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
