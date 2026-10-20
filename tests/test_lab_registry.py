import unittest

from krios.lab import EXPERIMENTS


class TestRegistry(unittest.TestCase):
    def test_registry_and_examples_are_labelled_synthetic(self):
        self.assertEqual(sorted(EXPERIMENTS), ["eclipse", "parallax", "pole", "shadow", "solarpath"])
        for name, cls in EXPERIMENTS.items():
            ex = cls.example()
            self.assertTrue(ex["metadata"]["synthetic"], name)
            self.assertIn(cls.rows_key, ex)


if __name__ == "__main__":
    unittest.main()
