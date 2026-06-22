import unittest

from MiniVectorDB.index.pq import OptimizedProductQuantizer, ProductQuantizer, ResidualQuantizer
from MiniVectorDB.index.scalar_quantization import ScalarQuantizer


class QuantizationTests(unittest.TestCase):
    def test_scalar_quantizer(self):
        quantizer = ScalarQuantizer(bits=4).fit([[0.0, 10.0], [1.0, 20.0]])
        codes = quantizer.encode([0.5, 15.0])
        decoded = quantizer.decode(codes)
        self.assertEqual(len(codes), 2)
        self.assertEqual(len(decoded), 2)

    def test_pq_and_residual(self):
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6], [0.2, 0.1, 0.0, 0.3]]
        pq = ProductQuantizer(parts=2, codebook_size=2).fit(vectors)
        codes = pq.encode(vectors[0])
        self.assertEqual(len(codes), 2)

        rq = ResidualQuantizer(stages=2, codebook_size=2).fit(vectors)
        rq_codes = rq.encode(vectors[0])
        self.assertEqual(len(rq_codes), 2)

        opq = OptimizedProductQuantizer().fit(vectors)
        payload = opq.encode(vectors[0])
        self.assertIn("pq", payload)


if __name__ == "__main__":
    unittest.main()

