"""Hierarchical-softmax output combination for fastText LID models.

Vendored from TigreGotico/lid176-onnx (``hs_tree.py`` and the ``HSCombiner``
class in ``lid176_hash.py``, https://huggingface.co/TigreGotico/lid176-onnx).

fastText models trained with ``loss=hs`` do not reduce to a flat softmax over
the output matrix: each row of the output matrix is a binary classifier for
one internal node of a Huffman tree built over the label frequencies, and a
label's probability is the product of the sigmoid (or 1-sigmoid) values along
its root-to-leaf path. Running the GlotLID/OpenLID ``MatMul -> Softmax``
recipe on such a model gives 0% agreement with fastText - the numbers come
out well-formed and completely meaningless.

So the ONNX graph for an hs model ends in ``Sigmoid`` over every Huffman node
and this module does the path walk. The tree is not serialized in fastText's
``.bin`` file; it is rebuilt deterministically from the label counts every
time the model loads. :func:`build_tree` reproduces that construction
(``HierarchicalSoftmaxLoss::buildTree`` in fastText's ``Loss.cc``) and the
published model repo bakes its result into ``hs_tree.json`` at export time,
so :class:`HSCombiner` only has to read that file. ``build_tree`` is kept
here for reproducibility: it is how ``hs_tree.json`` can be regenerated and
checked from a label-count list, without a fastText build.
"""

from __future__ import annotations

import json
from typing import List, Sequence, Tuple

import numpy as np


def build_tree(counts: Sequence[int]) -> Tuple[List[List[int]], List[List[bool]]]:
    """fastText's ``HierarchicalSoftmaxLoss::buildTree`` over label counts.

    Returns (paths, codes): for each label, the internal-node indices from
    leaf to root and the branch taken at each of them.
    """
    osz = len(counts)
    n = 2 * osz - 1
    parent = [-1] * n
    left = [-1] * n
    right = [-1] * n
    binary = [False] * n
    count = [1e15] * n
    for i in range(osz):
        count[i] = counts[i]

    leaf = osz - 1
    node = osz
    for i in range(osz, 2 * osz - 1):
        mini = [0, 0]
        for j in range(2):
            if leaf >= 0 and count[leaf] < count[node]:
                mini[j] = leaf
                leaf -= 1
            else:
                mini[j] = node
                node += 1
        left[i] = mini[0]
        right[i] = mini[1]
        count[i] = count[mini[0]] + count[mini[1]]
        parent[mini[0]] = i
        parent[mini[1]] = i
        binary[mini[1]] = True

    paths: List[List[int]] = []
    codes: List[List[bool]] = []
    for i in range(osz):
        path: List[int] = []
        code: List[bool] = []
        j = i
        while parent[j] != -1:
            path.append(parent[j] - osz)
            code.append(binary[j])
            j = parent[j]
        paths.append(path)
        codes.append(code)
    return paths, codes


class HSCombiner:
    """Turns per-Huffman-node sigmoids into per-label probabilities.

    The probabilities are accumulated in log space and exponentiated at the
    end: a path can be ~20 nodes deep, and multiplying twenty sigmoids
    directly underflows float32 for the unlikely labels.
    """

    def __init__(self, paths: List[List[int]], codes: List[List[bool]]):
        self.paths = paths
        self.codes = codes

    @classmethod
    def from_file(cls, path) -> "HSCombiner":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data["paths"], data["codes"])

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> "HSCombiner":
        return cls(*build_tree(counts))

    def __call__(self, node_probs: np.ndarray) -> np.ndarray:
        """node_probs: sigmoid(dot(hidden, node)) for every Huffman node.

        Returns one probability per label, in ``labels.json`` order.
        """
        log_f = np.log(np.clip(node_probs, 1e-12, 1.0))
        log_1mf = np.log(np.clip(1.0 - node_probs, 1e-12, 1.0))
        out = np.empty(len(self.paths), dtype=np.float64)
        for i, (path, code) in enumerate(zip(self.paths, self.codes)):
            s = 0.0
            for node, bit in zip(path, code):
                s += log_f[node] if bit else log_1mf[node]
            out[i] = s
        return np.exp(out)
