# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import pytest
import torch
from tests.test_utils import DummyTokenizer
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import PackedDataset


class DummyDataset(Dataset):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def __getitem__(self, index):
        if index >= 1000:
            raise IndexError()
        return {
            "tokens": [index] * self.sample_size,
            "labels": [index] * (self.sample_size - 1) + [CROSS_ENTROPY_IGNORE_IDX],
        }

    def __len__(self):
        return 1000


class DummyRealDataset(Dataset):
    def __init__(self):
        self.samples_list = [
            "This is a packing test",
            "A fantastic test. It should pack two samples.",
            "This one will not be fully packed.",
        ]
        self.tokenizer = DummyTokenizer()

    def __getitem__(self, index):
        tokens = self.tokenizer.encode(self.samples_list[index])
        labels = tokens[1:] + [CROSS_ENTROPY_IGNORE_IDX]
        return {"tokens": tokens, "labels": labels}

    def __len__(self):
        return len(self.samples_list)


class BoundaryDataset(Dataset):
    """Three raw samples with lengths 2, 3 and 7. Token/label values are chosen
    so that original samples can be identified and reconstructed after packing,
    truncation and padding."""

    def __init__(self):
        self.tokens = [
            [10, 11],
            [20, 21, 22],
            [30, 31, 32, 33, 34, 35, 36],
        ]

    def __getitem__(self, index):
        tokens = list(self.tokens[index])
        labels = [token + 100 for token in tokens]
        return {"tokens": tokens, "labels": labels}

    def __len__(self):
        return len(self.tokens)


def _reconstruct_samples(packs, field):
    """Walk the ``sample_boundaries`` graph of every pack in order and rebuild
    the original samples by concatenating the indicated ``[start, end)`` slices.
    Rows sharing a ``sample_idx`` (a sample truncated across packs) are joined."""
    chunks = {}
    order = []
    for pack in packs:
        for sample_idx, start, end in pack["sample_boundaries"].tolist():
            if sample_idx not in chunks:
                chunks[sample_idx] = []
                order.append(sample_idx)
            chunks[sample_idx].append(pack[field][start:end])
    return {
        sample_idx: torch.cat(chunks[sample_idx]) for sample_idx in order
    }



class TestPackedDataset:
    def _get_expected_seq_lens_and_input_pos(
        self, max_seq_len, sample_size, split_across_pack
    ):
        """
        Generate expected seq lens and position ids for given max sequence
        length and sample length
        """
        num_samples, remainder = divmod(max_seq_len, sample_size)
        seq_lens = [sample_size] * num_samples
        if split_across_pack and remainder > 0:
            num_samples += 1
        input_pos = [list(range(sample_size)) for i in range(1, num_samples + 1)]
        input_pos = list(itertools.chain(*input_pos))

        # Emulate seq len and position id padding
        if remainder > 0:
            if not split_across_pack:
                input_pos.extend(list(range(sample_size, sample_size + remainder)))
            seq_lens.extend([remainder])

        return torch.tensor(seq_lens), torch.tensor(input_pos[:max_seq_len])

    def _calculate_num_packs(
        self, dataset_size, max_seq_len, sample_size, split_across_pack, max_packs
    ):
        # First see how many samples we can fit in a single pack
        num_samples_per_pack, remainder = divmod(max_seq_len, sample_size)

        # If we split across pack (and the samples don't fit perfectly in max_seq_len), we can fit more
        if split_across_pack and remainder > 0:
            # Now we need the fractional to see how many we can partially fit in each pack
            num_samples_per_pack = max_seq_len / sample_size

        # If we don't split across pack, we will need more packs
        num_packs, remainder = divmod(dataset_size, num_samples_per_pack)

        # If there's leftover, we need to add one more pack
        if remainder > 0:
            num_packs += 1

        return num_packs if num_packs < max_packs else max_packs

    @pytest.mark.parametrize("max_seq_len", [25])
    @pytest.mark.parametrize("sample_size", [2, 5])
    @pytest.mark.parametrize("max_packs", [5, 200])
    @pytest.mark.parametrize("split_across_pack", [True, False])
    def test_packed_dataset(
        self, max_seq_len, sample_size, max_packs, split_across_pack
    ):
        dataset = DummyDataset(sample_size)
        packed = PackedDataset(
            dataset,
            max_seq_len=max_seq_len,
            max_packs=max_packs,
            split_across_pack=split_across_pack,
        )

        # Check we get right number of packs
        correct_num_packs = self._calculate_num_packs(
            len(dataset), max_seq_len, sample_size, split_across_pack, max_packs
        )
        assert len(packed) == correct_num_packs

        # Check all fields are same length
        assert (
            len(packed[0]["tokens"])
            == len(packed[0]["labels"])
            == len(packed[0]["input_pos"])
        )
        # Check that samples are packed correctly - very last individual sample
        # should have index value of the number of times dataset was iterated over
        if split_across_pack:
            # If we split samples, we'll know how many samples by taking the
            # full length and dividing by sample size
            last_index, remainder = divmod(len(packed) * max_seq_len, sample_size)
            # Account for remaining sample that didn't fit in window
            last_index = last_index if remainder > 0 else last_index - 1
        else:
            # If we don't split samples, we know how many samples by taking
            # how much fits in a single window and multiplying by max rows.
            # If there is a remainder, this will end up being a pad token.
            last_index = (
                (max_seq_len // sample_size) * len(packed) - 1
                if max_seq_len % sample_size == 0
                else 0
            )

        assert packed[-1]["tokens"][-1].item() == last_index

        (
            expected_seq_lens,
            expected_input_pos,
        ) = self._get_expected_seq_lens_and_input_pos(
            max_seq_len, sample_size, split_across_pack
        )

        torch.testing.assert_close(packed[0]["seq_lens"], expected_seq_lens)
        torch.testing.assert_close(packed[0]["input_pos"], expected_input_pos)

    @pytest.mark.parametrize("max_seq_len", [13])
    @pytest.mark.parametrize("sample_size", [14, 27, 40])
    @pytest.mark.parametrize("max_packs", [5, 200, 3100])
    @pytest.mark.parametrize("split_across_pack", [True])
    def test_chunked_case(self, max_seq_len, sample_size, max_packs, split_across_pack):
        dataset = DummyDataset(sample_size)
        packed = PackedDataset(
            dataset,
            max_seq_len=max_seq_len,
            max_packs=max_packs,
            split_across_pack=split_across_pack,
        )

        # Check we get right number of packs
        correct_num_packs = self._calculate_num_packs(
            len(dataset), max_seq_len, sample_size, split_across_pack, max_packs
        )
        assert len(packed) == correct_num_packs

        # Check all fields are same length
        assert all(
            len(pack["tokens"]) == len(pack["labels"]) == len(pack["input_pos"])
            for pack in packed
        )

        # Check that all sum(seq_lens) are equal to max_seq_len
        assert all(pack["seq_lens"].sum().item() == max_seq_len for pack in packed)

    def test_packed_dataset_real_data(self):
        expected_tokenized_prompts = [
            torch.tensor([0, 4, 2, 1, 7, 4, -1, 0, 1, 9]),
            torch.tensor([5, 2, 6, 4, 3, 8, -1, 0, 4, 3]),
            torch.tensor([4, 3, 2, 5, 7, -1, 0, 0, 0, 0]),
        ]
        expected_tokenized_labels = [
            torch.tensor([4, 2, 1, 7, 4, -1, -100, 1, 9, 5]),
            torch.tensor([2, 6, 4, 3, 8, -1, -100, 4, 3, 4]),
            torch.tensor([3, 2, 5, 7, -1, -100, -100, -100, -100, -100]),
        ]
        expected_seq_lens = [
            torch.tensor(
                [7, 3],
            ),
            torch.tensor(
                [7, 3],
            ),
            torch.tensor(
                [6, 4],
            ),
        ]
        expected_input_pos = [
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2]),
            torch.tensor([3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
            # Padded position ids cannot go beyond max seq_len - 1
            torch.tensor([3, 4, 5, 6, 7, 8, 9, 9, 9, 9]),
        ]
        packed = PackedDataset(
            DummyRealDataset(),
            max_seq_len=10,
            split_across_pack=True,
        )

        for i in range(len(packed)):
            prompt, label, seq_lens, input_pos = (
                packed[i]["tokens"],
                packed[i]["labels"],
                packed[i]["seq_lens"],
                packed[i]["input_pos"],
            )
            torch.testing.assert_close(prompt, expected_tokenized_prompts[i])
            torch.testing.assert_close(label, expected_tokenized_labels[i])
            torch.testing.assert_close(input_pos, expected_input_pos[i])
            torch.testing.assert_close(seq_lens, expected_seq_lens[i])

    def test_pad_pack(self):
        padding_idx = -8
        ignore_idx = -100  # Same as CROSS_ENTROPY_IGNORE_IDX
        pack = {
            "tokens": [2, 5],
            "labels": [3, 7],
            "seq_lens": [1, 1],
            # Let the first token be the end of the previous sample (pos 8),
            # and the second token the start of the next sample (pos 0). Collate
            # should continue from 0 -> 1, 2, ...
            "input_pos": [8, 0],
        }

        dataset = DummyDataset(2)
        packed = PackedDataset(
            dataset,
            max_seq_len=4,
        )

        pack = packed._convert_to_tensors(pack)
        padded = packed._pad_pack(pack, padding_idx=padding_idx)

        padded_input = padded["tokens"]
        padded_label = padded["labels"]
        padded_input_pos = padded["input_pos"]
        padded_seq_lens = padded["seq_lens"]

        torch.testing.assert_close(
            padded_input, torch.tensor([2, 5, padding_idx, padding_idx])
        )
        torch.testing.assert_close(
            padded_label, torch.tensor([3, 7, ignore_idx, ignore_idx])
        )
        torch.testing.assert_close(padded_input_pos, torch.tensor([8, 0, 1, 2]))
        torch.testing.assert_close(padded_seq_lens, torch.tensor([1, 1, 2]))

    def test_pack_errors_if_sample_too_long(self):
        dataset = DummyDataset(8)
        with pytest.raises(ValueError, match="Dataset sample is too long"):
            PackedDataset(
                dataset,
                max_seq_len=4,
            )

    def test_empty_sample_boundaries_shape(self):
        # A boundary list without any original samples converts to a (0, 3)
        # tensor. The real packing loop never pads an empty pack, so we only
        # exercise the conversion here.
        dataset = DummyDataset(2)
        packed = PackedDataset(dataset, max_seq_len=4)
        pack = packed._convert_to_tensors(
            {
                "tokens": [1],
                "labels": [1],
                "input_pos": [0],
                "seq_lens": [1],
                "sample_boundaries": [],
            }
        )
        assert pack["sample_boundaries"].shape == (0, 3)
        # Pads the single-token pack; padding adds no boundary.
        padded = packed._pad_pack(pack, padding_idx=0)
        torch.testing.assert_close(
            padded["sample_boundaries"], torch.empty((0, 3), dtype=torch.long)
        )

    def test_sample_boundaries_truncation_2_3_7(self):
        """Acceptance test for the sample boundary graph using raw samples of
        lengths 2, 3 and 7 packed into max_seq_len=5 packs with
        ``split_across_pack=True`` (the length-7 sample is truncated across
        packs). We check exact boundaries, that padding tokens belong to no
        original sample, and that the original samples (tokens and labels) can
        be fully reconstructed from the boundary graph.
        """
        padding_idx = 0
        packed = PackedDataset(
            BoundaryDataset(),
            max_seq_len=5,
            padding_idx=padding_idx,
            split_across_pack=True,
        )

        # Pack 0: S0 (len 2) + S1 (len 3), exactly fills the pack
        # Pack 1: first 5 tokens of S2 (len 7) - truncated here
        # Pack 2: remaining 2 tokens of S2 + padding
        expected_tokens = [
            [10, 11, 20, 21, 22],
            [30, 31, 32, 33, 34],
            [35, 36, padding_idx, padding_idx, padding_idx],
        ]
        expected_labels = [
            [110, 111, 120, 121, 122],
            [130, 131, 132, 133, 134],
            [135, 136, CROSS_ENTROPY_IGNORE_IDX,
             CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX],
        ]
        expected_input_pos = [
            [0, 1, 0, 1, 2],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ]
        expected_seq_lens = [
            [2, 3],
            [5],
            [2, 3],
        ]
        expected_boundaries = [
            [[0, 0, 2], [1, 2, 5]],
            [[2, 0, 5]],
            [[2, 0, 2]],
        ]

        for i, pack in enumerate(packed):
            torch.testing.assert_close(
                pack["tokens"], torch.tensor(expected_tokens[i])
            )
            torch.testing.assert_close(
                pack["labels"], torch.tensor(expected_labels[i])
            )
            torch.testing.assert_close(
                pack["input_pos"], torch.tensor(expected_input_pos[i])
            )
            torch.testing.assert_close(
                pack["seq_lens"], torch.tensor(expected_seq_lens[i])
            )
            torch.testing.assert_close(
                pack["sample_boundaries"],
                torch.tensor(expected_boundaries[i]),
            )

        # Boundary invariants: half-open spans within the pack, no overlap, and
        # padding tokens are never covered by a boundary.
        covered = torch.zeros(5, dtype=torch.bool)
        for pack in packed:
            for _, start, end in pack["sample_boundaries"]:
                assert 0 <= start.item() < end.item() <= 5
                assert not covered[start:end].any()
                covered[start:end] = True
            assert (
                pack["tokens"][~covered] == padding_idx
            ).all(), "tokens outside the boundary graph must be padding"
            covered.zero_()

        # Truncation: sample 2 is split across packs 1 and 2, and its rows are
        # contiguous and in pack order.
        s2_rows = torch.cat(
            [
                pack["sample_boundaries"]
                [pack["sample_boundaries"][:, 0] == 2]
                for pack in packed
            ]
        )
        assert s2_rows[:, 0].tolist() == [2, 2]
        assert s2_rows[0, 1].item() == 0 and s2_rows[0, 2].item() == 5
        assert s2_rows[1, 1].item() == 0 and s2_rows[1, 2].item() == 2
        assert (s2_rows[:, 2] - s2_rows[:, 1]).sum().item() == 7

        # Reconstruction: following the boundary graph recovers every original
        # sample's tokens and labels exactly (and only) once.
        reconstructed_tokens = _reconstruct_samples(packed, "tokens")
        reconstructed_labels = _reconstruct_samples(packed, "labels")
        original = BoundaryDataset()
        assert sorted(reconstructed_tokens) == list(range(len(original)))
        for idx, expected in enumerate(original.tokens):
            torch.testing.assert_close(
                reconstructed_tokens[idx], torch.tensor(expected)
            )
            torch.testing.assert_close(
                reconstructed_labels[idx],
                torch.tensor([token + 100 for token in expected]),
            )

    def test_sample_boundaries_no_split(self):
        """Same lengths 2, 3, 7 with split_across_pack=False and max_seq_len=7:
        the length-7 sample keeps its own pack (no truncation) and packs are
        padded; padding belongs to no original sample."""
        padding_idx = 0
        packed = PackedDataset(
            BoundaryDataset(),
            max_seq_len=7,
            padding_idx=padding_idx,
            split_across_pack=False,
        )

        expected_tokens = [
            [10, 11, 20, 21, 22, padding_idx, padding_idx],
            [30, 31, 32, 33, 34, 35, 36],
        ]
        expected_labels = [
            [110, 111, 120, 121, 122,
             CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX],
            [130, 131, 132, 133, 134, 135, 136],
        ]
        expected_boundaries = [
            [[0, 0, 2], [1, 2, 5]],
            [[2, 0, 7]],
        ]

        for i, pack in enumerate(packed):
            torch.testing.assert_close(
                pack["tokens"], torch.tensor(expected_tokens[i])
            )
            torch.testing.assert_close(
                pack["labels"], torch.tensor(expected_labels[i])
            )
            torch.testing.assert_close(
                pack["sample_boundaries"],
                torch.tensor(expected_boundaries[i]),
            )
            # Last two positions of pack 0 are padding and uncovered
            covered = torch.zeros(7, dtype=torch.bool)
            for _, start, end in pack["sample_boundaries"]:
                covered[start:end] = True
            assert (pack["tokens"][~covered] == padding_idx).all()

        reconstructed_tokens = _reconstruct_samples(packed, "tokens")
        reconstructed_labels = _reconstruct_samples(packed, "labels")
        original = BoundaryDataset()
        for idx, expected in enumerate(original.tokens):
            torch.testing.assert_close(
                reconstructed_tokens[idx], torch.tensor(expected)
            )
            torch.testing.assert_close(
                reconstructed_labels[idx],
                torch.tensor([token + 100 for token in expected]),
            )

    def test_sample_boundaries_real_data(self):
        packed = PackedDataset(
            DummyRealDataset(),
            max_seq_len=10,
            split_across_pack=True,
        )
        # samples tokenize to lengths 10, 10, 6 (see test_packed_dataset_real_data);
        # packing split_across_pack=True with max_seq_len=10 splits each long
        # sample across two packs (7 + 3), and the final pack is padded.
        expected_boundaries = [
            [[0, 0, 7], [1, 7, 10]],
            [[1, 0, 7], [2, 7, 10]],
            [[2, 0, 6]],
        ]
        for i, pack in enumerate(packed):
            torch.testing.assert_close(
                pack["sample_boundaries"],
                torch.tensor(expected_boundaries[i]),
            )

        # Reconstructed tokens equal the original tokenized samples, truncated
        # to the non-padding region of the last pack.
        reconstructed = _reconstruct_samples(packed, "tokens")
        ds = DummyRealDataset()
        for i, sample in enumerate(ds):
            torch.testing.assert_close(
                reconstructed[i], torch.tensor(sample["tokens"])
            )
        # Last 4 positions of the last pack are padding and uncovered.
        last_pack = packed[-1]
        covered = torch.zeros(10, dtype=torch.bool)
        for _, start, end in last_pack["sample_boundaries"]:
            covered[start:end] = True
        assert (last_pack["tokens"][~covered] == 0).all()
