# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the OpenPI checkpoint convertors."""

import numpy as np
import pytest

from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi import _unwrap_value_leaves


def test_jax_value_wrappers_are_unwrapped():
    kernel = np.zeros((4, 2), dtype=np.float32)
    bias = np.ones(2, dtype=np.float32)
    tree = {
        "PaliGemma": {
            "img": {"pos_embedding": {"value": kernel}},
            "llm": {"layers": {"Dense_0": {"kernel": {"value": kernel}}}},
        },
        "action_in_proj": {"kernel": {"value": kernel}, "bias": {"value": bias}},
    }

    unwrapped = _unwrap_value_leaves(tree)

    assert unwrapped["PaliGemma"]["img"]["pos_embedding"] is kernel
    assert unwrapped["PaliGemma"]["llm"]["layers"]["Dense_0"]["kernel"] is kernel
    assert unwrapped["action_in_proj"]["kernel"] is kernel
    assert unwrapped["action_in_proj"]["bias"] is bias


def test_lora_keys_resolve_einsum_and_feed_forward_pairs():
    from rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg import lora_keys

    assert lora_keys("llm/layers/attn/q_einsum/lora_a") == (
        "llm/layers/attn/q_einsum/w",
        "llm/layers/attn/q_einsum/lora_a",
        "llm/layers/attn/q_einsum/lora_b",
    )
    assert lora_keys("llm/layers/mlp/gating_einsum_lora_a") == (
        "llm/layers/mlp/gating_einsum",
        "llm/layers/mlp/gating_einsum_lora_a",
        "llm/layers/mlp/gating_einsum_lora_b",
    )
    assert lora_keys("llm/layers/attn/q_einsum/lora_b") is None
    assert lora_keys("llm/layers/mlp/linear") is None


def test_merge_lora_into_base_adds_the_scaled_delta():
    from rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg import (
        merge_lora_into_base,
    )

    # (2, 3) base weight with a rank-1 adapter, as the stacked JAX einsum stores it.
    base = np.zeros((2, 3), dtype=np.float32)
    lora_a = np.array([[1.0], [2.0]], dtype=np.float32)
    lora_b = np.array([[1.0, 10.0, 100.0]], dtype=np.float32)
    flat = {
        "llm/layers/attn/q_einsum/w": base,
        "llm/layers/attn/q_einsum/lora_a": lora_a,
        "llm/layers/attn/q_einsum/lora_b": lora_b,
    }

    merged, count = merge_lora_into_base(flat, scaling=0.5)

    assert count == 1
    assert "llm/layers/attn/q_einsum/lora_a" not in merged
    assert "llm/layers/attn/q_einsum/lora_b" not in merged
    np.testing.assert_allclose(
        merged["llm/layers/attn/q_einsum/w"], 0.5 * (lora_a @ lora_b)
    )


def test_merge_lora_into_base_rejects_a_mismatched_pair():
    from rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg import (
        merge_lora_into_base,
    )

    flat = {
        "llm/layers/mlp/gating_einsum": np.zeros((2, 4, 6), dtype=np.float32),
        "llm/layers/mlp/gating_einsum_lora_a": np.zeros((2, 4, 3), dtype=np.float32),
        "llm/layers/mlp/gating_einsum_lora_b": np.zeros((2, 5, 6), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="multiplies into"):
        merge_lora_into_base(flat, scaling=1.0)

    with pytest.raises(KeyError, match="lora_b"):
        merge_lora_into_base(
            {
                "llm/layers/mlp/gating_einsum": np.zeros((1, 2, 3), dtype=np.float32),
                "llm/layers/mlp/gating_einsum_lora_a": np.zeros(
                    (1, 2, 4), dtype=np.float32
                ),
            },
            scaling=1.0,
        )


def test_subtask_classifier_state_transposes_the_jax_kernels():
    from rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg import (
        subtask_classifier_state,
    )

    head = {
        "norm": {"scale": np.arange(4, dtype=np.float32), "bias": np.zeros(4)},
        "hidden": {
            "kernel": np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
            "bias": np.zeros(3, dtype=np.float32),
        },
        "out": {
            "kernel": np.arange(3 * 2, dtype=np.float32).reshape(3, 2),
            "bias": np.zeros(2, dtype=np.float32),
        },
    }

    mapped = subtask_classifier_state(head)

    assert (mapped["width"], mapped["hidden_dim"], mapped["num_classes"]) == (4, 3, 2)
    state = mapped["state"]
    np.testing.assert_array_equal(
        state["subtask_classifier.norm.weight"], head["norm"]["scale"]
    )
    np.testing.assert_array_equal(
        state["subtask_classifier.hidden.weight"], head["hidden"]["kernel"].T
    )
    np.testing.assert_array_equal(
        state["subtask_classifier.out.weight"], head["out"]["kernel"].T
    )


def test_subtask_classifier_state_checks_the_class_count():
    from rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg import (
        subtask_classifier_state,
    )

    head = {
        "norm": {"scale": np.zeros(4, dtype=np.float32), "bias": np.zeros(4)},
        "hidden": {"kernel": np.zeros((4, 2)), "bias": np.zeros(2)},
        "out": {"kernel": np.zeros((2, 13)), "bias": np.zeros(13)},
    }
    assert subtask_classifier_state(head)["num_classes"] == 13
    with pytest.raises(ValueError, match="num-subtask-classes"):
        subtask_classifier_state(head, num_classes=7)
    with pytest.raises(KeyError, match="norm"):
        subtask_classifier_state({"hidden": head["hidden"], "out": head["out"]})


def test_find_norm_stats_prefers_the_checkpoint_root(tmp_path):
    from rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg import find_norm_stats

    assert find_norm_stats(tmp_path) is None

    nested = tmp_path / "assets" / "org" / "dataset"
    nested.mkdir(parents=True)
    staged = nested / "norm_stats.json"
    staged.write_text("{}")
    assert find_norm_stats(tmp_path) == staged

    root_stats = tmp_path / "norm_stats.json"
    root_stats.write_text("{}")
    assert find_norm_stats(tmp_path) == root_stats
