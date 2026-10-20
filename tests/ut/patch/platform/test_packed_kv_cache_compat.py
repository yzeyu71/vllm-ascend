# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.platform import patch_kv_cache_utils as cache_patch


@pytest.mark.parametrize("use_fallback", [False, True])
def test_packed_groups_use_matching_eagle_annotation_api(use_fallback):
    config = SimpleNamespace()
    specs = {"layer": object()}
    groups = [object()]

    def annotate(config_arg, specs_arg, groups_arg, *, use_trailing_layer_fallback):
        assert use_trailing_layer_fallback is use_fallback

    predicate = MagicMock(return_value=use_fallback)
    annotator = MagicMock(side_effect=annotate)
    warning = MagicMock()
    upstream = SimpleNamespace(
        _annotate_eagle_groups=annotator,
        _warn_if_unannotated_eagle_mamba=warning,
        _uses_trailing_mtp_layers=predicate,
    )
    with (
        patch.object(cache_patch, "_kv_cache_planning", upstream),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups

    predicate.assert_called_once_with(config)
    assert annotator.call_args.args == (config, specs, groups)
    warning.assert_called_once_with(config, groups)


def test_packed_groups_tolerate_removed_eagle_annotation_api():
    """The packed-group hook must not require the #42309-removed API.

    vLLM #42309 drops ``_annotate_eagle_groups``/``_uses_trailing_mtp_layers``
    from main and annotates EAGLE groups in ``get_kv_cache_configs`` instead.
    """
    config = SimpleNamespace()
    specs = {"layer": object()}
    groups = [object()]
    with (
        # No EAGLE-annotation API, mirroring a post-#42309 main module.
        patch.object(cache_patch, "_kv_cache_planning", SimpleNamespace()),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups
