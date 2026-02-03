import pytest

from pathlib import Path
from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.adapters.base import AdapterContext


@pytest.mark.smoke
def test_malaria_adapter_smoke():
    # This assumes tests are run from repo root and raw/ exists.
    ctx = AdapterContext(raw_root=Path("raw"))
    ad = MalariaAdapter()

    classes = list(ad.iter_classes(ctx))
    assert len(classes) >= 1

    images = list(ad.iter_images(ctx))
    assert len(images) > 0

    anns = list(ad.iter_annotations(ctx))
    assert len(anns) > 0
