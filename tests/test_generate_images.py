import numpy as np
from PIL import Image

from src.data_generation.generate_images import DEFECT_FUNCS, _add_dent, _base_surface


def test_base_surface_shape_and_range():
    rng = np.random.default_rng(1)
    surface = _base_surface(64, rng)
    assert surface.shape == (64, 64, 3)
    assert surface.dtype == np.uint8
    assert surface.min() >= 0 and surface.max() <= 255


def test_base_surface_is_reproducible_with_same_seed():
    surface1 = _base_surface(64, np.random.default_rng(7))
    surface2 = _base_surface(64, np.random.default_rng(7))
    assert np.array_equal(surface1, surface2)


def test_all_defect_types_modify_the_image():
    """Every defect function must actually change pixel values -- a no-op
    defect function would silently produce mislabeled 'defective' images
    that are pixel-identical to good ones."""
    rng = np.random.default_rng(3)
    for name, fn in DEFECT_FUNCS.items():
        surface = _base_surface(64, rng)
        img = Image.fromarray(surface)
        original = np.array(img).copy()
        fn(img, rng)
        modified = np.array(img)
        assert not np.array_equal(original, modified), f"{name} did not modify the image"


def test_dent_darkens_a_local_region():
    rng = np.random.default_rng(5)
    surface = _base_surface(128, rng)
    img = Image.fromarray(surface)
    before_mean = np.array(img).mean()
    _add_dent(img, rng)
    after_mean = np.array(img).mean()
    # A dent adds a dark blob -- mean brightness should decrease
    assert after_mean < before_mean
