import numpy as np

from src.data_generation.generate_images import _add_dent, _base_surface
from src.features.extract_features import extract_features


def test_extract_features_returns_expected_dimensionality(project_config):
    rng = np.random.default_rng(1)
    image = _base_surface(project_config["data"]["image_size"], rng)
    features = extract_features(image, project_config)

    assert features.ndim == 1
    assert len(features) > 0
    assert np.isfinite(features).all()


def test_extract_features_is_deterministic(project_config):
    rng = np.random.default_rng(2)
    image = _base_surface(project_config["data"]["image_size"], rng)

    f1 = extract_features(image, project_config)
    f2 = extract_features(image, project_config)
    np.testing.assert_array_equal(f1, f2)


def test_extract_features_differs_between_good_and_defective(project_config):
    rng = np.random.default_rng(4)
    size = project_config["data"]["image_size"]
    good_image = _base_surface(size, rng)

    from PIL import Image

    img = Image.fromarray(good_image)
    _add_dent(img, rng)
    defective_image = np.array(img)

    f_good = extract_features(good_image, project_config)
    f_defective = extract_features(defective_image, project_config)

    assert not np.array_equal(f_good, f_defective)
    # The last 6 dims are intensity stats -- a dent should measurably lower
    # the minimum pixel value captured there.
    assert f_defective[-4] <= f_good[-4]  # index -4 corresponds to 'min'
