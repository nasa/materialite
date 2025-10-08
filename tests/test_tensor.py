import numpy as np
import pytest
from materialite.tensor import _default_dims, order_dims
from numpy.linalg import inv
from numpy.testing import assert_allclose, assert_array_equal

from materialite import (
    Order2SymmetricTensor,
    Order2Tensor,
    Order4SymmetricTensor,
    Orientation,
    Scalar,
    Vector,
)

NUM_POINTS = 2
NUM_SLIP_SYSTEMS = 4
NUM_TIME_INCREMENTS = 3


def check_mul(tensors1, tensors2, output_type, dims, subscripts):
    t1t2 = tensors1 * tensors2
    assert isinstance(t1t2, output_type)
    assert t1t2.indices_str == dims
    assert_allclose(
        t1t2.components, np.einsum(subscripts, tensors1.components, tensors2.components)
    )


def check_mul_cartesian(tensors1, tensors2, output_type, dims, subscripts):
    t1t2 = tensors1 * tensors2
    assert isinstance(t1t2, output_type)
    assert t1t2.indices_str == dims
    if output_type == Scalar:
        output = t1t2.components
    else:
        output = t1t2.cartesian
    assert_allclose(
        output, np.einsum(subscripts, tensors1.cartesian, tensors2.cartesian)
    )


def check_matmul_cartesian(tensors1, tensors2, output_type, dims, subscripts):
    t1t2 = tensors1 @ tensors2
    assert isinstance(t1t2, output_type)
    assert t1t2.indices_str == dims
    if output_type == Scalar:
        output = t1t2.components
    else:
        output = t1t2.cartesian
    assert_allclose(
        output, np.einsum(subscripts, tensors1.cartesian, tensors2.cartesian)
    )


def check_add(tensors1, tensors2):
    tensors = tensors1 + tensors2
    ref_value = tensors1.cartesian + tensors2.cartesian
    return tensors, ref_value


def check_sub(tensors1, tensors2):
    tensors = tensors1 - tensors2
    ref_value = tensors1.cartesian - tensors2.cartesian
    return tensors, ref_value


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def matrices(rng):
    return rng.random((NUM_POINTS, NUM_SLIP_SYSTEMS, 3, 3))


@pytest.fixture
def all_vectors(rng):
    return rng.random((NUM_POINTS, NUM_SLIP_SYSTEMS, 3))


@pytest.fixture
def vectors(all_vectors):
    return Vector(all_vectors, "ps")


@pytest.fixture
def vectors_p(all_vectors):
    return Vector(all_vectors[:, 0, :], "p")


@pytest.fixture
def vectors_s(all_vectors):
    return Vector(all_vectors[0, :, :], "s")


@pytest.fixture
def vector(all_vectors):
    return Vector(all_vectors[0, 0, :])


@pytest.fixture
def symmetric_matrices(matrices):
    return 0.5 * (matrices + np.einsum("psij -> psji", matrices))


@pytest.fixture
def sym_tensors(symmetric_matrices):
    return Order2SymmetricTensor.from_cartesian(symmetric_matrices, "ps")


@pytest.fixture
def sym_tensors_p(symmetric_matrices):
    return Order2SymmetricTensor.from_cartesian(symmetric_matrices[:, 0, :, :], "p")


@pytest.fixture
def sym_tensors_s(symmetric_matrices):
    return Order2SymmetricTensor.from_cartesian(symmetric_matrices[0, :, :, :], "s")


@pytest.fixture
def sym_tensor(symmetric_matrices):
    return Order2SymmetricTensor.from_cartesian(symmetric_matrices[0, 0, :, :])


@pytest.fixture
def o2_tensors(matrices):
    return Order2Tensor(matrices, "ps")


@pytest.fixture
def o2_tensors_p(matrices):
    return Order2Tensor(matrices[:, 0, :, :], "p")


@pytest.fixture
def o2_tensors_s(matrices):
    return Order2Tensor(matrices[0, :, :, :], "s")


@pytest.fixture
def o2_tensor(matrices):
    return Order2Tensor(matrices[0, 0, :, :])


@pytest.fixture
def scalars():
    data = (
        np.arange(NUM_POINTS * NUM_SLIP_SYSTEMS).reshape((NUM_POINTS, NUM_SLIP_SYSTEMS))
        + 1
    )
    return Scalar(data, "ps")


@pytest.fixture
def scalars_p():
    data = np.arange(NUM_POINTS) + 1
    return Scalar(data, "p")


@pytest.fixture
def scalars_s():
    data = np.arange(NUM_SLIP_SYSTEMS) + 1
    return Scalar(data, "s")


@pytest.fixture
def scalars_pst():
    data = (
        np.arange(NUM_POINTS * NUM_SLIP_SYSTEMS * NUM_TIME_INCREMENTS).reshape(
            (NUM_POINTS, NUM_SLIP_SYSTEMS, NUM_TIME_INCREMENTS)
        )
        + 1
    )
    return Scalar(data, "pst")


@pytest.fixture
def scalars_stp(scalars_pst):
    return scalars_pst.reorder("stp")


@pytest.fixture
def transversely_isotropic_stiffness_matrix():
    return np.array(
        [
            [252.0, 152.0, 152.0, 0.0, 0.0, 0.0],
            [152.0, 252.0, 152.0, 0.0, 0.0, 0.0],
            [152.0, 152.0, 202.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 90.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 90.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        ]
    )


@pytest.fixture
def isotropic_stiffness_matrix():
    return np.array(
        [
            [252.0, 72.0, 72.0, 0.0, 0.0, 0.0],
            [72.0, 252.0, 72.0, 0.0, 0.0, 0.0],
            [72.0, 72.0, 252.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 90.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 90.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
        ]
    )


@pytest.fixture
def mtex_stiffness():
    return np.array(
        [
            [320.5, 68.2, 71.6, 0, 0, 0],
            [68.2, 196.5, 76.8, 0, 0, 0],
            [71.6, 76.8, 233.5, 0, 0, 0],
            [0, 0, 0, 64, 0, 0],
            [0, 0, 0, 0, 77, 0],
            [0, 0, 0, 0, 0, 78.7],
        ]
    )


@pytest.fixture
def stiffness_matrices(
    transversely_isotropic_stiffness_matrix, isotropic_stiffness_matrix
):
    return np.array(
        [
            [transversely_isotropic_stiffness_matrix] * NUM_SLIP_SYSTEMS,
            [
                transversely_isotropic_stiffness_matrix,
                isotropic_stiffness_matrix,
            ]
            * int(NUM_SLIP_SYSTEMS / 2),
        ]
    )


@pytest.fixture
def minor_sym_tensor():
    return Order4SymmetricTensor.from_transverse_isotropic_constants(
        252, 152, 152, 202, 90
    )


@pytest.fixture
def minor_sym_tensor_cubic():
    return Order4SymmetricTensor.from_cubic_constants(C11=252.0, C12=72.0, C44=90.0)


@pytest.fixture
def minor_sym_tensor_iso():
    return Order4SymmetricTensor.from_isotropic_constants(220.0, 90.0)


@pytest.fixture
def minor_sym_tensors_p(minor_sym_tensor):
    return minor_sym_tensor.repeat(NUM_POINTS)


@pytest.fixture
def minor_sym_tensors(stiffness_matrices):
    return Order4SymmetricTensor.from_voigt(stiffness_matrices, "ps")


@pytest.fixture
def scalar():
    return Scalar(0)


def test_error_if_inconsistent_dimensions(matrices, symmetric_matrices):
    with pytest.raises(ValueError):
        _ = Order2Tensor(matrices, "p")

    with pytest.raises(ValueError):
        _ = Order2SymmetricTensor.from_cartesian(symmetric_matrices, "p")

    with pytest.raises(ValueError):
        _ = Scalar(np.arange(8).reshape((2, 4)), "p")

    with pytest.raises(ValueError):
        _ = Order4SymmetricTensor(stiffness_matrices, "p")

    with pytest.raises(ValueError):
        _ = Vector(np.arange(12).reshape((2, 2, 3)), "p")


def test_default_dims():
    assert _default_dims(0) == ""
    assert _default_dims(1) == "p"
    assert _default_dims(2) == "ps"
    assert _default_dims(3) == "pst"
    assert _default_dims(4) == "psta"
    assert _default_dims(5) == "pstab"

    # Ensure no duplicate characters (p and s don't reappear)
    dims = _default_dims(20)
    assert len(set(dims)) == len(dims)  # All unique


def test_order_dims():

    assert order_dims("ab", "bc") == "abc"
    assert order_dims("ac", "b") == "acb"
    assert order_dims("", "abc") == "abc"
    assert order_dims("abc", "") == "abc"
    assert order_dims("ps", "sp") == "ps"
    assert order_dims("ps", "psa") == "psa"
    assert order_dims("abc", "bcd") == "abcd"
    assert order_dims("ab", "cd") == "abcd"
    assert order_dims("abd", "dbc") == "abdc"
    assert order_dims("abcd", "dcba") == "abcd"
    assert order_dims("cab", "abc") == "cab"
    assert order_dims("zyx", "abc") == "zyxabc"
    assert order_dims("a", "bcd") == "abcd"
    assert order_dims("abd", "c") == "abdc"


def test_idempotence(scalars, vectors, sym_tensors, o2_tensors, minor_sym_tensors):
    def check_equal(t1, t2):
        assert isinstance(t1, type(t2))
        assert t1.dims_str == t2.dims_str
        assert t1.indices_str == t2.indices_str
        assert_allclose(t1.components, t2.components)

    scalars1 = Scalar(scalars)
    vectors1 = Vector(vectors)
    o2_tensors1 = Order2Tensor(o2_tensors)
    sym_tensors1 = Order2SymmetricTensor(sym_tensors)
    minor_sym_tensors1 = Order4SymmetricTensor(minor_sym_tensors)

    check_equal(scalars1, scalars)
    check_equal(vectors1, vectors)
    check_equal(o2_tensors1, o2_tensors)
    check_equal(sym_tensors1, sym_tensors)
    check_equal(minor_sym_tensors1, minor_sym_tensors)


def test_init_vector_ps(vectors, all_vectors):
    assert vectors.indices_str == "psj"
    assert_allclose(vectors.components, all_vectors)


def test_init_vector(vector, all_vectors):
    assert vector.indices_str == "j"
    assert_allclose(vector.components, all_vectors[0, 0, :])


def test_init_vector_defaults(all_vectors):
    assert Vector(all_vectors).dims_str == "ps"
    assert Vector(all_vectors[0, :, :]).dims_str == "p"
    assert Vector(all_vectors[0, 0, :]).dims_str == ""


def test_init_symmetric_tensor_ps(sym_tensors, symmetric_matrices):
    assert sym_tensors.indices_str == "psn"
    assert_allclose(sym_tensors.cartesian, symmetric_matrices)


def test_init_symmetric_tensor(sym_tensor, symmetric_matrices):
    assert sym_tensor.indices_str == "n"
    assert_allclose(sym_tensor.cartesian, symmetric_matrices[0, 0, :, :])


def test_init_symmetric_tensor_defaults(symmetric_matrices):
    assert Order2SymmetricTensor.from_cartesian(symmetric_matrices).dims_str == "ps"
    assert (
        Order2SymmetricTensor.from_cartesian(symmetric_matrices[0, ...]).dims_str == "p"
    )
    assert (
        Order2SymmetricTensor.from_cartesian(symmetric_matrices[0, 0, ...]).dims_str
        == ""
    )


def test_init_order2_tensor_ps(o2_tensors, matrices):
    assert o2_tensors.indices_str == "psij"
    assert_allclose(o2_tensors.cartesian, matrices)


def test_init_order2_tensor(o2_tensor, matrices):
    assert o2_tensor.indices_str == "ij"
    assert_allclose(o2_tensor.cartesian, matrices[0, 0, :, :])


def test_init_order2_tensor_defaults(matrices):
    assert Order2Tensor(matrices).dims_str == "ps"
    assert Order2Tensor(matrices[0, ...]).dims_str == "p"
    assert Order2Tensor(matrices[0, 0, ...]).dims_str == ""


def test_init_scalar_ps(scalars):
    assert scalars.indices_str == "ps"
    assert_allclose(
        scalars.components,
        np.arange(NUM_POINTS * NUM_SLIP_SYSTEMS).reshape((NUM_POINTS, NUM_SLIP_SYSTEMS))
        + 1,
    )


def test_init_scalar(scalar):
    assert scalar.indices_str == ""
    assert_allclose(scalar.components, 0)


def test_init_scalar_defaults():
    data = np.arange(NUM_POINTS * NUM_SLIP_SYSTEMS).reshape(
        NUM_POINTS, NUM_SLIP_SYSTEMS
    )
    assert Scalar(data).dims_str == "ps"
    assert Scalar(data[0, :]).dims_str == "p"
    assert Scalar(data[0, 0]).dims_str == ""


def test_init_minor_sym_tensors(
    minor_sym_tensors,
    minor_sym_tensors_p,
    transversely_isotropic_stiffness_matrix,
    stiffness_matrices,
):
    assert minor_sym_tensors_p.indices_str == "pmn"
    assert_allclose(
        minor_sym_tensors_p.voigt, [transversely_isotropic_stiffness_matrix] * 2
    )

    assert minor_sym_tensors.indices_str == "psmn"
    assert_allclose(minor_sym_tensors.voigt, stiffness_matrices)


def test_init_minor_sym_tensor(
    minor_sym_tensor,
    minor_sym_tensor_cubic,
    minor_sym_tensor_iso,
    transversely_isotropic_stiffness_matrix,
    isotropic_stiffness_matrix,
):
    assert minor_sym_tensor.indices_str == "mn"
    mandel = transversely_isotropic_stiffness_matrix.copy()
    mandel[3:, 3:] = mandel[3:, 3:] * 2
    assert_allclose(minor_sym_tensor.voigt, transversely_isotropic_stiffness_matrix)
    assert_allclose(minor_sym_tensor.mandel, mandel)
    assert_allclose(minor_sym_tensor_cubic.voigt, isotropic_stiffness_matrix)
    assert_allclose(minor_sym_tensor_iso.voigt, isotropic_stiffness_matrix)


def test_init_minor_sym_tensor_defaults(stiffness_matrices):
    assert Order4SymmetricTensor.from_voigt(stiffness_matrices).dims_str == "ps"
    assert Order4SymmetricTensor.from_voigt(stiffness_matrices[0, ...]).dims_str == "p"
    assert (
        Order4SymmetricTensor.from_voigt(stiffness_matrices[0, 0, ...]).dims_str == ""
    )


def test_from_list(vectors_p, vectors_s, vectors):
    vectors_p_list = [Vector(c) for c in vectors_p.components]
    vectors_s_list = [Vector(c) for c in vectors_s.components]
    vectors_list = [Vector(c, "s") for c in vectors.components]
    vectors_p_new = vectors_p.from_list(vectors_p_list)
    assert_array_equal(vectors_p_new.components, vectors_p.components)
    assert vectors_p_new.dims_str == "p"
    assert isinstance(vectors_p_new, Vector)

    vectors_s_to_p = Vector.from_list(vectors_s_list)
    assert_array_equal(vectors_s_to_p.components, vectors_s.components)
    assert vectors_s_to_p.dims_str == "p"
    assert isinstance(vectors_s_to_p, Vector)

    vectors_new = Vector.from_list(vectors_list)
    assert_array_equal(vectors_new.components, vectors.components)
    assert vectors_new.dims_str == "ps"


def test_from_stack(vectors_p, vectors):
    vectors_p_list = [Vector(v) for v in vectors_p]
    vectors_p_stacked = Vector.from_stack(vectors_p_list, new_dim="p")
    assert_array_equal(vectors_p_stacked.components, vectors_p.components)
    assert vectors_p_stacked.dims_str == vectors_p.dims_str

    vectors_list = [Vector(v) for v in vectors]
    vectors_stacked = Vector.from_stack(vectors_list, new_dim="p")
    assert_array_equal(vectors_stacked.components, vectors.components)
    assert vectors_stacked.dims_str == vectors.dims_str

    vectors_stacked_t = Vector.from_stack(vectors_list, new_dim="t", axis=1)
    assert_array_equal(np.moveaxis(vectors_stacked_t.components, 1, 0), vectors.components)
    assert vectors_stacked_t.dims_str == "st"


def test_get_item(
    scalars,
    o2_tensors,
    sym_tensors,
    minor_sym_tensors,
    vectors,
):
    for t in [
        scalars,
        o2_tensors,
        sym_tensors,
        minor_sym_tensors,
        vectors,
    ]:
        t0 = t[1:, 1:]
        assert_array_equal(t0.components, t.components[1:, 1:])
        assert t0.dims_str == "ps"

        t1 = t[:, 0]
        assert_array_equal(t1.components, t.components[:, 0])
        assert t1.dims_str == "p"
        assert_array_equal(t1[1].components, t.components[1, 0])

        t2 = t[0, :]
        assert_array_equal(t2.components, t.components[0, :])
        assert t2.dims_str == "s"
        assert_array_equal(t2[1].components, t.components[0, 1])

        t3 = t[0, 0]
        assert_array_equal(t3.components, t.components[0, 0])
        with pytest.raises(ValueError):
            t3[0]
        assert t3.dims_str == ""

        t4 = t[0]
        assert_array_equal(t4.components, t.components[0])
        assert t4.dims_str == "s"

        t5 = t[0, [0, 1]]
        assert_array_equal(t5.components, t.components[0, [0, 1]])
        assert t5.dims_str == "s"

        t6 = t[[0, 1], 0]
        assert_array_equal(t6.components, t.components[[0, 1], 0])
        assert t6.dims_str == "p"

        t7 = t[:, :]
        assert_array_equal(t7.components, t.components)
        assert t7.dims_str == "ps"

        t8 = t[:]
        assert_array_equal(t8.components, t.components)
        assert t8.dims_str == "ps"

        t9 = t[[0, 1]]
        assert_array_equal(t9.components, t.components[[0, 1]])

        t10 = t[0, np.array([0, 1])]
        assert_array_equal(t10.components, t.components[0, [0, 1]])

        t11 = t[np.array([0, 1])]
        assert_array_equal(t11.components, t.components[[0, 1]])

        with pytest.raises(ValueError):
            _ = t[:, :, 0]

        with pytest.raises(ValueError):
            _ = t[0, :, 0]

        with pytest.raises(ValueError):
            _ = t[0, 0, 0]

        with pytest.raises(ValueError):
            _ = t[..., 0]


def test_broadcast_with_different_dims(scalars, scalars_p):
    scalars_sp = scalars.reorder("sp")

    result = scalars + scalars_sp  # "ps" + "sp" -> "ps"

    assert result.dims_str == "ps"
    expected_components = scalars.components + scalars_sp.components.T
    assert_allclose(result.components, expected_components)


def test_repeat_and_writability(vectors):

    repeated = Scalar.zero().repeat((NUM_POINTS, NUM_SLIP_SYSTEMS))
    result = np.zeros((NUM_POINTS, NUM_SLIP_SYSTEMS))

    assert_allclose(repeated.components, result)
    assert repeated.dims_str == "ps"
    assert repeated.components.flags.writeable == True

    # Do not allow an already dimensioned tensor to be repeated
    with pytest.raises(ValueError, match="Cannot repeat.*already has dimensions"):
        vectors.repeat(5)


def test_reorder_dims(scalars_pst):

    reordered = scalars_pst.reorder("tps")

    assert reordered.dims_str == "tps"
    assert_allclose(reordered.components.transpose(1, 2, 0), scalars_pst.components)

    # Test error for mismatched dimension naming
    with pytest.raises(ValueError, match="must contain the same dimensions"):
        scalars_pst.reorder("xyz")


def test_add(
    sym_tensors,
    sym_tensor,
    o2_tensors,
    o2_tensor,
    scalars,
    vectors,
):
    r2_sym = [
        sym_tensors,
        sym_tensor,
        o2_tensors,
        o2_tensor,
    ]
    for t1 in r2_sym:
        for t2 in r2_sym:
            tensors, ref_value = check_add(t1, t2)
            assert_allclose(tensors.cartesian, ref_value)

    assert_allclose(
        (scalars + scalars).components,
        scalars.components + scalars.components,
    )
    assert_allclose(
        (vectors + vectors).components, vectors.components + vectors.components
    )
    with pytest.raises(TypeError):
        _ = sym_tensors + scalars
    with pytest.raises(TypeError):
        _ = o2_tensors + scalars
    with pytest.raises(TypeError):
        _ = scalars + sym_tensors
    with pytest.raises(TypeError):
        _ = vectors + sym_tensors


def test_add_broadcast(
    scalars,
    scalars_p,
    scalars_pst,
    scalars_stp,
    sym_tensors,
    sym_tensors_p,
    o2_tensors,
    o2_tensors_p,
    vectors,
    vectors_p,
):
    result = scalars.components + scalars_p.components[:, np.newaxis]
    assert_allclose((scalars + scalars_p).components, result)
    assert_allclose((scalars_p + scalars).components, result)

    expected = scalars_pst + scalars_stp
    result = scalars_pst.components + scalars_stp.components.transpose(
        2, 0, 1
    )  # "pst" + "stp" -> "pst"
    assert_allclose(expected.components, result)
    assert expected.dims_str == "pst"

    result_sym = sym_tensors.cartesian + sym_tensors_p.cartesian[:, np.newaxis, :, :]
    assert_allclose((sym_tensors + sym_tensors_p).cartesian, result_sym)

    result_r2 = o2_tensors.components + o2_tensors_p.components[:, np.newaxis, :, :]
    assert_allclose((o2_tensors + o2_tensors_p).cartesian, result_r2)

    result_r2_sym = o2_tensors.components + sym_tensors_p.cartesian[:, np.newaxis, :, :]
    assert_allclose((o2_tensors + sym_tensors_p).components, result_r2_sym)

    result_sym_r2 = sym_tensors.cartesian + o2_tensors_p.components[:, np.newaxis, :, :]
    assert_allclose((sym_tensors + o2_tensors_p).components, result_sym_r2)

    result_vector = vectors.components + vectors_p.components[:, np.newaxis, :]
    assert_allclose((vectors + vectors_p).components, result_vector)


def test_sub(
    sym_tensors,
    sym_tensor,
    o2_tensors,
    o2_tensor,
    scalars,
    vectors,
):
    r2_sym = [
        sym_tensors,
        sym_tensor,
        o2_tensors,
        o2_tensor,
    ]
    for t1 in r2_sym:
        for t2 in r2_sym:
            tensors, ref_value = check_sub(t1, t2)
            assert_allclose(tensors.cartesian, ref_value, atol=1e-14)

    assert_allclose((scalars - scalars).components, 0.0, atol=1e-14)
    assert_allclose((vectors - vectors).components, 0.0, atol=1e-14)
    with pytest.raises(TypeError):
        _ = sym_tensors - scalars
    with pytest.raises(TypeError):
        _ = o2_tensors - scalars
    with pytest.raises(TypeError):
        _ = scalars - sym_tensors
    with pytest.raises(TypeError):
        _ = vectors - sym_tensors


def test_sub_broadcast(
    scalars,
    scalars_p,
    sym_tensors,
    sym_tensors_p,
    o2_tensors,
    o2_tensors_p,
    vectors,
    vectors_p,
):
    result = scalars.components - scalars_p.components[:, np.newaxis]
    assert_allclose((scalars - scalars_p).components, result)
    assert_allclose((scalars_p - scalars).components, -result)

    result_sym = sym_tensors.cartesian - sym_tensors_p.cartesian[:, np.newaxis, :, :]
    assert_allclose((sym_tensors - sym_tensors_p).cartesian, result_sym, atol=1e-14)

    result_r2 = o2_tensors.components - o2_tensors_p.components[:, np.newaxis, :, :]
    assert_allclose((o2_tensors - o2_tensors_p).cartesian, result_r2, atol=1e-14)

    result_r2_sym = o2_tensors.components - sym_tensors_p.cartesian[:, np.newaxis, :, :]
    assert_allclose((o2_tensors - sym_tensors_p).components, result_r2_sym, atol=1e-14)

    result_sym_r2 = sym_tensors.cartesian - o2_tensors_p.components[:, np.newaxis, :, :]
    assert_allclose((sym_tensors - o2_tensors_p).components, result_sym_r2, atol=1e-14)

    result_vector = vectors.components + vectors_p.components[:, np.newaxis, :]
    assert_allclose((vectors + vectors_p).components, result_vector, atol=1e-14)


def test_mul_vector_vector(vectors, vectors_p, vectors_s, vector):
    check_mul(vectors, vectors, Scalar, "ps", "psi, psi -> ps")
    check_mul(vectors, vectors_p, Scalar, "ps", "psi, pi -> ps")
    check_mul(vectors, vectors_s, Scalar, "ps", "psi, si -> ps")
    check_mul(vectors_s, vectors, Scalar, "ps", "si, psi -> ps")
    check_mul(vectors, vector, Scalar, "ps", "psi, i -> ps")


def test_mul_scalar_scalar(scalars, scalars_p, scalars_s, scalar):
    check_mul(scalars, scalars, Scalar, "ps", "ps, ps -> ps")
    check_mul(scalars, scalars_p, Scalar, "ps", "ps, p -> ps")
    check_mul(scalars, scalars_s, Scalar, "ps", "ps, s -> ps")
    check_mul(scalars_s, scalars, Scalar, "ps", "s, ps -> ps")
    check_mul(scalars, scalar, Scalar, "ps", "ps, -> ps")


def test_mul_symmetric_symmetric(sym_tensors, sym_tensors_p, sym_tensors_s, sym_tensor):
    check_mul(sym_tensors, sym_tensors, Scalar, "ps", "psn, psn -> ps")
    check_mul(sym_tensors, sym_tensors_p, Scalar, "ps", "psn, pn -> ps")
    check_mul(sym_tensors, sym_tensors_s, Scalar, "ps", "psn, sn -> ps")
    check_mul(sym_tensors_s, sym_tensors, Scalar, "ps", "sn, psn -> ps")
    check_mul(sym_tensors, sym_tensor, Scalar, "ps", "psn, n -> ps")


def test_mul_symmetric_order2(
    sym_tensors,
    o2_tensors,
    o2_tensors_p,
    o2_tensors_s,
    o2_tensor,
):
    check_mul_cartesian(sym_tensors, o2_tensors, Scalar, "ps", "psij, psij -> ps")
    check_mul_cartesian(sym_tensors, o2_tensors_p, Scalar, "ps", "psij, pij -> ps")
    check_mul_cartesian(sym_tensors, o2_tensors_s, Scalar, "ps", "psij, sij -> ps")
    check_mul_cartesian(o2_tensors_s, sym_tensors, Scalar, "ps", "sij, psij -> ps")
    check_mul_cartesian(sym_tensors, o2_tensor, Scalar, "ps", "psij, ij -> ps")


def test_mul_scalar_vector(vectors, vectors_p, vectors_s, scalars):
    check_mul(scalars, vectors, Vector, "psj", "ps, psj -> psj")
    check_mul(scalars, vectors_p, Vector, "psj", "ps, pj -> psj")
    check_mul(scalars, vectors_s, Vector, "psj", "ps, sj -> psj")


def test_mul_vector_scalar(vectors, vectors_p, vectors_s, scalars):
    check_mul(vectors, scalars, Vector, "psj", "psj, ps -> psj")
    check_mul(vectors_p, scalars, Vector, "psj", "pj, ps -> psj")
    check_mul(vectors_s, scalars, Vector, "psj", "sj, ps -> psj")


def test_mul_symmetric_scalar(sym_tensors, sym_tensors_p, sym_tensor, scalars):
    check_mul(sym_tensors, scalars, Order2SymmetricTensor, "psn", "psn, ps -> psn")
    check_mul(sym_tensors_p, scalars, Order2SymmetricTensor, "psn", "pn, ps -> psn")
    check_mul(sym_tensor, scalars, Order2SymmetricTensor, "psn", "n, ps -> psn")


def test_mul_order2_order2(o2_tensors, o2_tensors_p, o2_tensors_s, o2_tensor):
    check_mul(o2_tensors, o2_tensors, Scalar, "ps", "psij, psij -> ps")
    check_mul(o2_tensors, o2_tensors_p, Scalar, "ps", "psij, pij -> ps")
    check_mul(o2_tensors, o2_tensors_s, Scalar, "ps", "psij, sij -> ps")
    check_mul(o2_tensors_s, o2_tensors, Scalar, "ps", "sij, psij -> ps")
    check_mul(o2_tensors, o2_tensor, Scalar, "ps", "psij, ij -> ps")


def test_mul_order2_scalar(o2_tensors, o2_tensors_p, o2_tensor, scalars):
    check_mul(o2_tensors, scalars, Order2Tensor, "psij", "psij, ps -> psij")
    check_mul(o2_tensors_p, scalars, Order2Tensor, "psij", "pij, ps -> psij")
    check_mul(o2_tensor, scalars, Order2Tensor, "psij", "ij, ps -> psij")


def test_mul_minor_sym(
    scalars,
    scalar,
    o2_tensor,
    sym_tensor,
    minor_sym_tensors,
    minor_sym_tensors_p,
    minor_sym_tensor,
):
    with pytest.raises(TypeError):
        _ = o2_tensor * minor_sym_tensor
    with pytest.raises(TypeError):
        _ = minor_sym_tensor * o2_tensor
    with pytest.raises(TypeError):
        _ = sym_tensor * minor_sym_tensor
    with pytest.raises(TypeError):
        _ = minor_sym_tensor * sym_tensor

    check_mul(
        minor_sym_tensors, scalars, Order4SymmetricTensor, "psmn", "psmn, ps -> psmn"
    )
    check_mul(
        scalars, minor_sym_tensors, Order4SymmetricTensor, "psmn", "ps, psmn -> psmn"
    )
    check_mul(minor_sym_tensors, scalar, Order4SymmetricTensor, "psmn", "psmn, -> psmn")

    check_mul_cartesian(
        minor_sym_tensors_p, minor_sym_tensors_p, Scalar, "p", "pijkl, pijkl -> p"
    )
    check_mul_cartesian(
        minor_sym_tensors_p, minor_sym_tensor, Scalar, "p", "pijkl, ijkl -> p"
    )
    check_mul_cartesian(
        minor_sym_tensors, minor_sym_tensors, Scalar, "ps", "psijkl, psijkl -> ps"
    )


def test_matmul_minor_sym(
    minor_sym_tensors_p,
    minor_sym_tensor,
    sym_tensors,
    sym_tensors_p,
    sym_tensor,
    o2_tensors,
    o2_tensors_p,
    o2_tensor,
):
    check_matmul_cartesian(
        minor_sym_tensors_p,
        minor_sym_tensor,
        Order4SymmetricTensor,
        "pmn",
        "pijqr, qrkl -> pijkl",
    )
    assert_allclose(
        (minor_sym_tensors_p @ minor_sym_tensor).components,
        (minor_sym_tensor @ minor_sym_tensors_p).components,
    )

    check_matmul_cartesian(
        minor_sym_tensors_p,
        sym_tensors,
        Order2SymmetricTensor,
        "psn",
        "pijkl, pskl -> psij",
    )
    check_matmul_cartesian(
        minor_sym_tensors_p,
        sym_tensors_p,
        Order2SymmetricTensor,
        "pn",
        "pijkl, pkl -> pij",
    )
    check_matmul_cartesian(
        minor_sym_tensors_p, sym_tensor, Order2SymmetricTensor, "pn", "pijkl, kl -> pij"
    )

    check_matmul_cartesian(
        minor_sym_tensors_p,
        o2_tensors,
        Order2SymmetricTensor,
        "psn",
        "pijkl, pskl -> psij",
    )
    check_matmul_cartesian(
        minor_sym_tensors_p,
        o2_tensors_p,
        Order2SymmetricTensor,
        "pn",
        "pijkl, pkl -> pij",
    )
    check_matmul_cartesian(
        minor_sym_tensors_p, o2_tensor, Order2SymmetricTensor, "pn", "pijkl, kl -> pij"
    )


def test_matmul_r2_sym(
    o2_tensors,
    o2_tensor,
    sym_tensors,
    sym_tensor,
    vectors,
    vector,
):
    check_matmul_cartesian(
        o2_tensors, o2_tensors, Order2Tensor, "psij", "psik, pskj -> psij"
    )
    check_matmul_cartesian(
        o2_tensors, o2_tensor, Order2Tensor, "psij", "psik, kj -> psij"
    )
    check_matmul_cartesian(
        o2_tensors, sym_tensors, Order2Tensor, "psij", "psik, pskj -> psij"
    )
    check_matmul_cartesian(o2_tensors, vectors, Vector, "psj", "psij, psj -> psi")
    check_matmul_cartesian(o2_tensors, vector, Vector, "psj", "psij, j -> psi")
    check_matmul_cartesian(
        sym_tensors, sym_tensors, Order2SymmetricTensor, "psn", "psik, pskj -> psij"
    )
    check_matmul_cartesian(
        sym_tensors, sym_tensor, Order2Tensor, "psij", "psik, kj -> psij"
    )
    check_matmul_cartesian(sym_tensors, vectors, Vector, "psj", "psij, psj -> psi")
    check_matmul_cartesian(sym_tensors, vector, Vector, "psj", "psij, j -> psi")


def test_scalar_mul_div(
    scalars,
    sym_tensors,
    o2_tensors,
    vectors,
):
    c = 2.0
    assert_allclose((scalars * c).components, scalars.components * c)
    assert_allclose((scalars / c).components, scalars.components / c)
    assert_allclose((sym_tensors * c).components, sym_tensors.components * c)
    assert_allclose((sym_tensors / c).components, sym_tensors.components / c)
    assert_allclose((o2_tensors * c).components, o2_tensors.components * c)
    assert_allclose((o2_tensors / c).components, o2_tensors.components / c)
    assert_allclose((vectors * c).components, vectors.components * c)
    assert_allclose((vectors / c).components, vectors.components / c)


def test_div(
    sym_tensors,
    o2_tensors,
    vectors,
    scalars,
):

    ones_scalar = np.ones((NUM_POINTS, NUM_SLIP_SYSTEMS))
    assert_allclose((scalars / scalars).components, ones_scalar)
    assert_allclose(
        (sym_tensors / scalars).components,
        sym_tensors.components / scalars.components[:, :, np.newaxis],
    )
    assert_allclose(
        (o2_tensors / scalars).components,
        o2_tensors.components / scalars.components[:, :, np.newaxis, np.newaxis],
    )
    assert_allclose(
        (vectors / scalars).components,
        vectors.components / scalars.components[:, :, np.newaxis],
    )


def test_mean(
    scalars,
    vectors,
    vectors_p,
    o2_tensors,
    sym_tensors,
    minor_sym_tensors,
    o2_tensors_s,
    sym_tensors_s,
    minor_sym_tensor,
):
    scalars_mn_s = np.mean(scalars.components, axis=1)
    vectors_mn_p = np.mean(vectors.components, axis=0)
    vectors_mn_s = np.mean(vectors.components, axis=1)
    vectors_p_mn = np.mean(vectors_p.components, axis=0)
    r2_mn_s = np.mean(o2_tensors.components, axis=1)
    sym_mn_s = np.mean(sym_tensors.cartesian, axis=1)
    minor_sym_mn_s = np.mean(minor_sym_tensors.cartesian, axis=1)

    assert_allclose(scalars.mean("s").components, scalars_mn_s)
    assert_allclose(vectors.mean("s").components, vectors_mn_s)
    assert_allclose(o2_tensors.mean("s").components, r2_mn_s)
    assert_allclose(sym_tensors.mean("s").cartesian, sym_mn_s)
    assert_allclose(minor_sym_tensors.mean("s").cartesian, minor_sym_mn_s)

    assert_allclose(vectors.mean("p").components, vectors_mn_p)
    assert_allclose(vectors.mean("s").components, vectors_mn_s)
    assert_allclose(vectors_p.mean().components, vectors_p_mn)

    with pytest.raises(ValueError):
        _ = o2_tensors_s.mean("p")
    with pytest.raises(ValueError):
        _ = sym_tensors_s.mean("p")
    with pytest.raises(ValueError):
        _ = minor_sym_tensor.mean("p")


def test_sum(vectors, vectors_p, vectors_s):
    assert_allclose(vectors.sum().components, np.sum(vectors.components, axis=0))
    assert_allclose(vectors.sum("p").components, np.sum(vectors.components, axis=0))
    assert_allclose(vectors_s.sum().components, np.sum(vectors_s.components, axis=0))
    assert_allclose(vectors_p.sum().components, np.sum(vectors_p.components, axis=0))


def test_neg(scalars, vectors, o2_tensors, sym_tensors, minor_sym_tensors):
    assert_allclose((-scalars).components, -(scalars.components))
    assert_allclose((-vectors).components, -(vectors.components))
    assert_allclose((-o2_tensors).components, -(o2_tensors.components))
    assert_allclose((-sym_tensors).components, -(sym_tensors.components))
    assert_allclose((-minor_sym_tensors).components, -(minor_sym_tensors.components))


def test_shape(vectors, vectors_p, vectors_s, vector):
    assert vectors.shape == vectors.components.shape[:2]
    assert vectors_p.shape == vectors_p.components.shape[:1]
    assert vectors_s.shape == vectors_s.components.shape[:1]
    assert vector.shape == ()


def test_abs(scalars):
    assert_allclose(scalars.abs.components, np.abs(scalars.components))


def test_sqrt(scalars):
    assert_allclose(scalars.sqrt.components, np.sqrt(scalars.components))


def test_cosh(scalars):
    assert_allclose(scalars.cosh.components, np.cosh(scalars.components))


def test_max(scalars):
    assert_allclose(scalars.max().components, np.max(scalars.components, axis=0))
    assert_allclose(scalars.max("s").components, np.max(scalars.components, axis=1))


def test_pow(scalars):
    assert_allclose((scalars**2.0).components, scalars.components**2.0)
    assert_allclose(
        (scalars**scalars).components,
        scalars.components**scalars.components,
    )


def test_apply(scalars):
    assert_allclose(scalars.apply(np.cos).components, np.cos(scalars.components))


def test_from_tensor_product(vectors, vectors_p, vectors_s, vector):
    vall_vall = np.einsum("psi, psj -> psij", vectors.components, vectors.components)
    vp_vs = np.einsum("pi, sj -> psij", vectors_p.components, vectors_s.components)
    vp_v = np.einsum("pi, j -> pij", vectors_p.components, vector.components)
    v_vp = np.einsum("i, pj -> pij", vector.components, vectors_p.components)
    v_v = np.outer(vector.components, vector.components)
    assert_allclose(
        Order2Tensor.from_tensor_product(vectors, vectors).components, vall_vall
    )
    assert_allclose(
        Order2Tensor.from_tensor_product(vectors_p, vectors_s).components, vp_vs
    )
    assert_allclose(
        Order2Tensor.from_tensor_product(vectors_p, vector).components, vp_v
    )
    assert_allclose(
        Order2Tensor.from_tensor_product(vector, vectors_p).components, v_vp
    )
    assert_allclose(Order2Tensor.from_tensor_product(vector, vector).components, v_v)


def test_cross_product(vectors, vectors_p, vectors_s, vector):
    vp_vs = np.cross(vectors_p.components[:, np.newaxis, :], vectors_s.components[np.newaxis, :, :])
    v_vall = np.cross(vector.components, vectors.components)
    assert_allclose(vectors.cross(vectors).components, np.zeros(vectors.components.shape), atol=1.0e-14)
    assert_allclose(vectors_p.cross(vectors_s).components, vp_vs, atol=1.0e-14)
    assert_allclose(vector.cross(vectors).components, v_vall, atol=1.0e-14)
    assert vectors_p.cross(vectors_s).dims_str == "ps"


def test_outer_vectors(vectors, vectors_p, vectors_s, vector):
    vall_vall = np.einsum("psi, psj -> psij", vectors.components, vectors.components)
    vp_vs = np.einsum("pi, sj -> psij", vectors_p.components, vectors_s.components)
    vp_v = np.einsum("pi, j -> pij", vectors_p.components, vector.components)
    v_vp = np.einsum("i, pj -> pij", vector.components, vectors_p.components)
    v_v = np.einsum("i, j -> ij", vector.components, vector.components)

    assert_allclose(vectors.outer(vectors).components, vall_vall)
    assert_allclose(vectors_p.outer(vectors_s).components, vp_vs)
    assert_allclose(vectors_p.outer(vector).components, vp_v)
    assert_allclose(vector.outer(vectors_p).components, v_vp)
    assert_allclose(vector.outer(vector).components, v_v)


def test_outer_sym(sym_tensors, sym_tensors_p, sym_tensors_s, sym_tensor):
    vall_vall = np.einsum(
        "psi, psj -> psij", sym_tensors.components, sym_tensors.components
    )
    vp_vs = np.einsum(
        "pi, sj -> psij", sym_tensors_p.components, sym_tensors_s.components
    )
    vp_v = np.einsum("pi, j -> pij", sym_tensors_p.components, sym_tensor.components)
    v_vp = np.einsum("i, pj -> pij", sym_tensor.components, sym_tensors_p.components)

    assert_allclose(sym_tensors.outer(sym_tensors).components, vall_vall)
    assert_allclose(sym_tensors_p.outer(sym_tensors_s).components, vp_vs)
    assert_allclose(sym_tensors_p.outer(sym_tensor).components, vp_v)
    assert_allclose(sym_tensor.outer(sym_tensors_p).components, v_vp)


def test_zero():
    zeros = np.zeros((3, 3))
    assert_allclose(Scalar.zero().components, 0)
    assert_allclose(Vector.zero().components, np.zeros(3))
    assert_allclose(Order2Tensor.zero().components, zeros)
    assert_allclose(Order2SymmetricTensor.zero().cartesian, zeros)
    assert_allclose(Order4SymmetricTensor.zero().components, np.zeros((6, 6)))


def test_repeat():
    p_shape = 2
    s_shape = 3
    ps_shape = (p_shape, s_shape)

    assert_allclose(Scalar.zero().repeat(ps_shape).components, np.zeros(ps_shape))
    assert_allclose(Scalar.zero().repeat(p_shape).components, np.zeros(p_shape))
    assert_allclose(Scalar.zero().repeat(s_shape, "s").components, np.zeros(s_shape))

    assert_allclose(Vector.zero().repeat(ps_shape).components, np.zeros((*ps_shape, 3)))
    assert_allclose(Vector.zero().repeat(p_shape).components, np.zeros((p_shape, 3)))
    assert_allclose(
        Vector.zero().repeat(s_shape, "s").components, np.zeros((s_shape, 3))
    )

    assert_allclose(
        Order2Tensor.zero().repeat(ps_shape).components, np.zeros((*ps_shape, 3, 3))
    )
    assert_allclose(
        Order2Tensor.zero().repeat(p_shape).components, np.zeros((p_shape, 3, 3))
    )
    assert_allclose(
        Order2Tensor.zero().repeat(s_shape, "s").components, np.zeros((s_shape, 3, 3))
    )

    assert_allclose(
        Order2SymmetricTensor.zero().repeat(ps_shape).components,
        np.zeros((*ps_shape, 6)),
    )
    assert_allclose(
        Order2SymmetricTensor.zero().repeat(p_shape).components, np.zeros((p_shape, 6))
    )
    assert_allclose(
        Order2SymmetricTensor.zero().repeat(s_shape, "s").components,
        np.zeros((s_shape, 6)),
    )

    assert_allclose(
        Order4SymmetricTensor.zero().repeat(ps_shape).components,
        np.zeros((*ps_shape, 6, 6)),
    )
    assert_allclose(
        Order4SymmetricTensor.zero().repeat(p_shape).components,
        np.zeros((p_shape, 6, 6)),
    )
    assert_allclose(
        Order4SymmetricTensor.zero().repeat(s_shape, "s").components,
        np.zeros((s_shape, 6, 6)),
    )

    assert Scalar.zero().repeat(s_shape, "s").dims_str == "s"
    assert Vector.zero().repeat(s_shape, "s").dims_str == "s"
    assert Order2Tensor.zero().repeat(s_shape, "s").dims_str == "s"
    assert Order2SymmetricTensor.zero().repeat(s_shape, "s").dims_str == "s"
    assert Order4SymmetricTensor.zero().repeat(s_shape, "s").dims_str == "s"


def test_repeat_when_tensor_already_has_dimensions():
    with pytest.raises(ValueError, match="Cannot repeat.*already has dimensions"):
        Scalar.zero().repeat(2).repeat(2)
    with pytest.raises(ValueError, match="Cannot repeat.*already has dimensions"):
        Vector.zero().repeat(2).repeat(2)
    with pytest.raises(ValueError, match="Cannot repeat.*already has dimensions"):
        Order2Tensor.zero().repeat(2).repeat(2)
    with pytest.raises(ValueError, match="Cannot repeat.*already has dimensions"):
        Order2SymmetricTensor.zero().repeat(2).repeat(2)
    with pytest.raises(ValueError, match="Cannot repeat.*already has dimensions"):
        Order4SymmetricTensor.zero().repeat(2).repeat(2)


def test_identity():
    i3 = np.identity(3)
    i6 = np.identity(6)
    assert_allclose(Order2Tensor.identity().components, i3)
    assert_allclose(Order2SymmetricTensor.identity().cartesian, i3)
    assert_allclose(Order4SymmetricTensor.identity().components, i6)


def test_random_unit(rng):
    expected_vector = np.array([-0.68014832, 0.60367164, -0.41590722])
    expected_vectors = np.array(
        [[-0.32868482, -0.09555077, -0.93959371], [-0.87883624, 0.41692759, 0.23198761]]
    )
    assert_allclose(Vector.random_unit(rng=rng).components, expected_vector)
    assert_allclose(Vector.random_unit(2, rng=rng).components, expected_vectors)


def test_unit(vectors):
    norm = lambda x: np.sqrt(np.einsum("psi, psi -> ps", x, x))
    actual_norm = (vectors / vectors.norm).components
    expected_norm = vectors.components / norm(vectors.components)[:, :, np.newaxis]
    assert_allclose(actual_norm, expected_norm)


def test_inverse(o2_tensors, o2_tensor, sym_tensors, sym_tensor, minor_sym_tensors):
    def invert(x):
        inverses = np.zeros(x.shape)
        for i, xi in enumerate(x):
            for j, xij in enumerate(xi):
                inverses[i, j, :, :] = inv(xij)
        return inverses

    r2_inverses = invert(o2_tensors.components)
    r2_inverse = inv(o2_tensor.components)
    sym_inverses = invert(sym_tensors.cartesian)
    sym_inverse = inv(sym_tensor.cartesian)
    minor_inverses = invert(minor_sym_tensors.components)

    assert_allclose(o2_tensors.inverse.components, r2_inverses)
    assert_allclose(o2_tensor.inverse.components, r2_inverse)
    assert_allclose(sym_tensors.inverse.cartesian, sym_inverses)
    assert_allclose(sym_tensor.inverse.cartesian, sym_inverse)
    assert_allclose(minor_sym_tensors.inverse.components, minor_inverses)


def test_transpose(o2_tensors, o2_tensor, sym_tensors, sym_tensor):
    assert_allclose(
        o2_tensors.T.components, np.einsum("psij -> psji", o2_tensors.components)
    )
    assert_allclose(o2_tensor.T.components, np.einsum("ij -> ji", o2_tensor.components))
    assert_allclose(sym_tensors.T.components, sym_tensors.components)
    assert_allclose(sym_tensor.T.components, sym_tensor.components)


def test_sym(o2_tensors, sym_tensors):
    assert_allclose(o2_tensors.sym.components, sym_tensors.components)


def test_norm(o2_tensors, sym_tensors, minor_sym_tensors):
    norm = lambda x: np.sqrt(np.einsum("psij, psij -> ps", x, x))
    norm4 = lambda x: np.sqrt(np.einsum("psijkl, psijkl -> ps", x, x))
    r2_norm = o2_tensors.norm
    sym_norm = sym_tensors.norm
    minor_sym_norm = minor_sym_tensors.norm
    assert isinstance(r2_norm, Scalar)
    assert isinstance(sym_norm, Scalar)
    assert isinstance(minor_sym_norm, Scalar)
    assert_allclose(r2_norm.components, norm(o2_tensors.components))
    assert_allclose(sym_norm.components, norm(sym_tensors.cartesian))
    assert_allclose(minor_sym_norm.components, norm4(minor_sym_tensors.cartesian))


def test_trace(o2_tensors, sym_tensors):
    def trace(x):
        traces = np.zeros(x.shape[:2])
        for i, xi in enumerate(x):
            for j, xij in enumerate(xi):
                traces[i, j] = xij[0, 0] + xij[1, 1] + xij[2, 2]
        return traces

    o2_traces = o2_tensors.trace
    sym_traces = sym_tensors.trace
    assert isinstance(o2_traces, Scalar)
    assert isinstance(sym_traces, Scalar)
    assert_allclose(o2_traces.components, trace(o2_tensors.components))
    assert_allclose(sym_traces.components, trace(sym_tensors.cartesian))


def test_dev(o2_tensors, sym_tensors):
    def dev(x):
        identity = np.identity(3)
        dev_part = np.zeros(x.shape)
        for i, xi in enumerate(x):
            for j, xij in enumerate(xi):
                dev_part[i, j, :, :] = xij - np.trace(xij) * identity / 3.0
        return dev_part

    r2_dev = o2_tensors.dev
    sym_dev = sym_tensors.dev
    assert isinstance(r2_dev, Order2Tensor)
    assert isinstance(sym_dev, Order2SymmetricTensor)
    assert_allclose(r2_dev.components, dev(o2_tensors.components))
    assert_allclose(sym_dev.cartesian, dev(sym_tensors.cartesian))


def test_symmetric_tensor_voigt_mandel(sym_tensors, symmetric_matrices):
    s = symmetric_matrices
    strain_voigt = np.array(
        [
            s[..., 0, 0],
            s[..., 1, 1],
            s[..., 2, 2],
            2 * s[..., 1, 2],
            2 * s[..., 0, 2],
            2 * s[..., 0, 1],
        ]
    )
    strain_voigt = np.moveaxis(strain_voigt, 0, -1)
    stress_voigt = np.array(
        [
            s[..., 0, 0],
            s[..., 1, 1],
            s[..., 2, 2],
            s[..., 1, 2],
            s[..., 0, 2],
            s[..., 0, 1],
        ]
    )
    stress_voigt = np.moveaxis(stress_voigt, 0, -1)
    mandel = stress_voigt.copy()
    mandel[:, :, 3:] = mandel[:, :, 3:] * np.sqrt(2)

    assert_allclose(sym_tensors.strain_voigt, strain_voigt)
    assert_allclose(sym_tensors.stress_voigt, stress_voigt)
    assert_allclose(sym_tensors.mandel, mandel)

    assert_allclose(
        Order2SymmetricTensor.from_strain_voigt(strain_voigt, "ps").components,
        sym_tensors.components,
    )
    assert_allclose(
        Order2SymmetricTensor.from_stress_voigt(stress_voigt, "ps").components,
        sym_tensors.components,
    )


def test_rotations_identity(o2_tensors, sym_tensors, vectors, minor_sym_tensors):
    o_identity = Orientation.identity()
    r2_crystal = o2_tensors.to_crystal_frame(o_identity)
    r2_specimen = o2_tensors.to_specimen_frame(o_identity)
    sym_crystal = sym_tensors.to_crystal_frame(o_identity)
    sym_specimen = sym_tensors.to_specimen_frame(o_identity)
    vec_crystal = vectors.to_crystal_frame(o_identity)
    vec_specimen = vectors.to_specimen_frame(o_identity)
    minor_specimen = minor_sym_tensors.to_specimen_frame(o_identity)
    assert_allclose(r2_crystal.components, o2_tensors.components)
    assert_allclose(r2_specimen.components, o2_tensors.components)
    assert_allclose(sym_crystal.components, sym_tensors.components)
    assert_allclose(sym_specimen.components, sym_tensors.components)
    assert_allclose(vec_crystal.components, vectors.components)
    assert_allclose(vec_specimen.components, vectors.components)
    assert_allclose(minor_specimen.components, minor_sym_tensors.components)


@pytest.mark.parametrize("num_orientations", [1, 2])
def test_rotations_with_inverse(
    o2_tensors,
    sym_tensors,
    vectors,
    minor_sym_tensors,
    rng,
    num_orientations,
    o2_tensor,
    sym_tensor,
    vector,
    minor_sym_tensor,
):
    tensors = [
        o2_tensors,
        sym_tensors,
        vectors,
        minor_sym_tensors,
        o2_tensor,
        sym_tensor,
        vector,
        minor_sym_tensor,
    ]
    o = Orientation.random(num_orientations, rng=rng)
    for t in tensors:
        if t.dims_str == "" and num_orientations == 2:
            expected = [t.components] * 2
        else:
            expected = t.components
        t_rotated = t.to_crystal_frame(o).to_specimen_frame(o)
        assert_allclose(t_rotated.components, expected, atol=1e-12)


def test_rotations_with_inverse_s_dimension(
    o2_tensors_p,
    sym_tensors_p,
    vectors_p,
    minor_sym_tensors_p,
    o2_tensors,
    sym_tensors,
    vectors,
    minor_sym_tensors,
):
    o = Orientation.random(shape=(NUM_POINTS, NUM_SLIP_SYSTEMS))
    tensors_p = [o2_tensors_p, sym_tensors_p, vectors_p, minor_sym_tensors_p]
    for t in tensors_p:
        expected = np.repeat(t.components[:, np.newaxis, ...], NUM_SLIP_SYSTEMS, axis=1)
        t_rotated = t.to_crystal_frame(o).to_specimen_frame(o)
        assert_allclose(t_rotated.components, expected, atol=1e-12)

    tensors = [o2_tensors, sym_tensors, vectors, minor_sym_tensors]
    for t in tensors:
        expected = t.components
        t_rotated = t.to_crystal_frame(o).to_specimen_frame(o)
        assert_allclose(t_rotated.components, expected, atol=1e-12)


def test_rotations_stress_strain(sym_tensors, minor_sym_tensors, rng):
    o = Orientation.random(2, rng=rng)
    sym_rotations = (
        minor_sym_tensors @ sym_tensors.to_crystal_frame(o)
    ).to_specimen_frame(o)
    minor_sym_rotations = minor_sym_tensors.to_specimen_frame(o) @ sym_tensors
    assert_allclose(sym_rotations.components, minor_sym_rotations.components)
    sym_rotations2 = (
        minor_sym_tensors @ sym_tensors.to_specimen_frame(o)
    ).to_crystal_frame(o)
    minor_sym_rotations2 = minor_sym_tensors.to_crystal_frame(o) @ sym_tensors
    assert_allclose(sym_rotations2.components, minor_sym_rotations2.components)


def test_directional_moduli(mtex_stiffness):
    num_repeats = 2
    stiffnesses = Order4SymmetricTensor.from_voigt(mtex_stiffness).repeat(num_repeats)
    direction = Vector(np.array([1.0, 1.0, 1.0]))
    assert_allclose(
        stiffnesses.directional_modulus(direction).components,
        np.array([183.0467131980609] * num_repeats),
    )


def test_directional_moduli_multiple_directions(mtex_stiffness):
    num_repeats = 2
    stiffnesses = Order4SymmetricTensor.from_voigt(mtex_stiffness).repeat(num_repeats)
    directions = Vector(np.array([[1.0, 1.0, 1.0]] * num_repeats), "s")
    expected_moduli = np.array([[183.0467131980609] * num_repeats] * num_repeats)
    assert_allclose(
        stiffnesses.directional_modulus(directions).components,
        expected_moduli,
    )


def test_directional_bulk_moduli(isotropic_stiffness_matrix):
    num_repeats = 2
    stiffnesses = Order4SymmetricTensor.from_voigt(isotropic_stiffness_matrix).repeat(
        num_repeats
    )
    direction = Vector(np.array([1.0, 1.0, 1.0]))
    assert_allclose(
        stiffnesses.directional_bulk_modulus(direction).components,
        np.array([132.0] * num_repeats),
    )


def test_directional_shear_moduli(isotropic_stiffness_matrix):
    num_repeats = 2
    stiffnesses = Order4SymmetricTensor.from_voigt(isotropic_stiffness_matrix).repeat(
        num_repeats
    )
    normal = Vector(np.array([1.0, 1.0, 1.0]))
    direction = Vector(np.array([1.0, -1.0, 0.0]))
    assert_allclose(
        stiffnesses.directional_shear_modulus(
            normal=normal,
            direction=direction,
        ).components,
        np.array([90.0] * num_repeats),
    )


def test_directional_poissons_ratio(isotropic_stiffness_matrix):
    num_repeats = 2
    stiffnesses = Order4SymmetricTensor.from_voigt(isotropic_stiffness_matrix).repeat(
        num_repeats
    )
    transverse = Vector(np.array([1.0, 1.0, 1.0]))
    axial = Vector(np.array([1.0, -1.0, 0.0]))
    assert_allclose(
        stiffnesses.directional_poissons_ratio(
            transverse_direction=transverse,
            axial_direction=axial,
        ).components,
        np.array([2.0 / 9.0] * num_repeats),
    )
