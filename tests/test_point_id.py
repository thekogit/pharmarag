import uuid
from src.vector_store import make_point_id

def test_point_id_is_a_stable_uuid():
    a = make_point_id("Aspirin 81 mg once daily", {})
    b = make_point_id("Aspirin 81 mg once daily", {})
    assert a == b
    uuid.UUID(a)
