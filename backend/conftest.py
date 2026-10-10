"""Shared test fixtures, and the guard rails that keep tests off the real database.

The deploy pipeline runs `docker compose run --rm backend pytest -v`, where
DATABASE_URL already points at the *production* database. Two layers keep test
data from ever landing there:

1. DATABASE_URL is overwritten below - before `database.py` is imported and
   builds its engine - with an address nothing listens on. `create_engine` is
   lazy, so this is harmless until something actually tries to connect, at
   which point it fails loudly instead of quietly writing to production.
2. The `fake_db` fixture is autouse, so *every* test gets an in-memory session
   whether or not it asks for one. Isolation no longer depends on each test
   author remembering to opt in.
"""

import math
import os

# Must happen before the imports below: auth.py, database.py and main.py all
# read environment variables at import time.
os.environ["DATABASE_URL"] = "postgresql+psycopg2://unused:unused@127.0.0.1:1/unused"
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.elements import BinaryExpression, False_, Grouping, True_
from types import SimpleNamespace

from database import get_db
from main import app
from models import Document

API_KEY = os.environ["API_KEY"]
ADMIN_API_KEY = os.environ["ADMIN_API_KEY"]

# A unit vector, so cosine distance against it is well defined. (A zero vector
# has no direction, and pgvector's cosine distance to it is NaN.)
DEFAULT_EMBEDDING = [1.0] + [0.0] * 1535

# pgvector renders cosine_distance as the `<=>` operator.
COSINE_OP = "<=>"


def cosine_distance(a, b):
    """Mirror of pgvector's cosine_distance: 1 - cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return 1.0 - dot / (norm_a * norm_b)


def _is_cosine_expression(element):
    return (
        isinstance(element, BinaryExpression)
        and getattr(element.operator, "opstring", None) == COSINE_OP
    )


def _query_embedding_from(entities):
    """Recover the query vector from a `cosine_distance(...).label('distance')`
    entity, so the fake can compute per-row distances."""
    for entity in entities:
        element = getattr(entity, "element", None)
        if _is_cosine_expression(element):
            return element.right.value
    return None


def _criterion_value(criterion):
    """Pull the literal value out of a `Column == value` BinaryExpression.
    Boolean literals compile to True_()/False_() singletons rather than a
    regular BindParameter, so they need special-casing."""
    right = criterion.right
    if isinstance(right, True_):
        return True
    if isinstance(right, False_):
        return False
    return right.value


class FakeQuery:
    """Stand-in for a SQLAlchemy Query supporting the subset of the API that
    main.py uses (filter/order_by/offset/limit/first/all), evaluated in memory
    against whatever has been added to the FakeSession so far."""

    def __init__(self, rows, entities, query_embedding=None):
        self._rows = list(rows)
        self._entities = entities
        self._query_embedding = (
            query_embedding
            if query_embedding is not None
            else _query_embedding_from(entities)
        )

    def _derive(self, rows):
        return FakeQuery(rows, self._entities, self._query_embedding)

    def _distance(self, row):
        if self._query_embedding is None:
            return 0.0
        return cosine_distance(row.embedding, self._query_embedding)

    def filter(self, *criteria):
        rows = self._rows
        for criterion in criteria:
            left = criterion.left
            if isinstance(left, Grouping) and _is_cosine_expression(left.element):
                # A distance cutoff: cosine_distance(embedding, qvec) <= limit.
                vector = left.element.right.value
                cutoff = criterion.right.value
                compare = criterion.operator
                rows = [
                    r
                    for r in rows
                    if compare(cosine_distance(r.embedding, vector), cutoff)
                ]
            else:
                key = left.key
                value = _criterion_value(criterion)
                rows = [r for r in rows if getattr(r, key) == value]
        return self._derive(rows)

    def order_by(self, *args):
        if not args:
            return self
        key = args[0]
        if isinstance(key, str):
            if key == "distance" and self._query_embedding is not None:
                # NaN sorts unpredictably, so push those rows to the end.
                def sort_key(row):
                    d = self._distance(row)
                    return (math.isnan(d), d if not math.isnan(d) else 0.0)

                return self._derive(sorted(self._rows, key=sort_key))
            return self
        column = getattr(key, "key", None)
        if column is not None:
            return self._derive(sorted(self._rows, key=lambda r: getattr(r, column)))
        return self

    def offset(self, n):
        return self._derive(self._rows[n:])

    def limit(self, n):
        return self._derive(self._rows[:n])

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return [self._project(r) for r in self._rows]

    def _project(self, row):
        if len(self._entities) == 1 and self._entities[0] is Document:
            return row
        values = {}
        for entity in self._entities:
            key = getattr(entity, "key", None) or getattr(entity, "name", None)
            values[key] = self._distance(row) if key == "distance" else getattr(row, key)
        return SimpleNamespace(**values)


class FakeSession:
    """Stand-in for a SQLAlchemy Session that only supports
    add/commit/refresh/query, so tests never need a database connection."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)

    def query(self, *entities):
        return FakeQuery(self.added, entities)


@pytest.fixture(autouse=True)
def fake_db():
    session = FakeSession()
    app.dependency_overrides[get_db] = lambda: session
    yield session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Autouse so no test can reach the real OpenAI API either. Tests that
    care about specific vectors reassign `mock_embeddings.return_value`."""
    with pytest.MonkeyPatch.context() as mp:
        import unittest.mock

        mocked = unittest.mock.MagicMock(return_value=list(DEFAULT_EMBEDDING))
        mp.setattr("main.get_embedding", mocked)
        yield mocked


@pytest.fixture
def client():
    return TestClient(app)
