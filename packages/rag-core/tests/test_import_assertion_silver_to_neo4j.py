"""Tests for importing assertion silver-set rows into Neo4j."""

from scripts.import_assertion_silver_to_neo4j import import_silver_assertions


class _Result:
    def __init__(self, rows=None, single=None):
        self._rows = rows or []
        self._single = single or {}

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._single


class _Session:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        return _Result(single={"total": 0})


class _Driver:
    def __init__(self):
        self.session_obj = _Session()

    def session(self, **kwargs):
        return self.session_obj


def test_import_silver_assertions_writes_assertion_properties_to_relationship():
    driver = _Driver()
    rows = [
        {
            "text": "目前不能排除冠心病。",
            "entity": "冠心病",
            "label": "speculated",
            "start": 6,
            "end": 9,
            "weak_label": "affirmed",
            "model_label": "affirmed",
            "rule_label": "speculated",
            "model_confidence": 0.91,
            "rule_confidence": 0.85,
            "needs_review": True,
            "confidence": 0.6,
            "cue": "不能排除",
            "source": "unit-test",
        }
    ]

    summary = import_silver_assertions(rows, driver, dataset="unit_silver", batch_size=1)

    assert summary == {"dataset": "unit_silver", "imported": 1}
    query, params = driver.session_obj.calls[0]
    assert "MERGE (ph)-[r:MENTIONED_IN]->(pa)" in query
    assert "r.assertion_status = row.assertion_status" in query
    assert "DETACH DELETE" not in query
    assert params["rows"][0]["assertion_status"] == "speculated"
    assert params["rows"][0]["needs_review"] is True
    assert params["rows"][0]["dataset"] == "unit_silver"
