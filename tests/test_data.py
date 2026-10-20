import json

import pytest

from slm.data import filters, synthetic
from slm.data.contamination import ContaminationIndex
from slm.data.dedup import ExactDedup, MinHashLSH
from slm.data.pii import redact
from slm.data.registry import Registry, Review, Source, classify_license, evaluate
from slm.data.shards import split_of


def reviewed(**kw) -> Source:
    base = dict(id="s", name="s", url="u", role="pretrain", license="MIT", revision="abc",
                review=Review(provenance_verified=True, license_verified=True, pii_reviewed=True))
    base.update(kw)
    return Source(**base)


@pytest.mark.parametrize("lic,kind", [
    ("MIT", "permissive"), ("CC-BY-4.0", "permissive"), ("CC-BY-SA-4.0", "share-alike"),
    ("CC-BY-NC-4.0", "forbidden"), ("CC-BY-ND-4.0", "forbidden"), ("CC-BY-NC-SA-4.0", "forbidden"),
    ("proprietary", "unknown"),
])
def test_classify_license(lic, kind):
    assert classify_license(lic) == kind


def test_policy_requires_review_and_revision():
    assert evaluate(reviewed()).usable
    assert not evaluate(reviewed(revision=None)).usable
    assert not evaluate(reviewed(review=Review())).usable
    assert not evaluate(reviewed(license="CC-BY-NC-4.0")).usable
    assert not evaluate(reviewed(license="mixed")).usable
    assert not evaluate(reviewed(status="comparison-only")).usable
    d = evaluate(reviewed(license="mixed", component_licenses=["CC-BY-4.0", "CC-BY-SA-4.0"]))
    assert d.usable and any("share-alike" in o for o in d.obligations)


def test_project_registry_audit_is_consistent():
    reg = Registry.load("configs/data/sources.yaml")
    decisions, errors = reg.audit()
    assert errors == []
    usable = {d.source_id for d in decisions if d.usable}
    assert usable == {"synthetic-prose-v1", "synthetic-arithmetic-v1"}
    for source_id in ("wikipedia-en", "wikipedia-fr"):
        source = reg[source_id]
        assert source.license_options == ["CC-BY-SA-3.0", "GFDL"]
        assert source.status == "candidate"
        decision = evaluate(source)
        assert not decision.usable
        assert "license not verified by a reviewer" in decision.blockers
        assert "privacy review not completed" in decision.blockers


def test_quality_filters():
    good = {"text": " ".join(["The library records the history of the town carefully."] * 3
                             + ["Each reviewer compares results with earlier measurements."] * 3)}
    assert filters.gopher_quality(min_words=10)(good) is None
    assert filters.gopher_quality()({"text": "too short"}) == "too_few_words"
    assert filters.gopher_quality(min_words=3)({"text": "### ### ### ### word"}) is not None
    rep = {"text": "\n".join(["same line here"] * 10 + ["other"])}
    assert filters.repetition()(rep) == "dup_line_frac"


def test_normalize():
    doc = filters.normalize({"text": "été  \r\n\n\n\nfin  "})
    assert doc["text"] == "été\n\nfin"


def test_exact_and_near_dedup():
    ex = ExactDedup()
    assert ex.check("a", "Hello   World") is None
    assert ex.check("b", "hello world") == "a"
    base = " ".join(f"word{i}" for i in range(300))
    near = base.replace("word150", "changed")
    other = " ".join(f"token{i}" for i in range(300))
    lsh = MinHashLSH(num_perm=128, bands=16, threshold=0.8)
    assert lsh.check("a", base) is None
    hit = lsh.check("b", near)
    assert hit is not None and hit[0] == "a" and hit[1] > 0.9
    assert lsh.check("c", other) is None


def test_pii_ip_and_iban_precision():
    for text in ["edict preserved in the Digest (48.22.7.1).", "version numbers start from 1.0.0.1 in the UI",
                 "(clause 5.2.4.1.1)", "UUID=\"AT32UC3B0512AU\"", "list=PLB03EA9545DD188C3"]:
        out, counts = redact(text)
        assert counts["ip"] == counts["iban"] == 0, out
    out, counts = redact("the server at 93.184.216.34 answered; logs show 93.184.216.35:8080")
    assert counts["ip"] == 2
    out, counts = redact("IBAN FR76 3000 6000 0112 3456 7890 189 and GB82WEST12345698765432")
    assert counts["iban"] == 2, out
    out, counts = redact("IBAN FR76 3000 6000 0112 3456 7890 188")  # wrong check digits
    assert counts["iban"] == 0
    out, counts = redact("AT32UC3B0512AU\nAT32UC3B164\nAB12CDEF\nGHIJ KLMN")  # line breaks crashed the check
    assert counts["iban"] == 0


def test_pii_redaction():
    text, counts = redact("Write to jane.doe@example.org or call +33 6 12 34 56 78 from IP 8.8.8.8, not 192.168.1.1.")
    assert "<EMAIL>" in text and "<PHONE>" in text and "<IP_ADDRESS>" in text
    assert "192.168.1.1" in text
    assert counts["email"] == 1 and counts["ip"] == 1
    same, counts = redact("In 2024, 1500 people attended.")
    assert same == "In 2024, 1500 people attended." and sum(counts.values()) == 0


@pytest.mark.parametrize("phone", ["+33 6 12 34 56 78", "+44 (0)20 7946 0958", "06 12 34 56 78",
                                   "01.23.45.67.89", "(555) 123-4567", "555-123-4567", "+1 555 123 4567"])
def test_pii_phone_formats(phone):
    text, counts = redact(f"Appelez le {phone} demain.")
    assert counts["phone"] == 1 and "<PHONE>" in text


# False positives observed on Wikipedia with an earlier, generic pattern.
@pytest.mark.parametrize("text", [
    "time scales were defined to read JD 2443144.5003725 (1 January 1977 00:00:32",
    "RTMP 86.205.001 was found in 1986. RTMP 99.120.007 was found in 1996",
    "set in 2021 at 38 picokelvin (pK), or 0.000000000038 of a kelvin",
    "risk of new cerebral infarction [RR = 0.62 (95% CI 0.48 0.80)]",
    "archiveurl=https://web.archive.org/web/20160304123456/https://example.org",
    "Phil.-Hist. Cl., 164 (1910) 501–528 online",
    "an acre may be declared as exactly 4,046.8564224 square metres",
    "Revolutions: A Continental History, 1750–1850 (2018) 340 pp",
    "Viking 2001 {ISBN 978-0-670-03035-8} an 85-page essay",
    "Molex 51021-0200 Mini-SPOX connector",
    "population 1 234 567 in 2020, 2 345 678 in 2021",
])
def test_pii_keeps_ordinary_numbers(text):
    out, counts = redact(text)
    assert counts["phone"] == 0, out


def test_contamination():
    idx = ContaminationIndex(n=5)
    idx.add("bench:1", "What is the capital city of France and why is it famous")
    assert idx.matches("Trivia: what is the capital city of France? Paris.") == {"bench:1": 3}
    assert idx.matches("Completely unrelated text about rivers and bridges in spring") == {}


def test_split_is_deterministic():
    ids = [f"doc{i}" for i in range(5000)]
    splits = [split_of(i, 0.1) for i in ids]
    assert splits == [split_of(i, 0.1) for i in ids]
    assert 0.07 < splits.count("val") / len(ids) < 0.13


def test_synthetic_is_deterministic_and_verifiable():
    assert synthetic.prose(5, seed=3) == synthetic.prose(5, seed=3)
    tasks = synthetic.arithmetic(20, seed=1)
    assert all(synthetic.verify_arithmetic(t, f" {t['answer']} done") for t in tasks)
    assert not synthetic.verify_arithmetic(tasks[0], "no number")


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from slm.data.pipeline import run

    raw = tmp_path / "raw.jsonl"
    docs = synthetic.prose(40, seed=0)
    docs.append({**docs[0], "id": "dup"})
    docs.append({"id": "short", "text": "tiny"})
    docs.append({"id": "pii", "text": docs[1]["text"] + " Contact: someone@example.com"})
    leaked = "the quick violet heron counted seventeen lanterns beside the frozen canal at dawn"
    docs.append({"id": "leak", "text": docs[2]["text"] + " " + leaked})
    synthetic.write_jsonl(docs, raw)
    evalf = tmp_path / "eval.jsonl"
    synthetic.write_jsonl([{"id": "e1", "text": f"Question: {leaked}?"}], evalf)
    cfg = tmp_path / "pipe.yaml"
    cfg.write_text(json.dumps({
        "name": "t",
        "registry": "configs/data/sources.yaml",
        "inputs": [{"source": "synthetic-prose-v1", "path": str(raw)}],
        "steps": [{"normalize": {}}, {"min_length": {"min_chars": 50}}, {"pii": {}}, {"exact_dedup": {}},
                  {"decontaminate": {"n": 8, "eval_files": [{"path": str(evalf)}]}}],
        "output_dir": str(tmp_path / "out"),
    }))
    m = run(cfg)
    st = m["stats"]
    assert st["docs_in"] == 44
    assert st["dropped"]["min_length"] == {"too_short": 1}
    assert st["dropped"]["exact_dedup"] == {"exact_duplicate": 1}
    assert st["dropped"]["decontaminate"] == {"benchmark_overlap": 1}
    assert st["pii_redactions"] == {"email": 1}
    assert st["docs_out"] == 41
    assert m["inputs"][0]["policy"]["usable"]
    lines = (tmp_path / "out" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_pipeline_refuses_unapproved_source(tmp_path):
    from slm.data.pipeline import run

    raw = tmp_path / "raw.jsonl"
    synthetic.write_jsonl(synthetic.prose(3), raw)
    cfg = tmp_path / "pipe.yaml"
    cfg.write_text(json.dumps({"registry": "configs/data/sources.yaml", "output_dir": str(tmp_path / "o"),
                               "inputs": [{"source": "fineweb-edu", "path": str(raw)}], "steps": []}))
    with pytest.raises(PermissionError):
        run(cfg)
    assert run(cfg, allow_unapproved=True)["allow_unapproved"] is True
