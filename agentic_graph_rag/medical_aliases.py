"""Deterministic medical query alias expansion."""

from __future__ import annotations

import re


MEDICAL_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("2型糖尿病", "二型糖尿病", "T2DM", "Type 2 Diabetes Mellitus"),
    ("糖尿病肾病", "DN", "Diabetic Nephropathy", "糖尿病肾损害"),
    ("终末期肾病", "ESRD", "End-Stage Renal Disease", "尿毒症"),
    ("SGLT-2抑制剂", "SGLT-2i", "钠-葡萄糖协同转运蛋白2抑制剂", "列净类药物"),
    ("ACEI", "血管紧张素转换酶抑制剂", "普利类药物"),
    ("ARB", "血管紧张素受体拮抗剂", "沙坦类药物"),
    ("ARNI", "血管紧张素受体脑啡肽酶抑制剂", "沙库巴曲缬沙坦"),
    ("HFrEF", "射血分数降低的心衰", "收缩性心衰"),
    ("LVEF", "左室射血分数", "心脏泵血能力"),
    ("MRA", "醛固酮受体拮抗剂", "保钾利尿剂"),
    ("STEMI", "ST段抬高型心肌梗死", "急性心梗"),
    ("NSTEMI", "非ST段抬高型心肌梗死", "非ST段抬高型急性冠脉综合征"),
    ("PCI", "经皮冠状动脉介入治疗", "支架手术"),
    ("DAPT", "双联抗血小板治疗", "双抗"),
    ("DES", "药物洗脱支架", "药物支架"),
    ("BMS", "裸金属支架", "普通支架"),
    ("噻嗪类利尿剂", "HCTZ", "氢氯噻嗪"),
    ("袢利尿剂", "呋塞米", "Furosemide"),
    ("NOAC", "新型口服抗凝药", "直接口服抗凝药"),
    ("INR", "国际标准化比值", "凝血功能指标"),
    ("CAP", "社区获得性肺炎", "院外肺炎"),
    ("CKD", "慢性肾脏病", "慢性肾功能不全"),
    ("eGFR", "估算肾小球滤过率", "肾功能指标"),
    ("UACR", "尿白蛋白/肌酐比值", "尿蛋白指标"),
    ("COPD", "慢性阻塞性肺疾病", "慢阻肺"),
    ("FEV1", "第一秒用力呼气容积"),
    ("FVC", "用力肺活量"),
    ("mMRC", "呼吸困难量表"),
    ("CAT", "COPD评估测试"),
    ("GOLD分级", "GOLD标准"),
    ("ICS", "吸入糖皮质激素"),
    ("AECOPD", "急性加重"),
    ("SABA", "短效β2受体激动剂"),
    ("LABA", "长效β2受体激动剂"),
    ("LAMA", "长效抗胆碱能药物"),
)


def expand_medical_aliases(query: str) -> str:
    """Append known medical aliases when the user query mentions one surface form."""
    text = query.strip()
    if not text:
        return query

    additions: list[str] = []
    seen = {_normalize_alias_token(token) for token in _extract_existing_terms(text)}
    for group in MEDICAL_ALIAS_GROUPS:
        if not any(_contains_surface(text, alias) for alias in group):
            continue
        for alias in group:
            normalized = _normalize_alias_token(alias)
            if normalized and normalized not in seen:
                additions.append(alias)
                seen.add(normalized)

    if not additions:
        return query
    return f"{text}\n\nMedical aliases: {', '.join(additions)}"


def _contains_surface(text: str, surface: str) -> bool:
    if _has_cjk(surface):
        return surface in text
    return bool(re.search(rf"(?<![0-9A-Za-z_]){re.escape(surface)}(?![0-9A-Za-z_])", text, re.IGNORECASE))


def _extract_existing_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[A-Za-z0-9][A-Za-z0-9_+./-]*", text)
    return [term for term in terms if term.strip()]


def _normalize_alias_token(text: str) -> str:
    lowered = text.casefold().strip()
    return re.sub(r"[\s\-_/.]+", "", lowered)


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
