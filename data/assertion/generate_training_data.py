"""
Assertion Detection 训练数据生成器
- 合成数据扩充（病历首页/出院小结/门诊记录风格）
- 规则弱标注（从模板生成）
- 数据增强（同义词替换 + 实体替换 + 句序变体）

输出: synthetic_assertion_full.jsonl (2000+ 条)
"""

import json
import random
import re
from itertools import product

random.seed(42)

# ============================================================
# 实体库
# ============================================================

DISEASES = [
    "高血压", "2型糖尿病", "冠心病", "心房颤动", "心力衰竭", "慢性肾脏病",
    "慢性阻塞性肺疾病", "支气管哮喘", "肺炎", "肺结核", "肺栓塞", "脑梗死",
    "脑出血", "帕金森病", "癫痫", "系统性红斑狼疮", "类风湿关节炎", "痛风",
    "甲状腺功能亢进", "甲状腺功能减退", "肝硬化", "慢性乙型肝炎", "胃溃疡",
    "急性胰腺炎", "胆囊结石", "肾病综合征", "尿路感染", "前列腺增生",
    "骨质疏松症", "腰椎间盘突出", "强直性脊柱炎", "银屑病", "白血病",
    "淋巴瘤", "多发性骨髓瘤", "贫血", "血小板减少症", "深静脉血栓形成",
    "主动脉夹层", "心肌梗死", "心肌炎", "心包炎", "扩张型心肌病",
    "肥厚型心肌病", "二尖瓣狭窄", "主动脉瓣狭窄", "肺动脉高压",
    "糖尿病肾病", "糖尿病视网膜病变", "糖尿病足", "甲状腺癌", "肺癌",
    "胃癌", "肝癌", "结直肠癌", "乳腺癌", "前列腺癌", "膀胱癌",
    "溃疡性结肠炎", "克罗恩病", "重症肌无力", "多发性硬化症",
    "系统性血管炎", "干燥综合征", "白塞病", "IgA肾病", "膜性肾病",
]

SYMPTOMS = [
    "胸闷气短", "心悸", "头晕", "头痛", "咳嗽咳痰", "咯血", "呼吸困难",
    "腹痛", "腹泻", "恶心呕吐", "黑便", "血便", "水肿", "蛋白尿",
    "血尿", "尿频尿急", "关节疼痛", "腰痛", "肌肉酸痛", "皮疹",
    "发热", "盗汗", "体重下降", "乏力", "食欲减退", "视物模糊",
    "吞咽困难", "声音嘶哑", "肢体麻木", "肢体无力", "晕厥",
    "胸痛", "心绞痛", "夜间阵发性呼吸困难", "端坐呼吸", "咳血",
    "双下肢水肿", "颈静脉怒张", "肝脾肿大", "淋巴结肿大",
    "皮肤黄染", "瘀斑", "紫绀", "杵状指",
]

DRUGS = [
    "二甲双胍", "格列美脲", "胰岛素", "恩格列净", "达格列净",
    "阿司匹林", "氯吡格雷", "替格瑞洛", "华法林", "利伐沙班", "达比加群",
    "阿托伐他汀", "瑞舒伐他汀", "依折麦布",
    "卡托普利", "依那普利", "缬沙坦", "氯沙坦", "沙库巴曲缬沙坦",
    "美托洛尔", "比索洛尔", "卡维地洛",
    "氨氯地平", "硝苯地平", "地尔硫卓",
    "呋塞米", "螺内酯", "氢氯噻嗪", "托拉塞米",
    "左甲状腺素钠", "甲巯咪唑", "丙硫氧嘧啶",
    "泼尼松", "甲泼尼龙", "环磷酰胺", "甲氨蝶呤", "来氟米特",
    "青霉素", "头孢曲松", "莫西沙星", "万古霉素", "美罗培南",
    "奥美拉唑", "雷贝拉唑", "铝碳酸镁",
    "噻托溴铵", "沙美特罗", "布地奈德", "孟鲁司特",
    "秋水仙碱", "别嘌醇", "非布司他",
    "肝素", "低分子肝素", "阿加曲班",
    "利妥昔单抗", "托珠单抗", "英夫利昔单抗",
]

TESTS = [
    "血常规", "尿常规", "肝功能", "肾功能", "血糖", "HbA1c", "血脂",
    "心肌酶谱", "肌钙蛋白", "BNP", "NT-proBNP", "D-二聚体",
    "凝血功能", "INR", "CRP", "降钙素原", "血沉",
    "抗核抗体", "抗dsDNA抗体", "类风湿因子", "抗CCP抗体", "补体C3",
    "甲状腺功能", "TSH", "FT4", "甲胎蛋白", "CA125", "CEA",
    "血培养", "痰培养", "尿培养", "HIV抗体", "乙肝五项",
    "心电图", "心脏超声", "胸部CT", "腹部CT", "头颅MRI",
    "肺功能", "FEV1/FVC", "骨密度", "肌电图", "脑电图",
]

SURGERIES = [
    "冠状动脉支架植入术", "冠状动脉搭桥术", "心脏瓣膜置换术",
    "射频消融术", "起搏器植入术", "PCI",
    "胆囊切除术", "阑尾切除术", "胃大部切除术", "肠段切除术",
    "甲状腺全切术", "子宫全切术", "肾移植手术",
    "腹腔镜手术", "胸腔镜手术", "开颅手术",
    "关节置换术", "椎间盘手术", "骨折内固定术",
]

FAMILY_MEMBERS = ["父亲", "母亲", "其父", "其母", "兄弟", "姐姐", "一级亲属"]

# ============================================================
# 否定/推测/条件/历史 触发词
# ============================================================

NEGATION_CUES = ["无", "未见", "否认", "排除", "未发现", "阴性", "未提示", "不支持", "未触及", "未闻及", "未检出"]
SPECULATION_CUES = ["疑似", "考虑", "可能", "不除外", "待排", "倾向于", "不能排除", "考虑为"]
CONDITIONAL_CUES = ["若", "如果", "当", "在……时"]
HISTORICAL_CUES = ["既往", "病史", "曾患", "曾因", "术后", "既往诊断", "既往使用", "年前"]

# ============================================================
# 句式模板
# ============================================================

# --- 病历首页风格 ---
ADMISSION_TEMPLATES_AFFIRMED = [
    "入院诊断：1.{entity}。",
    "主要诊断：{entity}。",
    "入院时主诉{symptom}{duration}，诊断{entity}。",
    "患者因{symptom}入院，确诊{entity}。",
    "入院查体：{entity}体征明显。",
]

ADMISSION_TEMPLATES_NEGATED = [
    "否认{entity}病史。",
    "既往史：无{entity}。",
    "个人史：否认{entity}。",
    "过敏史：无{entity}过敏。",
    "入院查体未见{entity}。",
]

# --- 出院小结风格 ---
DISCHARGE_TEMPLATES_AFFIRMED = [
    "出院诊断：{entity}。",
    "住院期间确诊{entity}，予以{drug}治疗。",
    "出院时{entity}控制稳定。",
    "住院期间发现{entity}，已处理。",
    "出院带药：{drug}，用于{entity}治疗。",
]

DISCHARGE_TEMPLATES_NEGATED = [
    "出院时{test}复查正常，排除{entity}。",
    "住院期间排除{entity}诊断。",
    "出院时无{symptom}等{entity}表现。",
    "复查{test}未见{entity}征象。",
]

# --- 门诊记录风格 ---
CLINIC_TEMPLATES_AFFIRMED = [
    "门诊复查：{entity}控制尚可。",
    "患者自诉{symptom}反复发作，考虑{entity}诊断明确。",
    "今日门诊随访，{entity}病情稳定。",
    "门诊查{test}示{entity}。",
    "继续{drug}治疗{entity}。",
]

CLINIC_TEMPLATES_SPECULATED = [
    "门诊就诊，{symptom}原因考虑{entity}可能。",
    "建议进一步检查，不除外{entity}。",
    "门诊初步印象：疑似{entity}。",
    "{symptom}待排{entity}，建议住院检查。",
    "门诊考虑为{entity}，需进一步确认。",
]

CLINIC_TEMPLATES_CONDITIONAL = [
    "嘱患者若出现{symptom}，需立即停用{drug}。",
    "告知：如果{test}异常，应调整{drug}剂量。",
    "若{entity}加重，建议住院治疗。",
    "当{test}超标时，{drug}需减量。",
    "如果合并{entity}，禁用{drug}。",
]

# --- 通用模板 ---
HISTORICAL_TEMPLATES = [
    "既往有{entity}病史{duration}。",
    "患者{duration}前确诊{entity}。",
    "曾因{entity}住院治疗。",
    "既往行{surgery}。",
    "{duration}前行{surgery}，术后恢复良好。",
    "既往使用{drug}治疗{entity}。",
    "曾患{entity}，已治愈。",
]

FAMILY_TEMPLATES = [
    "{member}有{entity}病史。",
    "{member}患有{entity}。",
    "{member}因{entity}去世。",
    "家族史：{member}有{entity}。",
    "家族中有{entity}患者。",
    "{member}确诊{entity}。",
]

# --- 难例模板（一句多标签）---
HARD_TEMPLATES = [
    ("患者确诊{e1}，但否认{e2}病史。", [("affirmed", "确诊"), ("negated", "否认")]),
    ("既往有{e1}，目前无{e2}症状。", [("historical", "既往"), ("negated", "无")]),
    ("患者有{e1}，不除外合并{e2}。", [("affirmed", "有"), ("speculated", "不除外")]),
    ("{member}有{e1}，患者本人否认{e2}。", [("family_history", ""), ("negated", "否认")]),
    ("既往有{e1}术后，目前{e2}控制稳定。", [("historical", "既往"), ("affirmed", "目前")]),
    ("患者无{e1}，但{test}异常，考虑{e2}可能。", [("negated", "无"), ("speculated", "考虑")]),
    ("曾因{e1}住院，目前无复发，但不排除{e2}。", [("historical", "曾因"), ("speculated", "不排除")]),
    ("若合并{e1}，{drug}应禁用。", [("conditional", "若"), ("conditional", "若")]),
]

DURATIONS = ["3天", "1周", "2周", "1月", "3月", "半年", "1年", "2年", "3年", "5年", "10年", "20年"]


def find_entity_span(text, entity):
    idx = text.find(entity)
    if idx < 0:
        return None, None
    return idx, idx + len(entity)


def make_sample(text, entity, label, cue, difficulty="easy"):
    start, end = find_entity_span(text, entity)
    if start is None:
        return None
    return {
        "text": text,
        "entity": entity,
        "label": label,
        "start": start,
        "end": end,
        "cue": cue,
        "domain": "medical",
        "source": "synthetic",
        "difficulty": difficulty,
    }


def generate_affirmed_samples(n=400):
    samples = []
    all_templates = (
        ADMISSION_TEMPLATES_AFFIRMED
        + DISCHARGE_TEMPLATES_AFFIRMED
        + CLINIC_TEMPLATES_AFFIRMED
    )
    for _ in range(n):
        tmpl = random.choice(all_templates)
        entity = random.choice(DISEASES + SYMPTOMS)
        drug = random.choice(DRUGS)
        symptom = random.choice(SYMPTOMS)
        test = random.choice(TESTS)
        duration = random.choice(DURATIONS)
        text = tmpl.format(entity=entity, drug=drug, symptom=symptom, test=test, duration=duration)
        cue = ""
        for kw in ["确诊", "诊断", "发现", "示", "合并", "存在", "主诉", "自诉", "提示"]:
            if kw in text and text.find(kw) < text.find(entity):
                cue = kw
                break
        s = make_sample(text, entity, "affirmed", cue)
        if s:
            samples.append(s)
    return samples


def generate_negated_samples(n=350):
    samples = []
    all_templates = ADMISSION_TEMPLATES_NEGATED + DISCHARGE_TEMPLATES_NEGATED
    neg_patterns = [
        "患者无{entity}。",
        "否认{entity}病史。",
        "查体未见{entity}。",
        "{test}未见{entity}。",
        "排除{entity}诊断。",
        "{test}阴性，不支持{entity}。",
        "患者否认{entity}及相关症状。",
        "未发现{entity}征象。",
        "患者无{entity}等不适。",
    ]
    all_templates += neg_patterns
    for _ in range(n):
        tmpl = random.choice(all_templates)
        entity = random.choice(DISEASES + SYMPTOMS)
        test = random.choice(TESTS)
        symptom = random.choice(SYMPTOMS)
        drug = random.choice(DRUGS)
        text = tmpl.format(entity=entity, test=test, symptom=symptom, drug=drug)
        cue = ""
        for kw in NEGATION_CUES:
            if kw in text:
                cue = kw
                break
        s = make_sample(text, entity, "negated", cue)
        if s:
            samples.append(s)
    return samples


def generate_speculated_samples(n=300):
    samples = []
    all_templates = CLINIC_TEMPLATES_SPECULATED + [
        "{symptom}原因考虑{entity}可能。",
        "不除外{entity}。",
        "疑似{entity}，建议进一步检查。",
        "{symptom}待排{entity}。",
        "不能排除{entity}诊断。",
        "倾向于{entity}诊断。",
        "考虑为{entity}活动期。",
        "{test}异常，可能与{entity}有关。",
    ]
    for _ in range(n):
        tmpl = random.choice(all_templates)
        entity = random.choice(DISEASES)
        symptom = random.choice(SYMPTOMS)
        test = random.choice(TESTS)
        drug = random.choice(DRUGS)
        text = tmpl.format(entity=entity, symptom=symptom, test=test, drug=drug)
        cue = ""
        for kw in SPECULATION_CUES:
            if kw in text:
                cue = kw
                break
        s = make_sample(text, entity, "speculated", cue)
        if s:
            samples.append(s)
    return samples


def generate_conditional_samples(n=250):
    samples = []
    all_templates = CLINIC_TEMPLATES_CONDITIONAL + [
        "若eGFR低于30，应禁用{drug}。",
        "如果出现{symptom}，需立即停用{drug}。",
        "当{test}超过正常上限时，{drug}需减量。",
        "在{entity}急性期，禁用{drug}。",
        "若合并{entity}，{drug}为禁忌。",
        "如果{test}持续异常，应停用{drug}。",
        "当合并{entity}时，需调整{drug}剂量。",
    ]
    for _ in range(n):
        tmpl = random.choice(all_templates)
        entity = random.choice(DISEASES)
        drug = random.choice(DRUGS)
        symptom = random.choice(SYMPTOMS)
        test = random.choice(TESTS)
        text = tmpl.format(entity=entity, drug=drug, symptom=symptom, test=test)
        # 标注 drug 为 conditional
        cue = ""
        for kw in ["若", "如果", "当", "在"]:
            if kw in text:
                cue = kw
                break
        s = make_sample(text, drug, "conditional", cue)
        if s:
            samples.append(s)
    return samples


def generate_historical_samples(n=300):
    samples = []
    for _ in range(n):
        tmpl = random.choice(HISTORICAL_TEMPLATES)
        entity = random.choice(DISEASES)
        drug = random.choice(DRUGS)
        surgery = random.choice(SURGERIES)
        duration = random.choice(DURATIONS)
        text = tmpl.format(entity=entity, drug=drug, surgery=surgery, duration=duration)
        # 决定标注哪个实体
        target = entity if "{entity}" in tmpl else surgery
        if target not in text:
            target = entity
        cue = ""
        for kw in HISTORICAL_CUES:
            if kw in text:
                cue = kw
                break
        s = make_sample(text, target, "historical", cue)
        if s:
            samples.append(s)
    return samples


def generate_family_samples(n=200):
    samples = []
    for _ in range(n):
        tmpl = random.choice(FAMILY_TEMPLATES)
        entity = random.choice(DISEASES)
        member = random.choice(FAMILY_MEMBERS)
        text = tmpl.format(entity=entity, member=member)
        cue = member if member in text else "家族"
        s = make_sample(text, entity, "family_history", cue)
        if s:
            samples.append(s)
    return samples


def generate_hard_samples(n=400):
    """生成一句多标签的难例"""
    samples = []
    for _ in range(n):
        tmpl_text, label_info = random.choice(HARD_TEMPLATES)
        e1 = random.choice(DISEASES)
        e2 = random.choice(DISEASES)
        while e2 == e1:
            e2 = random.choice(DISEASES)
        member = random.choice(FAMILY_MEMBERS)
        drug = random.choice(DRUGS)
        test = random.choice(TESTS)
        text = tmpl_text.format(e1=e1, e2=e2, member=member, drug=drug, test=test)

        # 标注 e1
        label1, cue1 = label_info[0]
        s1 = make_sample(text, e1, label1, cue1, difficulty="hard")
        if s1:
            samples.append(s1)

        # 标注 e2
        if len(label_info) > 1:
            label2, cue2 = label_info[1]
            s2 = make_sample(text, e2, label2, cue2, difficulty="hard")
            if s2:
                samples.append(s2)

    return samples


# ============================================================
# 数据增强：同义词替换
# ============================================================

SYNONYM_MAP = {
    "否认": ["无", "未诉", "自诉无"],
    "无": ["否认", "未见", "未诉"],
    "未见": ["无", "未发现", "未提示"],
    "排除": ["除外", "不考虑"],
    "考虑": ["怀疑", "倾向于"],
    "疑似": ["可能为", "考虑"],
    "不除外": ["不能排除", "待排"],
    "既往": ["既往史", "过去"],
    "曾因": ["曾患", "既往因"],
    "确诊": ["诊断为", "明确诊断"],
    "若": ["如果", "假如"],
    "如果": ["若", "假如"],
    "当": ["在……时", "一旦"],
}


def augment_synonym(samples, n=200):
    """对已有样本做同义词替换增强"""
    augmented = []
    candidates = [s for s in samples if s["cue"] in SYNONYM_MAP]
    if not candidates:
        return augmented
    for _ in range(n):
        original = random.choice(candidates)
        cue = original["cue"]
        synonyms = SYNONYM_MAP[cue]
        new_cue = random.choice(synonyms)
        new_text = original["text"].replace(cue, new_cue, 1)
        if original["entity"] not in new_text:
            continue
        s = make_sample(new_text, original["entity"], original["label"], new_cue, original["difficulty"])
        if s:
            augmented.append(s)
    return augmented


# ============================================================
# 主流程
# ============================================================

def main():
    all_samples = []

    print("生成 affirmed 样本...")
    all_samples.extend(generate_affirmed_samples(400))

    print("生成 negated 样本...")
    all_samples.extend(generate_negated_samples(350))

    print("生成 speculated 样本...")
    all_samples.extend(generate_speculated_samples(300))

    print("生成 conditional 样本...")
    all_samples.extend(generate_conditional_samples(250))

    print("生成 historical 样本...")
    all_samples.extend(generate_historical_samples(300))

    print("生成 family_history 样本...")
    all_samples.extend(generate_family_samples(200))

    print("生成 hard 难例...")
    all_samples.extend(generate_hard_samples(400))

    print("数据增强：同义词替换...")
    all_samples.extend(augment_synonym(all_samples, 300))

    # 去重
    seen = set()
    unique = []
    for s in all_samples:
        key = (s["text"], s["entity"], s["label"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    random.shuffle(unique)

    # 写入文件
    output_path = r"c:\Document\Python\agentic-graph-rag\data\assertion\synthetic_assertion_full.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in unique:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计
    labels = {}
    diffs = {}
    for s in unique:
        labels[s["label"]] = labels.get(s["label"], 0) + 1
        diffs[s["difficulty"]] = diffs.get(s["difficulty"], 0) + 1

    print(f"\n=== 生成完成 ===")
    print(f"总条数: {len(unique)}")
    print(f"标签分布: {dict(sorted(labels.items()))}")
    print(f"难度分布: {dict(sorted(diffs.items()))}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
