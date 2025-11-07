import json
import os
import re
import sys
from typing import List, Dict, Any, Tuple

# ============================================================
# 0. Lexicons / resources  （已移除 COLOR_WORDS）
# ============================================================

FEMININE_HINTS = [
    "woman","girl","mother","mom","grandmother","grandma","lady","female",
    "sister","aunt","princess","queen","wife","girlfriend","waitress","daughter"
]
MASCULINE_HINTS = [
    "man","boy","father","dad","grandfather","grandpa","male","brother",
    "uncle","king","husband","boyfriend","son","waiter"
]
HUMAN_HINTS = [
    "man","woman","boy","girl","child","kid","person","people","teacher",
    "grandmother","grandfather","grandma","grandpa","mother","father",
    "mom","dad","police","officer","driver","waiter","waitress","doctor",
    "nurse","worker","dancer","student","chef","cook","lady","gentleman",
    "woman","man","girl","boy","grandmother","grandfather","teacher",
    "grandma","grandpa","mother","father","child","kid","person"
]
ANIMAL_HINTS = [
    "dog","cat","bird","fish","horse","puppy","kitten","animal","hamster",
    "rabbit","bunny","turtle","lizard","parrot","duck","chicken","cow",
    "sheep","goat","pony","kitten","puppy"
]

# ============================================================
# 1. Basic helpers
# ============================================================

def load_text(path: str) -> str:
    """Read raw description from a .txt file and normalize whitespace."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str) -> List[str]:
    """
    Roughly split into sentences on .?!
    We keep punctuation but return cleanish pieces.
    """
    parts = re.split(r"([.?!])", text)
    sents = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if sent:
            sents.append((sent + punct).strip(" "))
    return sents


# ============================================================
# 2. Mention extraction and typing (for pronoun resolution)
# ============================================================

def normalize_np(article: str, rest_tokens: List[str]) -> str:
    """
    Build canonical NP like:
      "the grandmother", "the kite", "the wooden chair"
    Rules:
    - force first word to "the"
    - 不再删除颜色词或任何形容词；原样保留 rest_tokens
    """
    if not rest_tokens:
        return ""
    phrase = "the " + " ".join(rest_tokens)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def guess_gender_animacy(np: str) -> Tuple[str, str]:
    """
    Return (gender, animacy)
      gender in {"feminine","masculine","unknown"}
      animacy in {"human","animal","object"}
    """
    low = np.lower()

    # default
    animacy = "object"

    # human?
    for w in HUMAN_HINTS:
        if re.search(r"\b"+re.escape(w)+r"\b", low):
            animacy = "human"
            break
    # if not human, maybe animal?
    if animacy == "object":
        for w in ANIMAL_HINTS:
            if re.search(r"\b"+re.escape(w)+r"\b", low):
                animacy = "animal"
                break

    # gender (only meaningful if human)
    gender = "unknown"
    if animacy == "human":
        for w in FEMININE_HINTS:
            if re.search(r"\b"+re.escape(w)+r"\b", low):
                gender = "feminine"
                break
        if gender == "unknown":
            for w in MASCULINE_HINTS:
                if re.search(r"\b"+re.escape(w)+r"\b", low):
                    gender = "masculine"
                    break

    return gender, animacy


def extract_mentions(sentence: str) -> List[Dict[str, Any]]:
    """
    Find noun phrases of the form:
       (a|an|the) <up to ~4 following tokens>
    Example:
       "a tall old grandmother"
       "the small bicycle"
       "a bright kite"
       "the wooden chair"
       "a fish"
    We'll:
      - canonicalize to "the grandmother", "the kite", ...
      - attach gender/animacy tags
    （不再做颜色删除）
    """
    mentions = []
    tokens = sentence.split()

    for i, tok in enumerate(tokens):
        if re.fullmatch(r"(a|an|the)", tok, flags=re.IGNORECASE):
            tail = tokens[i+1:i+5]  # grab up to 4 tokens after article
            if not tail:
                continue
            np_norm = normalize_np("the", tail)
            if not np_norm:
                continue
            gender, animacy = guess_gender_animacy(np_norm)
            mentions.append({
                "np": np_norm,
                "gender": gender,
                "animacy": animacy,
            })

    return mentions


# ============================================================
# 3. Pronoun resolution
# ============================================================

def pick_referent(pronoun: str, memory: List[Dict[str, Any]], possessive: bool) -> str:
    """
    Find the best referent in memory for a given pronoun.
    We scan memory from newest to oldest (reverse order).
    Rules:
      - he/him/his → prefer masculine humans, fallback any human
      - she/her     → prefer feminine humans, fallback any human
      - it/its      → prefer animal/object
    If possessive=True, append "'s".
    If nothing matches, return the pronoun unchanged.
    """
    p = pronoun.lower()

    def match(entry: Dict[str,Any]) -> bool:
        g = entry["gender"]
        a = entry["animacy"]

        if p in ["he","him","his"]:
            if a == "human" and (g == "masculine" or g == "unknown"):
                return True
            return False

        if p in ["she","her"]:
            if a == "human" and (g == "feminine" or g == "unknown"):
                return True
            return False

        if p in ["it","its"]:
            if a in ["animal","object"]:
                return True
            return False

        return False

    for entry in reversed(memory):
        if match(entry):
            return entry["np"] + "'s" if possessive else entry["np"]

    return pronoun  # fallback: leave as-is


def resolve_pronouns_all(sentences: List[str]) -> List[str]:
    """
    Global coreference resolution across sentences.

    memory is a running list of mentions:
      memory = [
         {"np": "the grandmother", "gender": "feminine", "animacy": "human"},
         {"np": "the kite",        "gender": "unknown",  "animacy": "object"},
         ...
      ]

    For each sentence:
      1. Extract mentions from the original sentence, append to memory.
      2. Replace possessive pronouns (his/her/its) using memory.
      3. Replace subject/object pronouns (he/him/she/her/it) using memory.
      4. Cleanup "the the X".
      5. Extract mentions again from the resolved sentence and append to memory.
    """
    memory: List[Dict[str, Any]] = []
    resolved_sentences = []

    for sent in sentences:
        # Step 1: learn new mentions from the raw sentence
        memory.extend(extract_mentions(sent))

        # Step 2: possessive pronouns first
        def poss_sub(match):
            pron = match.group(0)
            return pick_referent(pron, memory, possessive=True)

        tmp = re.sub(r"\bhis\b|\bher\b|\bits\b", poss_sub, sent, flags=re.IGNORECASE)

        # Step 3: subject/object pronouns
        def subj_obj_sub(match):
            pron = match.group(0)
            return pick_referent(pron, memory, possessive=False)

        tmp = re.sub(r"\bhe\b|\bhim\b|\bshe\b|\bher\b|\bit\b", subj_obj_sub, tmp, flags=re.IGNORECASE)

        # Step 4: cleanup accidental "the the"
        tmp = re.sub(r"\bthe\s+the\b", "the ", tmp, flags=re.IGNORECASE)

        # Step 5: learn any new explicit mentions created by replacement
        memory.extend(extract_mentions(tmp))

        resolved_sentences.append(tmp)

    return resolved_sentences


# ============================================================
# 4. High-level preprocessing （已移除与颜色相关的处理）
# ============================================================

def preprocess_text(text: str) -> str:
    """
    1. Split text into sentences.
    2. Run global pronoun resolution across the original sentences（不删颜色词）。
    3. Join back into one paragraph string.
    """
    sents = split_sentences(text)
    sents_resolved = resolve_pronouns_all(sents)
    cleaned = " ".join(sents_resolved)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ============================================================
# 5. Scene splitting into T1/T2/T3...
# ============================================================

def naive_clause_split(text: str) -> List[str]:
    """
    First split: break text on .?!
    Then split long sentences on ', and/then/while/as' style coordinators.
    """
    parts = re.split(r"([.?!])", text)
    merged = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if not sent:
            continue
        merged.append((sent + punct).strip())

    splitter = re.compile(
        r",\s+(and then|and|then|while|as)\b|;\s*(and then|and|then)\b",
        flags=re.IGNORECASE,
    )

    fine_chunks = []
    for sent in merged:
        start = 0
        for m in splitter.finditer(sent):
            cut_point = m.start()
            left = sent[start:cut_point].strip(", ; ")
            if left:
                fine_chunks.append(left)
            start = m.start(0)
        tail = sent[start:].strip(", ; ")
        if tail:
            fine_chunks.append(tail)

    cleaned_chunks = []
    for ch in fine_chunks:
        ch = ch.strip()
        ch = re.sub(r"^[,;]\s*", "", ch)
        ch = ch.rstrip(" .!?;,")
        if ch:
            cleaned_chunks.append(ch)

    return cleaned_chunks


def refine_scenes(chunks: List[str]) -> List[str]:
    """
    Second pass:
    If a chunk still has an " and " joining two clear action clauses,
    and there's no comma (so it's likely two sequential beats),
    split it into two.
    """
    scenes: List[str] = []

    for ch in chunks:
        if " and " in ch and "," not in ch:
            parts = re.split(r"\band\b", ch, maxsplit=1)
            if len(parts) == 2:
                left = parts[0].strip(" ,;")
                right = "and " + parts[1].strip(" ,;")

                def looks_like_clause(s: str) -> bool:
                    return bool(re.search(
                        r"\b(run|runs|running|wave|waves|waving|wag|wags|wagging|"
                        r"kneel|kneels|kneeling|pet|pets|petting|walk|walks|walking|"
                        r"stand|stands|standing|sit|sits|sitting|hold|holds|holding|"
                        r"look|looks|looking|tell|tells|telling|fly|flies|flying|move|moves|moving)\b",
                        s,
                        flags=re.IGNORECASE
                    ))

                if looks_like_clause(left) and looks_like_clause(right):
                    if left:
                        scenes.append(left)
                    if right:
                        scenes.append(right)
                    continue

        scenes.append(ch)

    # Normalize capitalization / trailing commas
    normed = []
    for s in scenes:
        s = s.strip()
        s = s.rstrip(",")
        if re.match(r"^[a-z]", s) and not re.match(r"^(and|then)\b", s, re.IGNORECASE):
            s = s[0].upper() + s[1:]
        normed.append(s)

    return normed


def build_timeline(scenes: List[str]) -> List[Dict[str, str]]:
    """
    Assign T1, T2, T3... and add final period.
    """
    timeline = []
    for idx, scene in enumerate(scenes, start=1):
        scene_txt = scene.strip()
        if not scene_txt.endswith("."):
            scene_txt += "."
        timeline.append({
            "time": f"T{idx}",
            "scene": scene_txt
        })
    return timeline


def save_json(data: List[Dict[str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============================================================
# 6. main
# ============================================================

DEFAULT_IN_PATH = "data/input_txt/input.txt"
DEFAULT_OUT_PATH = "data/input_json/graph_spec.json"

def main():
    # 用法：
    #   1) 直接运行（使用默认路径）
    #        python scene_graph_builder.py
    #   2) 传入自定义路径
    #        python scene_graph_builder.py <input_txt> <output_json>

    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    elif len(sys.argv) == 1:
        in_path = DEFAULT_IN_PATH
        out_path = DEFAULT_OUT_PATH
    else:
        print("Usage: python scene_split.py [<input_txt> <output_json>]")
        sys.exit(1)

    # 检查输入是否存在
    if not os.path.exists(in_path):
        print(f"[Error] Input file not found: {in_path}")
        sys.exit(1)

    # 确保输出目录存在
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. 读取原始文本
    raw_text = load_text(in_path)

    # 2. 预处理：仅做跨句代词回填（不做任何“颜色删除”）
    preprocessed_text = preprocess_text(raw_text)

    # 3. 场景拆分
    coarse_chunks = naive_clause_split(preprocessed_text)
    refined = refine_scenes(coarse_chunks)

    # 4. 生成时间线
    timeline = build_timeline(refined)

    # 5. 保存 JSON
    save_json(timeline, out_path)
    print(f"[OK] Scene spec saved to: {out_path}")


if __name__ == "__main__":
    main()
