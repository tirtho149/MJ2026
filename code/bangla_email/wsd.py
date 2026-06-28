"""Bangla Word Sense Disambiguation (WSD).

A self-contained, reproducible WSD benchmark for Bangla homonyms/polysemes plus
an LLM-based disambiguator.  Many Bangla words carry several unrelated senses
(e.g. চাল = uncooked rice / a clever move / a roof); resolving them needs the
surrounding context.  This is directly relevant to email understanding: the same
token means different things in a `promotions` vs a `primary` email.

The task framing is standard lexical-sample WSD:
    given (target word, context sentence, candidate sense glosses)
    -> choose the correct sense id.

Everything here is gold-labelled and hand-authored (no external download), so the
benchmark is fully reproducible.  ``run_wsd.py`` scores Qwen on it zero-shot and
few-shot.
"""

from __future__ import annotations

import random as _random

from . import config

# ── Lexicon: ambiguous Bangla word -> senses, each with gold example contexts ──
# sense = {id, bn (gloss), en (gloss), examples:[context sentences using THIS sense]}
WSD_LEXICON = {
    "কল": [
        {"id": "kol_tap",     "bn": "পানির কল / ট্যাপ",      "en": "water tap / faucet",
         "examples": ["রান্নাঘরের কলটি নষ্ট হয়ে গেছে, সারাক্ষণ পানি পড়ছে।",
                      "কল ছেড়ে হাত ধুয়ে নাও।"]},
        {"id": "kol_machine", "bn": "যন্ত্র / কারখানা",       "en": "machine / mill",
         "examples": ["নতুন কলে চাল উৎপাদন শুরু হয়েছে।",
                      "কারখানার কলগুলো সকাল থেকে চলছে।"]},
    ],
    "পাতা": [
        {"id": "pata_leaf", "bn": "গাছের পাতা", "en": "leaf of a plant",
         "examples": ["শীতকালে গাছের পাতা ঝরে পড়ে।", "সবুজ পাতায় শিশির জমেছে।"]},
        {"id": "pata_page", "bn": "বইয়ের পাতা", "en": "page of a book",
         "examples": ["বইয়ের প্রথম পাতাটি ছিঁড়ে গেছে।", "খাতার শেষ পাতায় লিখে রাখো।"]},
    ],
    "চাল": [
        {"id": "chal_rice", "bn": "চাল (চাউল)", "en": "uncooked rice",
         "examples": ["বাজারে চালের দাম আবার বেড়েছে।", "দুই কেজি চাল কিনে আনো।"]},
        {"id": "chal_move", "bn": "চাল / কৌশল", "en": "a move / ploy",
         "examples": ["তার এই চালটা কেউ বুঝতে পারেনি।", "দাবার শেষ চালে সে জিতে গেল।"]},
        {"id": "chal_roof", "bn": "ঘরের চাল / ছাদ", "en": "roof",
         "examples": ["টিনের চালে বৃষ্টির শব্দ হচ্ছে।", "ঝড়ে ঘরের চাল উড়ে গেছে।"]},
    ],
    "বল": [
        {"id": "bol_ball", "bn": "বল (খেলার)", "en": "ball",
         "examples": ["শিশুরা মাঠে বল খেলছে।", "বলটি গোলপোস্টে ঢুকে গেল।"]},
        {"id": "bol_strength", "bn": "শক্তি / বল", "en": "strength / force",
         "examples": ["অসুখের পর শরীরে আর বল নেই।", "সব বল দিয়ে দড়িটা টানো।"]},
        {"id": "bol_speak", "bn": "বলা (আদেশ)", "en": "speak / say (imperative)",
         "examples": ["সত্যি কথাটা আমাকে বল।", "জোরে বল, শুনতে পাচ্ছি না।"]},
    ],
    "তাল": [
        {"id": "tal_fruit", "bn": "তাল (ফল)", "en": "palm fruit",
         "examples": ["পাকা তাল দিয়ে মা পিঠা বানিয়েছেন।", "গাছ থেকে তাল পড়ল।"]},
        {"id": "tal_rhythm", "bn": "সুরের তাল / ছন্দ", "en": "rhythm / beat",
         "examples": ["গানের তালে তালে সবাই নাচছে।", "ঢোলের তাল মিলিয়ে গাইছে।"]},
    ],
    "সোনা": [
        {"id": "sona_gold", "bn": "স্বর্ণ", "en": "gold (metal)",
         "examples": ["সোনার দাম আজ রেকর্ড ছুঁয়েছে।", "বিয়েতে সোনার গয়না দেওয়া হলো।"]},
        {"id": "sona_dear", "bn": "আদরের সম্বোধন", "en": "darling (term of endearment)",
         "examples": ["সোনা, তাড়াতাড়ি খেয়ে নাও।", "কেঁদো না সোনা, আমি আছি।"]},
    ],
    "ফল": [
        {"id": "phol_fruit", "bn": "ফল (খাদ্য)", "en": "fruit",
         "examples": ["প্রতিদিন টাটকা ফল খাওয়া ভালো।", "বাজার থেকে মৌসুমি ফল এনেছি।"]},
        {"id": "phol_result", "bn": "ফলাফল", "en": "result / outcome",
         "examples": ["পরীক্ষার ফল আগামীকাল প্রকাশিত হবে।", "পরিশ্রমের ফল সে পেয়েছে।"]},
    ],
    "হার": [
        {"id": "har_necklace", "bn": "গলার হার", "en": "necklace",
         "examples": ["সে গলায় সোনার হার পরেছে।", "হারটি খুব সুন্দর দেখাচ্ছে।"]},
        {"id": "har_defeat", "bn": "পরাজয়", "en": "defeat / loss",
         "examples": ["শেষ ম্যাচে দল হার মেনেছে।", "এত সহজে হার মানব না।"]},
        {"id": "har_rate", "bn": "হার (অনুপাত)", "en": "rate / ratio",
         "examples": ["ব্যাংকের সুদের হার বেড়েছে।", "বেকারত্বের হার কমেছে।"]},
    ],
    "মান": [
        {"id": "man_quality", "bn": "গুণমান", "en": "quality / standard",
         "examples": ["এই পণ্যের মান যথেষ্ট ভালো।", "শিক্ষার মান উন্নত করতে হবে।"]},
        {"id": "man_honor", "bn": "মান-সম্মান", "en": "honor / dignity",
         "examples": ["সবার সামনে তার মান রক্ষা পেল।", "মান-সম্মানের ভয়ে সে চুপ রইল।"]},
        {"id": "man_value", "bn": "মান (গাণিতিক)", "en": "value (mathematics)",
         "examples": ["সমীকরণে x এর মান নির্ণয় করো।", "চলকটির মান শূন্য।"]},
    ],
    "পান": [
        {"id": "pan_betel", "bn": "পান (পাতা)", "en": "betel leaf",
         "examples": ["খাওয়ার পর সে একটা পান মুখে দিল।", "দোকানে পান-সুপারি বিক্রি হয়।"]},
        {"id": "pan_drink", "bn": "পান করা", "en": "to drink",
         "examples": ["বিশুদ্ধ পানি পান করুন।", "প্রতিদিন পর্যাপ্ত পানি পান করা জরুরি।"]},
    ],
    "দল": [
        {"id": "dol_team", "bn": "দল (খেলার)", "en": "team / group",
         "examples": ["আমাদের দল ফাইনালে উঠেছে।", "দলের সব খেলোয়াড় পরিশ্রমী।"]},
        {"id": "dol_party", "bn": "রাজনৈতিক দল", "en": "political party",
         "examples": ["নির্বাচনে নতুন দল অংশ নিচ্ছে।", "দলটি ক্ষমতায় এসেছে।"]},
    ],
    "মাথা": [
        {"id": "matha_head", "bn": "মাথা (দেহ)", "en": "head (body part)",
         "examples": ["সকাল থেকে আমার মাথা ব্যথা করছে।", "রোদে মাথা ঢেকে বের হও।"]},
        {"id": "matha_leader", "bn": "প্রধান / কর্তা", "en": "leader / chief",
         "examples": ["সে এই দলের মাথা।", "পরিবারের মাথা হিসেবে দায়িত্ব নিয়েছে।"]},
    ],
    "আম": [
        {"id": "am_mango", "bn": "আম (ফল)", "en": "mango (fruit)",
         "examples": ["গরমকালে পাকা আম খেতে দারুণ লাগে।", "গাছে অনেক কাঁচা আম ধরেছে।"]},
        {"id": "am_common", "bn": "আম / সাধারণ", "en": "common / general",
         "examples": ["এটি আম জনতার দাবি।", "আম মানুষের কথা আগে ভাবতে হবে।"]},
    ],
    "কাল": [
        {"id": "kal_day", "bn": "আগামীকাল / গতকাল", "en": "tomorrow / yesterday",
         "examples": ["কাল সকালে আমরা দেখা করব।", "কাল রাতে খুব বৃষ্টি হয়েছিল।"]},
        {"id": "kal_era", "bn": "কাল / যুগ", "en": "era / age",
         "examples": ["সেই কালে মানুষ চিঠি লিখে যোগাযোগ করত।", "এ যুগের সাথে সেই কালের অনেক তফাত।"]},
    ],
    "পর": [
        {"id": "por_after", "bn": "পরে / এরপর", "en": "after / next",
         "examples": ["খাওয়ার পর একটু হেঁটে নিও।", "ছুটির পর স্কুল আবার খুলবে।"]},
        {"id": "por_stranger", "bn": "পর / অন্য", "en": "stranger / not one's own",
         "examples": ["আপন আর পর চেনা বড় কঠিন।", "এত বছর পরেও সে আমাকে পর ভাবে।"]},
    ],
    "সার": [
        {"id": "sar_fertilizer", "bn": "সার (কৃষি)", "en": "fertilizer",
         "examples": ["জমিতে সঠিক মাত্রায় সার দিতে হবে।", "ইউরিয়া সারের দাম আবার বেড়েছে।"]},
        {"id": "sar_essence", "bn": "সার / মূল কথা", "en": "essence / gist",
         "examples": ["পুরো বইটির সার কয়েক লাইনেই বলা যায়।", "বক্তৃতার সারটুকু আমি বুঝেছি।"]},
    ],
    "কর": [
        {"id": "kor_tax", "bn": "কর (খাজনা)", "en": "tax",
         "examples": ["এ বছরের আয়কর সময়মতো জমা দিয়েছি।", "জমির খাজনা ও কর পরিশোধ করো।"]},
        {"id": "kor_do", "bn": "করা (আদেশ)", "en": "do (imperative)",
         "examples": ["কাজটা এখনই কর।", "বসে না থেকে কিছু একটা কর।"]},
    ],
    "বাস": [
        {"id": "bash_bus", "bn": "বাস (যানবাহন)", "en": "bus (vehicle)",
         "examples": ["প্রতিদিন বাসে করে অফিসে যাই।", "শেষ বাসটা আজ মিস করেছি।"]},
        {"id": "bash_dwell", "bn": "বাস করা / বসবাস", "en": "to dwell / reside",
         "examples": ["সে ছোটবেলা থেকে এই গ্রামে বাস করে।", "শহরে বাস করা বেশ ব্যয়বহুল।"]},
    ],
    "ছাড়": [
        {"id": "chhar_discount", "bn": "ছাড় (মূল্যহ্রাস)", "en": "discount",
         "examples": ["ঈদ উপলক্ষে দোকানে বড় ছাড় চলছে।", "এই পণ্যে ২০% ছাড় দেওয়া হচ্ছে।"]},
        {"id": "chhar_release", "bn": "ছাড়া / মুক্ত করা", "en": "release / let go",
         "examples": ["দড়িটা একটু ছাড়ো।", "আমার হাত ছাড়, ব্যথা লাগছে।"]},
    ],
    "ঘর": [
        {"id": "ghor_room", "bn": "ঘর (কক্ষ/বাড়ি)", "en": "room / house",
         "examples": ["আমাদের নতুন ঘরে অনেক আলো-বাতাস।", "ঘরের ভেতরে ঢুকে দরজা বন্ধ করো।"]},
        {"id": "ghor_cell", "bn": "ঘর (ছকের ঘর)", "en": "cell (of a table/grid)",
         "examples": ["ছকের প্রতিটি ঘর সংখ্যা দিয়ে পূরণ করো।", "ক্যালেন্ডারের ঘরে তারিখ লেখা আছে।"]},
    ],
    "মুখ": [
        {"id": "mukh_face", "bn": "মুখ (চেহারা)", "en": "face / mouth",
         "examples": ["আনন্দে তার মুখ উজ্জ্বল হয়ে উঠল।", "মুখ ধুয়ে নাশতা করতে এসো।"]},
        {"id": "mukh_opening", "bn": "মুখ (প্রবেশমুখ)", "en": "opening / mouth (of river etc.)",
         "examples": ["নদীর মুখে নৌকাগুলো নোঙর করা।", "গুহার মুখটা বেশ সরু।"]},
    ],
    "হাত": [
        {"id": "hat_hand", "bn": "হাত (অঙ্গ)", "en": "hand",
         "examples": ["খাওয়ার আগে হাত ধুয়ে নাও।", "তার হাতে একটা বই দেখলাম।"]},
        {"id": "hat_cubit", "bn": "হাত (দৈর্ঘ্যের একক)", "en": "cubit (length unit)",
         "examples": ["কাপড়টা প্রায় তিন হাত লম্বা।", "দড়িটা পাঁচ হাত হবে।"]},
    ],
    "পাল": [
        {"id": "pal_herd", "bn": "পাল (পশুপাখির দল)", "en": "herd / flock",
         "examples": ["মাঠে গরুর পাল চরছে।", "আকাশে পাখির পাল উড়ে গেল।"]},
        {"id": "pal_sail", "bn": "পাল (নৌকার)", "en": "sail",
         "examples": ["বাতাস পেয়ে নৌকার পাল ফুলে উঠল।", "মাঝি পাল তুলে দিল।"]},
    ],
    "টান": [
        {"id": "tan_pull", "bn": "টান (আকর্ষণ-বল)", "en": "pull / tug",
         "examples": ["দড়িটায় জোরে টান দাও।", "ঘুড়ির সুতোয় হঠাৎ টান পড়ল।"]},
        {"id": "tan_affection", "bn": "টান (মমতা)", "en": "affection / attachment",
         "examples": ["মায়ের প্রতি তার গভীর টান।", "জন্মভূমির জন্য মনে একটা টান অনুভব করি।"]},
    ],
    "ছবি": [
        {"id": "chhobi_picture", "bn": "ছবি (চিত্র/ফটো)", "en": "picture / photo",
         "examples": ["দেয়ালে একটি সুন্দর ছবি ঝুলছে।", "ক্যামেরায় কয়েকটা ছবি তুললাম।"]},
        {"id": "chhobi_movie", "bn": "ছবি (চলচ্চিত্র)", "en": "movie / film",
         "examples": ["নতুন ছবিটি আগামী শুক্রবার মুক্তি পাবে।", "হলে গিয়ে নতুন ছবি দেখলাম।"]},
    ],
    "বার": [
        {"id": "bar_times", "bn": "বার (দফা/সংখ্যা)", "en": "times (count)",
         "examples": ["আমি তিন বার চেষ্টা করেছি।", "বার বার একই ভুল কোরো না।"]},
        {"id": "bar_weekday", "bn": "বার (সপ্তাহের দিন)", "en": "day of the week",
         "examples": ["আজ সপ্তাহের কোন বার?", "শুক্রবার আমাদের ছুটির বার।"]},
    ],
    "পদ": [
        {"id": "pod_post", "bn": "পদ (চাকরির)", "en": "post / position",
         "examples": ["সে ম্যানেজার পদে নিয়োগ পেয়েছে।", "এই পদের জন্য আবেদন করেছি।"]},
        {"id": "pod_verse", "bn": "পদ (কবিতার চরণ)", "en": "verse / line of a poem",
         "examples": ["কবিতার প্রথম পদটি আবৃত্তি করো।", "গানের এই পদটি খুব জনপ্রিয়।"]},
    ],
}


def build_instances():
    """Flatten the lexicon into gold (word, context, sense_id) instances."""
    inst = []
    for word, senses in WSD_LEXICON.items():
        for s in senses:
            for ex in s["examples"]:
                inst.append({"word": word, "context": ex, "gold": s["id"],
                             "senses": senses})
    return inst


def _options_block(senses):
    return "\n".join(f'{i+1}. {s["bn"]} ({s["en"]})' for i, s in enumerate(senses))


WSD_SYSTEM = (
    "তুমি একজন বাংলা ভাষা বিশেষজ্ঞ। একটি শব্দের একাধিক অর্থ থাকতে পারে; বাক্যের "
    "প্রসঙ্গ দেখে সঠিক অর্থটি বেছে নাও। শুধুমাত্র সঠিক অর্থের নম্বরটি লেখো — অন্য কিছু নয়।"
)


def build_wsd_messages(word, context, senses, demos=None):
    """Chat prompt for one WSD instance; optional few-shot demos."""
    parts = []
    if demos:
        parts.append("উদাহরণ:")
        for d in demos:
            opts = _options_block(d["senses"])
            ans = next(i + 1 for i, s in enumerate(d["senses"]) if s["id"] == d["gold"])
            parts.append(f'বাক্য: "{d["context"]}"\nশব্দ: {d["word"]}\nঅর্থসমূহ:\n{opts}\nউত্তর: {ans}\n')
    opts = _options_block(senses)
    parts.append(f'বাক্য: "{context}"\nশব্দ: {word}\nঅর্থসমূহ:\n{opts}\nউত্তর:')
    return [
        {"role": "system", "content": WSD_SYSTEM},
        {"role": "user",   "content": "\n".join(parts)},
    ]


def parse_choice(text, n_senses):
    """Extract the chosen sense index (1-based) from the model output."""
    import re
    for tok in re.findall(r"[০-৯0-9]+", text):
        tok = tok.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
        v = int(tok)
        if 1 <= v <= n_senses:
            return v
    return None


def select_demos(instances, target, k, rng):
    """k few-shot demos drawn from OTHER words (no leakage of the target word)."""
    pool = [d for d in instances if d["word"] != target["word"]]
    rng.shuffle(pool)
    # one demo per distinct word, up to k
    seen, demos = set(), []
    for d in pool:
        if d["word"] in seen:
            continue
        seen.add(d["word"]); demos.append(d)
        if len(demos) >= k:
            break
    return demos


def evaluate(llm, SamplingParams, instances, k=0, seed=config.SEED):
    """Zero/few-shot WSD accuracy; returns (metrics, per_instance)."""
    rng = _random.Random(seed)
    params = SamplingParams(temperature=0.0, max_tokens=8)
    msgs, metas = [], []
    for inst in instances:
        demos = select_demos(instances, inst, k, rng) if k else None
        msgs.append(build_wsd_messages(inst["word"], inst["context"], inst["senses"], demos))
        metas.append(inst)
    outs = llm.chat(msgs, params)

    correct, total, per_word = 0, 0, {}
    records = []
    for inst, o in zip(metas, outs):
        choice = parse_choice(o.outputs[0].text, len(inst["senses"]))
        pred = inst["senses"][choice - 1]["id"] if choice else "unknown"
        ok = pred == inst["gold"]
        correct += int(ok); total += 1
        pw = per_word.setdefault(inst["word"], [0, 0])
        pw[0] += int(ok); pw[1] += 1
        records.append({"word": inst["word"], "context": inst["context"],
                        "gold": inst["gold"], "pred": pred, "correct": ok})
    acc = correct / total if total else 0.0
    return {"k": k, "accuracy": acc, "n": total,
            "per_word_acc": {w: c / n for w, (c, n) in per_word.items()}}, records
