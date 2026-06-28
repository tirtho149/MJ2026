"""Shared configuration for the Bangla email pipeline.

Single source of truth for the label scheme, generation targets, prompt facets
and filesystem paths.  Imported by every other module so the synthetic-generation
notebook and the classification notebook can no longer drift apart.
"""

from __future__ import annotations

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve everything relative to the package so the code runs the same whether
# launched from an sbatch job, an srun shell, or a login node.
PKG_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.environ.get("BANGLA_DATA_DIR", os.path.join(REPO_DIR, "data"))

# The real corpus (github.com/tariqulnwu/Bangla-Email-Dataset), cloned next to
# the package.  ``data.load_raw`` also accepts the canonical xlsx the original
# notebook expected, or falls back to the built-in seed for offline smoke tests.
RAW_CSV   = os.environ.get("BANGLA_RAW_CSV",  os.path.join(REPO_DIR, "_dataset_src", "Bangla_Email_Dataset.csv"))
RAW_XLSX  = os.environ.get("BANGLA_RAW_XLSX", os.path.join(REPO_DIR, "Bangla_Email_Dataset_RAW.xlsx"))

# ── Label scheme (integer target <-> category name) ───────────────────────────
# These integers are the ``Label`` column of the GitHub CSV; the mapping matches
# the original notebook's CATEGORY_TARGET exactly so nothing downstream shifts.
CATEGORY_TARGET = {
    "primary":    0,
    "spam":       1,
    "updates":    2,
    "important":  3,
    "promotions": 4,
    "social":     5,
}
CATEGORIES   = list(CATEGORY_TARGET.keys())
TARGET_CATEGORY = {v: k for k, v in CATEGORY_TARGET.items()}   # int -> name

# Bangla display labels (used by plots / reports).
CATEGORY_BN = {
    "primary":    "প্রাথমিক",
    "updates":    "আপডেট",
    "spam":       "স্প্যাম",
    "promotions": "প্রচার",
    "social":     "সামাজিক",
    "important":  "গুরুত্বপূর্ণ",
}

# ── Models ────────────────────────────────────────────────────────────────────
# Default generation/classification model.  Smoke jobs override with the 1.5B.
DEFAULT_MODEL = os.environ.get("BANGLA_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SMOKE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"

SEED = 42

# ── Balancing plan ────────────────────────────────────────────────────────────
PER_CLASS_TARGET = 3000          # every class up-sampled to this; no real data dropped

# Per-category generation length (token budget), matched to real lengths.
MAX_TOKENS = {
    "social":     80,
    "promotions": 150,
    "primary":    150,
    "spam":       150,
    "updates":    170,
    "important":  240,
}

# Per-category natural character-length bounds for validation.
LEN_BOUNDS = {
    "social":     (15, 220),
    "promotions": (25, 320),
    "primary":    (15, 700),
    "spam":       (15, 500),
    "updates":    (15, 600),
    "important":  (15, 900),
}

OVERGEN            = 1.30   # over-generate 30 % to absorb validation + dedup drops
NEAR_DUP_THRESHOLD = 0.90   # char-ngram TF-IDF cosine above this -> reject
MAX_ROUNDS         = 6      # safety cap on the loop-until-target generator

# ── Category definitions (kept consistent across both notebooks) ──────────────
CATEGORY_DEF = {
    "primary":    "ব্যক্তিগত, সরাসরি ইমেইল — বন্ধু/সহকর্মী/পরিবার/শিক্ষকের কাছ থেকে দৈনন্দিন যোগাযোগ।",
    "spam":       "অবাঞ্ছিত, সন্দেহজনক বা প্রতারণামূলক ইমেইল — লটারি, ভুয়া পুরস্কার, ফিশিং।",
    "updates":    "স্বয়ংক্রিয় বিজ্ঞপ্তি — লেনদেন, রসিদ, কনফার্মেশন, অ্যাপ/সেবা আপডেট।",
    "important":  "উচ্চ-অগ্রাধিকার, জরুরি বা সরকারি/ব্যাংক/স্বাস্থ্য সংক্রান্ত গুরুত্বপূর্ণ ইমেইল।",
    "promotions": "মার্কেটিং ইমেইল — ব্র্যান্ডের ছাড়, অফার, ভাউচার, নিউজলেটার।",
    "social":     "সোশ্যাল মিডিয়া বিজ্ঞপ্তি — ফেসবুক/ইউটিউব/ইনস্টাগ্রাম ইত্যাদির নোটিফিকেশন।",
}

# ── Facet banks: randomized concrete details fed into each prompt ─────────────
# Sampling one value per facet anchors every email on different concrete entities
# -> diversity without repeating the few-shot examples.
FACETS = {
    "primary": {
        "relation": ["ঘনিষ্ঠ বন্ধু", "সহকর্মী", "বড় ভাই", "ছোট বোন", "বিশ্ববিদ্যালয়ের শিক্ষক",
                     "প্রতিবেশী", "পুরোনো ক্লাসমেট", "কাজিন", "রুমমেট", "অফিসের ম্যানেজার"],
        "intent":   ["দেখা করার পরিকল্পনা", "একটি সাহায্য চাওয়া", "একটি খবর জানানো",
                     "অনুষ্ঠানে আমন্ত্রণ", "একটি প্রশ্ন", "বই/ নোট ধার চাওয়া",
                     "কাজের অগ্রগতি জানানো", "দেরির জন্য দুঃখ প্রকাশ", "একটি প্রস্তাব", "ধন্যবাদ জানানো"],
        "name":     ["রনি", "সাকিব", "তানিয়া", "মেহেদী", "নুসরাত", "আরিফ", "ফারজানা", "শাওন", "রিয়া", "হাসান"],
        "place":    ["ঢাকা", "খুলনা", "চট্টগ্রাম", "রাজশাহী", "সিলেট", "বরিশাল", "রংপুর", "কুমিল্লা"],
    },
    "spam": {
        "scheme":  ["লটারি জেতা", "বিনামূল্যে আইফোন", "বিদেশি চাকরির অফার", "সহজ লোন",
                    "ক্রিপ্টো বিনিয়োগে দ্বিগুণ লাভ", "অ্যাকাউন্ট স্থগিত — লিঙ্কে ক্লিক",
                    "উপহার দাবি করুন", "OTP শেয়ার করতে বলা", "ভুয়া ব্যাংক রিফান্ড", "নকল কুরিয়ার ফি"],
        "lure":    ["৫০ লক্ষ টাকা", "১০ হাজার ডলার", "একটি নতুন গাড়ি", "বিনামূল্যে রিচার্জ",
                    "বিশাল ক্যাশব্যাক", "সীমিত সময়ের সুযোগ"],
        "urgency": ["আজই", "২৪ ঘণ্টার মধ্যে", "এখনই", "শেষ সুযোগ", "দ্রুত"],
    },
    "updates": {
        "service": ["bKash", "Nagad", "Rocket", "Dutch-Bangla Bank", "Pathao", "Foodpanda",
                    "Daraz", "গ্রামীণফোন", "রবি", "বিদ্যুৎ বিল (DPDC)", "WASA", "BRAC Bank"],
        "event":   ["একটি লেনদেন সম্পন্ন", "পেমেন্ট কনফার্মেশন", "রিচার্জ সফল", "অর্ডার শিপড",
                    "বিল পরিশোধ", "অ্যাকাউন্ট ব্যালেন্স", "ডেলিভারি আপডেট", "অ্যাপ আপডেট উপলব্ধ",
                    "পাসওয়ার্ড পরিবর্তিত", "নতুন লগইন শনাক্ত"],
        "amount":  ["৳ ৫০০", "৳ ১,২৫০", "৳ ৩,৪৯৯", "৳ ৮৯৯", "৳ ১২,০০০", "৳ ২৫০", "৳ ৭৫০"],
        "date":    ["১২ জুন", "০৩ মার্চ", "২৮ জানুয়ারি", "১৫ আগস্ট", "০৯ ডিসেম্বর"],
    },
    "important": {
        "topic":  ["ব্যাংক নিরাপত্তা সতর্কতা", "NID / সরকারি বিজ্ঞপ্তি", "কোভিড টিকার তথ্য",
                   "যাচাইকরণ কোড (verification code)", "ভর্তি / পরীক্ষার ফলাফল",
                   "অফিসিয়াল নোটিশ", "ভিসা / পাসপোর্ট আপডেট", "বীমা দাবি", "আদালতের নোটিশ",
                   "চাকরির নিয়োগপত্র"],
        "org":    ["Bangladesh Bank", "স্বাস্থ্য অধিদপ্তর", "Google", "বিশ্ববিদ্যালয় কর্তৃপক্ষ",
                   "NBR", "পাসপোর্ট অধিদপ্তর", "আপনার অফিস", "BTCL"],
        "action": ["দ্রুত যাচাই করুন", "নথি সঙ্গে আনুন", "কোডটি গোপন রাখুন",
                   "নির্ধারিত তারিখে উপস্থিত হন", "অবিলম্বে যোগাযোগ করুন"],
    },
    "promotions": {
        "brand":    ["Aarong", "Daraz", "Bata", "Foodpanda", "Le Reve", "Sajgoj", "Yellow",
                     "Pizza Hut", "Star Cineplex", "Pickaboo", "Chaldal", "Apex"],
        "product":  ["নতুন কালেকশন", "ঈদ অফার", "ফ্ল্যাশ সেল", "কম্বো প্যাক", "উইন্টার সেল",
                     "গিফট ভাউচার", "বিউটি প্রোডাক্ট", "জুতার কালেকশন"],
        "discount": ["৫০% ছাড়", "৩০% পর্যন্ত ছাড়", "কিনলে একটি ফ্রি", "৳৫০০ ক্যাশব্যাক",
                     "ফ্রি ডেলিভারি", "২৫% ডিসকাউন্ট কোড"],
        "deadline": ["আজ রাত ১২টা পর্যন্ত", "শুধু এই সপ্তাহে", "৩ দিন বাকি", "স্টক শেষ হওয়ার আগে"],
    },
    "social": {
        "platform": ["Facebook", "YouTube", "Instagram", "LinkedIn", "WhatsApp গ্রুপ",
                     "Twitter (X)", "TikTok", "Messenger"],
        "action":   ["একটি ফ্রেন্ড রিকোয়েস্ট", "গ্রুপে নতুন পোস্ট", "সাবস্ক্রাইব করার অনুরোধ",
                     "একটি কমেন্ট", "আপনাকে মেনশন", "প্রোফাইল ভিজিট", "নতুন ফলোয়ার",
                     "একটি ইভেন্টের আমন্ত্রণ", "একটি ছবিতে ট্যাগ", "লাইভে এসেছে"],
        "who":      ["একজন পরিচিত", "আপনার বন্ধু রনি", "'Barishal University Students' গ্রুপ",
                     "একটি পেজ 'Bangla Tech'", "১৮ জন", "একজন সহকর্মী"],
    },
}

# ── Prompts ───────────────────────────────────────────────────────────────────
GEN_SYSTEM_PROMPT = (
    "তুমি একজন বাংলাদেশি ইমেইল লেখক। বাস্তব, স্বাভাবিক বাংলায় ছোট ইমেইল লেখো — "
    "ঠিক যেমন বাংলাদেশের সাধারণ মানুষ লেখে। দরকার হলে ইংরেজি ব্র্যান্ড/পণ্যের নাম ও সংখ্যা "
    "স্বাভাবিকভাবে মিশিয়ে দাও। প্রতিটি ইমেইল আলাদা ও অনন্য হবে — আগের মতো নয়।"
)

CLASSIFY_SYSTEM_PROMPT = """You are an expert Bangla email classifier.
Classify the Bangla email text into exactly one of these 6 categories:
- primary    (প্রাথমিক)   : personal, direct, important personal emails
- updates    (আপডেট)     : automated notifications, receipts, confirmations
- spam       (স্প্যাম)    : unwanted, suspicious, phishing emails
- promotions (প্রচার)    : marketing, discounts, offers, newsletters
- social     (সামাজিক)   : social media notifications, community emails
- important  (গুরুত্বপূর্ণ) : high-priority, urgent, official emails

Respond with ONLY the category name in lowercase English. Nothing else."""

CATEGORIES_VALID = set(CATEGORIES)
