from google import genai
from google.genai import types
import unicodedata
import re
import time
import os

# =========================
# CONFIG
# =========================
MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
MAX_CHUNK_WORDS = 150
MAX_OUTPUT_TOKENS = 4096
OVERLAP_WORDS = 25
INPUT_FOLDER = r"C:\Users\smwaj\fyp_text\test_video"  # root folder containing category subfolders

# =========================
# CUSTOM WORD MAP
# =========================
WORD_MAP = {
    "عاصف": "آصف",
    "اسف": "آصف",
    "عاصم": "آصف",   # Asim → Asif
    "اسم": "آصف",    # common misspelling of Asim
    "قران": "قرآن",
    "يه": "یہ",
    "كيا": "کیا",
}

# =========================
# AUTH
# =========================
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")
client = genai.Client(api_key=api_key)


# =========================
# HELPER FUNCTIONS
# =========================

def normalize_urdu(text):
    return unicodedata.normalize("NFC", text)

def is_english(text):
    letters = re.findall(r"[a-zA-Z]", text)
    return len(letters) / max(len(text), 1) > 0.5

def apply_word_map(text):
    words = text.split()
    return " ".join(WORD_MAP.get(w, w) for w in words)

def split_text_into_chunks(text, max_words=MAX_CHUNK_WORDS):
    words = text.split()
    chunk_ranges = []
    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        if end < len(words):
            best_break = -1
            for j in range(end - 1, i + max(int(max_words * 0.3), 1) - 1, -1):
                if any(words[j].endswith(p) for p in ['۔', '!', '?', '؟']):
                    best_break = j + 1
                    break
            if best_break > i:
                chunk_ranges.append((i, best_break))
                i = best_break
            else:
                chunk_ranges.append((i, end))
                i = end
        else:
            chunk_ranges.append((i, end))
            i = end
    chunks = []
    for start, end in chunk_ranges:
        chunks.append((" ".join(words[start:end]), start, end))
    return chunks

def build_chunks_with_context(chunks, all_words, overlap=OVERLAP_WORDS):
    result = []
    for idx, (chunk_text, start, end) in enumerate(chunks):
        parts = []
        if start > 0:
            ctx_start = max(0, start - overlap)
            prev_ctx = " ".join(all_words[ctx_start:start])
            parts.append(f"[سابقہ سیاق - صرف حوالے کے لیے، اسے درست نہ کریں]\n{prev_ctx}\n")
        parts.append(f"[START]\n{chunk_text}\n[END]")
        if end < len(all_words):
            ctx_end = min(len(all_words), end + overlap)
            next_ctx = " ".join(all_words[end:ctx_end])
            parts.append(f"\n[اگلا سیاق - صرف حوالے کے لیے، اسے درست نہ کریں]\n{next_ctx}")
        prompt_text = "\n".join(parts)
        core_word_count = len(chunk_text.split())
        result.append((prompt_text, core_word_count))
    return result

def build_prompt(chunk_with_markers, word_count):
    return f"""
آپ ایک اردو پروف ریڈر ہیں۔ نیچے دیا گیا اردو متن ایک تقریر کی ٹرانسکرپشن ہے۔

آپ کا کام:
1. ہجے کی غلطیاں درست کریں
2. گرامر درست کریں
3. جملوں کی ساخت بہتر کریں
4. انگریزی الفاظ کو اردو میں تبدیل کریں (جہاں ممکن ہو)

⚠️ بہت اہم ہدایات:
- صرف [START] اور [END] کے درمیان والا متن درست کریں۔
- سابقہ سیاق اور اگلا سیاق صرف حوالے کے لیے ہے — اسے اپنے جواب میں شامل نہ کریں۔
- متن میں تقریباً {word_count} الفاظ ہیں۔ آپ کا جواب بھی اتنا ہی لمبا ہونا چاہیے۔
- کوئی جملہ حذف نہ کریں۔
- متن کو مختصر نہ کریں۔
- کوئی نیا مواد شامل نہ کریں۔
- صرف درست شدہ متن واپس کریں — نہ [START]/[END] ٹیگز، نہ وضاحت۔

متن:
{chunk_with_markers}
"""

def clean_model_output(text):
    for marker in [
        "[START]",
        "[END]",
        "[سابقہ سیاق - صرف حوالے کے لیے، اسے درست نہ کریں]",
        "[اگلا سیاق - صرف حوالے کے لیے، اسے درست نہ کریں]",
    ]:
        text = text.replace(marker, "")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def generate_chunk(prompt_text, core_word_count, chunk_num, total_chunks):
    full_prompt = build_prompt(prompt_text, core_word_count)
    best_output = None
    for attempt in range(MAX_RETRIES):
        print(f"  Attempt {attempt+1}/{MAX_RETRIES} …")
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.15,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
            )
            text_out = response.text
            cleaned = clean_model_output(text_out)
            cleaned = apply_word_map(cleaned)
            wc = len(cleaned.split())
            ratio = wc / max(core_word_count,1)
            print(f"   → Output {wc} words | ratio {ratio:.2f}")
            if ratio >= 0.75:
                return cleaned
            best_output = cleaned
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            time.sleep(2.5)
    print(f"  ⚠️ Using best attempt for chunk {chunk_num}.")
    return best_output or ""

def process_text(raw_text):
    if is_english(raw_text):
        print("Detected English → skipping Gemini")
        return raw_text
    processed_text = normalize_urdu(raw_text)
    all_words = processed_text.split()
    chunks = split_text_into_chunks(processed_text)
    chunks_with_context = build_chunks_with_context(chunks, all_words)
    final_output_parts = []
    total = len(chunks_with_context)
    for i, (prompt_text, core_word_count) in enumerate(chunks_with_context, start=1):
        print(f"\n--- Chunk {i}/{total} ---")
        corrected = generate_chunk(prompt_text, core_word_count, i, total)
        final_output_parts.append(corrected)
    final_output = " ".join(final_output_parts)
    final_output = re.sub(r"\s+", " ", final_output).strip()
    final_output = apply_word_map(final_output)
    return final_output

# =========================
# SINGLE FILE REFINE — for pipeline use
# =========================

def refine_single_file(txt_path: str) -> str:
    """
    Refine a single transcript file using Gemini.
    Returns the path of the output '_final.txt' file.
    Called by the automation pipeline orchestrator.
    """
    import pathlib
    p = pathlib.Path(txt_path)
    base_name = p.stem
    output_path = p.parent / (base_name + "_final.txt")

    if output_path.exists():
        print(f"[Refining] Skipping (already refined): {p.name}")
        return str(output_path)

    print(f"[Refining] Processing: {p.name}")
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    final_output = process_text(raw_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"[Refining] Saved → {output_path.name}")
    return str(output_path)


# =========================
# BATCH FOLDER PROCESSING (manual use)
# =========================

def main():
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if not filename.endswith(".txt") or filename.endswith("_final.txt"):
                continue
            input_path = os.path.join(root, filename)
            base_name = os.path.splitext(filename)[0]
            output_filename = base_name + "_final.txt"
            output_path = os.path.join(root, output_filename)
            if os.path.exists(output_path):
                print(f"Skipping existing file → {output_filename}")
                continue
            print("\n===================================")
            print(f"Processing file: {filename}")
            print("===================================")
            with open(input_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            final_output = process_text(raw_text)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_output)
            print(f"Saved → {output_filename}")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()