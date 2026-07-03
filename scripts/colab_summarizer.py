import os
import json
import torch
import re
from tqdm import tqdm

# We use Qwen2.5-7B-Instruct because of its outstanding multilingual/Russian performance and JSON formatting.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def clean_json_response(text: str) -> dict:
    """Extracts JSON substring from the LLM output and parses it."""
    text = text.strip()
    # Remove markdown code block markers if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding the first '{' and last '}'
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(text[start_idx:end_idx+1])
            except json.JSONDecodeError:
                pass
        return None

def main():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("WARNING: Running without GPU. This will be extremely slow!")

    # Base paths
    input_file = "chunks_with_embeddings.json"
    output_file = "chunks_with_summaries.json"
    progress_source_file = None

    # Kaggle environment setup
    if os.path.exists("/kaggle/working"):
        input_file = "/kaggle/input/datasets/timurx/chunks-with-embeddings/chunks_with_embeddings.json"
        progress_source_file = "/kaggle/input/datasets/timurx/chunks-with-summaries/chunks_with_summaries.json"
        output_file = "/kaggle/working/chunks_with_summaries.json"
        print("Kaggle environment detected.")
        print(f"  Input chunks: {input_file}")
        print(f"  Input progress source (read-only): {progress_source_file}")
        print(f"  Output checkpoint (writable): {output_file}")

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure chunks_with_embeddings.json is uploaded/accessible.")
        return

    # Load input dataset
    print(f"Loading chunks from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    # Load existing progress (checkpointing)
    processed_chunks = []
    processed_ids = set()

    # Determine where to load progress from:
    # 1. First check if there is writable checkpoint progress from this current session
    # 2. Fall back to reading the pre-uploaded checkpoint from the input dataset
    progress_file_to_load = None
    if os.path.exists(output_file):
        progress_file_to_load = output_file
    elif progress_source_file and os.path.exists(progress_source_file):
        progress_file_to_load = progress_source_file

    if progress_file_to_load:
        print(f"Found existing progress file '{progress_file_to_load}'. Loading progress...")
        try:
            with open(progress_file_to_load, "r", encoding="utf-8") as f:
                processed_chunks = json.load(f)
            # Create a unique key for each chunk to check if processed (e.g., video_id + start_time)
            for c in processed_chunks:
                key = f"{c['video_id']}_{c.get('start_time')}"
                processed_ids.add(key)
            print(f"Resuming from checkpoint. Already processed: {len(processed_chunks)} / {len(chunks)} chunks.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")
            processed_chunks = []
            processed_ids = set()

    # Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # We load in 4-bit precision using bitsandbytes if available (perfect for Google Colab T4)
    # Install: pip install bitsandbytes accelerate
    try:
        import bitsandbytes
        print("bitsandbytes detected. Loading model in 4-bit precision...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16
        )
    except ImportError:
        print("bitsandbytes not installed. Loading model in default half-precision (float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )

    # Prompt template
    system_instruction = (
        "Ты — профессиональный ИИ-ассистент, который извлекает краткую суть лекций по психологии отношений. "
        "Твоя задача — прочитать фрагмент и вернуть результат СТРОГО в формате JSON с двумя полями:\n"
        "- \"summary\": краткое саммари фрагмента (1-2 предложения, обобщающие главную суть в утвердительной форме).\n"
        "- \"key_points\": список из 2-5 важнейших тезисов/мыслей из текста.\n\n"
        "Отвечай только валидным JSON-объектом. Никакого дополнительного текста или вводных слов."
    )

    save_interval = 10  # Save progress every 10 chunks
    
    # Process chunks in loop
    pbar = tqdm(chunks, desc="Processing chunks")
    for idx, chunk in enumerate(pbar):
        key = f"{chunk['video_id']}_{chunk.get('start_time')}"
        if key in processed_ids:
            continue

        text = chunk["text"]
        user_prompt = (
            f"Фрагмент текста:\n"
            f"\"\"\"\n{text}\n\"\"\"\n\n"
            f"Сгенерируй JSON с саммари и ключевыми тезисами."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Apply chat template
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([inputs_text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Extract generated response
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse response
            parsed_data = clean_json_response(response)
            if parsed_data and "summary" in parsed_data and "key_points" in parsed_data:
                chunk["summary"] = parsed_data["summary"]
                chunk["key_points"] = parsed_data["key_points"]
            else:
                print(f"\n[Warning] Failed to parse JSON for chunk {idx}. LLM response was:\n{response}")
                # Fallback placeholder values
                chunk["summary"] = chunk["text"][:120] + "..."
                chunk["key_points"] = [chunk["text"][:80] + "..."]

        except Exception as e:
            print(f"\n[Error] Failed to process chunk {idx}: {e}")
            chunk["summary"] = chunk["text"][:120] + "..."
            chunk["key_points"] = [chunk["text"][:80] + "..."]

        processed_chunks.append(chunk)
        processed_ids.add(key)

        # Save checkpoint periodically
        if len(processed_chunks) % save_interval == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed_chunks, f, ensure_ascii=False, indent=2)

    # Final Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete! Output saved to: {output_file}")

if __name__ == "__main__":
    main()
