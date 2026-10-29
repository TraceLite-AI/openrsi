"""从 Hugging Face 下载 GSM8K，转成 GSM8KEnv 读的 jsonl（不装 verl 时用）。

  python prepare_gsm8k.py ~/data/gsm8k-jsonl
"""
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

out = Path(sys.argv[1] if len(sys.argv) > 1 else "gsm8k").expanduser()
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    f = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
    rows = pd.read_parquet(f)[["question", "answer"]].to_dict("records")
    (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(split, len(rows))
print(out)
