"""自带训练器（anyrsi_weights.train）在真实的极小模型上训练：SFT、DPO、KTO、LoRA 合并、经 WeightsEvolvable 调用。

需要 torch 和 transformers（LoRA 那条还要 peft），缺了就跳过。CI 里有一个单独的 job 装了 CPU 版的 torch。
模型是随机初始化的 2 层 GPT-2，加一个现场训练的小 BPE 分词器，不用下载任何东西。
"""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from anyrsi import Patch  # noqa: E402
from anyrsi_weights import WeightsEvolvable, write_rows  # noqa: E402
from anyrsi_weights.train import encode, seq_logprob, train  # noqa: E402

PROMPT = [{"role": "user", "content": "is the sky blue"}]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["<|user|>\nis the sky blue\n<|assistant|>\nyes it is\nno never\n"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=120, special_tokens=["<unk>", "<eos>", "<pad>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>")
    fast.save_pretrained(d)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(fast), n_positions=64, n_embd=32, n_layer=2, n_head=2)).save_pretrained(d)
    return d


def logp(model_dir, completion):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    ids, start = encode(tok, PROMPT, completion, 64)
    with torch.no_grad():
        return float(seq_logprob(model, ids, start))


def test_sft_learns_the_answer(tiny, tmp_path):
    data = write_rows([{"messages": [*PROMPT, {"role": "assistant", "content": "yes it is"}]}] * 4, tmp_path / "s.jsonl")
    report = train("sft", str(tiny), str(data), str(tmp_path / "out"), {"epochs": 15, "lr": 5e-3})
    assert report["loss_per_epoch"][-1] < report["loss_per_epoch"][0]
    assert logp(tmp_path / "out", "yes it is") > logp(tiny, "yes it is")
    assert json.loads((tmp_path / "out" / "anyrsi_train.json").read_text())["method"] == "sft"


def test_dpo_moves_probability_from_the_wrong_answer_to_the_right_one(tiny, tmp_path):
    data = write_rows([{"prompt": PROMPT, "chosen": "yes it is", "rejected": "no never"}] * 4, tmp_path / "d.jsonl")
    train("dpo", str(tiny), str(data), str(tmp_path / "out"), {"epochs": 10, "lr": 5e-3, "beta": 0.5})
    before = logp(tiny, "yes it is") - logp(tiny, "no never")
    after = logp(tmp_path / "out", "yes it is") - logp(tmp_path / "out", "no never")
    assert after > before + 1.0


def test_kto_learns_from_unpaired_good_and_bad_answers(tiny, tmp_path):
    rows = [{"messages": [*PROMPT, {"role": "assistant", "content": "yes it is"}], "label": True}] * 3 + \
           [{"messages": [*PROMPT, {"role": "assistant", "content": "no never"}], "label": False}] * 3
    train("kto", str(tiny), str(write_rows(rows, tmp_path / "k.jsonl")), str(tmp_path / "out"),
          {"epochs": 10, "lr": 5e-3, "beta": 0.5, "batch": 6})
    before = logp(tiny, "yes it is") - logp(tiny, "no never")
    after = logp(tmp_path / "out", "yes it is") - logp(tmp_path / "out", "no never")
    assert after > before + 1.0


def test_lora_dpo_is_merged_into_a_plain_model(tiny, tmp_path):
    pytest.importorskip("peft")
    data = write_rows([{"prompt": PROMPT, "chosen": "yes it is", "rejected": "no never"}] * 4, tmp_path / "d.jsonl")
    report = train("dpo", str(tiny), str(data), str(tmp_path / "out"),
                   {"epochs": 10, "lr": 2e-2, "beta": 0.5, "lora": {"r": 4, "alpha": 8, "target_modules": ["c_attn"]}})
    assert report["lora_merged"] and not (tmp_path / "out" / "adapter_config.json").exists()
    after = logp(tmp_path / "out", "yes it is") - logp(tmp_path / "out", "no never")
    assert after > logp(tiny, "yes it is") - logp(tiny, "no never")


def test_weights_evolvable_runs_the_builtin_trainer_and_can_roll_back(tiny, tmp_path):
    w = WeightsEvolvable(str(tiny), tmp_path / "w", trainer="builtin")
    data = write_rows([{"prompt": PROMPT, "chosen": "yes it is", "rejected": "no never"}] * 2, tmp_path / "d.jsonl")
    snap = w.apply(Patch(target="weights", ops=[{"op": "train", "method": "dpo", "data": str(data),
                                                 "config": {"epochs": 2, "lr": 1e-3}}]))
    model = w.export(snap.id)[0]["text"]
    assert snap.parent_ids == ["ckpt_root"] and (tmp_path / "w" / "checkpoints").exists()
    assert json.loads(open(f"{model}/anyrsi_train.json").read())["method"] == "dpo"
    w.rollback("ckpt_root")
    assert w.export()[0]["text"] == str(tiny)


def test_unknown_method_and_empty_data_are_refused(tiny, tmp_path):
    with pytest.raises(ValueError, match="没有数据"):
        train("sft", str(tiny), str(write_rows([], tmp_path / "e.jsonl")), str(tmp_path / "o"))
    with pytest.raises(ValueError, match="不支持"):
        train("ppo", str(tiny), str(write_rows([{"x": 1}], tmp_path / "x.jsonl")), str(tmp_path / "o"))


def test_log_probability_counts_only_the_answer_tokens(tiny):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny)
    model = AutoModelForCausalLM.from_pretrained(tiny).eval()
    ids, start = encode(tok, PROMPT, "yes it is", 64)
    assert ids[start:] == tok("yes it is", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        manual = sum(float(lp[i - 1, ids[i]]) for i in range(start, len(ids)))
        assert float(seq_logprob(model, ids, start)) == pytest.approx(manual, abs=1e-4)
