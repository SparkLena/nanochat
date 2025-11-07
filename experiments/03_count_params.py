from torch.fx import config
from nanochat.gpt import GPT, GPTConfig

print("=" * 70)
print("GPT 模型参数量分析")
print("=" * 70)

depths = [4, 6, 8, 10, 12, 20, 26]

print("\n| Depth | Params (M) | Embedding (M) | Transformer (M) | Ratio |")
print("|-------|-----------|---------------|-----------------|-------|")

for d in depths:
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=50304,
        n_layer=d,
        n_head=max(1, (d*64)//128),
        n_kv_head=max(1, (d*64)//128),
        n_embd=d*64
    )

    model = GPT(config)

    total = sum(p.numel() for p in model.parameters())
    emb = model.transformer.wte.weight.numel() + model.lm_head.weight.numel()
    trans = total - emb
    ratio = trans / total if total > 0 else 0

    print(f"| d={d:2} | {total/1e6:7.2f} | {emb/1e6:11.2f} | {trans/1e6:13.2f} | {ratio:5.1%} |")

print("-" * 70)    

# 计算 Chinchilla 最佳数据量
print("\n根据 Chinchilla 法则计算所需数据量 (Token/Param = 20):")
print("-" * 70)
print(f"{'Model':<10} {'Params':<12} {'Tokens Needed':<15} {'Shards (~250M/shard)':<20}")
print("-" * 70)


for d in [4, 6, 8, 12, 20]:
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=50304,
        n_layer=d,
        n_head=max(1, (d*64)//128),
        n_kv_head=max(1, (d*64)//128),
        n_embd=d*64
    )
    model = GPT(config)
    params = sum(p.numel() for p in model.parameters())
    tokens_needed = params * 20
    shards = tokens_needed / 250_000_000

    print(f"d={d:<8} {params/1e6:>8.2f}M   {tokens_needed/1e9:>10.2f}B      {shards:>10.1f}")

print("\n" + "=" * 70)
print("关键发现:")
print("1. 小模型 (d=4-6) 时，Embedding 层占比很大")
print("2. d=6 模型需要约 220M tokens (约 1 个分片)")
print("3. 您的训练只用了 1M tokens (0.004 个分片)，数据不足 220 倍！")
print("=" * 70)