import time
import torch
import argparse
import tiktoken

from datasets import load_dataset
from gpt2 import GPT2Model


# fmt: off
def parse_arguments():
    parser = argparse.ArgumentParser(description=("Pre-train GPT model."))
    parser.add_argument("--block_size", type=int, default=64) #256)
    parser.add_argument("--batch_size", type=int, default=16) #64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="corbt/all-recipes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embed_dim", type=int, default=96) #384)
    parser.add_argument("--num_heads", type=int, default=2) #6)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=2) #6)

    return parser.parse_args()
# fmt: on


def get_batch(data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, val_data, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(
                val_data, block_size, batch_size, device
            )  # get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    args = parse_arguments()

    train_samples = 100
    val_samples = 20

    dataset = load_dataset(args.dataset)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = len(enc._mergeable_ranks)

    text = "RECIPE: " + "\n\nRECIPE: ".join(dataset["train"][:train_samples]["input"])
    train_data = torch.tensor(enc.encode(text), dtype=torch.long)

    text = "RECIPE: " + "\n\nRECIPE: ".join(
        dataset["train"][train_samples : train_samples + val_samples]["input"]
    )
    val_data = torch.tensor(enc.encode(text), dtype=torch.long)

    model = GPT2Model(
        args.embed_dim,
        args.num_heads,
        vocab_size,
        args.max_position_embeddings,
        args.num_layer,
        use_cache=False,
    ).to(args.device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = time.time()

    for iter in range(args.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(
                model,
                args.eval_iters,
                val_data,
                args.block_size,
                args.batch_size,
                args.device,
            )  # estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, args.device)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    print(enc.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    end = time.time()
    print(f"Training took {end - start}")


if __name__ == "__main__":
    main()
