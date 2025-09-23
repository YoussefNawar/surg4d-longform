"""
Train Qwen feature autoencoder (inspired by splattalk)
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Autoencoder_dataset
from model_qwen import QwenAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--language_name', type=str, default='qwen_features')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cos_weight', type=float, default=0.0)
    parser.add_argument('--eval_after', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--latent_dim', type=int, default=256)
    args = parser.parse_args()

    data_dir = os.path.join(args.dataset_path, args.language_name)
    os.makedirs(f'ckpt/{args.model_name}', exist_ok=True)

    dataset = Autoencoder_dataset(data_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = QwenAutoencoder(input_dim=3584, latent_dim=args.latent_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tb = SummaryWriter(f'ckpt/{args.model_name}')

    global_step = 0
    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x = batch.to(args.device, dtype=torch.float32)
            x_rec = model(x)
            loss = F.mse_loss(x_rec, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tb.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

        # Eval
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(args.device, dtype=torch.float32)
                x_rec = model(x)
                val_loss_sum += F.mse_loss(x_rec, x, reduction='sum').item()
        val_loss = val_loss_sum / len(val_dataset)
        tb.add_scalar('val/loss', val_loss, epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'ckpt/{args.model_name}/best_ckpt.pth')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'ckpt/{args.model_name}/{epoch+1}_ckpt.pth')


if __name__ == '__main__':
    main()


