import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Config as cfg
from src.data_loader import LibriSpeechDataset, collate_fn
from src.model import CTCASR
from src.utils import decode_indices_to_text, calculate_wer

def train():
    train_dataset = LibriSpeechDataset(cfg.DATA_ROOT, train=True)
    val_dataset = LibriSpeechDataset(cfg.DATA_ROOT, train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = CTCASR().to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    ctc_loss = torch.nn.CTCLoss(blank=cfg.BLANK_IDX, reduction='mean', zero_infinity=True)

    best_wer = float('inf')
    for epoch in range(1, cfg.EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            mfccs, targets, input_lens, target_lens = batch
            mfccs = mfccs.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            input_lens = input_lens.to(cfg.DEVICE)
            target_lens = target_lens.to(cfg.DEVICE)

            logits = model(mfccs, input_lens)
            log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = ctc_loss(log_probs, targets, input_lens, target_lens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_refs = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                mfccs, targets, input_lens, target_lens = batch
                mfccs = mfccs.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)
                input_lens = input_lens.to(cfg.DEVICE)
                target_lens = target_lens.to(cfg.DEVICE)

                logits = model(mfccs, input_lens)
                log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss = ctc_loss(log_probs, targets, input_lens, target_lens)
                val_loss += loss.item()

                for i in range(mfccs.size(0)):
                    pred_indices = torch.argmax(logits[i], dim=-1).cpu().numpy()
                    pred_text = decode_indices_to_text(pred_indices)
                    ref_len = target_lens[i].item()
                    ref_indices = targets[i][:ref_len].cpu().numpy()
                    ref_text = decode_indices_to_text(ref_indices)
                    all_preds.append(pred_text)
                    all_refs.append(ref_text)

        avg_val_loss = val_loss / len(val_loader)
        wer = calculate_wer(" ".join(all_preds), " ".join(all_refs))
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | WER: {wer:.2%}")

        torch.save(model.state_dict(), f"{cfg.MODEL_SAVE_DIR}/ctc_epoch_{epoch}.pt")
        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), f"{cfg.MODEL_SAVE_DIR}/ctc_best.pt")
            print(f"✅ New best model saved! WER: {best_wer:.2%}")

if __name__ == "__main__":
    train()