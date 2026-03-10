import torch
import math

def calculate_bpb(loss):
    """
    Calculates Bits Per Base (BPB) from Cross Entropy Loss.
    BPB = Loss / ln(2)
    """
    return loss / math.log(2)

def calculate_perplexity(loss):
    """
    Calculates Perplexity from Cross Entropy Loss.
    PPL = exp(Loss)
    """
    return torch.exp(loss)

def calculate_accuracy(logits, labels):
    """
    Calculates accuracy of predictions.
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).float()
    return correct.mean()

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.total_tokens = 0
        self.total_correct = 0
        self.steps = 0

    def update(self, loss, logits, labels):
        batch_size = labels.size(0)
        seq_len = labels.size(1)
        num_tokens = batch_size * seq_len
        
        self.total_loss += loss.item() * num_tokens
        self.total_tokens += num_tokens
        
        preds = torch.argmax(logits, dim=-1)
        self.total_correct += (preds == labels).sum().item()
        self.steps += 1

    def get_metrics(self):
        avg_loss = self.total_loss / max(1, self.total_tokens)
        accuracy = self.total_correct / max(1, self.total_tokens)
        bpb = calculate_bpb(avg_loss)
        ppl = calculate_perplexity(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "bpb": bpb,
            "perplexity": ppl
        }
