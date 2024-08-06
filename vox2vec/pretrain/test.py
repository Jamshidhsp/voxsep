import torch
import torch.nn.functional as F

def custom_contrastive_loss(x1, x2, temperature=0.1):
    """
    Compute contrastive loss given anchor-positive and anchor-negative pairs.
    
    Args:
    - x1: Tensor of shape (N, D) containing N vectors (anchors and positives).
    - x2: Tensor of shape (N, D) containing N negative vectors for each anchor in x1.
    - temperature: Temperature scaling parameter for the softmax.
    
    Returns:
    - loss: The computed contrastive loss.
    """
    N, D = x1.size()
    
    # Normalize vectors to unit length
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    
    # Compute similarities within x1 (positive pairs) and between x1 and x2 (negative pairs)
    positive_similarities = torch.mm(x1_norm, x1_norm.t()) / temperature
    negative_similarities = torch.mm(x1_norm, x2_norm.t()) / temperature
    
    # Mask to exclude self-comparisons from positives
    mask = torch.eye(N, dtype=torch.bool)
    positive_similarities.masked_fill_(mask, float('-inf'))
    
    # Log-sum-exp across negatives for each anchor
    negatives_logsumexp = torch.logsumexp(negative_similarities, dim=1)
    
    # For positives, since each vector in x1 is positive with each other, we take the log-sum-exp excluding self
    positives_logsumexp = torch.logsumexp(positive_similarities, dim=1)
    
    # The loss is the negative log likelihood of correct positives over all pairs
    loss = torch.mean(negatives_logsumexp - positives_logsumexp)
    
    return loss

# Example tensors
x1 = torch.randn(126, 16)
x2 = torch.randn(126, 16)

# Compute the loss
loss = custom_contrastive_loss(x1, x2)
print(f"Custom Contrastive Loss: {loss.item()}")
