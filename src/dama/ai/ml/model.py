"""Neural network model for move scoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class BoardEncoder(nn.Module):
    """CNN encoder for the board state."""

    def __init__(self, embedding_size: int = 128, num_blocks: int = 4, channels: int = 64):
        super().__init__()

        self.input_conv = nn.Conv2d(BOARD_PLANES, channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # Final layers to produce embedding
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * 8 * 8, embedding_size)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        Encode board state.

        Args:
            board: Tensor of shape (batch, BOARD_PLANES, 8, 8)

        Returns:
            Tensor of shape (batch, embedding_size)
        """
        x = F.relu(self.input_bn(self.input_conv(board)))

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class MoveScorer(nn.Module):
    """MLP to score a move given board embedding and move features."""

    def __init__(self, embedding_size: int = 128, hidden_size: int = 64):
        super().__init__()

        input_size = embedding_size + MOVE_FEATURE_SIZE

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, board_embedding: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
        """
        Score a move.

        Args:
            board_embedding: Tensor of shape (batch, embedding_size)
            move_features: Tensor of shape (batch, MOVE_FEATURE_SIZE)

        Returns:
            Tensor of shape (batch, 1) - score for each move
        """
        x = torch.cat([board_embedding, move_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Clamp scores to prevent overflow in float16 (AMP) - range [-50, 50] is safe for softmax
        x = torch.clamp(x, min=-50.0, max=50.0)
        return x


class ValueHead(nn.Module):
    """Value head that predicts expected game outcome from board embedding.
    
    Outputs a scalar in [-1, 1] representing expected result:
      +1 = win for current player, -1 = loss, 0 = draw.
    """

    def __init__(self, embedding_size: int = 128, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, board_embedding: torch.Tensor) -> torch.Tensor:
        """Predict value from board embedding.
        
        Args:
            board_embedding: (batch, embedding_size)
        Returns:
            (batch,) value predictions in [-1, 1]
        """
        x = F.relu(self.fc1(board_embedding))
        x = torch.tanh(self.fc2(x))  # tanh to bound output to [-1, 1]
        return x.squeeze(-1)


class MoveScorerNet(nn.Module):
    """
    Complete move scoring network with optional value head.

    Takes a board state and a batch of move features,
    returns a score for each move (policy) and optionally a value estimate.
    """

    def __init__(self, embedding_size: int = 128, num_blocks: int = 4, hidden_size: int = 64, channels: int = 64,
                 value_head_enabled: bool = False, value_head_hidden: int = 128):
        super().__init__()

        # Store architecture params for checkpoint serialization
        self.arch_params = {
            'embedding_size': embedding_size,
            'num_blocks': num_blocks,
            'hidden_size': hidden_size,
            'channels': channels,
            'value_head_enabled': value_head_enabled,
            'value_head_hidden': value_head_hidden,
        }

        self.board_encoder = BoardEncoder(embedding_size, num_blocks, channels)
        self.move_scorer = MoveScorer(embedding_size, hidden_size)

        # Optional value head for TD/value learning
        self.value_head_enabled = value_head_enabled
        if value_head_enabled:
            self.value_head = ValueHead(embedding_size, value_head_hidden)
        else:
            self.value_head = None
        
        # Initialize weights for numerical stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling to prevent large values."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        board: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Score moves for a batch of positions.

        Args:
            board: Tensor of shape (batch_size, BOARD_PLANES, 8, 8)
            move_features: Tensor of shape (total_moves, MOVE_FEATURE_SIZE)
                           where total_moves = sum(move_counts)
            move_counts: Tensor of shape (batch_size,) - number of moves per position

        Returns:
            Tensor of shape (total_moves,) - score for each move
        """
        batch_size = board.shape[0]

        # Encode all boards
        board_embeddings = self.board_encoder(board)  # (batch_size, embedding_size)

        # Expand board embeddings to match moves
        # Create indices to repeat board embeddings for each move
        indices = torch.repeat_interleave(
            torch.arange(batch_size, device=board.device),
            move_counts
        )
        expanded_embeddings = board_embeddings[indices]  # (total_moves, embedding_size)

        # Score all moves
        scores = self.move_scorer(expanded_embeddings, move_features)  # (total_moves, 1)

        return scores.squeeze(-1)

    def forward_with_value(
        self,
        board: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor
    ) -> tuple:
        """
        Score moves AND predict position value for a batch.

        Returns:
            (policy_scores, value_predictions) where:
            - policy_scores: (total_moves,) scores for each move
            - value_predictions: (batch_size,) value in [-1, 1] per position
        """
        batch_size = board.shape[0]

        # Encode all boards (shared backbone)
        board_embeddings = self.board_encoder(board)  # (batch_size, embedding_size)

        # Policy head: expand embeddings and score moves
        indices = torch.repeat_interleave(
            torch.arange(batch_size, device=board.device),
            move_counts
        )
        expanded_embeddings = board_embeddings[indices]
        scores = self.move_scorer(expanded_embeddings, move_features).squeeze(-1)

        # Value head
        if self.value_head is not None:
            values = self.value_head(board_embeddings)  # (batch_size,)
        else:
            values = torch.zeros(batch_size, device=board.device)

        return scores, values

    def forward_padded(
        self,
        board: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score moves using padded move features (training-optimized path).

        Runs MLP on ALL padded positions (including padding slots) using 3D
        nn.Linear ops (batched matmul), then masks invalid slots to -inf.
        This keeps all tensor shapes fixed, enabling torch.compile to capture
        the full forward pass in CUDAGraphs — eliminating kernel launch overhead.
        The MLP is <10% of total compute (CNN dominates), so computing padding
        slots costs less than the dynamic-shape overhead of skipping them.

        Args:
            board: (batch_size, BOARD_PLANES, 8, 8)
            move_features: (batch_size, max_moves, MOVE_FEATURE_SIZE) padded
            move_counts: (batch_size,) valid moves per position

        Returns:
            (batch_size, max_moves) scores, -inf for padding slots
        """
        batch_size = board.shape[0]
        max_moves = move_features.shape[1]

        board_embeddings = self.board_encoder(board)  # (batch, emb)

        # Expand board embeddings to (batch, max_moves, emb) — strided view, no copy.
        # nn.Linear natively supports (..., features) via batched matmul,
        # so we can work in 3D without reshape/contiguous.
        expanded = board_embeddings.unsqueeze(1).expand(-1, max_moves, -1)
        combined = torch.cat([expanded, move_features], dim=-1)  # (batch, max_moves, emb+feat)

        # MLP on all positions — fixed shapes, CUDAGraph-friendly
        x = F.relu(self.move_scorer.fc1(combined))
        x = F.relu(self.move_scorer.fc2(x))
        x = self.move_scorer.fc3(x)
        scores = torch.clamp(x.squeeze(-1), min=-50.0, max=50.0)  # (batch, max_moves)

        # Mask invalid (padding) slots to -inf
        arange = torch.arange(max_moves, device=board.device)
        valid_mask = arange.unsqueeze(0) < move_counts.unsqueeze(1)
        scores = scores.masked_fill(~valid_mask, float('-inf'))

        return scores

    def forward_padded_with_value(
        self,
        board: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor,
    ) -> tuple:
        """Padded forward with value head (training-optimized path).

        Fixed-size MLP on all positions — CUDAGraph-friendly.
        See forward_padded docstring for rationale.
        """
        batch_size = board.shape[0]
        max_moves = move_features.shape[1]

        board_embeddings = self.board_encoder(board)

        # Full-pad MLP (fixed shapes, no boolean indexing)
        expanded = board_embeddings.unsqueeze(1).expand(-1, max_moves, -1)
        combined = torch.cat([expanded, move_features], dim=-1)

        x = F.relu(self.move_scorer.fc1(combined))
        x = F.relu(self.move_scorer.fc2(x))
        x = self.move_scorer.fc3(x)
        scores = torch.clamp(x.squeeze(-1), min=-50.0, max=50.0)

        arange = torch.arange(max_moves, device=board.device)
        valid_mask = arange.unsqueeze(0) < move_counts.unsqueeze(1)
        scores = scores.masked_fill(~valid_mask, float('-inf'))

        if self.value_head is not None:
            values = self.value_head(board_embeddings)
        else:
            values = torch.zeros(batch_size, device=board.device)

        return scores, values

    def score_single(self, board: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
        """
        Score moves for a single position (convenience method).

        Args:
            board: Tensor of shape (1, BOARD_PLANES, 8, 8) or (BOARD_PLANES, 8, 8)
            move_features: Tensor of shape (num_moves, MOVE_FEATURE_SIZE)

        Returns:
            Tensor of shape (num_moves,) - score for each move
        """
        if board.dim() == 3:
            board = board.unsqueeze(0)

        num_moves = move_features.shape[0]

        # Encode board
        board_embedding = self.board_encoder(board)  # (1, embedding_size)

        # Expand for all moves (.contiguous() ensures a real copy for efficient matmul)
        expanded_embedding = board_embedding.expand(num_moves, -1).contiguous()

        # Score moves
        scores = self.move_scorer(expanded_embedding, move_features)

        return scores.squeeze(-1)


def create_model(
    embedding_size: int = 128,
    num_blocks: int = 4,
    hidden_size: int = 64,
    channels: int = 64,
    value_head_enabled: bool = False,
    value_head_hidden: int = 128,
) -> MoveScorerNet:
    """Create a new model with given architecture parameters."""
    return MoveScorerNet(
        embedding_size=embedding_size,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        channels=channels,
        value_head_enabled=value_head_enabled,
        value_head_hidden=value_head_hidden,
    )


def _infer_arch_from_state_dict(state_dict: dict) -> dict:
    """Infer architecture params from state_dict tensor shapes.

    Used for old checkpoints that were saved without arch_params metadata.
    """
    w = state_dict.get('board_encoder.input_conv.weight')
    channels = w.shape[0] if w is not None else 64

    num_blocks = 0
    while f'board_encoder.blocks.{num_blocks}.conv1.weight' in state_dict:
        num_blocks += 1
    num_blocks = num_blocks or 4

    b = state_dict.get('board_encoder.fc.bias')
    embedding_size = b.shape[0] if b is not None else 128

    b2 = state_dict.get('move_scorer.fc2.bias')
    hidden_size = b2.shape[0] if b2 is not None else 64

    value_head_enabled = 'value_head.fc1.weight' in state_dict
    vh = state_dict.get('value_head.fc1.bias')
    value_head_hidden = vh.shape[0] if vh is not None else 128

    return {
        'channels': channels,
        'num_blocks': num_blocks,
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'value_head_enabled': value_head_enabled,
        'value_head_hidden': value_head_hidden,
    }


def load_model(path: str, device: torch.device = None) -> MoveScorerNet:
    """Load a model from a checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # weights_only=False needed for checkpoint dicts; safe since we control the saved files
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle models saved after torch.compile() - strip "_orig_mod." prefix
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    # Read architecture params from checkpoint, or infer from state_dict shapes
    # for old checkpoints saved before arch_params was introduced.
    arch = checkpoint.get('arch_params', None)
    if not arch:
        arch = _infer_arch_from_state_dict(state_dict)
    model = create_model(
        embedding_size=arch.get('embedding_size', 128),
        num_blocks=arch.get('num_blocks', 4),
        hidden_size=arch.get('hidden_size', 64),
        channels=arch.get('channels', 64),
        value_head_enabled=arch.get('value_head_enabled', False),
        value_head_hidden=arch.get('value_head_hidden', 128),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def save_model(model: MoveScorerNet, path: str, **kwargs) -> None:
    """Save a model checkpoint with architecture params for portable loading."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'arch_params': getattr(model, 'arch_params', {
            'embedding_size': 128, 'num_blocks': 4, 'hidden_size': 64, 'channels': 64,
        }),
        **kwargs
    }
    torch.save(checkpoint, path)
