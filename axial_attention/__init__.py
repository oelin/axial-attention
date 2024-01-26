import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


class Attention(nn.Module):
    """Attention.

    Example
    -------
    >>> module = Attention(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x, mask=None)
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        """

        super().__init__()

        self.heads = heads
        
        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.linear_1(x)
        q, k, v = rearrange(x, '... t (n h e) -> n ... h t e', n=3, h=self.heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.linear_2(rearrange(x, '... h t e -> ... t (h e)'))

        return x


class AxialAttention(nn.Module):
    """Axial attention.

    Example
    -------
    >>> module = AxialAttention(
    ...     embedding_dimension=256,
    ...     heads=16,
    ...     axis=-2,  # Mix along the penultimate axis.
    ... )
    >>> x = torch.randn((1, ..., 10, 256))
    >>> x = module(x)  # Shape: (1, ..., 10, 256).
    """

    def __init__(
        self, 
        *, 
        embedding_dimension: int, 
        heads: int,
        axis: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        axis : int
            The axis to mix along. It's assumed that the last axis is the
            embedding axis.
        """

        super().__init__()

        self.axis = axis

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x.transpose(self.axis, -2)  # Transpose to the sequence axis.
        x = self.attention(x, mask=mask)
        x = x.transpose(self.axis, -2)  # Transpose back.

        return x
