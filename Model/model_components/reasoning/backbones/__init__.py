"""Swappable backbones for the reasoning band (issue #98).

``head_on_896`` (the default, :class:`~..reasoning_band.ReasoningBand`) trains
small heads on the World Model's Encoded Visual History.  The variants in this
package replace that input path — e.g. a frozen tiny-VLM reading the front
camera directly (Moondream2, proposed by @m-zain-khawaja in #98).
"""

from .moondream_frozen import MoondreamReasoningBranch

__all__ = ["MoondreamReasoningBranch"]
