import logging
import torch
import torch.nn.functional as F
from model import MattingNetwork

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

#PRETRAINED_MODEL = "./model/rvm_mobilenetv3_scripted.pt"
TORCHSCRIPT_MODEL = "./Cattery/RVM_Nuke.pt"



def load_rvm():
    """Load the Torchscript MattingNetwork model"""

    model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
    model.load_state_dict(torch.load('./model_weights/rvm_mobilenetv3.pth'))
    # model = torch.jit.load(PRETRAINED_MODEL)
    # model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model


class MattingModelNuke(torch.nn.Module):
    """
    A Nuke-compatible wrapper for the Robust Video Matting model.

    Ensures the input tensor matches Nuke's inference node format (1, inChan, inH, inW)
    and the output tensor follows (1, outChan, outH, outW).
    """

    def __init__(self):
        super().__init__()
        self.model = load_rvm()
        self.model_half = load_rvm().half()

    def forward(self, x):
        """
        Forward pass of RVM.

        Args:
            x (torch.Tensor): Input tensor of shape (1, channels, height, width).

        Returns:
            torch.Tensor: Matte output (1, 1, height, width)
        """
        b, c, h, w = x.shape  # Nuke provides (1, inChan, inH, inW)
        dtype = x.dtype
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        # Padding
        padding_factor = 14
        pad_h = ((h - 1) // padding_factor + 1) * padding_factor
        pad_w = ((w - 1) // padding_factor + 1) * padding_factor
        pad_dims = (0, pad_w - w, 0, pad_h - h)
        x = F.pad(x, pad_dims)

        # Ensure input is normalized for RVM (expects 3-channel RGB)
        if c == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        elif c != 3:
            raise ValueError(f"Expected input with 3 channels (RGB), but got {c} channels")

        # Use appropriate model precision
        if dtype == torch.float16:
            mask = self.model_half((x))[1]
        else:
            mask = self.model((x))[1]

        return mask[:, :1, :h, :w].contiguous()  # Output matte (1, 1, h, w)


def trace_rvm(model_file=TORCHSCRIPT_MODEL):
    """
    Traces the RVM model using MattingModelNuke and saves it as a TorchScript model.
    """
    with torch.jit.optimized_execution(True):
        rvm_nuke = torch.jit.script(MattingModelNuke().eval().requires_grad_(False))
        rvm_nuke.save(model_file)
        LOGGER.info(rvm_nuke.code)
        LOGGER.info(rvm_nuke.graph)
        LOGGER.info("Traced model saved: %s", model_file)


if __name__ == "__main__":
    trace_rvm()
