# RVC Test Environment

## Environment Info

- **Conda Environment Name**: `rvc_test`
- **Environment Path**: `E:\Anaconda\envs\rvc_test`
- **Python Version**: 3.10.20
- **PyTorch Version**: 2.1.2+cu121
- **Torchvision Version**: 0.16.2+cu121
- **Torchaudio Version**: 2.1.2+cu121
- **CUDA Version (torch)**: 12.1

## GPU Verification

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **GPU Count**: 1
- **CUDA Capability**: (8, 9)
- **CUDA Available**: True
- **CUDA Tensor Computation**: PASSED

## Environment Status

- **Status**: Ready for RVC integration
- **Created**: 2026-07-30

## Notes

- NumPy version 2.2.6 installed (torch 2.1.2 compiled with NumPy 1.x)
- Warning present but does not block CUDA functionality
- Recommend downgrading numpy to <2 for full compatibility: `pip install "numpy<2"`

## Next Steps

1. Install RVC dependencies (fairseq, praat-parselmouth, etc.)
2. Clone RVC repository
3. Download pre-trained models
4. Test RVC inference pipeline

## Commands Used

```bash
# Create environment
conda create -y -n rvc_test python=3.10

# Install PyTorch with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Verify environment
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
