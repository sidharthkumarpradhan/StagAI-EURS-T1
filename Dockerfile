# Team StagAI, Track 1. Built from the repository root (with network); the container then runs with no network.
# Same PyTorch as our own runs (2.5.1). The frozen DINOv3 ViT-L/16 backbone of the teachers is not in our checkpoints:
# it is fetched here, at build time, into a read-only Hugging Face cache, and loaded offline at run time.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV HF_HOME=/opt/hf
RUN python -c "import timm; timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1
COPY . /pkg
WORKDIR /pkg
