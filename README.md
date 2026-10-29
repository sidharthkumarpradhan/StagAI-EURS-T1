# StagAI: EURS @ MaCVi WACV 2027, Track 1 reproduction

Reported testing-phase submission: **19062** (`6e2b4bf589d1`, composite 0.637).

```
repro.yaml           team StagAI, track eurs-t1, submission 19062
METHOD.md            approach, training data, external data
run.py               main(model_dir, input_dir, output_dir, device)
src/                 macvi library (models, data, evaluation) and restorers.py
weights/             weights_link.yaml: one public .tar.gz (teachers/, restorers/, key_affines.npz)
Dockerfile           PyTorch 2.5.1 image + pinned packages; bakes the DINOv3 backbone into an offline HF cache
tools/               predict_track1.py (our original rebuild script, also packages the zip) and the packaging script
repro_check.py, docker/   from the organizers' template
```

```bash
python repro_check.py . --download-weights
docker build -t stagai-t1 .
docker run --rm --network none --gpus all -v "$PWD/weights":/pkg/weights:ro -v /path/to/degraded:/in:ro -v /tmp/out:/out \
    stagai-t1 python /pkg/repro_check.py /pkg --run --input /in --output /out
```

Code: MIT (see LICENSE). Weights: trained by team StagAI; the DINOv3 backbone is used under the DINOv3 License.
