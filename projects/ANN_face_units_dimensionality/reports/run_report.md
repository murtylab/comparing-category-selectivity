# Face-selective unit dimensionality — run report

Generated: 2026-07-12T23:16:24

## Run configuration

- fROI: FFA, t-threshold: 7
- Stimuli per dataset: 1468 (AFHQv2 test in full; CelebA-HQ val and ImageNet val subsampled to match, fixed seed)
- Random-matched baseline repeats: 5 (seed=42)
- Device: mps, num_workers: 0, batch_size: 256
- Layers swept (all ReLU layers, relu1 + relu2 per block): layer1.0.relu1, layer1.0.relu2, layer1.1.relu1, layer1.1.relu2, layer2.0.relu1, layer2.0.relu2, layer2.1.relu1, layer2.1.relu2, layer3.0.relu1, layer3.0.relu2, layer3.1.relu1, layer3.1.relu2, layer4.0.relu1, layer4.0.relu2, layer4.1.relu1, layer4.1.relu2

## Face-selective unit counts per layer

| Model | Layer | Layer index | # face-selective | # total units |
|---|---|---|---|---|
| resnet18_random | layer1.0.relu1 | 1 | 779 | 200704 |
| resnet18_random | layer1.0.relu2 | 2 | 513 | 200704 |
| resnet18_random | layer1.1.relu1 | 3 | 559 | 200704 |
| resnet18_random | layer1.1.relu2 | 4 | 471 | 200704 |
| resnet18_random | layer2.0.relu1 | 5 | 157 | 100352 |
| resnet18_random | layer2.0.relu2 | 6 | 115 | 100352 |
| resnet18_random | layer2.1.relu1 | 7 | 52 | 100352 |
| resnet18_random | layer2.1.relu2 | 8 | 91 | 100352 |
| resnet18_random | layer3.0.relu1 | 9 | 52 | 50176 |
| resnet18_random | layer3.0.relu2 | 10 | 54 | 50176 |
| resnet18_random | layer3.1.relu1 | 11 | 11 | 50176 |
| resnet18_random | layer3.1.relu2 | 12 | 30 | 50176 |
| resnet18_random | layer4.0.relu1 | 13 | 7 | 25088 |
| resnet18_random | layer4.0.relu2 | 14 | 4 | 25088 |
| resnet18_random | layer4.1.relu1 | 15 | 1 | 25088 |
| resnet18_random | layer4.1.relu2 | 16 | 2 | 25088 |
| resnet18_trained | layer1.0.relu1 | 1 | 1433 | 200704 |
| resnet18_trained | layer1.0.relu2 | 2 | 536 | 200704 |
| resnet18_trained | layer1.1.relu1 | 3 | 92 | 200704 |
| resnet18_trained | layer1.1.relu2 | 4 | 208 | 200704 |
| resnet18_trained | layer2.0.relu1 | 5 | 49 | 100352 |
| resnet18_trained | layer2.0.relu2 | 6 | 160 | 100352 |
| resnet18_trained | layer2.1.relu1 | 7 | 62 | 100352 |
| resnet18_trained | layer2.1.relu2 | 8 | 224 | 100352 |
| resnet18_trained | layer3.0.relu1 | 9 | 499 | 50176 |
| resnet18_trained | layer3.0.relu2 | 10 | 1042 | 50176 |
| resnet18_trained | layer3.1.relu1 | 11 | 689 | 50176 |
| resnet18_trained | layer3.1.relu2 | 12 | 1222 | 50176 |
| resnet18_trained | layer4.0.relu1 | 13 | 456 | 25088 |
| resnet18_trained | layer4.0.relu2 | 14 | 643 | 25088 |
| resnet18_trained | layer4.1.relu1 | 15 | 550 | 25088 |
| resnet18_trained | layer4.1.relu2 | 16 | 1044 | 25088 |

## Effective dimensionality summary

| Model | Layer | Dataset | Unit type | # units | ED (mean) | ED (std) |
|---|---|---|---|---|---|---|
| resnet18_random | layer1.0.relu1 | afhqv2 | face_selective | 779 | 2.49 | 0.00 |
| resnet18_random | layer1.0.relu1 | afhqv2 | random_matched | 779 | 25.57 | 0.59 |
| resnet18_random | layer1.0.relu1 | celebahq | face_selective | 779 | 2.26 | 0.00 |
| resnet18_random | layer1.0.relu1 | celebahq | random_matched | 779 | 15.31 | 1.69 |
| resnet18_random | layer1.0.relu1 | imagenet | face_selective | 779 | 2.62 | 0.00 |
| resnet18_random | layer1.0.relu1 | imagenet | random_matched | 779 | 20.35 | 1.91 |
| resnet18_random | layer1.0.relu2 | afhqv2 | face_selective | 513 | 2.72 | 0.00 |
| resnet18_random | layer1.0.relu2 | afhqv2 | random_matched | 513 | 26.11 | 1.11 |
| resnet18_random | layer1.0.relu2 | celebahq | face_selective | 513 | 2.28 | 0.00 |
| resnet18_random | layer1.0.relu2 | celebahq | random_matched | 513 | 15.42 | 1.97 |
| resnet18_random | layer1.0.relu2 | imagenet | face_selective | 513 | 3.05 | 0.00 |
| resnet18_random | layer1.0.relu2 | imagenet | random_matched | 513 | 23.94 | 1.72 |
| resnet18_random | layer1.1.relu1 | afhqv2 | face_selective | 559 | 2.86 | 0.00 |
| resnet18_random | layer1.1.relu1 | afhqv2 | random_matched | 559 | 34.28 | 2.11 |
| resnet18_random | layer1.1.relu1 | celebahq | face_selective | 559 | 2.42 | 0.00 |
| resnet18_random | layer1.1.relu1 | celebahq | random_matched | 559 | 19.77 | 2.20 |
| resnet18_random | layer1.1.relu1 | imagenet | face_selective | 559 | 3.35 | 0.00 |
| resnet18_random | layer1.1.relu1 | imagenet | random_matched | 559 | 27.70 | 1.79 |
| resnet18_random | layer1.1.relu2 | afhqv2 | face_selective | 471 | 3.03 | 0.00 |
| resnet18_random | layer1.1.relu2 | afhqv2 | random_matched | 471 | 29.16 | 3.12 |
| resnet18_random | layer1.1.relu2 | celebahq | face_selective | 471 | 2.50 | 0.00 |
| resnet18_random | layer1.1.relu2 | celebahq | random_matched | 471 | 18.91 | 3.21 |
| resnet18_random | layer1.1.relu2 | imagenet | face_selective | 471 | 3.54 | 0.00 |
| resnet18_random | layer1.1.relu2 | imagenet | random_matched | 471 | 24.94 | 0.76 |
| resnet18_random | layer2.0.relu1 | afhqv2 | face_selective | 157 | 2.50 | 0.00 |
| resnet18_random | layer2.0.relu1 | afhqv2 | random_matched | 157 | 21.98 | 1.24 |
| resnet18_random | layer2.0.relu1 | celebahq | face_selective | 157 | 2.14 | 0.00 |
| resnet18_random | layer2.0.relu1 | celebahq | random_matched | 157 | 14.98 | 0.69 |
| resnet18_random | layer2.0.relu1 | imagenet | face_selective | 157 | 2.93 | 0.00 |
| resnet18_random | layer2.0.relu1 | imagenet | random_matched | 157 | 17.47 | 1.91 |
| resnet18_random | layer2.0.relu2 | afhqv2 | face_selective | 115 | 2.79 | 0.00 |
| resnet18_random | layer2.0.relu2 | afhqv2 | random_matched | 115 | 19.91 | 2.02 |
| resnet18_random | layer2.0.relu2 | celebahq | face_selective | 115 | 2.31 | 0.00 |
| resnet18_random | layer2.0.relu2 | celebahq | random_matched | 115 | 14.50 | 2.83 |
| resnet18_random | layer2.0.relu2 | imagenet | face_selective | 115 | 3.16 | 0.00 |
| resnet18_random | layer2.0.relu2 | imagenet | random_matched | 115 | 17.73 | 2.65 |
| resnet18_random | layer2.1.relu1 | afhqv2 | face_selective | 52 | 3.97 | 0.00 |
| resnet18_random | layer2.1.relu1 | afhqv2 | random_matched | 52 | 13.71 | 1.64 |
| resnet18_random | layer2.1.relu1 | celebahq | face_selective | 52 | 2.79 | 0.00 |
| resnet18_random | layer2.1.relu1 | celebahq | random_matched | 52 | 11.55 | 1.28 |
| resnet18_random | layer2.1.relu1 | imagenet | face_selective | 52 | 4.37 | 0.00 |
| resnet18_random | layer2.1.relu1 | imagenet | random_matched | 52 | 12.22 | 2.28 |
| resnet18_random | layer2.1.relu2 | afhqv2 | face_selective | 91 | 2.14 | 0.00 |
| resnet18_random | layer2.1.relu2 | afhqv2 | random_matched | 91 | 16.91 | 2.46 |
| resnet18_random | layer2.1.relu2 | celebahq | face_selective | 91 | 1.85 | 0.00 |
| resnet18_random | layer2.1.relu2 | celebahq | random_matched | 91 | 12.21 | 1.86 |
| resnet18_random | layer2.1.relu2 | imagenet | face_selective | 91 | 2.47 | 0.00 |
| resnet18_random | layer2.1.relu2 | imagenet | random_matched | 91 | 15.32 | 2.64 |
| resnet18_random | layer3.0.relu1 | afhqv2 | face_selective | 52 | 3.25 | 0.00 |
| resnet18_random | layer3.0.relu1 | afhqv2 | random_matched | 52 | 14.09 | 0.93 |
| resnet18_random | layer3.0.relu1 | celebahq | face_selective | 52 | 2.46 | 0.00 |
| resnet18_random | layer3.0.relu1 | celebahq | random_matched | 52 | 12.21 | 1.40 |
| resnet18_random | layer3.0.relu1 | imagenet | face_selective | 52 | 3.70 | 0.00 |
| resnet18_random | layer3.0.relu1 | imagenet | random_matched | 52 | 13.11 | 0.77 |
| resnet18_random | layer3.0.relu2 | afhqv2 | face_selective | 54 | 2.95 | 0.00 |
| resnet18_random | layer3.0.relu2 | afhqv2 | random_matched | 54 | 12.87 | 1.52 |
| resnet18_random | layer3.0.relu2 | celebahq | face_selective | 54 | 2.63 | 0.00 |
| resnet18_random | layer3.0.relu2 | celebahq | random_matched | 54 | 10.21 | 1.79 |
| resnet18_random | layer3.0.relu2 | imagenet | face_selective | 54 | 3.37 | 0.00 |
| resnet18_random | layer3.0.relu2 | imagenet | random_matched | 54 | 11.28 | 2.82 |
| resnet18_random | layer3.1.relu1 | afhqv2 | face_selective | 11 | 4.15 | 0.00 |
| resnet18_random | layer3.1.relu1 | afhqv2 | random_matched | 11 | 4.63 | 0.71 |
| resnet18_random | layer3.1.relu1 | celebahq | face_selective | 11 | 2.51 | 0.00 |
| resnet18_random | layer3.1.relu1 | celebahq | random_matched | 11 | 4.46 | 0.79 |
| resnet18_random | layer3.1.relu1 | imagenet | face_selective | 11 | 4.60 | 0.00 |
| resnet18_random | layer3.1.relu1 | imagenet | random_matched | 11 | 4.31 | 0.80 |
| resnet18_random | layer3.1.relu2 | afhqv2 | face_selective | 30 | 2.81 | 0.00 |
| resnet18_random | layer3.1.relu2 | afhqv2 | random_matched | 30 | 9.18 | 2.22 |
| resnet18_random | layer3.1.relu2 | celebahq | face_selective | 30 | 2.65 | 0.00 |
| resnet18_random | layer3.1.relu2 | celebahq | random_matched | 30 | 9.57 | 2.80 |
| resnet18_random | layer3.1.relu2 | imagenet | face_selective | 30 | 3.54 | 0.00 |
| resnet18_random | layer3.1.relu2 | imagenet | random_matched | 30 | 9.08 | 2.43 |
| resnet18_random | layer4.0.relu1 | afhqv2 | face_selective | 7 | 2.45 | 0.00 |
| resnet18_random | layer4.0.relu1 | afhqv2 | random_matched | 7 | 2.14 | 0.97 |
| resnet18_random | layer4.0.relu1 | celebahq | face_selective | 7 | 2.19 | 0.00 |
| resnet18_random | layer4.0.relu1 | celebahq | random_matched | 7 | 2.35 | 1.08 |
| resnet18_random | layer4.0.relu1 | imagenet | face_selective | 7 | 2.56 | 0.00 |
| resnet18_random | layer4.0.relu1 | imagenet | random_matched | 7 | 2.30 | 1.13 |
| resnet18_random | layer4.0.relu2 | afhqv2 | face_selective | 4 | 1.79 | 0.00 |
| resnet18_random | layer4.0.relu2 | afhqv2 | random_matched | 4 | 2.22 | 0.50 |
| resnet18_random | layer4.0.relu2 | celebahq | face_selective | 4 | 1.64 | 0.00 |
| resnet18_random | layer4.0.relu2 | celebahq | random_matched | 4 | 1.88 | 0.58 |
| resnet18_random | layer4.0.relu2 | imagenet | face_selective | 4 | 2.03 | 0.00 |
| resnet18_random | layer4.0.relu2 | imagenet | random_matched | 4 | 2.09 | 0.64 |
| resnet18_random | layer4.1.relu1 | afhqv2 | face_selective | 1 | 1.00 | 0.00 |
| resnet18_random | layer4.1.relu1 | afhqv2 | random_matched | 1 | 1.00 | 0.00 |
| resnet18_random | layer4.1.relu1 | celebahq | face_selective | 1 | 1.00 | 0.00 |
| resnet18_random | layer4.1.relu1 | celebahq | random_matched | 1 | 1.00 | 0.00 |
| resnet18_random | layer4.1.relu1 | imagenet | face_selective | 1 | 1.00 | 0.00 |
| resnet18_random | layer4.1.relu1 | imagenet | random_matched | 1 | 1.00 | 0.00 |
| resnet18_random | layer4.1.relu2 | afhqv2 | face_selective | 2 | 1.26 | 0.00 |
| resnet18_random | layer4.1.relu2 | afhqv2 | random_matched | 2 | 1.42 | 0.33 |
| resnet18_random | layer4.1.relu2 | celebahq | face_selective | 2 | 1.87 | 0.00 |
| resnet18_random | layer4.1.relu2 | celebahq | random_matched | 2 | 1.42 | 0.36 |
| resnet18_random | layer4.1.relu2 | imagenet | face_selective | 2 | 1.20 | 0.00 |
| resnet18_random | layer4.1.relu2 | imagenet | random_matched | 2 | 1.45 | 0.38 |
| resnet18_trained | layer1.0.relu1 | afhqv2 | face_selective | 1433 | 23.58 | 0.00 |
| resnet18_trained | layer1.0.relu1 | afhqv2 | random_matched | 1433 | 195.02 | 6.66 |
| resnet18_trained | layer1.0.relu1 | celebahq | face_selective | 1433 | 25.96 | 0.00 |
| resnet18_trained | layer1.0.relu1 | celebahq | random_matched | 1433 | 107.71 | 5.72 |
| resnet18_trained | layer1.0.relu1 | imagenet | face_selective | 1433 | 31.02 | 0.00 |
| resnet18_trained | layer1.0.relu1 | imagenet | random_matched | 1433 | 61.01 | 4.48 |
| resnet18_trained | layer1.0.relu2 | afhqv2 | face_selective | 536 | 16.14 | 0.00 |
| resnet18_trained | layer1.0.relu2 | afhqv2 | random_matched | 536 | 186.57 | 5.71 |
| resnet18_trained | layer1.0.relu2 | celebahq | face_selective | 536 | 24.61 | 0.00 |
| resnet18_trained | layer1.0.relu2 | celebahq | random_matched | 536 | 120.32 | 11.13 |
| resnet18_trained | layer1.0.relu2 | imagenet | face_selective | 536 | 32.06 | 0.00 |
| resnet18_trained | layer1.0.relu2 | imagenet | random_matched | 536 | 82.05 | 15.41 |
| resnet18_trained | layer1.1.relu1 | afhqv2 | face_selective | 92 | 28.35 | 0.00 |
| resnet18_trained | layer1.1.relu1 | afhqv2 | random_matched | 92 | 61.45 | 3.72 |
| resnet18_trained | layer1.1.relu1 | celebahq | face_selective | 92 | 15.65 | 0.00 |
| resnet18_trained | layer1.1.relu1 | celebahq | random_matched | 92 | 51.71 | 3.28 |
| resnet18_trained | layer1.1.relu1 | imagenet | face_selective | 92 | 28.60 | 0.00 |
| resnet18_trained | layer1.1.relu1 | imagenet | random_matched | 92 | 37.00 | 14.06 |
| resnet18_trained | layer1.1.relu2 | afhqv2 | face_selective | 208 | 14.95 | 0.00 |
| resnet18_trained | layer1.1.relu2 | afhqv2 | random_matched | 208 | 133.31 | 2.80 |
| resnet18_trained | layer1.1.relu2 | celebahq | face_selective | 208 | 21.41 | 0.00 |
| resnet18_trained | layer1.1.relu2 | celebahq | random_matched | 208 | 115.14 | 4.61 |
| resnet18_trained | layer1.1.relu2 | imagenet | face_selective | 208 | 18.98 | 0.00 |
| resnet18_trained | layer1.1.relu2 | imagenet | random_matched | 208 | 91.23 | 11.22 |
| resnet18_trained | layer2.0.relu1 | afhqv2 | face_selective | 49 | 9.39 | 0.00 |
| resnet18_trained | layer2.0.relu1 | afhqv2 | random_matched | 49 | 41.57 | 1.03 |
| resnet18_trained | layer2.0.relu1 | celebahq | face_selective | 49 | 13.26 | 0.00 |
| resnet18_trained | layer2.0.relu1 | celebahq | random_matched | 49 | 33.09 | 2.26 |
| resnet18_trained | layer2.0.relu1 | imagenet | face_selective | 49 | 21.24 | 0.00 |
| resnet18_trained | layer2.0.relu1 | imagenet | random_matched | 49 | 34.88 | 9.97 |
| resnet18_trained | layer2.0.relu2 | afhqv2 | face_selective | 160 | 30.35 | 0.00 |
| resnet18_trained | layer2.0.relu2 | afhqv2 | random_matched | 160 | 100.92 | 6.45 |
| resnet18_trained | layer2.0.relu2 | celebahq | face_selective | 160 | 24.13 | 0.00 |
| resnet18_trained | layer2.0.relu2 | celebahq | random_matched | 160 | 84.42 | 4.88 |
| resnet18_trained | layer2.0.relu2 | imagenet | face_selective | 160 | 58.22 | 0.00 |
| resnet18_trained | layer2.0.relu2 | imagenet | random_matched | 160 | 67.81 | 11.27 |
| resnet18_trained | layer2.1.relu1 | afhqv2 | face_selective | 62 | 10.72 | 0.00 |
| resnet18_trained | layer2.1.relu1 | afhqv2 | random_matched | 62 | 41.48 | 3.28 |
| resnet18_trained | layer2.1.relu1 | celebahq | face_selective | 62 | 12.71 | 0.00 |
| resnet18_trained | layer2.1.relu1 | celebahq | random_matched | 62 | 29.29 | 7.34 |
| resnet18_trained | layer2.1.relu1 | imagenet | face_selective | 62 | 30.18 | 0.00 |
| resnet18_trained | layer2.1.relu1 | imagenet | random_matched | 62 | 30.63 | 5.08 |
| resnet18_trained | layer2.1.relu2 | afhqv2 | face_selective | 224 | 48.34 | 0.00 |
| resnet18_trained | layer2.1.relu2 | afhqv2 | random_matched | 224 | 131.57 | 12.94 |
| resnet18_trained | layer2.1.relu2 | celebahq | face_selective | 224 | 49.88 | 0.00 |
| resnet18_trained | layer2.1.relu2 | celebahq | random_matched | 224 | 107.78 | 11.58 |
| resnet18_trained | layer2.1.relu2 | imagenet | face_selective | 224 | 90.87 | 0.00 |
| resnet18_trained | layer2.1.relu2 | imagenet | random_matched | 224 | 78.26 | 12.64 |
| resnet18_trained | layer3.0.relu1 | afhqv2 | face_selective | 499 | 59.64 | 0.00 |
| resnet18_trained | layer3.0.relu1 | afhqv2 | random_matched | 499 | 170.26 | 19.79 |
| resnet18_trained | layer3.0.relu1 | celebahq | face_selective | 499 | 58.85 | 0.00 |
| resnet18_trained | layer3.0.relu1 | celebahq | random_matched | 499 | 127.06 | 12.61 |
| resnet18_trained | layer3.0.relu1 | imagenet | face_selective | 499 | 146.46 | 0.00 |
| resnet18_trained | layer3.0.relu1 | imagenet | random_matched | 499 | 75.69 | 18.65 |
| resnet18_trained | layer3.0.relu2 | afhqv2 | face_selective | 1042 | 38.39 | 0.00 |
| resnet18_trained | layer3.0.relu2 | afhqv2 | random_matched | 1042 | 172.13 | 18.55 |
| resnet18_trained | layer3.0.relu2 | celebahq | face_selective | 1042 | 49.56 | 0.00 |
| resnet18_trained | layer3.0.relu2 | celebahq | random_matched | 1042 | 137.41 | 3.62 |
| resnet18_trained | layer3.0.relu2 | imagenet | face_selective | 1042 | 127.31 | 0.00 |
| resnet18_trained | layer3.0.relu2 | imagenet | random_matched | 1042 | 94.13 | 18.47 |
| resnet18_trained | layer3.1.relu1 | afhqv2 | face_selective | 689 | 22.00 | 0.00 |
| resnet18_trained | layer3.1.relu1 | afhqv2 | random_matched | 689 | 73.45 | 18.30 |
| resnet18_trained | layer3.1.relu1 | celebahq | face_selective | 689 | 22.88 | 0.00 |
| resnet18_trained | layer3.1.relu1 | celebahq | random_matched | 689 | 67.77 | 13.66 |
| resnet18_trained | layer3.1.relu1 | imagenet | face_selective | 689 | 62.99 | 0.00 |
| resnet18_trained | layer3.1.relu1 | imagenet | random_matched | 689 | 44.51 | 20.03 |
| resnet18_trained | layer3.1.relu2 | afhqv2 | face_selective | 1222 | 33.07 | 0.00 |
| resnet18_trained | layer3.1.relu2 | afhqv2 | random_matched | 1222 | 137.34 | 18.04 |
| resnet18_trained | layer3.1.relu2 | celebahq | face_selective | 1222 | 37.16 | 0.00 |
| resnet18_trained | layer3.1.relu2 | celebahq | random_matched | 1222 | 113.39 | 12.20 |
| resnet18_trained | layer3.1.relu2 | imagenet | face_selective | 1222 | 124.10 | 0.00 |
| resnet18_trained | layer3.1.relu2 | imagenet | random_matched | 1222 | 101.51 | 13.74 |
| resnet18_trained | layer4.0.relu1 | afhqv2 | face_selective | 456 | 25.81 | 0.00 |
| resnet18_trained | layer4.0.relu1 | afhqv2 | random_matched | 456 | 61.68 | 14.68 |
| resnet18_trained | layer4.0.relu1 | celebahq | face_selective | 456 | 37.48 | 0.00 |
| resnet18_trained | layer4.0.relu1 | celebahq | random_matched | 456 | 45.83 | 12.23 |
| resnet18_trained | layer4.0.relu1 | imagenet | face_selective | 456 | 101.50 | 0.00 |
| resnet18_trained | layer4.0.relu1 | imagenet | random_matched | 456 | 91.96 | 37.89 |
| resnet18_trained | layer4.0.relu2 | afhqv2 | face_selective | 643 | 33.13 | 0.00 |
| resnet18_trained | layer4.0.relu2 | afhqv2 | random_matched | 643 | 61.33 | 7.55 |
| resnet18_trained | layer4.0.relu2 | celebahq | face_selective | 643 | 45.09 | 0.00 |
| resnet18_trained | layer4.0.relu2 | celebahq | random_matched | 643 | 77.67 | 10.23 |
| resnet18_trained | layer4.0.relu2 | imagenet | face_selective | 643 | 119.50 | 0.00 |
| resnet18_trained | layer4.0.relu2 | imagenet | random_matched | 643 | 156.31 | 17.33 |
| resnet18_trained | layer4.1.relu1 | afhqv2 | face_selective | 550 | 18.59 | 0.00 |
| resnet18_trained | layer4.1.relu1 | afhqv2 | random_matched | 550 | 72.64 | 10.15 |
| resnet18_trained | layer4.1.relu1 | celebahq | face_selective | 550 | 34.88 | 0.00 |
| resnet18_trained | layer4.1.relu1 | celebahq | random_matched | 550 | 71.63 | 9.76 |
| resnet18_trained | layer4.1.relu1 | imagenet | face_selective | 550 | 110.80 | 0.00 |
| resnet18_trained | layer4.1.relu1 | imagenet | random_matched | 550 | 148.71 | 3.74 |
| resnet18_trained | layer4.1.relu2 | afhqv2 | face_selective | 1044 | 21.20 | 0.00 |
| resnet18_trained | layer4.1.relu2 | afhqv2 | random_matched | 1044 | 41.31 | 4.18 |
| resnet18_trained | layer4.1.relu2 | celebahq | face_selective | 1044 | 28.91 | 0.00 |
| resnet18_trained | layer4.1.relu2 | celebahq | random_matched | 1044 | 58.01 | 6.78 |
| resnet18_trained | layer4.1.relu2 | imagenet | face_selective | 1044 | 76.78 | 0.00 |
| resnet18_trained | layer4.1.relu2 | imagenet | random_matched | 1044 | 162.84 | 2.19 |