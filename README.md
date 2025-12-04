# Advanced Federated Learning Security Auditing

A comprehensive Python blueprint demonstrating **Model Poisoning Attacks** and **Server-side Defense Mechanisms** within a Federated Learning (FL) framework.

## Overview

This project simulates a realistic FL security scenario:
- **5 simulated clients** training a CNN on FashionMNIST
- **1 malicious client** executing a model poisoning attack
- **Robust server-side defense** using Norm Clipping + Coordinate-wise Median aggregation

## Features

### Attack Simulation
- **Model Poisoning Attack**: A malicious client amplifies its weight updates by a large factor (100x) to poison the global model
- Demonstrates how a single bad actor can significantly degrade model performance

### Defense Mechanisms
- **Norm Clipping**: Limits the L2 norm of client updates to prevent extreme values
- **Coordinate-wise Median Aggregation**: Uses median instead of mean to resist outliers
- Combined defense effectively mitigates the attack

## Requirements

- Python 3.8+
- PyTorch
- Flower (flwr)
- NumPy
- scikit-learn
- torchvision

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete security audit simulation:

```bash
python federated_learning_security_audit.py
```

The script will run three scenarios:
1. **Baseline**: No attack (all honest clients)
2. **Scenario A**: Attack WITHOUT defense (vulnerable FedAvg)
3. **Scenario B**: Attack WITH defense (Robust FedAvg)

## Expected Results

| Scenario | Final Accuracy | Status |
|----------|---------------|--------|
| Baseline (No Attack) | ~70-80% | ✅ Normal training |
| Attack WITHOUT Defense | Significantly degraded | ⚠️ Vulnerable |
| Attack WITH Defense | Close to baseline | ✅ Protected |

## Technical Architecture

### Model
- **SimpleCNN**: 2 convolutional layers + 2 fully connected layers for FashionMNIST classification

### Attack Logic (`malicious_client_update`)
```python
# Amplify the gradient update by a scale factor
update = new_params - old_params
poisoned_update = update * SCALE_FACTOR  # Default: 100x
poisoned_params = old_params + poisoned_update
```

### Defense Logic (`RobustFedAvg`)
```python
# 1. Norm Clipping
if norm(update) > threshold:
    update = update * (threshold / norm(update))

# 2. Coordinate-wise Median
aggregated = median(all_updates, axis=0)  # Instead of mean
```

## Configuration

Key parameters in the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLIENTS` | 5 | Number of FL clients |
| `NUM_ROUNDS` | 5 | FL training rounds |
| `MALICIOUS_CLIENT_ID` | 2 | Which client is malicious |
| `ATTACK_SCALE_FACTOR` | 100.0 | Attack amplification factor |
| `NORM_CLIP_THRESHOLD` | 10.0 | Max allowed update norm |

## License

MIT License - See [LICENSE](LICENSE) for details

## References

- [Flower Framework](https://flower.dev/)
- [Byzantine-Robust Aggregation](https://arxiv.org/abs/1703.02757)
- [Model Poisoning Attacks](https://arxiv.org/abs/1811.12470)