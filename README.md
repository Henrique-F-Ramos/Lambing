# Sheep Parturition Prediction

This repository contains scripts and notebooks for preprocessing sheep sensor data, training machine learning models, and deploying real-time classification via MQTT.

## File Overview

- **main.ipynb** - Jupyter Notebook for data preprocessing (finding max/min values for sheep data), model training, and initial testing.
- **new_filter.py** - Script for post-processing model outputs using a newly developed filtering method.
- **sender.py** - A simple MQTT sender that transmits sensor data.
- **receiver.py** - Receives MQTT data from `sender.py` and runs a trained single-second model on the data.
- **receiver_roll.py** - Receives MQTT data from `sender.py` and runs a trained sequential model on the data.
- **requirements.txt** - Lists all required Python packages and their versions for the virtual environment.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preprocessing & Model Training
Run the Jupyter notebook to preprocess data, train models, and conduct initial testing:
```bash
jupyter notebook main.ipynb
```

### Running MQTT Components

1. Start the sender:
   ```bash
   python sender.py
   ```

2. Start the receiver for single-second model:
   ```bash
   python receiver.py
   ```

3. Start the receiver for sequential model:
   ```bash
   python receiver_roll.py
   ```

### Post-processing Model Outputs
After running the models, apply the new filtering method:
```bash
python new_filter.py
```

## Notes
- Ensure the MQTT broker is properly configured before running `sender.py` and `receiver.py/receiver_roll.py`.
- Modify `main.ipynb`,`reviever.py` and `reciever_roll.py` as needed for different datasets or models.
- The `requirements.txt` file ensures version compatibility; install dependencies in a virtual environment to avoid conflicts.