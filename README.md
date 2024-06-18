# Optuna Stellarator Optimization

This project provides a command-line interface (CLI) for running hyperparameter optimization using Optuna. The optimization results can be stored in a local PostgreSQL database, which can be managed using Adminer.

## Prerequisites

- Docker
- Docker Compose
- Python Python 3.10.6

## Setup

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone git@github.com:abriosi/stellarator-optuna.git
cd stellarator-optuna
```

### 2. Start PostgreSQL and Adminer

Use Docker Compose to start the PostgreSQL and Adminer services:

```bash
docker-compose up -d
```

This will start two services:
- **PostgreSQL**: Running on port `5432`
- **Adminer**: Running on port `8080`

### 3. Install Python Dependencies

Install the dependencies using using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Optimization

You can run the optimization script with the following command:

```bash
python main.py --sampler <sampler_name> --trials <number_of_trials> --seed <random_seed> --storage <postgresql_url> --study-name <study_name>
```

#### Parameters

- `--sampler`: The sampler to use for optimization. Supported values are:
  - `RandomSampler`
  - `TPESampler`
  - `CmaEsSampler`
  - `GPSampler`
  - `PartialFixedSampler`
  - `NSGAIISampler`
  - `QMCSampler`
- `--trials`: Number of trials for optimization (default: 100).
- `--seed`: Seed for reproducibility (optional).
- `--storage`: Database URL for Optuna storage (optional). Use the format `postgresql://user:password@localhost:5432/optuna`.
- `--study-name`: Name of the study (required).

#### Example

```bash
python main.py --sampler RandomSampler --trials 200 --seed 42 --storage postgresql://user:password@localhost:5432/optuna --study-name my_study
```

### Accessing the Database with Adminer

Open your web browser and go to `http://localhost:8080`. Use the following credentials to log in:

- **System**: PostgreSQL
- **Server**: postgres (this is the name of the service defined in `docker-compose.yml`)
- **Username**: user
- **Password**: password
- **Database**: optuna

## Files

- `main.py`: The main Python script for running Optuna optimization.
- `docker-compose.yml`: Docker Compose configuration file to set up PostgreSQL and Adminer services.

## Stopping the Services

To stop the PostgreSQL and Adminer services, run:

```bash
docker-compose down
```

## Cleaning Up

To remove the Docker containers and volumes, run:

```bash
docker-compose down -v
```
