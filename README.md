# BEAM data analysis scripts

This repository contains Python code for analyzing simulation data generated from a Beam run, a mobility simulation framework. The code includes classes and functions for handling input directories, processing path traversal events, and generating useful output data.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Beam simulation outputs detailed information about the movement of agents (e.g., vehicles, pedestrians) in a simulated environment. The provided Python code focuses on processing and analyzing key aspects of the simulation data, offering functionalities such as:

- **Input Handling:** Classes like `BeamRunInputDirectory` facilitate the organization and retrieval of input data for analysis.

- **Event Data Processing:** Functions like `fixPathTraversals` enhance the raw path traversal events, adding additional columns for corrected occupancy, vehicle miles, passenger miles, and more.

- **Output Data Generation:** The code generates meaningful output data frames, including link statistics, mode-specific events, and mode-specific vehicle miles traveled (VMT).

## Project Structure

Explain the structure of the project directory and the role of each major file. Provide a brief overview of the key components.

- `src/`: Contains source code files.
  - `input.py`: Defines input-related classes.
  - `transformations.py`: Defines functions for data transformations.
- `analysis/`: Contains analysis scripts or notebooks.
- `README.md`: Documentation file (you are here).

## Installation

Explain how to install or set up the project dependencies, if any.

```bash
pip install -r requirements.txt
