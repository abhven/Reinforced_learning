# Reinforcement Learning - HW2 -Abhishek Venkataraman

This is the submission towards HW2 of EECS 598 Section 07. This file contains the instructions to reproduce the results obtained by me. 

## Setting up the working directory

### Folder structure

Dowload and unzip the frozen_lakes.tar.gz file from and unzip it using the command:
```
tar -xzf frozen_lakes.tar.gz 
```
Download and unzip the submission file abhven.zip in the same directory as frozen lakes folder. You have the following directory structure:
```
program directory
--hw2.py
--hw2_tests.py
--requirements.txt
--frozen_lakes
  --__init__.py
  --lakes_envs.py
```

### Setting up dependencies 

For this assignment, the libraries required are 
```
numpy
matplotlib
gym
```

To install the dependencies run the command:
```
pip install -r requirements.txt 
```

### Exection of code 
All the commands need to be executed in the program directory

For part 1, use the command
```
python hw2_tests.py 1
```

For part 2, use the command
```
python hw2_tests.py 2
```

For part 3, use the command
```
python hw2_tests.py 3
```

