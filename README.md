# Monitoring driver behaviour using smartphone sensors.
## Created as a part of team project 'ZPI'.
### Technologies:
* Python
* Python libraries: 
  * pandas, sklearn, statistics, numpy, matlab.engine
* Matlab
### I. Conducting experiments and gathering data.
Matlab mobile app was used to gather accelerometer and GPS data. Matlab mobile automatically uploads gathered data to the cloud. Then you can easily download the files on your machine.

<p align='center'>
<img align="center" width="300" height="500" src="https://play-lh.googleusercontent.com/-y6uiyXP3XyGVdlRt7AvDf8utdrbn4-X44EE0wmrnHgspS_AS0nxuW5OhMA1NpaVx_k=w1920-h977-rw">
</p>

### II. Data processing.
This phase consisted of several steps:
1. Converting *mat* files to *csv* format in Matlab.
2. Transforming the data into pandas dataframe object.
3. Importing the converted files and processing them in python using filtering and standarisation. 

### III. Feature extraction.
Preporcessed data is now ready for feature extraction. 

### IV. Maneuver classification.

### V. Generating a report for the user.
