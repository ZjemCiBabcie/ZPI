# Monitoring driver behaviour using smartphone sensors. 
Full project documentation (in Polish) available here: https://github.com/wkberezowski/ZPI/blob/Z3/PROJECT%20DOCUMENTATION%20IN%20POLISH.pdf.

## Created as a part of team project 'ZPI'.
### Technologies:
* Python
* Python libraries: 
  * pandas, sklearn, statistics, numpy, matlab.engine
* Matlab
### I. Conducting experiments and gathering data.
Matlab mobile app was used to gather accelerometer and GPS data. 
### II. Data processing.
Used filtering and standarisation for data preprocessing.
### III. Feature extraction.
Performed feature extaction to detect braking and accelerating maneuvers.
### IV. Maneuver classification.
Classified detected maneuvers into agressive and normal.
### V. Generating a report for the user.
The report contained information such as: number of aggresive and normal maneuvers or average and top speed.
