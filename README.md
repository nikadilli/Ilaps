# Ilaps
**python data reduction and imaging application for LA-ICP-MS**

## About
LA-ICP-MS is an extremely powerful analytical technique for spatially resolved analysis of solid samples. It produces complex, time-dependent signals which require different approach than signals produced by solution sample introduction. 

Ilaps (**I**maging of **L**aser **a**blation **p**lasma **s**pectrometry) is a python application aimed on data created with laser ablation inductively coupled plasma mass spectrometry. It is suited for a spot and line analysis as well as an elemental imaging. The data can be imported directly from mass spectrometry data acquisition as .csv, .asc or .xlsx files.

## Run from code 
To run the application, all necessary dependences must be installed. Ilaps is a python desktop aplication, therefore it is necessary to install python and required libraries before the first time the app is started.

1. Python 3 is necessary to run the code. Download [here](https://www.python.org/downloads/) and follow the instalation.
2. Dowload or clone this repository.
3. Open terminal/cmd and navigate to the Ilaps folder.
  * `cd path/to/folder/Ilaps`
4. Create virtual enviroment. 
  * `python -m venv venvname`
5. Activate virtual enviroment. 
  * Windows `call venvname/Scripts/activate.bat`
  * Linux/Mac `source venvname/bin/activate`
6. Instal python libraries required for Ilaps.
  * `pip install -r requirements.txt`
7. Run Ilaps from python.
  * `python GUI.py`

If everything is already installed, follow only steps 3., 5. and 7. to run Ilaps. 
After each major release it is recomended to download the new version of Ilaps.
