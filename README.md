## LaQuiniela of LaLiga

Team members: 

Manuel Arnau Fernández - NIU: 1597487

Pere Bancells i Blazquez - NIU: 1563650

Harry Wolimba Hall - NIU: 1733432

Jordi Ren - NIU: 1739708

This repo contains a ML project to predict the outcome of a matchday in LaLiga. The data used is in ```laliga.sqlite```.

### Repository structure

```
quiniela/
  ├─── analysis/				# Jupyter Notebooks used to explore the data
  │          ...
  ├─── logs/					# Logs of the program are written
  │          ...
  ├─── models/					# The place were trained models are stored
  │          ...
  ├─── quiniela/				# Main Python package
  │          ...
  ├─── reports/					# The place to save HTML / CSV / Excel reports
  │          ...
  ├─── .gitignore
  ├─── cli.py					# Main executable. Entrypoint for CLI
  ├─── laliga.sqlite			# The database
  ├─── README.md
  ├─── requirements.txt			# List of libraries needed to run the project
  └─── settings.py				# General parameters of the program
```

### How to run it

You are provided with a fully-functional dummy model. Once you've installed dependences (```pip install -r requirements.txt```), which is only Pandas in this dummy case, you can try it yourself:

```console
foo@bar:~$ python cli.py train --training_seasons 2010:2020
Model succesfully trained and saved in ./models/my_quiniela.model
foo@bar:~$ python cli.py predict 2021-2022 1 3
Matchday 3 - LaLiga - Division 1 - Season 2021-2022
======================================================================
         RCD Mallorca          vs            Espanyol            --> X
           Valencia            vs             Alavés             --> X
        Celta de Vigo          vs            Athletic            --> X
        Real Sociedad          vs            Levante             --> X
           Elche CF            vs           Sevilla FC           --> X
          Real Betis           vs          Real Madrid           --> X
          Barcelona            vs             Getafe             --> X
           Cádiz CF            vs           CA Osasuna           --> X
        Rayo Vallecano         vs           Granada CF           --> X
       Atlético Madrid         vs           Villarreal           --> X
```

Here, we call ```train``` to train the model using seasons from 2010 to 2020, and then we perfom a prediction of 3rd matchday of 2021-2022 season at 1st Division using ```predict```. Of course, that's a terrible prediction: that's why it's a dummy model!! Call to ```train``` did literally nothing, and ```predict``` always return ```X ```. It is your job to make something interesting.

Check out options on ```train``` and ```predict``` using ```-h``` option. You are free to add any other argument that you find necessary.

### Data

The data is provided as a SQLite3 database. This database contains the following tables:

   * ```Matches```: All the matches played between seasons 1928-1929 and 2021-2022 with the date and score. Columns are ```season```,	```division```, ```matchday```, ```date```, ```time```, ```home_team```, ```away_team```, ```score```. Have in mind there is no time information for many of them and also that it contains matches still not played from current season.
   * ```Predictions```: The table for you to insert your predictions. It is initially empty. Columns are ```season```,	 ```timestamp```, ```division```, ```matchday```, ```home_team```, ```away_team```, ```prediction```.

The data source is [Transfermarkt](https://www.transfermarkt.com/), and it was scraped using Python's library BeautifulSoup4.

