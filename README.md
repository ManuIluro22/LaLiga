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
  ├─── laliga.sqlite			      # The database
  ├─── README.md
  ├─── requirements.txt			      # List of libraries needed to run the project
  └─── settings.py				# General parameters of the program
```

### How to run it

Down below you can see an output snip. Once you've installed dependences (```pip install -r requirements.txt```) you can try it yourself:

```console
user@user:~/RI/laliga/la-quiniela$ python3 cli.py train --training_seasons 2005:2020 --model_name gbc_long
Model succesfully trained and saved in /home/manu/RI/laliga/la-quiniela/models/gbc_long
user@user:~/RI/laliga/la-quiniela$ python cli.py predict 2020-2021 1 25 --model_name gbc_long
Loading matchday 25 in season 2020-2021, division 1...
Matchday 25 - LaLiga - Division 1 - Season 2020-2021
===============================================================================================
            Alavés             vs           CA Osasuna           --> X --- confidence: 36.70%
        Celta de Vigo          vs        Real Valladolid         --> 1 --- confidence: 34.54%
           Cádiz CF            vs           Real Betis           --> X --- confidence: 34.86%
            Getafe             vs            Valencia            --> 2 --- confidence: 34.42%
          Granada CF           vs            Elche CF            --> 1 --- confidence: 35.61%
           Levante             vs            Athletic            --> X --- confidence: 33.66%
         Real Madrid           vs         Real Sociedad          --> 1 --- confidence: 41.74%
           SD Eibar            vs           SD Huesca            --> X --- confidence: 35.23%
          Sevilla FC           vs           Barcelona            --> 2 --- confidence: 39.82%
          Villarreal           vs        Atlético Madrid         --> 2 --- confidence: 41.44%
```

Here, we call ```train``` to train the model using seasons from 2005 to 2020, and then we perfom a prediction of 25th matchday of 2021-2022 season at 1st Division using ```predict```. We can see that apart from the prediction, it also shows how much confident is the model on it.

Check out options on ```train``` and ```predict``` using ```-h``` option.

### Data

The data is provided as a SQLite3 database. This database contains the following tables:

   * ```Matches```: All the matches played between seasons 1928-1929 and 2021-2022 with the date and score. Columns are ```season```,	```division```, ```matchday```, ```date```, ```time```, ```home_team```, ```away_team```, ```score```. Have in mind there is no time information for many of them and also that it contains matches still not played from current season.
   * ```Predictions```: The table for you to insert your predictions. It is initially empty. Columns are ```season```,	 ```timestamp```, ```division```, ```matchday```, ```home_team```, ```away_team```, ```prediction```.

The data source is [Transfermarkt](https://www.transfermarkt.com/), and it was scraped using Python's library BeautifulSoup4.

