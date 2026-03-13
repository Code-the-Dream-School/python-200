# mini-etl module

This is a tiny module for experimenting with GitHub Copilot.

The goal is to make the tests pass.

## Run tests
From this folder:

    python -m pytest -q

Or if you prefer very quiet tests:

    python -m pytest -q --disable-warnings --tb=no

## Run the script
From your terminal, inside the folder:

    python mini_etl.py

This will run the mini ETL pipeline on the sample CSV file `sample_data.csv` and print the daily summary to the console. The first time you run it, you should see errors. The goal is to use github copilot to help you fix things. See the README for the lesson for more details. 

