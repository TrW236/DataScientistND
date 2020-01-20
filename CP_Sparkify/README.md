# Sparkify Project

This project is one of the capstone projects provided by Udacity `Data Scientist Nanodegree`.

## Folder Structure

```
.
├── img
├── mini_sparkify_event_data.json
├── README.md
└── Sparkify.ipynb
```

* note that file `mini_sparkify_event_data.json` contains the raw data to be analyzed. This file can be obtained from `Udacity`, and is `not` publicly available.

* `Sparkify.ipynb` is the main project file containing all the codes needed for analysis.

## Main Libraries Used

* Plotting Libs: `plotly`, `matplotlib`, `seaborn`
* Data Processing Lib: `pandas`, `numpy`, `pyspark`

## Project Motivation

Data are massively generated every day around the world, especially in the field of business related to the internet.  The event data about users' actions can be analyzed, and the results can be some guides for making business decisions.

This project is the capstone project of Data Scientist Nanodegree. Udacity provides a sample-dataset recording user-events, on which we can work. 

## Project Summary

The main workflow of this project is presented in the following graph:

<img src="./img/Sparkify_workflow.png" alt="workflow" width="140">

Through doing this project, I learned many skills relating to using the framework Pyspark, and have an interesting result in completing the project.

<img src="./img/bar_importance.png" alt="BarPlot Importance" width="700">

The result shows that the number of days from registration to the event is very relevant, determining whether this user churn or not. We could study further why this feature is important. After an appropriate study, we could make some changes to the business and improve the service quality, so that more users could be gained, which means more potential revenues for the company.

* I have written a [report](./report.md) describing this project in detail.

## Acknowledgements

*    Udacity Data Scientist Nanodegree
