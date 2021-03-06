# Choose-Your-Own-Mushroom-Classification-Project

Data Source: https://archive.ics.uci.edu/ml/datasets/mushroom

This Mushroom Classification project is for the online Harvard Data Science Capstone course. 

Mushroom_Project.R - R script for performing supervised classification machine learning tasks. This script performs 5 machine learning tasks. It prints accuracy comparison data frame and top 5 feature comparison data frame as outputs along with 5 tree based models in the Workspace.

Mushroom_Project.Rmd - mushroom report in Rmd format. It provides project overview, data analysis, model approach, result, and conclusions. It also has a reference section for all relevant mushroom domain knowledge and machine learning information links.

Mushroom_Project.pdf - mushroom report in PDF format. It contains the rendered figure, table, and text of Mushroom_Project.Rmd without the code chunks.

mushroom_dataset.csv - mushroom data set downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data as a csv file.

**Note:**  
Both the R script and the Rmd will take a while to finish.  

Some seeding issues prevent full reproducibility as noted on for example: https://github.com/mlr-org/mlr/issues/938  

random forest with recursive feature elimination tends to produce different results on different systems, this can cause slight different feature results for model 3 to model 5  
