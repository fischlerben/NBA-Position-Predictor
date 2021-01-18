# NBA Position Predictor
![NBA](https://cdn.vox-cdn.com/thumbor/ytC-ZCsT-G-M1Fscy7oZUeZE9X0=/1400x788/filters:format(png)/cdn.vox-cdn.com/uploads/chorus_asset/file/19725578/TheRinger_Top25NBAPlayers_2.png)

This Machine Learning example uses 15 seasons (2005-2020) of NBA player statistics (the features) to predict the position of each player (the target).  Machine Learning models include Decision Trees, Random Forests, SVMs and Gradient Boosting.  An example PCA transformation of X-data is included as well.  Each model is then evaluated and compared to the others to determine which Machine Learning model works best for this particular set of NBA data.

Note: Stats are per 36 minutes, *not* per game.  This ensures that the players' statistics are a fairer representation of the position's output.  For instance, if a player only averages 5 min/game, his per game stats would be representative of that *player's* individual production, but for the purposes of predicting their position, does not say much.  Stats per 36 minutes scales the data and makes it more meaningful.

CSV Datasets obtained from Basketball-Reference.com

---

## Data Pre-Processing:
Pre-processing mainly consisted of cleaning up the data and filtering the dataset to only show players that matched a certain criteria.  Each season was read in as a separate CSV, and then concatenated into one single DataFrame that I could work from.  Next, I filtered the DataFrame to only include players that had a minimum of 820 minutes played/season (10 min/game * 82 games/season).  I did this because rows containing less than this acceptable level of Minutes Played would unfairly skew the data.  For instance, if a player only plays 1 minute for a team and scores 2 points, his Points Per 36 Minutes would be 72; clearly not a fair representation.

    original_df = original_df[original_df.MP >= 820]

I then sliced that DataFrame to only include relevant features (dropping columns such as Name, Age and Team; features that aren't actually player statistics and have no bearing on their position).  Next was to convert mixed positions, like "C-PF" and "PF-SF" into one of the 5 main basketball positions:

    df = df.replace("C-PF","C")

This was done for every combination.  Last thing to do was to round all columns to 2 decimal places, for simplicity's sake:

    df = df.round({'PTS': 2})
    
This was done for every column.  My original dataframe consisted of 7089 rows and 29 columns; after filtering players and dropping irrelevant columns, I was left with a dataframe of 4051 rows and 21 columns.  This meant 4051 rows of player statistics for my model to work with, along with 20 features for the model to test (20 features, 1 target column).  Here is a sample of the dataframe, showing only 5 rows, but all columns:

![OriginalDF](/Screenshots/OriginalDF.png?raw=true)

---

## Pre-Model Visualizations:
### Summary Stats by Position:

    summary_df = df.groupby('Pos').mean()

![SummaryTable](/Screenshots/SummaryTable.png?raw=true)
#### We can glean useful information from the above chart.  We see that in terms of, say, PTS, there is not much difference position-by-position: each averages about 15 points/36 minutes.  However we see serious differences in a few categories, which are the ones we would expect; most namely, TRB (Total Rebounds) and AST (Assists).  While point guards take the ball up the court and distribute the ball, leading to assists (PG's average 6.2 assists while C's average 2.1 assists), centers are taller and have a more rebound-heavy responsibility (C's average 10.1 rebounds while PG's average 3.9 rebounds).  This gives a good idea about what to expect in the model, especially in terms of what the model deems to be the most important features that help determine a player's position.

### Simple Bar Chart:

    bar_chart_df = summary_df[['PTS', 'TRB', 'AST', 'STL', 'BLK']]
    bar_chart_df.plot(kind='bar', figsize = (12, 8), title='Bar Chart of Main Stats across all 5 Positions')

![BarChart](/Screenshots/BarChart.png?raw=true)
#### This simple bar plot shows the five main basketball statistics - Points, Rebounds, Assists, Steals and Blocks - and how they are distributed among the five positions.  As mentioned above, points/36 minutes do not differ that greatly position-by-position; rebounds and assists, however, vary greatly.  We can also see in yellow that centers average signficantly more blocks than other positions - blocks in general are hard to come by, so while centers average only 1.6 blocks/36 minutes, point guards average only .26 blocks/36 minutes, meaning in this particular dataset, centers average 144% more blocks/36 minutes than do point guards.  This feautre could potentially be deemed one of the more important ones by the upcoming models.

### Seaborn Pair Chart:

    sns_df = df[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'Pos']].head(300)
    sns_df = sns_df.reset_index()
    sns_df = sns_df.drop('index', axis=1)

    sns_plot = sns.pairplot(sns_df, hue='Pos', size=2)
    sns_plot
    
![SeabornChart](/Screenshots/SeabornChart.png?raw=true)

#### Explanation.




### Decision Tree 1 Confusion Matrix:
![DT1_cm](/Screenshots/DT1_cm.png?raw=true)

### Decision Tree 1 Specific Predictions:
![specific_pred_1](/Screenshots/specific_pred_1.png?raw=true)

### Feature Importance Table:
![importance_table](/Screenshots/importance_table.png?raw=true)

### Decision Tree 2 Confusion Matrix:
![DT2_cm](/Screenshots/DT2_cm.png?raw=true)

### Decision Tree 1 Results vs. Decision Tree 2 Results:
![DT1_DT2_comp](/Screenshots/DT1_DT2_comp.png?raw=true)

### Decision Tree 3 Confusion Matrix:
![DT3_cm](/Screenshots/DT3_cm.png?raw=true)

### Decision Tree 3 Specific Predictions:
![specific_pred_2](/Screenshots/specific_pred_2.png?raw=true)

### Decision Tree 1 Results vs. Decision Tree 3 Results:
![DT1_DT3_comp](/Screenshots/DT1_DT3_comp.png?raw=true)

### Random Forest Confusion Matrix:
![RF_cm](/Screenshots/RF_cm.png?raw=true)

### Random Forest Specific Predictions:
![specific_pred_3](/Screenshots/specific_pred_3.png?raw=true)

### Random Forest Results vs. Decision Tree 3 Results:
![DT3_RF_comp](/Screenshots/DT3_RF_comp.png?raw=true)