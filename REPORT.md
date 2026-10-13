# Assignment 1 Report
- **María Martínez González**
- **18752**
- **mmg.ieu2022@student.ie.edu**
- **BBADBA 5B**

## Dataset
I use a public Spotify tabular dataset from Kaggle with songs from 1921 to 2020. Each row is one song. For each song we know audio features, which are numbers spotify calculates from the sound like: danceable, how energetic and how loud the song is. Also other characteristics like artist ,title, release date and a popularity score that goes from 0 to 100. As i mentioned, it is a tabular dataset so one row per song and one colum per feature.

Why did I remove and not use all the years? Old songs have a very low popularity score. The average for songs from 1921 to 1950 is 1.9, while for 2000 to 2020 it is about 55. That does not mean old songs sound worse. The score is based on recent plays, so old songs just get fewer. If I kept them, the model would mainly learn "is this song old?" instead of anything about the music. So I kept only 2000 to 2020.

How did I clean the dataset and make it smaller? The original file is 28 MB, which is too big for this project, so I did this once, in a separate script make_sample.py:
- I removed duplicates (the same artist with the same title).
- I kept at most 850 songs per year, chosen at random. This way no year counts more than another.
- The result is 17,850 songs (2.8 MB). This is the file the project uses. The original 28 MB file is not in the repository.
- One more cleaning step that i will explain later is that I found that songs from June 2020 have faulty popularity scores and removed them (131 songs). The model is trained on the remaining 17,719 songs. The details are in the next section.

## Business / real-life framing

**The decision.** Imagine a music label that receives many new songs but only has budget to promote some of them, for example with playlist promotion. The model gives each new song a score, so the label can rank the songs and promote the ones at the top. So the model is not there to say "this song will have exactly 63 popularity". It is there to help choose.

**What the model predicts.** Not the raw popularity, but how popular a song is compared with the other songs from the same year: target = popularity - average popularity of that year. If the number is positive, the song did better than the typical song of its year. If it is negative, it did worse. Why did I do it like this? ecause the average popularity goes up every year: 45.8 in 2000 and 69.4 in 2019. Part of this is just how the score is measured (recent songs get more recent plays), not that songs are getting better. A label that compares songs released at the same time does not care about that general rise. It only cares about which songs beat the others. Without this change, the model would mostly learn "is the song recent?" again.

**How this decided the pipeline:**
- **I split the data by time, not randomly.** In real life, the label never knows the future. A random split would let the model learn from songs released after the ones it is tested on, which is cheating. So I used:
  - **Train: 2000 to 2015** (13,600 songs). The model learns here.
  - **Validation: 2016 to 2017** (1,700 songs). I only use it to choose settings.
  - **Test: 2018 to May 2020** (2,419 songs). I look at it only at the end, to see
    how the model does on songs it has never seen.
- **I do not use the year as an input.** For a brand-new song, the year gives no information about how it sounds.
- **Everything the model learns from data, it learns only from the training years.** This includes the scaling of the numbers and the artist information. If not, information from the future would leak into the model.
- **I chose the metrics because of the decision.** The label needs a good ranking, so besides the usual errors (MAE, RMSE and R²) I use a business number, the **hit rate**: if we promote the top k% of songs, how many of them really didbetter than the typical song of their year? The dashboard has a slider to change k.

## Data preparation & feature engineering
**I found a data problem** in June 2020: I was checking how spread out the popularity is in each year. Every year had a spread between 5.8 and 8.8, except 2020, which had 18.3. That looked wrong, so I looked month by month. from Jan to May spread was between (6.4 and 7.6) which was normal but in June spread was 29.8. Also, 5.6% of the 2020 songs had a popularity of 5 or lower, while in 2018 and 2019 there were none. **Why do I think it happened?** I believe June songs were so new when the data was collected that they had not collected plays yet, so their score was still very low. So I removed the 131 June 2020 songs **before** calculating each year's average. This matters because the target is "popularity minus the average of the year". If I had kept June, it would pull the 2020 average down, and every other 2020 song would get an artificial bonus of about +5 in its target, just because of the error in june 2020 songs. **Solving the promblem was key** After removing the songs , the 2020 spread went from 18.3 to 7.0, like the other years. Also, the relation between the artist feature (explainedin the next part) and the target on the test set went from 0.034 to 0.178. So the broken data was hiding a real signal. The final dataset has **17,719 songs**.

### The features (30 columns in total)
- **Audio features.** Acousticness, danceability, energy, loudness, tempo and valence are the main audio features, there were already numerical so I used them as they were.
- **Song length.** Most songs last about 3 or 4 minutes, but a few last over an hour. Those few songs distort the numbers. The logarithm is a way of shrinking huge numbers a lot and small numbers only a little, so the very long songs get pulled back closer to the rest. "Skew" just measures how lopsided the data is. It went from 18 (very lopsided) to almost 0 (balanced).
- **Speechiness, instrumentalness and liveness.** Most songs are near zero and a few are high. I used the square root, which reduces the problem to make it work but does not fully fix it
- **Explicit and mode.** They are already yes/no (0 or 1), so I did nothing.
- **Musical Key.** Stored the key as a number from 0 to 11, but the number is only a label . If I gave the model the raw numbers, it would treat key 11 as “bigger” than key 2 and try to find a straight-line pattern (higher key number means more or less popular), which makes no sense. Instead of one column with a number, I made one column per key, each answering a yes/no question (0 or 1).
- **Artists and title.** I added the number of artists (maximum 5) and whether the title has “feat” or “remix”, because collaborations and remixes might behave differently. “5 or more is just ‘many’”. It is the same idea as the logarithm for song length: don’t let a few extreme cases dominate.
- **Release month.** as months are a loop December comes January again. numbers 1 to 12, the model would think December and January are far apart. So i used sine and cosine to write the position as a circle.
- **Songs with no month.** For about 7% of songs, we only know the year. Months were written as a position in the circle so songs with no month I put 0 and 0, which is the center of the circle. It points to no month at all, so it says “no information about the month” and doesn’t favor any season. I chose to remove the year as well because most of these songs were old so it would tell the model “this is an old song”.
- **Different Scales.** Putting everything on the same scale. Tempo is around 120, while danceability is between 0 and 1. tempo would seem more important just because its numbers are bigger. For each feature I subtracted its average and divided by its spread (standard deviation). The result is that every feature has an average of 0 and a typical size of 1. A tempo of 120 and a danceability of 0.6 both become numbers like 0.3 or -1.2, meaning “a bit above average” or “a bit below”.

### The artist score (the feature that matters most)

Audio features alone barely relate to the target (the best one has a correlation
of only 0.07). But the same artists appear many times (17,850 songs, 8,542
different artist names). So for each song I compute the **average target of the
artist's other songs**, pulled towards the overall average when the artist has
few songs (so one lucky song does not make an artist look great).

Two protections against cheating:
1. **Out-of-fold on training data.** The training songs are split in 5 groups. A
   song's artist score is computed from the *other 4 groups*, so a song never sees
   its own answer.
2. **Unknown artists get the neutral value** (the overall average). For validation
   and test songs, the score only uses training years. About 57% of validation
   songs and about 45% of test songs have an artist the model already knows.

**Proof it helps** (validation R², same model with and without these features):
about **-0.01 without** vs about **+0.015 with**. Without them, the model is
slightly worse than just guessing the average.

## Modeling: three implementations, one model

**The model:** **linear regression with a small penalty (Ridge)**. It is a simple
formula: each feature is multiplied by a weight, and the results are added up. I
chose it because the assignment asks for a simple model, and because it is easy
to explain: you can read the weights. The penalty shrinks weights that are mostly
noise, which is useful here because the signal is weak.

**Three ways to train the same thing, on the same split:**
1. **scikit-learn** (`Ridge`), which solves it directly with a formula.
2. **Manual PyTorch loop**, where I write each step myself: predict, measure the
   error, compute the direction to improve (the gradient), move the weights a
   little, repeat 3,000 times.
3. **Standard PyTorch** (`nn.Module` + `torch.optim`), the usual way, where PyTorch
   handles the steps.

**Settings I tuned (on validation only):**
- **Penalty strength:** I tried 9 values from 0.1 to 100,000. The best was **10,000**
  (error goes 5.942 at 0.1, 5.870 at 10,000, then up again to 5.883 at 30,000). That
  is a strong penalty, which tells me most features carry very little signal.
- **Learning rate** (size of each step, for the two PyTorch versions): I tried
  0.005, 0.02, 0.05 and 0.2. All gave exactly the same validation error (5.8695),
  so the choice does not matter here. The code kept 0.02 only because it came first
  in the tie. The problem is easy for gradient descent because the features are on
  the same scale.

**Results on the test set (2018 to May 2020), same for all three:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive baseline (predict the training average for every song) | 5.1165 | 6.4476 | 0.000 |
| scikit-learn Ridge | 4.9028 | 6.2882 | 0.0488 |
| Manual PyTorch loop | 4.9028 | 6.2882 | 0.0488 |
| Standard PyTorch | 4.9028 | 6.2882 | 0.0488 |

(Validation R² is 0.037 and training R² is 0.093.)

**Do the three agree? Yes.** The weights match to about **0.0000013** (1.3e-06),
which is rounding error, and all predictions and metrics are the same. Making this
true needed two details that I had to get right, not just hope for:
- sklearn's penalty adds up errors over all songs, but PyTorch's loss is an
  *average*. So the penalty must be divided by the number of songs (`alpha / n`) in
  the manual loop, and the optimizer's `weight_decay` must be `2 * alpha / n`
  (because of the derivative of a square).
- The intercept (the starting level) is **not** penalized in any of the three.

One thing that looks like a disagreement but is not: the final "loss" printed is
53.4 for the manual loop and 51.9 for the standard PyTorch version. The manual loss
includes the penalty term and `nn.MSELoss` does not (the penalty lives inside the
optimizer). The weights are identical, so the models are the same.

### What the model is good for: the promotion test

If the label promotes the top k% of test songs according to the model, how many
really beat their release-year group?

| Top k% promoted | Songs | Hit rate (model) | Hit rate (random pick) | Improvement | Average extra popularity vs cohort |
|---|---|---|---|---|---|
| 5% | 120 | 59.2% | 43.3% | x1.37 | +2.6 points |
| 20% | 483 | 57.3% | 43.3% | x1.33 | +2.0 points |
| 50% | 1,209 | 52.7% | 43.3% | x1.22 | +1.2 points |

(The "random" rate is 43.3% and not 50% because popularity is skewed: fewer than
half of the songs beat their year's average.)

**In plain words:** the model is a **modest helper for ranking**, not a predictor
of exact popularity. The more selective the label is, the better the hit rate. At
5% it only looks at 120 songs, so one or two songs move the number by about a
point, which means the order matters more than the exact figures.

## Limitations & next steps

1. **Very low R² (0.049).** Only about 5% of the differences between songs of the
   same year are explained. This is expected: audio features alone say little about
   success, and success depends on marketing, the artist's fame and luck. The
   prediction plot shows it clearly: predictions stay in a narrow band around 0
   while real values go from -12 to +26. *Next step:* add information that really
   drives popularity and is known before release (artist followers, label,
   marketing budget, playlist placements).
2. **The artist signal weakens over time.** The correlation of the artist score
   with the target falls from 0.31 (training) to 0.21 (validation) to 0.18 (test).
   Artists change, and about 55% of test songs come from artists the model has
   never seen, and for them the score is neutral. *Next step:* retrain regularly so
   new artists enter the table, and add artist-level features that exist for new
   artists too.
3. **The popularity label is a snapshot.** It was measured when the data was
   collected, not at release. That is why I use the cohort-relative target, but it
   is not a perfect fix: old songs have had years to be replayed. *Next step:* use
   popularity measured a fixed time (for example 30 days) after release.
4. **The June 2020 cause is a guess.** The numbers prove June labels are on a
   different scale (average 35.7 vs about 70), but I did not verify the reason. I
   also removed 131 songs. *Next step:* check the data collection date, or collect
   those songs again later.
5. **Some features are still skewed** after the transformations
   (`sqrt_instrumentalness` skew 2.44). A linear model does not need perfectly
   shaped features, but a model that captures non-linear effects (for example
   gradient boosting) could use them better. I stayed with linear models because
   that is what the assignment asks and because they are easy to explain.
6. **Spread differs between periods** (about 7.6 in training, 6.0 in validation,
   6.4 in test), so raw errors should not be compared across splits. I compare R²
   within each split.
7. **Settings were chosen on only two validation years**, and the model may favor
   those years. *Next step:* use several time-based validation windows.
8. **Training R² (0.093) is higher than test R² (0.049).** Some of this is real
   overfitting; some is because the artist score behaves a bit differently on
   training songs (out-of-fold) than on new ones.
9. **The promotion test counts a hit as "better than its cohort" and ignores
   money.** A real decision needs the cost of promoting a song and the value of a
   hit. *Next step:* turn the hit rate into expected profit with real numbers from
   the label.

## How to run

```
uv run python main.py train   # trains all three models and saves them in models/
uv run python main.py app     # opens the dashboard (loads models/, trains nothing)
```

The dashboard has three tabs: model comparison (metrics, predicted vs actual,
coefficients of the three methods), distributions (including the log-duration
fix), and the promotion simulator. The checks I used to find the June 2020 problem
and test the features are in `eda/`; run them with `uv run python -m eda.check_2020`
from the repo root.
