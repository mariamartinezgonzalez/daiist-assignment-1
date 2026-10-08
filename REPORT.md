# Assignment 1 Report
- **María Martínez González**
- **18752**
- **mmg.ieu2022@student.ie.edu**
- **BBADBA 5B**

## Dataset

I use the **Spotify Data 1921-2020** dataset from Kaggle. One row is one song, with
its audio features (for example danceability, energy, loudness) and a
**popularity score from 0 to 100**.

I only kept songs from **2000 to 2020**. In the data, the average popularity of
1921-1950 songs is 1.9, while for 2000-2020 it is about 55. That gap reflects how
old a song is, not how it sounds, so I cut the data.

To keep the repo small and fair:
- I removed duplicate songs (same artist and same title).
- I took at most **850 songs per year**, so no year weighs more than another.
- This gives **17,850 songs** (about 2.8 MB). The script that does this is
  `make_sample.py`. It is run once and is not part of training. The raw 28 MB file
  is not in the repo.

## Business / real-life framing

**The decision.** A music label receives many new songs and only has budget to
promote some of them (for example playlist promotion). The model gives each new
song a score so the label can **rank them and promote the top ones**.

**What the model predicts.** Not raw popularity, but **how popular a song is
compared with the other songs of its release year**:
`target = popularity - average popularity of that year`.
I did this because average popularity climbs from 45.8 (year 2000) to 69.4 (year
2019). Part of that is just how the popularity score is measured, not how good
the songs are. A label comparing songs released in the same period does not care
about that general rise, only about which songs beat their group. A positive
target means "more popular than the typical song of its year".

**How this decided the pipeline:**
- **Split by time, not at random.** A random split would let the model learn from
  songs released *after* the ones it is tested on, which cannot happen in real
  life. So: **train 2000-2015 (13,600 songs), validation 2016-2017 (1,700), test
  2018 to May 2020 (2,419).** The validation set is only for choosing settings.
  The test set is only used at the end.
- **I do not use `year` as an input.** For a brand-new song, the year carries no
  information about how it sounds.
- **Every number learned from data is learned from the training years only**
  (scaling, artist averages). Otherwise information from the future would leak in.
- **Which metric?** Because the label only needs a good *ranking*, I report the
  usual errors (MAE, RMSE, R²) and also a business number: **if we promote the top
  k% of songs, how often is a promoted song really better than its cohort?**
  (the "hit rate"). The dashboard has a slider for this.

## Data preparation & feature engineering

### A data-quality problem I found: June 2020

While checking the target by year, one year looked strange: the spread of
popularity in 2020 was 18.3, while every other year was between 5.8 and 8.8.
Looking month by month:

| 2020 songs | Average popularity | Spread |
|---|---|---|
| January to May | about 70-71 | 6.4 to 7.6 (normal) |
| **June** (131 songs) | **35.7** | **29.8** |

In 2020, 5.6% of songs had popularity 5 or lower, against 0% in 2018 and 2019.
My explanation (a guess I could not verify) is that June songs were so new when
the data was collected that they had not yet collected plays.

**What I did:** I removed the 131 June 2020 songs **before** computing the yearly
average. If I had kept them, they would drag the 2020 average down and every other
2020 song would get an artificial bonus of about +5 in its target. After removing
them, 2020's spread went from 18.3 to 7.0, and the artist signal on the test set
(see below) went from a correlation of 0.034 to 0.178. So the problem was hiding
a real signal. The final dataset has **17,719 songs**.

### The features (30 columns in total)

| Feature | Why |
|---|---|
| `acousticness`, `danceability`, `energy`, `loudness`, `tempo`, `valence` | Main audio features, used as they are. |
| `log(duration)` | Song length has a few extreme songs (up to 75 minutes). Its skew was 18.0 and became -0.09 after the log. The dashboard shows both pictures. |
| square root of `speechiness`, `instrumentalness`, `liveness` | These are mostly near zero with a long tail. The square root reduces the skew (2.65 to 1.79, 3.04 to 2.44, 2.24 to 1.29). It helps but does not fully fix it. I report that honestly. |
| `explicit`, `mode` | Already 0/1. |
| `key` as 11 yes/no columns | A musical key is a category, not a size (key 11 is not "bigger" than key 2). I left one key out on purpose, because with all 12 columns they always add up to 1 and repeat the intercept. |
| number of artists (maximum 5), "feat" in title, "remix" in title | Collaborations and remixes could behave differently. |
| release month as a circle (sine and cosine) | Month 12 is next to month 1. The circle keeps that. 7% of songs only have a year, so for them the month is set to a neutral value. I did **not** add a "month unknown" flag, because those songs are all old (15% of 2000 songs, 0% from 2015), so the flag would just act as a hidden "old song" signal. |
| **artist score** and **how many songs the artist has** | See below. This is the most useful feature. |

All features are then put on the same scale using the **training years' average
and spread only**.

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