# Assignment 1 Report
- **María Martínez González**
- **18752**
- **mmg.ieu2022@student.ie.edu**
- **BBADBA 5B** 

## Dataset

*I use the Spotify Data 1921-2020 dataset from Kaggle (one row = one song with
its audio features and popularity score). I restricted it to 2000-2020 because
in the data I saw that the average popularity of 1921-1950 songs is 1.9, while
for 2000-2020 it is about 55. That gap reflects how old a song is, not how it
sounds, so I cut the data to 2000-2020.*

## Business / real-life framing

*The label scores songs that are about to be released, so I train on 2000-2017
and test on 2018-2020. I drop `year` because it is not a useful signal for a
new release. I accept a low R² and explain it in the limitations section.*

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
