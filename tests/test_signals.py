import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.analysis.signals import bootstrap_confidence_interval, aggregate_signals
from src.analysis.feature_engineering import lexicon_score


def test_lexicon_score_bullish():
    assert lexicon_score("Nifty breakout, buy the rally") > 0


def test_lexicon_score_bearish():
    assert lexicon_score("Market crash, sell everything, panic") < 0


def test_lexicon_score_neutral_when_no_match():
    assert lexicon_score("Just watching the market today") == 0.0


def test_single_observation_ci_has_no_spread():
    lower, upper = bootstrap_confidence_interval(np.array([0.5]), np.array([1.0]))
    assert lower == upper == 0.5


def test_confidence_interval_widens_with_fewer_tweets():
    rng = np.random.default_rng(0)
    many_scores = rng.uniform(-1, 1, size=200)
    few_scores = many_scores[:5]

    lo_many, hi_many = bootstrap_confidence_interval(many_scores, np.ones_like(many_scores), iterations=200)
    lo_few, hi_few = bootstrap_confidence_interval(few_scores, np.ones_like(few_scores), iterations=200)

    assert (hi_few - lo_few) >= (hi_many - lo_many)


def test_aggregate_signals_empty_df():
    df = pd.DataFrame(columns=["timestamp", "lexicon_score", "engagement_score"])
    result = aggregate_signals(df)
    assert result.empty
