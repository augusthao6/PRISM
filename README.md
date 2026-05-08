# PRISM
Predictive Regime Integrated State Momentum

We present PRISM (Predictive Regime-Integrated State Momentum), a quantitative equity strategy that conditions a stacked machine learning forecast on macroeconomic regime and stock cluster identity, then translates the forecast into positions
through a four-step bounded portfolio constructor and a three-layer risk overlay. The
alpha pipeline combines eight cross-sectional factors with a Gaussian Hidden Markov
Model for SPY-derived regimes, two K-Means partitions (one over daily market-state
vectors, one over per-name long-run behavior), and an ExtraTrees-plus-Gradient-Boosting
stack whose per-cluster reliability is used to size positions. Across an in-sample window
from 2016 through 2021, PRISM produces a Sharpe ratio of 1.13 with an 11% maximum drawdown. Three independent out-of-sample windows—May to October 2023,
January to June 2025, and January to April 2026—yield Sharpe ratios of 1.0, 0.9, and
2.0 respectively, with maximum drawdowns of 4–7%. In two stress backtests, PRISM
lost 2.32% during the COVID-19 crash and 16.73% during the 2008 Global Financial
Crisis, against approximate SPY returns of −20% and −40% respectively. A leaveone-out ablation confirms that every component—volatility targeting, the per-cluster
reliability gate, the correlation-crowding cut, the name-cluster diversification, and the
regime detector—contributes positively to in-sample Sharpe and turnover, with the
regime detector and reliability gate the largest drivers of risk-adjusted performance.
