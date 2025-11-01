# NBA Effort in Continuous-Time

This is an extension to the stochastic differential games chapter of my Ph.D. disseration, *Adversarial Risk Analysis for Decision-Making in Sports*, which can be found [here][phd_dissertation].

The larger decision modeling requires multiple stages of modeling. There will also be a decision evaluation component which will require some predictive modeling as well. 

## Step 1 - Creating a proxy for team effort

The challenge to tackle is to determine how to measure "effort" in basketball. One possible proxy is the collection of hustle statistics that the NBA collects. While these are based on tracking data, tracking data generally isn't publicly available. So, we will use hustle statistics fetched from `stats.nba.com`. While there are definitely other more direct ways of measuring effort (perhaps using biometrics), hustle stats are the most readily available for each game. 

The next question is how do we take this collection of hustle measures to create a singular composite effort proxy measure. One approach is to perform a multi-stage regression. We first separate out the offensive and defensive hustle measures and try to assign weights in some data-driven way. We use offensive and defensive ratings to accomplish this. 

$$\text{Offensive Rating} = \theta_0^{Off} + \mathbf{X^{Off}}\boldsymbol{\theta^{Off}} + \epsilon^{Off}$$
$$\text{Defensive Rating} = \theta_0^{Def} + \mathbf{X^{Def}}\boldsymbol{\theta^{Def}} + \epsilon^{Def}$$

Why do we use offensive and defensive ratings as the target variables here? Well they are defined as 
$$\text{Offensive Rating} = \frac{\text{Points Scored}}{\text{Possessions}}\times 100, \,\,\, \text{Defensive Rating} = \frac{\text{Points Allowed}}{\text{Possessions}}\times 100$$
or in other words, points scored per 100 possessions and points allowed per 100 poessessions, respectively. Essentially these are scoring rates for the team of interest and its opponent. If we have this data at the individual game level, the home team's offensive rating is the same as the away team's defensive rating and vice versa. At the end of the day, we want to determine the contribution of effort to changes in the score of a single game, so using these as the targets nudges us in that direction. 

Once we estimate these models, we'll obtain composite offensive and defensive effort:

$$\text{Offensive Effort} = \mathbf{X}^{Off}\boldsymbol{\hat{\theta}^{Off}}$$
$$\text{Defensive Effort} = \mathbf{X}^{Def}\boldsymbol{\hat{\theta}^{Def}}$$

Now, we want to combine these into a single composite effort variable. To do so, we run another regression model now with net ratings as the response regressing on the above two predicts to again obtain optimal weights: 

$$\text{Net Rating} = \gamma_0 + \mathbf{X^{Off}}\boldsymbol{\gamma^{Off}} + \mathbf{X^{Def}}\boldsymbol{\gamma^{Def}} + \epsilon^{Net}$$

With this model estimated, we obtain the composite effort variable.

$$\text{Effort} = \mathbf{X^{Off}}\boldsymbol{\hat\gamma^{Off}} + \mathbf{X^{Def}}\boldsymbol{\hat\gamma^{Def}}$$

Again, why do we use net rating here as the target? Net rating is defined as 

$$\text{Net Rating} = \text{Offensive Rating} - \text{Defensive Rating}$$

We can specify it by team as well: 

$$\text{Home Net Rating} = \text{Home Offensive Rating} - \text{Home Defensive Rating} = \text{Home Offensive Rating} - \text{Away Offensive Rating}$$

We can similarly specify the away team net rating as well. The net rating now represents the scoring rate differential, which is *almost* exactly what we will want to model.

## Step 2 - Connecting effort to scoring 
The effort level strategy won't be chosen in a vacuum. It will be chosen to maximum the end-game score differential (while accounting for some risks) in the context of a *continuosly evolving* game. We model this continuous evolution with a system of stochastic differential equations (SDEs). The outcome of the game is determined by the score differential. As a result, we'd like to know how much effort contributes to the ebbs and flows of the game, i.e., the score differential dynamics. Chances are that if a strong team is playing a weak team, the strong team is going to win without significant extra effort. However, for more evenly matched teams, effort will likely be more of a determining factor (admittedly, this statement is opinion, not data-driven evidence). So, to connect effort to the score differential, we also want to account for other factors that determine the relative strength of the two teams in a particular matchup. Consider the SDE: 

$$dX_t = \text{drift} \cdot dt + \text{diffusion} \cdot dW_t$$

$X_t$ is the score differential at time $t$, which means that $dX_t$ represents infinitesmal changes in the score differential. While we can't technically take the derivative of the score differential with respect to time, we can intuitively think about $\frac{dX_t}{dt}$ in this manner. This represents the score differential rate which is the same as the difference in home and away scoring rates, which is exactly what net ratings attempt to capture. We will need to adjust things once we get down to modeling, but this is the intuition. The drift term determines the long-term trend while the diffusion determines how "rough" or volatile the path is. If we want to incorporate the effect of effort on the score differential and how it changes over time, we'll need to include the other determining factors as well. The simplest way to do this is to place them in the drift term. In the end, this SDE will look something like

$$dX_t = (\text{home advantage} + \Delta \text{strength} + \Delta \text{effort})dt + \text{diffusion}\cdot dW_t$$

In practice, we will have a couple more terms in the drift representing additional effort from the home and away teams, and the current drift terms will be constant values estimated according to a third regression model using the output of the Step 1. The estimated regression will produce coefficients for home advantage, relative team strengths using Dean Oliver's four factors, and relative team efforts. 

$$\text{Net Rating} = \beta_0 + \beta_1 X_{eFG\%} + \beta_2 X_{FTA Rate} + \beta_3 X_{OREB\%} + \beta_4 X_{TOV\%} + \beta_5 X_{effort} + \epsilon$$

Once estimated, each term in this model will be in the drift term of the SDE, including the intercept which will now represent the home advantage. There will be a handful of games at international venues which are considered neutral, but for the vast majority of games, there will be a home team. So while, it's not technically an intercept, it will effectively be one and will retain a useful interpretation. 

## Overall SDE Modeling Workflow
![model_workflow](/figures/SDE_workflow.png)

## Step 3 - Obtain optimal controls, value, and trajectories for both team

## Step 4 - Model over many games

## Step 5 - Evaluate usefulness and reality

[phd_dissertation]: https://scholarspace.library.gwu.edu/concern/gw_etds/9k41zf13m