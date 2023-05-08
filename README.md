# Optimal Feedback Control Model for Sensorimotor Collaboration Between Humans

We developed an optimal control model to predict the movement behaviors of humans in a collaborative reaching task. Here we briefly explain the experiments and the model. However, we have described the experiments and the model in detail in our manuscript......

## Experimental Description

In the experimental task, participants from a human pair attempted to jointly move the midpoint between their hands to a target in the horizontal plane. We conducted two experiments to test how each participant from the pair relies on haptic and/or visual feedback of their partner to complete the task. Visual feedback was enabled by displaying the hand position of their partner. Haptic feedback was enabled through a virtual spring connection between their hands. The location of the midpoint was not shown to the participants. In Experiment 1, we constrained hand movements and the location of the target to a single dimension (lateral). In Experiment 2, we we allowed participant hand movements in both dimensions (forward and lateral) while they reached forward towards a target. The general task setup is shown below.

<img src="/Plots/experimental setup.png" title="Experimental Task Setup" width="600">

## Modeling Description

We modeled each participant controlling their hand as a separate optimal feedback controller. The optimal controllers relied on state information of their partner through vision and/or haptics to estimate the location of the midpoint between the hands. Vision was characterized with greater feedback delay but a greater accuracy in comparison to haptic feedback. An iterative Linear Qudaratic Regulator and Extended Kalman Filter were used to respectively design control and state estimation. The model schematic for one participant is shown below.

<img src="/Plots/model schematic.png" title="Model Schematic" width="800">

## Modeling Scripts

The description of the scripts are as follows:

* Simulation notebooks: Used to run simulations, analyses and plotting. *'simulationcentre_exp1.py'* for Experiment 1 and *'simulationcenter_exp2.py'* for Experiment 2.
* Model scripts: consists of the model classes. *'models_exp1.py'* for Experiment 1 and *'models_exp2.py'* for Experiment 2.
* Model parameters: defines parameters for the model. *'modelparams_exp1.py'* for Experiment 1 and *'modelparams_exp2.py'* for Experiment 2.
* Plot parameters: defines parameters for the plots/figures. *'plot_parameters'* for Experiment 1 an Experiment 2.

## Running Simulations

* Open the model parameters script file for the concerned experiment and define the location for saving/loading simulation data.
* Open the plot parameters script file and define the location for saving plots.
* Open the simulation notebook and run the necessary simulations and analyses.


