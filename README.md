# Behavioural Data Analyses from Cavanagh et al. 2020

## Description
In 2020, we published a [paper](https://elifesciences.org/articles/53664) in eLife that analysed the choice behaviour of subjects performing a complex decision-making experiment. The behavioural analyses were originally performed in Matlab, and the code uploaded to the [original repository](https://github.com/elifesciences-publications/CavanaghLam2020CodeRepository). The purpose of this repository is to replicate the code for behavioural analyses in Python. 

### Analyses:

- Using logistic regression to predict choices and provide insight into a subject's decision-making strategy
- Using cross-validation to compare model fits
- Using optimisation algorithms to fit psychometric functions to choice data

## Repository 
Run <b>main.py</b> to produce the figures, which will output in PDF format. The functions performing the analyses are contained within <b>SeanFunc.py</b>. 
The following data files are required:
 - <i>NonDrugDayProcessedData_SubjectA_41_Sessions21-Aug-2020.mat</i>
 - <i>NonDrugDayProcessedData_SubjectH_63_Sessions21-Aug-2020.mat</i>

## Scientific Overview of the Project
A subanaesthetic dose of ketamine is an effective pharmacological model of psychosis; recapitulating positive, negative and cognitive symptoms. Ketamine also exacerbates symptoms in people with schizophrenia. Ketamine binds to a variety of receptors, but principally acts at the NMDA-R. To gain a greater understanding of the neural basis of psychosis, we designed a study incorporating behavioural analyses, electrophysiology, pharmacology, and neural network modelling.

Participants began by learning a challenging decision-making task - where they had to combine information across time. They had to make a decision about which side had the 'taller' average height:

![alt text](https://github.com/SCavanaghNeuro/SCavanaghNeuro.github.io/blob/main/images/example%20task.gif)

Behavioural analyses of control data revealed that mean sample height (unsurprisingly) was the primary driver of choices. More surprisingly, choices were also irrationally influenced by sample variability (to a lesser extent). This revealed that subjects showed a ‘pro-variance’ bias:

![alt text](https://github.com/SCavanaghNeuro/SCavanaghNeuro.github.io/blob/main/images/Ket_image1.jpg)


Next, we trained a biologically realistic neural network to perform the same decision-making task. We then interfered with the activity of NMDA-R at different points in the circuit, to see how it would affect decision-making:

![alt text](https://github.com/SCavanaghNeuro/SCavanaghNeuro.github.io/blob/main/images/Ket_image2.jpg)

Finally, we gave subjects either saline or ketamine, to compare the decision-making with our model predictions:

![alt text](https://github.com/SCavanaghNeuro/SCavanaghNeuro.github.io/blob/main/images/Ket_image3.jpg)

Take home point – Ketamine may act to lower the balance between excitation and inhibition in cortical decision circuits. This could lead to specific impairments in decision-making behaviour, and explain the decision-making deficit experienced in psychosis.
## Contributions
The paper was co-authored with Norman Lam, John Murray, Laurence Hunt, and Steve Kennerley.
The python code for generating figures in this repository was adapted from code written by Norman Lam.

## Links
- [Paper](https://elifesciences.org/articles/53664)
- [Dataset download](https://datadryad.org/stash/dataset/doi:10.5061/dryad.pnvx0k6k3)
- [Original code repository](https://github.com/elifesciences-publications/CavanaghLam2020CodeRepository)
- [Discussion of paper on Naked Scientists Podcast](https://www.thenakedscientists.com/articles/interviews/ketamine-mimics-schizophrenia)



