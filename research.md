---
layout: page
title: Research
permalink: /research/
description: Machine learning for metabolic modeling and scientific computing.
---

## FluxFormer

A Transformer surrogate pretrained on pFBA simulations of the full 2,712-reaction *E. coli* iML1515 metabolic model. FluxFormer represents reactions as tokens and predicts their fluxes from nutrient constraints.

The study compares six simulated pretraining distributions and transfers the frozen models to experimental prediction tasks. In the growth experiment, a small network learns to map medium composition to the surrogate’s inputs, using 110 experimental media and repeated 10-fold cross-validation.

{% include fluxformer-growth.html %}

The figure compares task-specific pretraining (A) with a broader nutrient distribution (E). Results shown are from the current manuscript draft.

[View code on GitHub](https://github.com/ruokokoski/metabolic-NN)
