---
layout: research
title: Research
permalink: /research/
description: Machine learning for metabolic modeling and scientific computing.
presentation: true
---

{% if site.show_research_slides %}
{% include research-presentation.html %}
{% else %}
<section id="fluxformer" class="research-project" aria-labelledby="project-title">
  <h2 id="project-title">FluxFormer</h2>
  <p>A Transformer surrogate for genome-scale metabolic modeling, learning from simulations to predict reaction fluxes and transfer to experimental tasks.</p>
  <a class="text-link" href="https://github.com/ruokokoski/metabolic-NN">View code <span aria-hidden="true">↗</span></a>
</section>
{% endif %}
