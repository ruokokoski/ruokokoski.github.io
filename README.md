# Research Notes and Insights

Personal research and project site hosted on GitHub Pages using Jekyll. The shared layouts, CSS, and small
theme scripts work without a JavaScript framework or external font downloads.

## Preview locally

You need Ruby and Bundler. On distributions with versioned executables, use
`bundle3.2` instead of `bundle` in the commands below.

From this repository, run the setup once:

```sh
bundle config set --local path vendor/bundle
bundle install
```

Start the preview:

```sh
bundle exec jekyll serve --livereload --host 127.0.0.1
```

Open <http://127.0.0.1:4000> in your browser. The server rebuilds when you edit
pages, layouts, styles, or scripts. Restart it after changing `_config.yml`.
Stop it with Ctrl+C. These commands do not commit, push, or publish anything.

If the checkout is on a remote machine, forward port 4000 using your editor's
Ports panel, or run `ssh -L 4000:127.0.0.1:4000 user@host` from your computer.
Then open the same local address. Forward port 35729 as well for LiveReload,
or refresh the page manually.

To check the generated site without starting a server:

```sh
bundle exec jekyll build --strict_front_matter
```

## Shared design

- `_layouts/default.html` supplies the header, footer, and main landmark.
- `_layouts/home.html`, `_layouts/page.html`, and `_layouts/post.html` share it.
- `css/override.css` defines colors, typography, responsive layouts, and code styles.
- Code blocks use Jekyll's bundled Rouge highlighter and share the theme palette.
- `_includes/theme-init.js` selects the theme before CSS loads to avoid a flash.
- `js/theme.js` offers Light / Dark selection and remembers the choice in browser
  storage. Dark is the default when no choice is saved.
- Without JavaScript, the site also defaults to dark.
- The homepage introduces Teemu and uses a local copy of the photo from
  <https://github.com/ruokokoski>. Replace `pics/teemu-ruokokoski.jpg` to update it.
- `research.md` contains a title page followed by eight slide outlines.
- `_includes/fluxformer-feature.html` adds the compact homepage project feature.

## Research presentation

Open `/research/` and choose **Present fullscreen** to present the current slide.
Use Previous / Next, arrow
keys, Page Up / Page Down, Home / End, the Jump to menu, or swipe horizontally.
Keyboard shortcuts apply while focus is inside the presentation; form controls
retain their usual behavior. **Overview** shows compact previews generated from
the actual slide content, with readable titles below them. Previews follow the
current theme and selected figures, and refresh whenever the overview opens.
**Present fullscreen** requests real browser fullscreen for the slide stage itself,
sized to the entire viewport without a maximum width or fixed aspect ratio.
If the browser blocks fullscreen, a message offers a link to open the presentation
in its own tab; a CSS-only expanded page is not substituted. Only the active slide is shown;
navigation bars and slide tools stay outside it. Navigate with the keyboard or swipes; exit with
Escape, a double-click, or a double-tap. The usual controls return on exit.

Slides have stable links such as `/research/#slide-architecture`. Browser Back
and Forward restore slide selection. **More detail** opens supporting text,
and **Expand visual** opens the selected visual in a dialog. The Results slide
has selectable visual placeholders. The title page, slide 2, **From waste to
value**, slide 3, **Microbial metabolism**, slide 4, **Metabolic models**,
slide 5, **The data problem**, slide 6, **Virtual experiments**, and slide 7,
**Generating simulated data**, slide 8, **FluxFormer architecture**, and slide 9,
**Training across nutrient distributions**, and slide 10,
**Broad sampling: D and E distributions**, are filled;
the remaining two slides are still skeletons. Content
slides share a heading above a text-and-figure layout. Each slide has a live date and 24-hour `hh:mm`
clock in its footer, using the `Europe/Helsinki` time zone.
All slide templates share the same proportional typography and footer size.
The stage has one persistent footer shared by all slides. In fullscreen it has
a reserved row outside the changing slide; only the content above it scrolls.
Sizing follows the native fullscreen box and typography follows its container
dimensions, avoiding dynamic viewport sizing during fullscreen transitions.
During fullscreen, the stage is temporarily mounted directly under the document
body, outside the site's clipped card. The footer has its own rendering layer.
On exit or a rejected request, the stage returns to its original position.

Edit `_data/fluxformer_slides.yml` to fill the slides. Keep each slide's `id`
stable so shared links continue working. Optional fields are:

```yaml
content: |
  Slide text in Markdown.
notes: |
  Speaker notes, available outside fullscreen.
details: |
  Supporting details in Markdown.
image: /pics/example.png
alt: A description of the figure.
caption: A short caption.
```

Use plain text for slide content unless emphasis is explicitly requested.
Figure source captions are added only when requested. Body text uses a shared
size across slides; keep any future content slides on that same scale.
All content slides have three green lines directly below their titles, scaled
proportionally to the title font. In light
mode the slide canvas, including fullscreen and overview previews, is white.
Use `layout: wide` for text above a full-width figure row; `equation: fba`
places the three-line FBA equation image, cropped from the thesis, beside the bullets on
desktop and below them on narrow screens, preserving its mathematical typeface.
Use `layout: stats` with `stats` entries (`value`, `label`, `caption`) for dataset
cards beneath the text. Slide 5 uses two cards for the 110 growth conditions
and 29 omics conditions, with captions and speaker notes. The compact cards
occupy 60% of the available width on desktop and stack on narrow screens.
Use `layout: figure` for a figure-only slide body, without a text placeholder.

For multiple selectable figures, add `visuals` entries with `id`, `label`, and
the same optional `image`, `alt`, and `caption` fields. Without JavaScript, all
slides and visuals appear in sequence. Styles and scripts are loaded only on
pages with `presentation: true` in their front matter.

## FluxFormer figures

`pics/fluxformer/side-stream-bioproduction.png` is an unmodified copy of
`metabolic-NN/insights/thesis/bioreactor_biorender.png` (3778 × 2597 pixels),
the original artwork for thesis Figure 1.1, printed page 1 / PDF page 11.
It is shown on slide 2; its white figure background keeps labels readable in
both themes.

`pics/fluxformer/microbial-metabolism.png` isolates the cell from the same
original artwork for slide 3. The source crop is 1500 × 1740 pixels at
`(2220, 300)`. Exterior reactor callout lines and the biomass arrow are masked
away; nutrient uptake, intracellular reactions, labels and secreted products
are retained. White margins are trimmed and a 24-pixel white border is added.

`pics/fluxformer/metabolic-models-fba.png` is rendered from the original
`metabolic-NN/insights/thesis/fba.pdf`, the artwork for thesis Figure 2.2
(printed page 8 / PDF page 18). The PDF is rendered with Poppler at 3200 pixels,
then its white margins are trimmed and a 16-pixel border is added. Equation 2.1
uses the original thesis crop described below, with the same objective and constraints.

`pics/fluxformer/virtual-experiments-workflow.png` is cropped directly from
Article Figure 1 on PDF page 4 of `pics/fluxformer/fluxformer_0911.pdf`.
The page is rendered with Poppler at 3200 pixels on its longest edge; the crop
is 2030 × 820 pixels at `(220, 270)`. All four panels are retained and the
article caption is excluded. This is the current FluxFormer artwork.

`pics/fluxformer/growth-model-a.png` and `growth-model-e.png` are the original
1673 × 1630 image panels extracted from Figure 4 (page 9) of the sibling
`metabolic-NN/tmp/fluxformer_0211.pdf`. They retain the original plot labels,
error bars, colors, and annotations. The website adds the Model A / Model E panel
headings and caption in HTML. Each plot links to its full-size image.

Source PDF SHA-256:
`6b349b08652f20cec90957db0a3208c925585d6f09165f375fe319d5da27d505`.

To reproduce the extraction using Poppler:

```sh
pdfimages -f 9 -l 9 -png ../metabolic-NN/tmp/fluxformer_0211.pdf /tmp/fluxformer-growth
cp /tmp/fluxformer-growth-000.png pics/fluxformer/growth-model-a.png
cp /tmp/fluxformer-growth-002.png pics/fluxformer/growth-model-e.png
```

These are manuscript-draft results. Update the panels and source provenance
together when the research figures change.

For a new page, use front matter such as:

```yaml
---
layout: page
title: About
description: A short introduction.
---
```

Existing post URLs and Markdown source content are preserved. The blog listing
continues to use `/archive.html`.

`pics/fluxformer/fba-equation.png` is Equation 2.1 cropped from thesis PDF
page 18 at 400 dpi, using the pixel rectangle `(1333, 2050, 584, 380)`.
The three-line formulation retains the thesis typeface and bold vector/matrix
notation. Its equation number and surrounding prose are excluded.

`pics/fluxformer/simulated-data-generation.png` is rendered from
`metabolic-NN/insights/thesis/fba_workflow.pdf`, the original artwork for thesis
Figure 3.3 (printed page 22 / PDF page 32). Poppler renders it at 3200 pixels
on the longest edge; white margins are trimmed and a 16-pixel border is added.
Slide 7 is inserted before the existing architecture skeleton, preserving its link.

Fullscreen figures in the standard text-and-figure layout fill the available
figure column and remaining slide height without a fixed pixel-height cap.
Their aspect ratios are preserved; mobile retains the stacked layout.

`pics/fluxformer/fluxformer-architecture.png` is cropped directly from the left
panel of Article Figure 2 on PDF page 5 of `pics/fluxformer/fluxformer_0911.pdf`.
Poppler renders the page at 3600 pixels on its longest edge; the crop rectangle
is `(240, 290, 800, 1180)`. It retains the complete architecture from Inputs to
Output X / Output c, excluding the experimental-task panels and article caption.

`pics/fluxformer/nutrient-distributions.png` is cropped directly from Article
Figure 3 on PDF page 7 of `pics/fluxformer/fluxformer_0911.pdf`. The page is
rendered at 3600 pixels on its longest edge; the crop is `(930, 300, 920, 650)`.
The condition spaces, model/data legend and model-set label are retained; the
article caption is excluded. Slide 9 precedes the existing results skeleton.

`pics/fluxformer/broad-de-distributions.png` is cropped from the supplied
`pics/fluxformer/broad_DE.png`, using `(0, 146, 1672, 795)`. The crop removes
the image title and blue separator while preserving the three panels and all
their content. Slide 10 uses this copy; the original image is unchanged.
