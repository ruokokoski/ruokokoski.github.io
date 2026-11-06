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
has selectable visual placeholders. Only the title page is filled; the remaining
eight slides are still skeletons. Each slide has a live date and 24-hour `hh:mm`
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
details: |
  Supporting details in Markdown.
image: /pics/example.png
alt: A description of the figure.
caption: A short caption.
```

For multiple selectable figures, add `visuals` entries with `id`, `label`, and
the same optional `image`, `alt`, and `caption` fields. Without JavaScript, all
slides and visuals appear in sequence. Styles and scripts are loaded only on
pages with `presentation: true` in their front matter.

## FluxFormer figures

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
