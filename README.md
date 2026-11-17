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

## FluxFormer walkthrough

`/fluxformer/` follows four recorded E. coli core conditions through token
initialization, attention, embedding/scalar propagation, the feed-forward
network, three-layer refinement, and final flux comparisons. Navigation is
Home → FluxFormer → Research → Blog. The walkthrough uses the site's existing
dark/light themes and native HTML, CSS, and JavaScript modules.

Only a small overview loads initially. Full 95-reaction layer traces load on
demand, with two decoded layers cached. Visitors can select conditions,
reactions, layers, heads, and individual features, and expand numerical details.
They cannot edit nutrient bounds or run model inference in the browser.

The public `data/fluxformer/` files are lossless presentation copies of the
verified traces under `_tools/fluxformer_traces/examples/`. To regenerate them:

```sh
python3 _tools/fluxformer_traces/prepare_web_data.py
```

See [_tools/fluxformer_traces/WEB.md](_tools/fluxformer_traces/WEB.md) for the
data contract, testing commands, browser requirements, and payload sizes.
`fluxformer.html`, `css/fluxformer.css`, and `js/fluxformer*.js` implement the page;
its assets load only on the FluxFormer page.
