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
- `research.md` is the placeholder for papers, results, and research projects.

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
