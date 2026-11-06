(function () {
  'use strict';

  var deck = document.querySelector('[data-presentation]');
  if (!deck) return;

  var slides = Array.from(deck.querySelectorAll('[data-slide]'));
  if (!slides.length) return;

  var stage = deck.querySelector('[data-presentation-stage]');
  var overview = deck.querySelector('[data-presentation-overview]');
  var overviewToggle = deck.querySelector('[data-overview-toggle]');
  var overviewCards = Array.from(deck.querySelectorAll('[data-go-to]'));
  var previous = deck.querySelector('[data-slide-previous]');
  var next = deck.querySelector('[data-slide-next]');
  var jump = deck.querySelector('[data-slide-jump]');
  var status = deck.querySelector('[data-slide-status]');
  var progress = deck.querySelector('[data-slide-progress]');
  var fullscreenToggle = deck.querySelector('[data-fullscreen-toggle]');
  var dialog = deck.querySelector('[data-visual-dialog]');
  var expandedVisual = deck.querySelector('[data-expanded-visual]');
  var current = 0;
  var overviewOpen = false;
  var expanded = false;
  var backgroundElements = [];
  var expandTrigger = null;

  function indexFor(id) {
    return slides.findIndex(function (slide) { return slide.dataset.slide === id; });
  }

  function indexFromURL() {
    var match = /^#slide-([a-z0-9-]+)$/.exec(window.location.hash);
    return match ? indexFor(match[1]) : -1;
  }

  function closeOverview() {
    overviewOpen = false;
    overview.hidden = true;
    stage.hidden = false;
    overviewToggle.setAttribute('aria-expanded', 'false');
    overviewToggle.textContent = 'Overview';
  }

  function showSlide(index, options) {
    options = options || {};
    current = Math.max(0, Math.min(index, slides.length - 1));
    closeOverview();
    slides.forEach(function (slide, i) { slide.hidden = i !== current; });
    var slide = slides[current];
    var id = slide.dataset.slide;
    deck.dataset.currentSlide = id;
    jump.value = id;
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    progress.value = current + 1;
    status.textContent = (current + 1) + ' / ' + slides.length + ' · ' + slide.querySelector('h3').textContent;
    overviewCards.forEach(function (card) {
      if (card.dataset.goTo === id) card.setAttribute('aria-current', 'step');
      else card.removeAttribute('aria-current');
    });
    if (options.updateURL && window.location.hash !== '#slide-' + id) {
      window.history.pushState(null, '', '#slide-' + id);
    }
    if (options.focus) slide.focus({ preventScroll: true });
  }

  function navigate(index) {
    showSlide(index, { updateURL: true, focus: true });
  }

  function restoreFromURL() {
    var index = indexFromURL();
    showSlide(index < 0 ? 0 : index, { focus: deck.contains(document.activeElement) });
  }

  previous.addEventListener('click', function () { navigate(current - 1); });
  next.addEventListener('click', function () { navigate(current + 1); });
  jump.addEventListener('change', function () { navigate(indexFor(jump.value)); });
  overviewCards.forEach(function (card) {
    card.addEventListener('click', function () { navigate(indexFor(card.dataset.goTo)); });
  });
  overviewToggle.addEventListener('click', function () {
    if (overviewOpen) {
      closeOverview();
      overviewToggle.focus();
    } else {
      overviewOpen = true;
      overview.hidden = false;
      stage.hidden = true;
      overviewToggle.setAttribute('aria-expanded', 'true');
      overviewToggle.textContent = 'Close overview';
      overviewCards[current].focus();
    }
  });

  var start = document.querySelector('[data-presentation-start]');
  if (start) start.addEventListener('click', function (event) {
    event.preventDefault();
    navigate(0);
    deck.scrollIntoView({ block: 'start' });
  });

  window.addEventListener('popstate', restoreFromURL);
  window.addEventListener('hashchange', restoreFromURL);

  function updateFullscreenControl() {
    var active = document.fullscreenElement === deck || expanded;
    fullscreenToggle.textContent = active ? 'Exit fullscreen' : 'Present fullscreen';
    fullscreenToggle.setAttribute('aria-pressed', String(active));
  }

  // Expanded mode also works when a browser or embed disallows native fullscreen.
  function setExpanded(value) {
    expanded = value;
    deck.classList.toggle('is-expanded', value);
    document.body.classList.toggle('presentation-expanded', value);
    if (value) {
      var element = deck;
      while (element.parentElement && element !== document.body) {
        Array.from(element.parentElement.children).forEach(function (sibling) {
          if (sibling !== element) {
            backgroundElements.push({ element: sibling, inert: sibling.inert });
            sibling.inert = true;
          }
        });
        element = element.parentElement;
      }
      deck.focus({ preventScroll: true });
    } else {
      backgroundElements.forEach(function (item) { item.element.inert = item.inert; });
      backgroundElements = [];
      fullscreenToggle.focus({ preventScroll: true });
    }
    updateFullscreenControl();
  }

  fullscreenToggle.addEventListener('click', async function () {
    if (expanded) {
      setExpanded(false);
      return;
    }
    try {
      if (document.fullscreenElement === deck) await document.exitFullscreen();
      else if (deck.requestFullscreen && document.fullscreenEnabled) await deck.requestFullscreen();
      else setExpanded(true);
    } catch (error) {
      setExpanded(true);
    }
  });
  document.addEventListener('fullscreenchange', function () {
    updateFullscreenControl();
    if (document.fullscreenElement === deck) slides[current].focus({ preventScroll: true });
    else fullscreenToggle.focus({ preventScroll: true });
  });

  deck.addEventListener('keydown', function (event) {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.target.closest('input, textarea, select, [contenteditable], summary, a, dialog')) return;
    if (event.key === 'Escape') {
      if (overviewOpen) {
        closeOverview();
        overviewToggle.focus();
      } else if (expanded) setExpanded(false);
      return;
    }
    if (overviewOpen) return;
    var target;
    if (event.key === 'ArrowRight' || event.key === 'PageDown') target = current + 1;
    if (event.key === 'ArrowLeft' || event.key === 'PageUp') target = current - 1;
    if (event.key === 'Home') target = 0;
    if (event.key === 'End') target = slides.length - 1;
    if (target !== undefined) {
      event.preventDefault();
      navigate(target);
    }
  });

  var touch = null;
  stage.addEventListener('touchstart', function (event) {
    touch = null;
    if (event.touches.length !== 1 || event.target.closest('button, select, summary, a, input, textarea')) return;
    touch = { x: event.touches[0].clientX, y: event.touches[0].clientY, time: Date.now() };
  }, { passive: true });
  stage.addEventListener('touchend', function (event) {
    if (!touch || event.changedTouches.length !== 1) return;
    var dx = event.changedTouches[0].clientX - touch.x;
    var dy = event.changedTouches[0].clientY - touch.y;
    if (Date.now() - touch.time < 800 && Math.abs(dx) > 70 && Math.abs(dx) > Math.abs(dy) * 1.5) {
      navigate(current + (dx < 0 ? 1 : -1));
    }
    touch = null;
  }, { passive: true });
  stage.addEventListener('touchcancel', function () { touch = null; }, { passive: true });

  slides.forEach(function (slide) {
    var panels = Array.from(slide.querySelectorAll('[data-visual]'));
    var select = slide.querySelector('[data-figure-select]');
    function showVisual(id) {
      panels.forEach(function (panel) { panel.hidden = panel.dataset.visual !== id; });
    }
    if (select) {
      showVisual(select.value);
      select.addEventListener('change', function () { showVisual(select.value); });
      slide.querySelector('[data-figure-control]').hidden = false;
    }
    var expandButton = slide.querySelector('[data-expand-visual]');
    if (dialog && typeof dialog.showModal === 'function') {
      expandButton.hidden = false;
      expandButton.addEventListener('click', function () {
        var panel = panels.find(function (item) { return !item.hidden; });
        if (!panel) return;
        var copy = panel.cloneNode(true);
        copy.removeAttribute('id');
        copy.querySelectorAll('[id]').forEach(function (item) { item.removeAttribute('id'); });
        copy.querySelectorAll('img').forEach(function (img) { img.loading = 'eager'; });
        expandedVisual.replaceChildren(copy);
        deck.querySelector('#visual-dialog-title').textContent = slide.querySelector('h3').textContent;
        expandTrigger = expandButton;
        dialog.showModal();
      });
    }
  });
  dialog.addEventListener('close', function () {
    expandedVisual.replaceChildren();
    if (expandTrigger) expandTrigger.focus({ preventScroll: true });
  });

  deck.querySelectorAll('[data-presentation-controls]').forEach(function (control) { control.hidden = false; });
  progress.hidden = false;
  deck.dataset.enhanced = 'true';
  restoreFromURL();
}());
