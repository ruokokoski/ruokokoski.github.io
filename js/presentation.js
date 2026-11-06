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
  var fullscreenError = deck.querySelector('[data-fullscreen-error]');
  var dialog = deck.querySelector('[data-visual-dialog]');
  var expandedVisual = deck.querySelector('[data-expanded-visual]');
  var current = 0;
  var overviewOpen = false;
  var expandTrigger = null;
  var exitingPresentation = false;

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
    slide.querySelector('.slide-content').scrollTop = 0;
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
    showSlide(index < 0 ? 0 : index, { focus: deck.contains(document.activeElement) || stage.contains(document.activeElement) });
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

  window.addEventListener('popstate', restoreFromURL);
  window.addEventListener('hashchange', restoreFromURL);

  function updateFullscreenControl() {
    var active = document.fullscreenElement === stage;
    deck.classList.toggle('is-presenting', active);
    stage.classList.toggle('is-presenting', active);
    if (active) closeOverview();
    fullscreenToggle.textContent = active ? 'Exit fullscreen' : 'Present fullscreen';
    fullscreenToggle.setAttribute('aria-pressed', String(active));
  }

  async function exitPresentation() {
    if (exitingPresentation) return;
    exitingPresentation = true;
    try {
      if (document.fullscreenElement === stage) await document.exitFullscreen();
    } finally {
      exitingPresentation = false;
    }
  }

  fullscreenToggle.addEventListener('click', async function () {
    fullscreenError.hidden = true;
    closeOverview();
    try {
      if (document.fullscreenElement === stage) await document.exitFullscreen();
      else if (stage.requestFullscreen && document.fullscreenEnabled) {
        await stage.requestFullscreen({ navigationUI: 'hide' });
        if (document.fullscreenElement !== stage) throw new Error('Fullscreen did not activate');
      } else throw new Error('Fullscreen unavailable');
    } catch (error) {
      fullscreenError.querySelector('a').href = window.location.href;
      fullscreenError.hidden = false;
    }
  });
  document.addEventListener('fullscreenchange', function () {
    updateFullscreenControl();
    if (document.fullscreenElement === stage) {
      stage.scrollTop = 0;
      slides[current].scrollTop = 0;
      slides[current].querySelector('.slide-content').scrollTop = 0;
      slides[current].focus({ preventScroll: true });
    } else fullscreenToggle.focus({ preventScroll: true });
  });

  document.addEventListener('keydown', function (event) {
    if (!deck.contains(event.target) && !stage.contains(event.target)) return;
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.target.closest('dialog')) return;
    if (event.key === 'Escape') {
      if (overviewOpen) {
        closeOverview();
        overviewToggle.focus();
      } else if (document.fullscreenElement === stage) exitPresentation();
      return;
    }
    if (event.target.closest('input, textarea, select, [contenteditable], summary, a')) return;
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
  var lastTap = null;
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
      lastTap = null;
    } else if (deck.classList.contains('is-presenting') && Date.now() - touch.time < 300 && Math.abs(dx) < 12 && Math.abs(dy) < 12) {
      if (lastTap && Date.now() - lastTap.time < 350 && Math.abs(touch.x - lastTap.x) < 30 && Math.abs(touch.y - lastTap.y) < 30) {
        exitPresentation();
        lastTap = null;
      } else lastTap = { x: touch.x, y: touch.y, time: Date.now() };
    } else {
      lastTap = null;
    }
    touch = null;
  }, { passive: true });
  stage.addEventListener('touchcancel', function () { touch = null; }, { passive: true });
  stage.addEventListener('dblclick', function (event) {
    if (deck.classList.contains('is-presenting') && !event.target.closest('a, button, input, textarea, select, [contenteditable]')) exitPresentation();
  });

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
    if (expandButton && dialog && typeof dialog.showModal === 'function') {
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

  var dateFormat = new Intl.DateTimeFormat('fi-FI', {
    timeZone: 'Europe/Helsinki', day: '2-digit', month: '2-digit', year: 'numeric'
  });
  var timeFormat = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Europe/Helsinki', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
  });
  var dates = Array.from(stage.querySelectorAll('[data-slide-date]'));
  var times = Array.from(stage.querySelectorAll('[data-slide-time]'));
  function updateClock() {
    var now = new Date();
    var date = dateFormat.format(now);
    var time = timeFormat.format(now);
    dates.forEach(function (element) {
      if (element.textContent !== date) {
        element.textContent = date;
        var parts = dateFormat.formatToParts(now);
        element.dateTime = ['year', 'month', 'day'].map(function (type) {
          return parts.find(function (part) { return part.type === type; }).value;
        }).join('-');
      }
    });
    times.forEach(function (element) {
      if (element.textContent !== time) {
        element.textContent = time;
        element.dateTime = time;
      }
    });
  }
  updateClock();
  window.setInterval(updateClock, 1000);
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) updateClock();
  });
  restoreFromURL();
}());
