(function () {
  'use strict';
  var root = document.documentElement;
  var control = document.querySelector('.theme-control');
  var select = document.getElementById('theme-select');
  var theme = root.dataset.theme || 'dark';

  function applyTheme() {
    root.dataset.theme = theme;
    if (select) select.value = theme;
  }

  applyTheme();
  if (control && select) {
    control.hidden = false;
    select.addEventListener('change', function () {
      theme = select.value;
      try {
        localStorage.setItem('color-theme', theme);
      } catch (error) { /* The selected theme still works for this page. */ }
      applyTheme();
    });
  }
  window.addEventListener('storage', function (event) {
    if (event.key !== 'color-theme' && event.key !== null) return;
    theme = event.newValue === 'light' || event.newValue === 'dark' ? event.newValue : 'dark';
    applyTheme();
  });
}());
