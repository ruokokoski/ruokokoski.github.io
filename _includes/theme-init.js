(function () {
  var theme = 'dark';
  try {
    var saved = localStorage.getItem('color-theme');
    if (saved === 'light' || saved === 'dark') theme = saved;
  } catch (error) { /* Keep the default dark theme when storage is blocked. */ }
  document.documentElement.dataset.theme = theme;
}());
