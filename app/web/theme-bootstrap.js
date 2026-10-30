(function() {
  try {
    var theme = localStorage.getItem("knight-theme") || "dark";
    document.documentElement.setAttribute("data-theme", theme);
  } catch (_) {}
})();
